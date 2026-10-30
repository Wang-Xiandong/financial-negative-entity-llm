"""
大模型推理 + 评估：零样本 / 少样本 / LoRA 微调后，统一在 test.csv 上算官方指标

GPU 上运行（在 Project_NLP 目录下）：
    M=/root/autodl-tmp/models
    python src/llm_infer.py --model $M/Qwen2.5-1.5B-Instruct --tag qwen1.5b_zero
    python src/llm_infer.py --model $M/Qwen2.5-1.5B-Instruct --tag qwen1.5b_few --shots 4
    python src/llm_infer.py --model $M/Qwen2.5-7B-Instruct   --tag qwen7b_zero
    python src/llm_infer.py --model $M/Qwen2.5-7B-Instruct   --tag qwen7b_few  --shots 4
    python src/llm_infer.py --model $M/Qwen2.5-7B-Instruct   --tag qwen7b_qlora --adapter outputs/llm/qwen7b_qlora
本地调试（CPU + 0.5B 小模型，只跑几条确认流程）：
    python src/llm_infer.py --model Qwen/Qwen2.5-0.5B-Instruct --tag debug --limit 6 --bs 2
"""

import argparse
import json
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_data import build_messages, parse_output
from metrics import ROOT, load_texts, official_score, remove_nested

OUT = ROOT / "outputs"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)                 # 本地模型目录或 HF 模型名
    p.add_argument("--tag", required=True)                   # 结果文件名
    p.add_argument("--shots", type=int, default=0)           # 0 = 零样本；4 = 用 fewshot.json 里的 4 条示例
    p.add_argument("--adapter", default=None)                # LoRA 权重目录（LLaMA-Factory 的 output_dir）
    p.add_argument("--split", default="test")
    p.add_argument("--bs", type=int, default=16)             # 一次并行生成多少条
    p.add_argument("--max_new", type=int, default=64)        # 最多生成多少 token；答案只是几个实体名，64 足够
    p.add_argument("--limit", type=int, default=0)           # >0 时只跑前 N 条（调试）
    return p.parse_args()


def load_model(args, device):
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"          # 生成任务必须左填充：右边要接着往后生成，padding 放右边会插在提示和答案之间
    # Qwen 权重本身就是 bf16；"auto" 按原精度加载（本机 CPU 上转 fp32 时 transformers 5 会段错误）
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto").to(device)
    except TypeError:                  # LLaMA-Factory 可能装的是旧版 transformers，参数名还叫 torch_dtype
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(device)
    if args.adapter:
        from peft import PeftModel     # LLaMA-Factory 安装时会自带 peft
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()   # 把 B·A 加回原权重 W，推理时和普通模型一样快
    model.eval()
    return tok, model


@torch.no_grad()                       # 推理不需要梯度，省显存
def generate_all(prompts, tok, model, args, device):
    """按长度排序后分批生成：同一批长度相近，padding 少，速度快。返回与 prompts 同顺序的输出。"""
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    outputs = [None] * len(prompts)
    for b in range(0, len(order), args.bs):
        idx = order[b: b + args.bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=args.max_new,
                             do_sample=False,           # 贪心解码：结果确定、可复现
                             temperature=None, top_p=None, top_k=None,   # 关掉模型默认的采样参数，避免警告
                             pad_token_id=tok.pad_token_id)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]    # 去掉提示部分，只保留新生成的
        for i, text in zip(idx, tok.batch_decode(new_tokens, skip_special_tokens=True)):
            outputs[i] = text
        print(f"\r已生成 {min(b + args.bs, len(order))}/{len(order)}", end="", flush=True)
    print()
    return outputs


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gold = load_texts(args.split)
    if args.limit:
        gold = gold.head(args.limit)
    shots = json.load(open(ROOT / "data" / "llm" / "fewshot.json", encoding="utf-8"))[:args.shots]

    tok, model = load_model(args, device)
    has_ent = gold["entity"].map(len) > 0                    # 没有实体的文本（极少数）不用问模型，直接判"无"
    sub = gold[has_ent]
    # apply_chat_template：把对话转成模型训练时见过的格式（<|im_start|>user ... <|im_end|> 等特殊标记）
    # enable_thinking=False 只对 Qwen3 有效（关掉思考模式），Qwen2.5 会忽略这个参数
    prompts = [tok.apply_chat_template(build_messages(e, c, shots), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
               for e, c in zip(sub["entity"], sub["content"])]
    print(f"{len(prompts)} 条待生成；示例提示末尾：\n...{prompts[0][-300:]}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    raw = generate_all(prompts, tok, model, args, device)
    elapsed = time.perf_counter() - t0

    # ---- 解析 → 文本级预测 ----
    raw_of = dict(zip(sub["id"], raw))
    rows, n_halluc, n_items = [], 0, 0
    for i, ents in zip(gold["id"], gold["entity"]):
        text = raw_of.get(i, "无")
        keys, status, h = parse_output(text, ents)
        n_halluc += h
        n_items += len(keys) + h
        rows.append({"id": i, "raw": text, "status": status,
                     "negative": int(len(keys) > 0), "key_entity": keys})
    pred = pd.DataFrame(rows)
    pred_rule = pred.assign(key_entity=pred["key_entity"].map(remove_nested))

    status = pred.loc[has_ent.values, "status"].value_counts(normalize=True).round(4).to_dict()
    results = {
        "method": f"{args.model.rstrip('/').split('/')[-1]} | shots={args.shots} | adapter={args.adapter}",
        "args": vars(args),
        args.split: official_score(gold, pred),              # 主结果：提示词里已写了嵌套规则，不再额外后处理
        f"{args.split}+rule": official_score(gold, pred_rule),   # 参考：再套一次 metrics 里的规则
        "output_status": status,                             # ok / loose / none / invalid 的比例
        "invalid_ratio": status.get("invalid", 0.0),         # 无法解析的比例（单独报告）
        "hallucination_ratio": round(n_halluc / n_items, 4) if n_items else 0.0,  # 输出了列表外名称的比例
        "infer_time_s": round(elapsed, 1),
        "infer_ms_per_1k_texts": round(elapsed / len(prompts) * 1000 * 1000, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if device == "cuda" else None,
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))

    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    with open(OUT / "results" / f"{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pred.assign(key_entity=pred["key_entity"].map(";".join)).to_csv(
        OUT / "preds" / f"{args.tag}_{args.split}_outputs.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
