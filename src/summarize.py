"""
汇总：读取 outputs/results/*.json，生成所有方法的对比表 outputs/results/summary.md

句对模型（TF-IDF / BERT）记录的是"每千对"推理耗时，大模型记录的是"每千条文本"；
这里统一换算成每千条文本（test 平均每条文本 1753/744 ≈ 2.36 个实体对），才能放在同一列比较。

运行（在 Project_NLP 目录下）：
    python src/summarize.py
"""

import json

from metrics import ROOT, load_pairs, load_texts

RES = ROOT / "outputs" / "results"
GPU_YUAN_PER_H = 1.58                      # AutoDL RTX 4080 SUPER 按量计费单价

# (结果文件名, 表中显示的方法名)；按"传统 → 判别式微调 → 提示词 → 生成式微调"排列
ROWS = [
    ("tfidf", "TF-IDF + LR"),
    ("bert", "BERT 句对分类（全量微调）"),
    ("roberta", "RoBERTa-wwm 句对分类（全量微调）"),
    ("qwen1.5b_zero", "Qwen2.5-1.5B 零样本"),
    ("qwen1.5b_few", "Qwen2.5-1.5B 4-shot"),
    ("qwen7b_zero", "Qwen2.5-7B 零样本"),
    ("qwen7b_few", "Qwen2.5-7B 4-shot"),
    ("qwen7b_qlora", "Qwen2.5-7B QLoRA 微调"),
]


def load(tag):
    with open(RES / f"{tag}.json", encoding="utf-8") as f:
        return json.load(f)


def train_time_s(tag, r):
    """句对模型的训练耗时记在结果里；QLoRA 是 LLaMA-Factory 训练的，耗时在它自己的 train_results.json 里。"""
    if "train_time_s" in r:
        return r["train_time_s"]
    path = ROOT / "outputs" / "llm" / tag / "train_results.json"
    if path.exists():
        return json.load(open(path, encoding="utf-8"))["train_runtime"]
    return 0.0                                  # 提示词方法不训练


def fmt_time(s):
    if s == 0:
        return "—"
    return f"{s:.1f} s" if s < 60 else f"{s / 60:.1f} min"


def main():
    pairs_per_text = len(load_pairs("test")) / len(load_texts("test"))
    n_test = len(load_texts("test"))

    lines = ["| 方法 | F1 | F1_s | F1_e | 无效输出 | 幻觉 | 训练耗时 | 推理 s/千条文本 | 峰值显存 GB | GPU 费用 ¥ |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, name in ROWS:
        if not (RES / f"{tag}.json").exists():
            continue                            # 还没跑的实验跳过
        r = load(tag)
        t = r["test"]                           # 主结果：句对模型用 val 上选好的阈值/规则；大模型不加规则
        if "infer_ms_per_1k_pairs" in r:        # 句对模型：每千对 → 每千条文本
            infer_s = r["infer_ms_per_1k_pairs"] * pairs_per_text / 1000
        else:
            infer_s = r["infer_ms_per_1k_texts"] / 1000
        train_s = train_time_s(tag, r)
        mem = r.get("peak_gpu_mem_gb")
        # 费用 = (训练 + 在 test 上推理一遍) 的 GPU 时长 × 单价；TF-IDF 在 CPU 上跑，不计
        cost = (train_s + infer_s * n_test / 1000) / 3600 * GPU_YUAN_PER_H if mem else 0.0
        lines.append(" | ".join([
            f"| {name}", f"{t['F1']:.4f}", f"{t['F1_s']:.4f}", f"{t['F1_e']:.4f}",
            f"{r['invalid_ratio']:.1%}" if "invalid_ratio" in r else "—",
            f"{r['hallucination_ratio']:.1%}" if "hallucination_ratio" in r else "—",
            fmt_time(train_s), f"{infer_s:.1f}",
            f"{mem:.1f}" if mem else "—(CPU)", f"{cost:.2f}" if mem else "—"]) + " |")

    table = "\n".join(lines)
    note = ("\n\n注：test 集 744 条文本 / 1753 个（实体, 文本）对。句对模型的峰值显存是训练时的，"
            "大模型是推理时的（QLoRA 训练峰值约 19 GB）。\n")
    print(table)
    with open(RES / "summary.md", "w", encoding="utf-8") as f:
        f.write(table + note)
    print(f"\n已保存 {RES / 'summary.md'}")


if __name__ == "__main__":
    main()
