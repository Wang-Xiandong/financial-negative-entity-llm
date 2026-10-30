"""
BERT 实体级句对分类微调：输入 [CLS] 实体 [SEP] 文本 [SEP]，判断该实体是否为负面主体

本地小批量测试（CPU，只验证流程能跑通，不看分数）：
    python src/train_bert.py --debug 200 --max_len 128 --epochs 1 --bs 8
    注：本机 CPU 版 torch 在 batch=32、长度 128 时会段错误（batch 8/16 正常，与代码无关），所以本地用 --bs 8

GPU 上正式训练：
    python src/train_bert.py                                          # bert-base-chinese
    python src/train_bert.py --model hfl/chinese-roberta-wwm-ext --tag roberta  # 换模型只需改模型名

国内下载模型慢的话，运行前设置镜像：  set HF_ENDPOINT=https://hf-mirror.com （Linux: export）
"""

import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from metrics import (ROOT, load_pairs, load_texts, official_score, pairs_to_pred, save_submission,
                     tune_on_val)

OUT = ROOT / "outputs"


# ----------------------------------------------------------------------------
# 1. 命令行参数：同一份代码，本地调试和 GPU 正式训练只改参数
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bert-base-chinese")    # HuggingFace 上的模型名
    p.add_argument("--tag", default="bert")                    # 结果文件名前缀，区分不同模型
    p.add_argument("--max_len", type=int, default=512)         # 由长度分布决定：512 只截断 7.4%
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)           # BERT 微调的常用学习率，太大会破坏预训练权重
    p.add_argument("--bs", type=int, default=32)               # batch size
    p.add_argument("--warmup", type=float, default=0.1)        # 前 10% 的步数学习率从 0 线性升到 lr
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", type=int, default=0)             # >0 时只用这么多训练对，快速跑通流程
    return p.parse_args()


# ----------------------------------------------------------------------------
# 2. 数据：DataFrame → 分词 → torch Dataset
# ----------------------------------------------------------------------------
def crop_around_entity(entity, content, max_chars):
    """
    文本太长时，直接截断可能把实体截掉（实体在第 600 个字，max_len=512 就看不到了）。
    所以：实体在前面 → 正常截前 max_chars 个字；实体在后面 → 以实体为中心截一段。
    中文 BERT 基本一字一个 token，所以按字数估算即可。
    """
    if len(content) <= max_chars:
        return content
    p = content.find(entity)
    if p == -1 or p + len(entity) <= max_chars:                # 找不到，或实体本来就在截断范围内
        return content[:max_chars]
    start = p - max_chars // 3                                  # 实体放在窗口前 1/3 处，后文留得多一些
    return content[start: start + max_chars]


class PairDataset(torch.utils.data.Dataset):
    """
    和 d2l 里的 SNLI 数据集类似：初始化时一次性把所有句对分好词，__getitem__ 只按下标取。
    分词器会自动生成：
        input_ids       每个 token 在词表里的编号，含 [CLS]=101、[SEP]=102
        token_type_ids  片段编号：实体部分为 0，文本部分为 1（对应 d2l 的 segments）
        attention_mask  1 = 真实 token，0 = padding（对应 d2l 的 valid_len）
    """

    def __init__(self, pairs, tokenizer, max_len):
        # 给 [CLS] [SEP] [SEP] 和实体本身留位置，剩下的给文本
        contents = [crop_around_entity(e, c, max_len - len(e) - 3)
                    for e, c in zip(pairs["entity"], pairs["content"])]
        # 传两个列表 → 句对模式；truncation="only_second"：超长时只截第二句（文本），实体名保持完整
        self.enc = tokenizer(list(pairs["entity"]), contents,
                             truncation="only_second", max_length=max_len)
        # 官方测试集没有标签 → None。注意不能用 -1 占位：样本里只要有 labels 字段，
        # Trainer.predict 也会拿它算交叉熵，-1 不在 [0, 2) 范围内会触发 CUDA 断言
        self.labels = pairs["label"].tolist() if "label" in pairs else None

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}           # 第 i 个样本的 input_ids 等
        if self.labels is not None:
            item["labels"] = self.labels[i]                     # Trainer 约定标签字段名叫 labels
        return item                                             # 这里还没 padding，由 collator 按 batch 补齐


def subset(pairs, gold, n_texts, seed):
    """调试模式：随机抽 n_texts 条文本，pairs 和标准答案同步取子集（否则官方指标会对不上）。"""
    ids = gold["id"].sample(n=min(n_texts, len(gold)), random_state=seed)
    return pairs[pairs["id"].isin(ids)].reset_index(drop=True), gold[gold["id"].isin(ids)]


# ----------------------------------------------------------------------------
# 3. 训练中每个 epoch 在 val 上的评估（实体级 F1，用来挑最好的 epoch）
# ----------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred                                  # logits: (样本数, 2)
    preds = logits.argmax(axis=-1)                              # 两个 logit 里大的那个就是预测类别
    return {"f1": f1_score(labels, preds), "acc": float((preds == labels).mean())}


def softmax_pos(logits):
    """logits → label=1 的概率。减去每行最大值再 exp，防止数值溢出（结果不变）。"""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e[:, 1] / e.sum(axis=1))


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    args = get_args()
    set_seed(args.seed)                                         # 固定 python/numpy/torch 的随机种子
    use_gpu = torch.cuda.is_available()
    print(f"设备：{'GPU ' + torch.cuda.get_device_name(0) if use_gpu else 'CPU'}")

    # ---- 数据 ----
    train, val, test = load_pairs("train"), load_pairs("val"), load_pairs("test")
    gold_val, gold_test = load_texts("val"), load_texts("test")
    if args.debug:
        train = train.sample(n=args.debug, random_state=args.seed).reset_index(drop=True)
        val, gold_val = subset(val, gold_val, args.debug // 4, args.seed)
        test, gold_test = subset(test, gold_test, args.debug // 4, args.seed)
    print(f"训练对 {len(train)}，val 对 {len(val)}，test 对 {len(test)}")

    # ---- 模型和分词器 ----
    # 第一次运行会从 HuggingFace 下载（约 400MB），之后从本地缓存读取
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 加载预训练 BERT + 一个随机初始化的分类层 Linear(768, 2)
    # 会打印 "Some weights ... newly initialized: classifier.weight/bias"，这是正常的：分类层就是要从零训练
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    ds_train = PairDataset(train, tokenizer, args.max_len)
    ds_val = PairDataset(val, tokenizer, args.max_len)
    ds_test = PairDataset(test, tokenizer, args.max_len)
    # 打印一条分词结果，对照 d2l：[CLS] 实体 [SEP] 文本 [SEP]，token_type_ids 前 0 后 1
    print("示例输入：", tokenizer.decode(ds_train[0]["input_ids"][:60]))
    print("token_type_ids：", ds_train[0]["token_type_ids"][:60])

    # ---- 训练配置 ----
    ckpt_dir = OUT / "checkpoints" / args.tag
    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,                # 评估不需要存梯度，可以用更大的 batch
        warmup_steps=args.warmup,                               # transformers 5.x：传 0~1 的小数表示占总步数的比例
        weight_decay=0.01,                                      # AdamW 的权重衰减（Trainer 默认优化器就是 AdamW）
        fp16=use_gpu,                                           # 半精度：GPU 上快约一倍、显存减半；CPU 不支持
        eval_strategy="epoch",                                  # 每个 epoch 结束在 val 上评估一次
        # 每个 epoch 存一次 checkpoint，训练结束自动加载 val 上 f1 最好的那个；调试模式不存，省时间和硬盘
        save_strategy="no" if args.debug else "epoch",
        load_best_model_at_end=not args.debug,
        metric_for_best_model="f1",
        save_total_limit=1,                                     # 只保留最好的一个 checkpoint（每个约 400MB）
        logging_steps=10 if args.debug else 50,                 # 每隔多少步打印一次 loss
        report_to="none",                                       # 不上传到 wandb 等平台
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        # 动态 padding：每个 batch 只补齐到该 batch 内最长的长度，而不是统一补到 max_len，快很多
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # ---- 训练 ----
    if use_gpu:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()                    # 内部就是：取 batch → 前向 → 交叉熵 → 反向 → AdamW 更新 → 调整学习率
    train_time = time.perf_counter() - t0

    # ---- 预测 val → 选阈值和规则；预测 test → 最终分数 ----
    val["prob"] = softmax_pos(trainer.predict(ds_val).predictions)
    best_t, use_rule = tune_on_val(val, gold_val)
    to_pred = lambda p, ids, t=best_t: pairs_to_pred(p, ids, threshold=t, use_rule=use_rule)

    t0 = time.perf_counter()
    test["prob"] = softmax_pos(trainer.predict(ds_test).predictions)
    infer_ms_per_1k = (time.perf_counter() - t0) / len(test) * 1000 * 1000

    results = {
        "method": f"{args.model} 句对微调",
        "args": vars(args),                                     # 把所有超参数一起记下来，方便复现
        "threshold": best_t,
        "use_rule": use_rule,
        "val": official_score(gold_val, to_pred(val, gold_val["id"])),
        "test@0.5": official_score(gold_test, to_pred(test, gold_test["id"], 0.5)),
        "test": official_score(gold_test, to_pred(test, gold_test["id"])),
        "train_time_s": round(train_time, 1),
        "infer_ms_per_1k_pairs": round(infer_ms_per_1k, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if use_gpu else None,
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # ---- 保存 ----
    suffix = "_debug" if args.debug else ""                     # 调试结果单独命名，不覆盖正式结果
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    with open(OUT / "results" / f"{args.tag}{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    test.to_csv(OUT / "preds" / f"{args.tag}{suffix}_test_pairs.csv", index=False, encoding="utf-8-sig")

    if not args.debug:
        trainer.save_model(str(OUT / "models" / args.tag))      # 保存验证集上最好的模型
        tokenizer.save_pretrained(str(OUT / "models" / args.tag))
        # 官方测试集 → 提交文件
        official = load_pairs("official_test")
        official["prob"] = softmax_pos(
            trainer.predict(PairDataset(official, tokenizer, args.max_len)).predictions)
        ids = load_texts("official_test")["id"]
        (OUT / "submissions").mkdir(parents=True, exist_ok=True)
        save_submission(to_pred(official, ids), OUT / "submissions" / f"{args.tag}_submit.csv")


if __name__ == "__main__":
    main()
