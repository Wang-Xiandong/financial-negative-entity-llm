"""
误差分析：模型在 test 上错在哪里

1. 切片统计：按"文本长度 / 实体是否在文本里 / 嵌套实体 / 实体个数"分组，比较各模型在每组里的错误率
2. 模型间对比：BERT 和 RoBERTa 错的是不是同一批样本（都错 → 难样本或标注噪声）
3. 导出错误样本：附上实体附近的上下文，用于人工抽查、归类原因
4. BERT vs QLoRA 大模型：错误重合、漏报/误报构成，以及人工归类的 BERT 错误里哪些类别被大模型修正

运行（在 Project_NLP 目录下）：
    python src/error_analysis.py
"""

import json

import pandas as pd

from metrics import ROOT, load_texts, pairs_to_pred

OUT = ROOT / "outputs" / "error_analysis"
MODELS = ["tfidf", "bert", "roberta"]           # 句对模型：输出逐对概率
LLM = "qwen7b_qlora"                            # 生成式模型：输出文本级的负面主体名单
SAMPLE_MODEL = "bert"                           # 从哪个模型的错误里抽样人工查看
N_SAMPLE = 40
SEED = 42


def final_decisions(tag, gold):
    """
    读取某个模型在 test 上的逐对概率，按它在 val 上选好的阈值和规则，
    得到每个（实体, 文本）对的【最终】判断（后处理之后的），和标准答案对比。
    """
    cfg = json.load(open(ROOT / "outputs" / "results" / f"{tag}.json", encoding="utf-8"))
    pairs = pd.read_csv(ROOT / "outputs" / "preds" / f"{tag}_test_pairs.csv",
                        encoding="utf-8-sig", dtype={"id": str})
    pred = pairs_to_pred(pairs, gold["id"], threshold=cfg["threshold"], use_rule=cfg["use_rule"])
    key_of = dict(zip(pred["id"], pred["key_entity"].map(set)))   # {id: 预测的负面主体集合}
    pairs["pred"] = [int(e in key_of[i]) for i, e in zip(pairs["id"], pairs["entity"])]
    pairs["wrong"] = (pairs["pred"] != pairs["label"]).astype(int)
    return pairs


def llm_wrong(tag, pairs):
    """
    大模型一次输出整段文本的负面主体名单，没有逐对概率。
    把名单拆回（实体, 文本）对：实体在名单里 → 预测为 1，否则 0，就能和句对模型逐对比较。
    """
    out = pd.read_csv(ROOT / "outputs" / "preds" / f"{tag}_test_outputs.csv", encoding="utf-8-sig",
                      dtype={"id": str}, keep_default_na=False)       # keep_default_na：空名单读成 ""，不是 NaN
    key_of = {i: set(k.split(";")) - {""} for i, k in zip(out["id"], out["key_entity"])}
    pred = [int(e in key_of.get(i, set())) for i, e in zip(pairs["id"], pairs["entity"])]
    return (pd.Series(pred, index=pairs.index) != pairs["label"]).astype(int)


def add_slices(pairs, gold):
    """给每个对打上几个"可能导致出错"的属性标签。"""
    ents_of = dict(zip(gold["id"], gold["entity"]))
    n_ent = pairs["id"].map(lambda i: len(ents_of[i]))
    L = pairs["content"].str.len()
    pairs["长度"] = pd.cut(L, [0, 256, 500, 10 ** 6], labels=["≤256", "256-500", ">500(会被截断)"])
    pairs["实体在文本中"] = [e in c for e, c in zip(pairs["entity"], pairs["content"])]
    pairs["嵌套关系"] = [
        "是短名(被包含)" if any(e != o and e in o for o in ents_of[i])
        else "是长名(包含别人)" if any(e != o and o in e for o in ents_of[i])
        else "无"
        for e, i in zip(pairs["entity"], pairs["id"])]
    pairs["实体个数"] = pd.cut(n_ent, [0, 1, 3, 100], labels=["1", "2-3", "≥4"])
    return pairs


def context(entity, content, w=60):
    """实体第一次出现位置前后各 w 个字；用【】把实体标出来，方便人工看。"""
    p = content.find(entity)
    if p == -1:
        return "（实体未出现）" + content[:2 * w]
    return content[max(0, p - w): p] + "【" + entity + "】" + content[p + len(entity): p + len(entity) + w]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gold = load_texts("test")

    # ---- 1. 三个模型的逐对最终判断，合到一张表 ----
    base = None
    for tag in MODELS:
        p = final_decisions(tag, gold)
        if base is None:
            base = add_slices(p[["id", "entity", "content", "label"]].copy(), gold)
        base[f"wrong_{tag}"] = p["wrong"].values             # 三个文件的行顺序相同（都来自 test_pairs.csv）
        base[f"prob_{tag}"] = p["prob"].round(3).values
    base["wrong_qlora"] = llm_wrong(LLM, base)

    # ---- 2. 切片错误率 ----
    wrong_cols = [f"wrong_{t}" for t in MODELS] + ["wrong_qlora"]
    tables = []
    for col in ["长度", "实体在文本中", "嵌套关系", "实体个数", "label"]:
        t = base.groupby(col, observed=True)[wrong_cols].mean().round(3)   # 每组的平均错误率
        t.insert(0, "样本数", base.groupby(col, observed=True).size())
        t.index = [f"{col}={v}" for v in t.index]
        tables.append(t)
    slices = pd.concat(tables)
    slices.loc["全部"] = [len(base)] + base[wrong_cols].mean().round(3).tolist()
    print("各切片的错误率（逐对，后处理之后）：\n", slices.to_string(), "\n")
    slices.to_csv(OUT / "slices.csv", encoding="utf-8-sig")

    # ---- 3. BERT 和 RoBERTa 错误的重合程度 ----
    b, r = base["wrong_bert"] == 1, base["wrong_roberta"] == 1
    overlap = {"BERT错": int(b.sum()), "RoBERTa错": int(r.sum()), "两个都错": int((b & r).sum()),
               "只BERT错": int((b & ~r).sum()), "只RoBERTa错": int((~b & r).sum())}
    print("错误重合：", overlap, "\n")

    # ---- 4. 导出错误样本，供人工归类 ----
    errs = base[base[f"wrong_{SAMPLE_MODEL}"] == 1].copy()
    errs["错误类型"] = errs["label"].map({1: "漏报FN(应为负面主体)", 0: "误报FP(不应是负面主体)"})
    errs["上下文"] = [context(e, c) for e, c in zip(errs["entity"], errs["content"])]
    gold_key = dict(zip(gold["id"], gold["key_entity"].map(";".join)))
    gold_ent = dict(zip(gold["id"], gold["entity"].map(";".join)))
    errs["全部实体"] = errs["id"].map(gold_ent)
    errs["标准答案主体"] = errs["id"].map(gold_key)
    errs["两个BERT都错"] = errs["wrong_roberta"] == 1
    errs["原因(人工填写)"] = ""
    cols = ["id", "entity", "错误类型", f"prob_{SAMPLE_MODEL}", "两个BERT都错", "长度", "嵌套关系",
            "全部实体", "标准答案主体", "上下文", "原因(人工填写)"]
    errs[cols].to_csv(OUT / f"{SAMPLE_MODEL}_all_errors.csv", index=False, encoding="utf-8-sig")
    print(f"{SAMPLE_MODEL} 共 {len(errs)} 个错误对：", errs["错误类型"].value_counts().to_dict())
    sample_path = OUT / f"{SAMPLE_MODEL}_errors_sample.csv"
    if sample_path.exists():                       # 已经人工填写过原因，不能覆盖
        print(f"{sample_path.name} 已存在（含人工归类），跳过导出\n")
    else:
        sample = errs.sample(n=min(N_SAMPLE, len(errs)), random_state=SEED)
        sample[cols].to_csv(sample_path, index=False, encoding="utf-8-sig")
        print(f"已导出 {sample_path.name}（{len(sample)} 条，供人工归类）\n")

    # ---- 5. BERT vs QLoRA：错误重合 + 漏报/误报构成 ----
    b, q = base["wrong_bert"] == 1, base["wrong_qlora"] == 1
    kind = base["label"].map({1: "漏报FN", 0: "误报FP"})              # 错的对：标签为 1 就是漏报，为 0 就是误报
    groups = {"BERT错": b, "QLoRA错": q, "两个都错": b & q,
              "只BERT错(QLoRA修正)": b & ~q, "只QLoRA错(新增)": ~b & q}
    cmp = pd.DataFrame({name: {"对数": int(m.sum()), **kind[m].value_counts().to_dict()}
                        for name, m in groups.items()}).T.fillna(0).astype(int)
    print("BERT vs QLoRA 错误对比：\n", cmp.to_string(), "\n")
    cmp.to_csv(OUT / "bert_vs_qlora.csv", encoding="utf-8-sig")

    # ---- 6. 人工归类的 BERT 错误样本，QLoRA 修正了哪些类别 ----
    s = pd.read_csv(sample_path, encoding="utf-8-sig", dtype={"id": str})
    s = s.merge(base[["id", "entity", "wrong_qlora"]], on=["id", "entity"], how="left")
    s["类别"] = s["原因(人工填写)"].str.split("[：:（(]").str[0]      # 只取冒号/括号前的类别名
    by_cat = s.groupby("类别").agg(样本数=("wrong_qlora", "size"), QLoRA仍错=("wrong_qlora", "sum"))
    by_cat["QLoRA修正"] = by_cat["样本数"] - by_cat["QLoRA仍错"]
    by_cat = by_cat.sort_values("QLoRA修正", ascending=False)
    print("人工归类的 BERT 错误，QLoRA 修正情况：\n", by_cat.to_string())
    by_cat.to_csv(OUT / "qlora_fix_by_category.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
