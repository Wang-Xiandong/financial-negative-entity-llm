"""
评估：后处理规则 + 官方指标（F1_s / F1_e / F1）

所有方法（TF-IDF、BERT、RoBERTa、Qwen）都用这个文件评估，保证对比公平。
典型用法（训练/推理完成之后）：

    from metrics import load_texts, pairs_to_pred, official_score
    gold = load_texts("test")                         # 标准答案（文本级）
    pairs["prob"] = model_predict(pairs)              # 模型给每个（实体, 文本）对的负面概率
    pred = pairs_to_pred(pairs, gold["id"])           # 对 → 文本级预测（含后处理）
    print(official_score(gold, pred))

直接运行本文件会演示一个手工小例子，并在 val 上跑几个"假模型"作为分数参照：
    python src/metrics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"


# ----------------------------------------------------------------------------
# 读取数据
# ----------------------------------------------------------------------------
def parse_entities(s):
    """'A;B' → ['A', 'B']；空值（没有负面主体时 csv 里是空）→ []"""
    return [e for e in s.split(";") if e] if isinstance(s, str) else []


def load_texts(split):
    """读取文本级数据（train/val/test），把 entity、key_entity 转回 list。"""
    df = pd.read_csv(DATA / f"{split}.csv", encoding="utf-8-sig", dtype={"id": str})
    for col in ["entity", "key_entity"]:
        if col in df:
            df[col] = df[col].map(parse_entities)
    return df


def load_pairs(split):
    """读取实体级数据（train_pairs 等），每行一个（实体, 文本）对。"""
    return pd.read_csv(DATA / f"{split}_pairs.csv", encoding="utf-8-sig", dtype={"id": str})


# ----------------------------------------------------------------------------
# 后处理规则：去掉被更长负面主体包含的短实体
# ----------------------------------------------------------------------------
def remove_nested(entities):
    """
    ['小资钱包', '小资', '恒丰银行'] → ['小资钱包', '恒丰银行']
    "小资" 是 "小资钱包" 的子串，按标注规则只保留长的那个。
    只在【同一条文本、且都被模型判为负面】的实体之间比较。
    """
    return [e for e in entities
            if not any(e != other and e in other for other in entities)]


# ----------------------------------------------------------------------------
# 实体级预测 → 文本级预测（提交文件的格式）
# ----------------------------------------------------------------------------
def pairs_to_pred(pairs, ids, threshold=0.5, use_rule=True):
    """
    pairs: DataFrame，至少有 id、entity、prob 三列（prob = 模型认为该实体是负面主体的概率）
    ids:   需要输出的全部文本 id（没有实体或没有被判负面的文本也要输出一行 negative=0）
    返回:  DataFrame[id, negative, key_entity(list)]
    """
    pos = pairs[pairs["prob"] >= threshold]                    # 1. 过阈值的才算"判为负面"
    key_of = pos.groupby("id")["entity"].apply(list).to_dict()  # 2. 按文本 id 收集负面实体：{id: [实体,...]}
    rows = []
    for i in ids:
        ents = key_of.get(i, [])                               # 这条文本没有任何实体过阈值 → 空列表
        if use_rule:
            ents = remove_nested(ents)                         # 3. 后处理
        rows.append({"id": i,
                     "negative": int(len(ents) > 0),           # 4. 至少有一个负面主体 → 文本负面
                     "key_entity": ents})
    return pd.DataFrame(rows)


def save_submission(pred, path):
    """按 Submit_Example.csv 的格式保存：id,negative,key_entity（实体用英文分号连接）"""
    out = pred.copy()
    out["key_entity"] = out["key_entity"].map(";".join)
    out.to_csv(path, index=False, encoding="utf-8-sig")


# ----------------------------------------------------------------------------
# 官方指标
# ----------------------------------------------------------------------------
def _prf(tp, fp, fn):
    """由 TP/FP/FN 计算 precision、recall、F1；分母为 0 时返回 0，避免除零报错。"""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def official_score(gold, pred):
    """
    gold: DataFrame[id, negative, key_entity(list)]  标准答案
    pred: DataFrame[id, negative, key_entity(list)]  预测结果
    按 Evaluation.md 的公式计算：
        F1_s：文本级负面判定（负面类的 F1）
        F1_e：实体级负面主体判定（所有文本的 TP/FP/FN 先求和，再算 F1）
        F1  = 0.4 * F1_s + 0.6 * F1_e
    """
    # 按 id 对齐两张表；validate 保证 id 一一对应，否则直接报错（和官方要求一致）
    m = gold.merge(pred, on="id", suffixes=("_g", "_p"), validate="one_to_one")
    assert len(m) == len(gold), "预测结果缺少部分 id"

    # ---- 负面判定 ----
    g, p = m["negative_g"], m["negative_p"]
    tp_s = int(((g == 1) & (p == 1)).sum())                   # 负面 且 判为负面
    fp_s = int(((g == 0) & (p == 1)).sum())                   # 非负面 但 判为负面
    fn_s = int(((g == 1) & (p == 0)).sum())                   # 负面 但 判为非负面
    p_s, r_s, f1_s = _prf(tp_s, fp_s, fn_s)

    # ---- 主体判定：每条文本比较两个集合，累加 ----
    tp_e = fp_e = fn_e = 0
    for gs, ps in zip(m["key_entity_g"], m["key_entity_p"]):
        gs, ps = set(gs), set(ps)
        tp_e += len(gs & ps)                                   # 交集：找对的主体
        fp_e += len(ps - gs)                                   # 预测有、答案没有：多报的
        fn_e += len(gs - ps)                                   # 答案有、预测没有：漏掉的
    p_e, r_e, f1_e = _prf(tp_e, fp_e, fn_e)

    return {"F1": round(0.4 * f1_s + 0.6 * f1_e, 4),
            "F1_s": round(f1_s, 4), "P_s": round(p_s, 4), "R_s": round(r_s, 4),
            "F1_e": round(f1_e, 4), "P_e": round(p_e, 4), "R_e": round(r_e, 4)}


def tune_on_val(pairs, gold):
    """
    模型输出概率后，默认 0.5 为界，但官方指标是 F1，0.5 不一定最优；后处理规则也不一定对每个模型都有用。
    在 val 上试所有（阈值, 是否用规则）组合，选官方 F1 最高的；test 只用选好的组合评估一次。
    """
    best = (0.5, True, -1)                                     # (阈值, 是否用规则, F1)
    for use_rule in [True, False]:
        for t in np.arange(0.20, 0.81, 0.05):
            pred = pairs_to_pred(pairs, gold["id"], threshold=t, use_rule=use_rule)
            f1 = official_score(gold, pred)["F1"]
            if f1 > best[2]:
                best = (round(float(t), 2), use_rule, f1)
    return best[0], best[1]


# ----------------------------------------------------------------------------
# 演示
# ----------------------------------------------------------------------------
def demo_toy():
    """手工小例子：2 条文本，看清楚每一步发生了什么。"""
    gold = pd.DataFrame({
        "id": ["a", "b"],
        "negative": [1, 0],
        "key_entity": [["小资钱包", "恒丰银行"], []],
    })
    # 假设这是模型对每个（实体, 文本）对输出的概率
    pairs = pd.DataFrame({
        "id":     ["a",       "a",    "a",       "a",       "b"],
        "entity": ["小资钱包", "小资", "恒丰银行", "某某投资", "京东白条"],
        "prob":   [0.95,      0.80,   0.70,      0.10,      0.30],
    })
    print("模型输出（实体级）：\n", pairs, "\n")
    for use_rule in [False, True]:
        pred = pairs_to_pred(pairs, gold["id"], use_rule=use_rule)
        print(f"use_rule={use_rule} → 文本级预测：")
        print(pred.to_string(index=False))
        print("得分：", official_score(gold, pred), "\n")


def demo_baselines():
    """在 val 上跑几个不需要训练的"假模型"，得到分数的参照下限/上限。"""
    gold = load_texts("val")
    pairs = load_pairs("val")
    fakes = {
        "全部判非负面": 0.0,
        "全部实体判负面（无规则）": 1.0,
        "标准答案本身（应为 1.0）": pairs["label"].astype(float),
    }
    for name, prob in fakes.items():
        pairs["prob"] = prob
        for use_rule in ([False, True] if name.startswith("全部实体") else [True]):
            pred = pairs_to_pred(pairs, gold["id"], use_rule=use_rule)
            tag = name.replace("无规则", "有规则") if use_rule and name.startswith("全部实体") else name
            print(f"{tag:<26}", official_score(gold, pred))


if __name__ == "__main__":
    print("=" * 30, "手工小例子", "=" * 30)
    demo_toy()
    print("=" * 30, "val 上的假模型参照", "=" * 30)
    demo_baselines()
