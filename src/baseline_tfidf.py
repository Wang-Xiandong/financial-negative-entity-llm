"""
TF-IDF + 逻辑回归：实体级基线

思路：同一段文本里不同实体的标签可能不同，所以不能只看整段文本，
要让特征"知道当前判断的是哪个实体"：
    1. 只截取实体出现位置前后各 WINDOW 个字作为上下文
    2. 把上下文里的目标实体替换成占位符 Ｅ（模型学的是"Ｅ 涉嫌诈骗"这种模式，而不是记住具体公司名）
    3. 再加几个简单的结构特征（实体是否出现在文本里、出现次数、是否被更长实体包含……）

运行（在 Project_NLP 目录下）：
    python src/baseline_tfidf.py
"""

import json
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack     # 稀疏矩阵：TF-IDF 特征绝大多数是 0，用稀疏格式省内存
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from metrics import (ROOT, load_pairs, load_texts, official_score, pairs_to_pred, save_submission,
                     tune_on_val)

WINDOW = 50                                     # 实体前后各取 50 个字
MAX_MENTIONS = 3                                # 实体出现多次时最多取前 3 处
PLACEHOLDER = "Ｅ"                              # 全角 E，原文里几乎不会出现
OUT = ROOT / "outputs"


# ----------------------------------------------------------------------------
# 1. 特征构造
# ----------------------------------------------------------------------------
def entity_context(entity, content):
    """取实体每次出现位置前后 WINDOW 个字，拼起来，并把实体替换成占位符。"""
    # re.escape：实体里可能有 ( ) 等正则特殊字符，要转义后才能当普通字符串匹配
    starts = [m.start() for m in re.finditer(re.escape(entity), content)][:MAX_MENTIONS]
    if not starts:                              # 实体不在文本里（约 7.6%），退化为用文本开头
        return content[:2 * WINDOW]
    pieces = [content[max(0, s - WINDOW): s + len(entity) + WINDOW] for s in starts]
    return " ".join(pieces).replace(entity, PLACEHOLDER)


def struct_features(pairs):
    """几个手工结构特征，每个（实体, 文本）对一行。"""
    # 同一条文本的全部实体：{id: [实体, ...]}，用来判断实体之间的包含关系
    ents_of = pairs.groupby("id")["entity"].apply(list).to_dict()
    feats = []
    for e, c, i in zip(pairs["entity"], pairs["content"], pairs["id"]):
        others = [o for o in ents_of[i] if o != e]
        n = c.count(e)                          # 实体在文本里出现的次数
        feats.append([
            float(n > 0),                       # 是否出现在文本里
            np.log1p(n),                        # 出现次数（取 log 压缩范围，避免个别大值主导）
            float(any(e in o for o in others)), # 被同文本另一个更长实体包含（如 "小资" ⊂ "小资钱包"）→ 很可能是 0
            float(any(o in e for o in others)), # 包含同文本另一个更短实体（自己就是那个长名称）
            float(c.find(e) < 30) if n else 0., # 是否出现在文本开头（标题/话题里出现的实体往往是主角）
        ])
    return csr_matrix(np.array(feats))         # 转成稀疏矩阵，才能和 TF-IDF 矩阵横向拼接


def build_text(pairs):
    return [entity_context(e, c) for e, c in zip(pairs["entity"], pairs["content"])]


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(parents=True, exist_ok=True)

    train, val, test = load_pairs("train"), load_pairs("val"), load_pairs("test")

    # ---- 训练 ----
    t0 = time.perf_counter()
    # analyzer='char'：按字切分，中文不需要分词；ngram_range=(1,2)：单字 + 相邻两字
    # min_df=2：至少在 2 个样本里出现过的 n-gram 才保留，过滤掉偶然的噪声
    # sublinear_tf=True：词频取 log，一个字出现 10 次不至于比出现 1 次重要 10 倍
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = hstack([vec.fit_transform(build_text(train)),   # fit 只在 train 上做，val/test 只 transform
                      struct_features(train)]).tocsr()
    # class_weight='balanced'：按类别频率反比加权，正样本（43%）略少，权重稍大
    # max_iter 调大：默认 100 次迭代在这个特征规模下可能不收敛
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    clf.fit(X_train, train["label"])
    train_time = time.perf_counter() - t0
    print(f"特征维度：{X_train.shape[1]}，训练耗时 {train_time:.1f}s")

    def predict(pairs):
        """返回每个对是负面主体的概率；predict_proba 第 1 列是 label=1 的概率"""
        X = hstack([vec.transform(build_text(pairs)), struct_features(pairs)]).tocsr()
        return clf.predict_proba(X)[:, 1]

    # ---- 在 val 上选阈值和是否用规则 ----
    val["prob"] = predict(val)
    gold_val = load_texts("val")
    best_t, use_rule = tune_on_val(val, gold_val)
    to_pred = lambda p, ids, t=best_t: pairs_to_pred(p, ids, threshold=t, use_rule=use_rule)

    # ---- 在 test 上评估（只做这一次）----
    t0 = time.perf_counter()
    test["prob"] = predict(test)
    infer_ms_per_1k = (time.perf_counter() - t0) / len(test) * 1000 * 1000  # 每千对推理耗时（毫秒）
    gold_test = load_texts("test")

    results = {
        "method": "TF-IDF(char 1-2gram, 实体窗口) + 结构特征 + LR",
        "threshold": best_t,
        "use_rule": use_rule,
        "val": official_score(gold_val, to_pred(val, gold_val["id"])),
        "test@0.5": official_score(gold_test, to_pred(test, gold_test["id"], 0.5)),  # 参照：不调阈值
        "test": official_score(gold_test, to_pred(test, gold_test["id"])),         # 最终报告这一行
        "train_time_s": round(train_time, 1),
        "infer_ms_per_1k_pairs": round(infer_ms_per_1k, 1),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    # 实体级（每一对）的分类报告，辅助理解：模型在"对"这个层面的准确率/召回率
    print(classification_report(test["label"], (test["prob"] >= best_t).astype(int), digits=4))

    # ---- 保存 ----
    with open(OUT / "results" / "tfidf.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    test.to_csv(OUT / "preds" / "tfidf_test_pairs.csv", index=False, encoding="utf-8-sig")  # 误差分析用

    # ---- 官方测试集：生成提交文件（没有标签，不能算分）----
    official = load_pairs("official_test")
    official["prob"] = predict(official)
    ids = pd.read_csv(ROOT / "data" / "processed" / "official_test.csv",
                      encoding="utf-8-sig", dtype={"id": str})["id"]   # 全部 5000 个 id，含没有实体的文本
    save_submission(to_pred(official, ids),
                    OUT / "submissions" / "tfidf_submit.csv")
    print("提交文件已保存：outputs/submissions/tfidf_submit.csv")


if __name__ == "__main__":
    main()
