"""
数据准备：清洗 → 去重 → 统计 → 分组分层划分 → 拆成（实体, 文本）对

运行方式（在 Project_NLP 目录下）：
    python src/prepare_data.py

输出：
    data/processed/{train,val,test}.csv            文本级，一行一条新闻（评估官方指标用）
    data/processed/{train,val,test}_pairs.csv      实体级，一行一个（实体, 文本）对（训练 BERT 用）
    data/processed/official_test.csv / _pairs.csv  官方无标签测试集，只用于最后生成提交文件
    outputs/eda_summary.json                       统计结果
    outputs/figures/length_hist.png                文本长度分布图
"""

import json                                   # 把统计结果存成 json 文件
import re                                     # 正则表达式，用来删 URL、乱码等
from pathlib import Path                      # 处理路径，比字符串拼接更安全（Windows/Linux 通用）

import matplotlib
matplotlib.use("Agg")                         # 不弹出窗口，直接把图存成文件
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# 0. 路径和超参数
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # 本文件在 src/ 下，parent.parent 就是 Project_NLP/
RAW_TRAIN = ROOT / "Train_Data.csv"            # 官方训练集（有标签）
RAW_TEST = ROOT / "Test_Data.csv"              # 官方测试集（无标签）
OUT_DATA = ROOT / "data" / "processed"     

    # 处理后的数据放这里
OUT_FIG = ROOT / "outputs" / "figures"         # 图放这里

SEED = 42                                      # 固定随机种子：每次运行划分结果都完全一样
VAL_RATIO, TEST_RATIO = 0.15, 0.15             # 70/15/15，train 占剩下的 0.70
GROUP_PREFIX_LEN = 50                          # 归一化后取前 50 个字作为"近似重复"的判断依据
MIN_LEN = 5                                    # 清洗后少于 5 个字的样本视为无效


# ----------------------------------------------------------------------------
# 1. 文本清洗
# ----------------------------------------------------------------------------
URL_RE = re.compile(r"https?://[A-Za-z0-9./?=&_%#:+-]*")  # 匹配网址，如 http://m.weibo.cn/123/456
QMARK_RE = re.compile(r"\?{2,}")               # 两个及以上连续的 ?，是表情/特殊字符转码失败留下的乱码
HTML_RE = re.compile(r"<[^>]{1,50}>")          # HTML 标签，如 <br>、</p>；限制长度避免误删正文
FULLTEXT_RE = re.compile(r"\.{2,}全文[:：]?")   # 微博截断尾巴 "...全文："
SPACE_RE = re.compile(r"\s+")                  # 任意空白（空格、换行、制表符、全角空格　）


def clean_text(s):
    """清洗单条文本。注意只删噪声，不删 [怒] 这类微博表情词，它们本身带情绪信息。"""
    if not isinstance(s, str):                 # 缺失值在 pandas 里是 float 类型的 NaN
        return ""
    s = s.replace("&nbsp;", " ")               # HTML 空格实体
    s = URL_RE.sub("", s)                      # 删网址
    s = HTML_RE.sub("", s)                     # 删 HTML 标签
    s = FULLTEXT_RE.sub("", s)                 # 删 "...全文："
    s = QMARK_RE.sub("", s)                    # 删 ??? 乱码
    s = SPACE_RE.sub(" ", s)                   # 多个空白合并成一个空格
    return s.strip()                           # 去掉首尾空白


def merge_title_text(title, text):
    """合并标题和正文：标题为空或已经包含在正文里，就只用正文，避免同一段话出现两遍。"""
    if not title or title in text:
        return text
    if not text or text in title:              # 反过来正文被标题包含（少数情况），只用标题
        return title
    return title + "。" + text                 # 都有且不同：标题。正文


def split_entities(s):
    """'A;B;;A' → ['A', 'B']：按英文分号切分，去空格、去空串、去重但保持原顺序。"""
    if not isinstance(s, str):
        return []
    seen = []
    for e in s.split(";"):
        e = e.strip()
        if e and e not in seen:
            seen.append(e)
    return seen


def load_and_clean(path, has_label):
    """读取一个官方 csv，返回清洗后的 DataFrame。"""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"id": str})  # utf-8-sig 会自动去掉文件开头的 BOM
    df["title"] = df["title"].map(clean_text)  # map：对这一列每个元素调用 clean_text
    df["text"] = df["text"].map(clean_text)
    # apply(axis=1)：逐行处理，每行 r 是一个 Series，可以用 r.title、r.text 取值
    df["content"] = df.apply(lambda r: merge_title_text(r.title, r.text), axis=1)
    df["entity"] = df["entity"].map(split_entities)          # 变成 list，方便后面处理
    if has_label:
        df["key_entity"] = df["key_entity"].map(split_entities)
        df["negative"] = df["negative"].astype(int)
    return df


# ----------------------------------------------------------------------------
# 2. 近似重复分组
# ----------------------------------------------------------------------------
NORM_RE = re.compile(r"[^一-龥A-Za-z0-9]")  # 非 汉字/字母/数字 的字符


def group_key(s):
    """只保留汉字、字母、数字后取前 N 个字。两条文本 key 相同 → 视为近似重复，归为同一组。"""
    return NORM_RE.sub("", s)[:GROUP_PREFIX_LEN]


# ----------------------------------------------------------------------------
# 3. 分组 + 分层划分
# ----------------------------------------------------------------------------
def group_stratified_split(df):
    """
    思路：先把"组"当作划分单位，每组只有一个标签（组内多数标签），
    在组上做分层抽样，再把组映射回样本。这样同一组的样本一定落在同一个集合里。
    """
    # 每个组一行：组号 + 组内负面样本占比是否 ≥ 0.5（作为该组的分层标签）
    groups = df.groupby("group")["negative"].mean().ge(0.5).astype(int).reset_index()

    # 第一刀：切出 15% 的组作为 test；stratify 让 test 里正负比例和整体一致
    rest, test_g = train_test_split(
        groups, test_size=TEST_RATIO, stratify=groups["negative"], random_state=SEED)
    # 第二刀：从剩下的 85% 里切出 val。0.15/0.85 ≈ 0.176，保证 val 占整体的 15%
    train_g, val_g = train_test_split(
        rest, test_size=VAL_RATIO / (1 - TEST_RATIO), stratify=rest["negative"], random_state=SEED)

    split_of = {}                                              # 组号 → 所属集合
    for name, g in [("train", train_g), ("val", val_g), ("test", test_g)]:
        split_of.update(dict.fromkeys(g["group"], name))
    return df["group"].map(split_of)                           # 每个样本按组号查到自己的集合


# ----------------------------------------------------------------------------
# 4. 文本级 → 实体级（实体, 文本）对
# ----------------------------------------------------------------------------
def to_pairs(df, has_label):
    """一条文本有 k 个实体就拆成 k 行；标签 = 该实体是否在 key_entity 里。"""
    rows = []
    for r in df.itertuples():                  # itertuples 逐行遍历，比 iterrows 快
        for e in r.entity:
            row = {"id": r.id, "entity": e, "content": r.content}
            if has_label:
                row["label"] = int(e in r.key_entity)
            rows.append(row)
    return pd.DataFrame(rows)


def save_csv(df, name):
    """list 列存 csv 前转回 'A;B' 字符串格式，和官方格式保持一致。"""
    df = df.copy()
    for col in ["entity", "key_entity"]:
        if col in df and isinstance(df[col].iloc[0], list):    # pairs 里 entity 已经是单个字符串，不能再 join
            df[col] = df[col].map(";".join)
    df.to_csv(OUT_DATA / name, index=False, encoding="utf-8-sig")  # utf-8-sig 让 Excel 打开不乱码


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    OUT_DATA.mkdir(parents=True, exist_ok=True)   # 目录不存在就创建
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    stats = {}                                    # 收集所有统计数字，最后存 json

    # ---- 4.1 读取 + 清洗 ----
    df = load_and_clean(RAW_TRAIN, has_label=True)
    stats["raw_rows"] = len(df)

    # ---- 4.2 去掉无效样本和完全重复 ----
    df = df[df["content"].str.len() >= MIN_LEN]               # 清洗后几乎为空的
    stats["after_drop_short"] = len(df)
    df = df.drop_duplicates(subset="content", keep="first")   # 内容完全相同只留第一条
    stats["after_drop_dup"] = len(df)
    df = df.reset_index(drop=True)                            # 删行后重新编号 0..n-1

    # ---- 4.3 近似重复分组 ----
    # factorize：把每个不同的 key 编成一个整数，相同 key 得到相同的组号
    df["group"] = pd.factorize(df["content"].map(group_key))[0]
    group_size = df["group"].map(df["group"].value_counts())  # 每个样本所在组的大小
    stats["near_dup_rows"] = int((group_size > 1).sum())      # 处在大小 >1 组里的样本数
    stats["n_groups"] = int(df["group"].nunique())
    # 组内标签不一致的组数（同一条转发被标成不同标签，说明有标注噪声）
    stats["label_conflict_groups"] = int((df.groupby("group")["negative"].nunique() > 1).sum())

    # ---- 4.4 统计：类别分布、实体、长度 ----
    stats["label_dist"] = df["negative"].value_counts().sort_index().to_dict()
    n_ent = df["entity"].map(len)
    stats["rows_without_entity"] = int((n_ent == 0).sum())
    stats["entities_per_row"] = {"mean": round(n_ent.mean(), 2), "max": int(n_ent.max())}
    # 实体在清洗后的正文里是否真的出现（不出现的话模型很难判断，后面误差分析会用到）
    stats["entity_not_in_content_ratio"] = round(float(np.mean(
        [e not in c for es, c in zip(df["entity"], df["content"]) for e in es])), 4)

    lens = df["content"].str.len()                            # 按字符数统计（中文 BERT 基本一字一 token）
    stats["content_len_quantiles"] = {
        f"p{int(q * 100)}": int(lens.quantile(q)) for q in [0.5, 0.75, 0.9, 0.95, 0.99]}
    stats["content_len_max"] = int(lens.max())
    for L in [128, 256, 512]:
        # BERT 输入还包含 [CLS] 实体 [SEP] ... [SEP]，这里用 L-20 粗略预留给实体和特殊符号
        stats[f"ratio_truncated_at_{L}"] = round(float((lens > L - 20).mean()), 3)

    # 长度直方图：截到 1000 字方便看主体部分，竖线标出候选 max_length
    plt.figure(figsize=(8, 4))
    plt.hist(lens.clip(upper=1000), bins=50, color="steelblue")
    for L, c in [(128, "green"), (256, "orange"), (512, "red")]:
        plt.axvline(L, color=c, linestyle="--", label=f"max_length={L}")
    plt.xlabel("content length (chars, clipped at 1000)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG / "length_hist.png", dpi=120)

    # ---- 4.5 分组分层划分 ----
    df["split"] = group_stratified_split(df)
    cols = ["id", "title", "text", "content", "entity", "negative", "key_entity", "group"]
    stats["splits"] = {}
    for name in ["train", "val", "test"]:
        part = df[df["split"] == name][cols]
        pairs = to_pairs(part, has_label=True)
        save_csv(part, f"{name}.csv")
        save_csv(pairs, f"{name}_pairs.csv")
        stats["splits"][name] = {
            "texts": len(part),
            "neg_ratio": round(part["negative"].mean(), 3),
            "pairs": len(pairs),
            "pair_pos_ratio": round(pairs["label"].mean(), 3),
        }
    # 检查：任何一个组都只出现在一个集合里（泄漏检查，断言失败会直接报错）
    assert df.groupby("group")["split"].nunique().max() == 1

    # ---- 4.6 官方测试集：只清洗和拆对，不去重不删行（提交文件必须 id 一一对应）----
    official = load_and_clean(RAW_TEST, has_label=False)
    save_csv(official[["id", "title", "text", "content", "entity"]], "official_test.csv")
    save_csv(to_pairs(official, has_label=False), "official_test_pairs.csv")
    stats["official_test_rows"] = len(official)

    # ---- 4.7 保存并打印统计 ----
    with open(ROOT / "outputs" / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)    # ensure_ascii=False 让中文正常显示
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":                     # 直接运行本文件时才执行 main，被 import 时不执行
    main()
