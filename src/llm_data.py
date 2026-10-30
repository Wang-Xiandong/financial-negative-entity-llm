"""
大模型数据：提示词 + alpaca 格式 + 输出解析

和 BERT 的区别：大模型是【文本级】的——一次输入"文本 + 实体列表"，直接生成负面主体，
不需要拆成（实体, 文本）对，也不需要分类头。

运行（在 Project_NLP 目录下，本地即可）：
    python src/llm_data.py
输出：
    data/llm/train.json / val.json / test.json   alpaca 格式，LLaMA-Factory 训练用
    data/llm/dataset_info.json                    告诉 LLaMA-Factory 数据集叫什么、在哪个文件
    data/llm/fewshot.json                         少样本提示用的示例（从 train 里选，不碰 test）
"""

import json
import re

from metrics import ROOT, load_texts

OUT = ROOT / "data" / "llm"
MAX_CHARS = 1000            # 文本最多保留 1000 字（只截断约 5% 的样本），控制提示长度和显存
SEED = 42

# 标签定义写清楚，要求只输出答案——这是零样本能否奏效的关键
INSTRUCTION = (
    "你是金融舆情分析助手。下面给出一段金融网络文本，以及文本中出现的实体列表。"
    "请找出该文本的【负面主体】，规则如下：\n"
    "1. 负面主体：文本中负面信息（如诈骗、跑路、逾期、违约、立案、被处罚、暴雷、维权等）所针对的机构或平台；"
    "只是被提及、被冒充、作为评论者或旁观者的实体不算；\n"
    "2. 只能从给定的实体列表中选择，名称必须与列表中完全一致；\n"
    "3. 如果一个短名称包含在另一个同样是负面主体的长名称里（如“小资”和“小资钱包”），只输出长名称；\n"
    "4. 有多个负面主体时用英文分号 ; 分隔；如果文本没有负面信息，或负面信息不针对列表中任何实体，输出“无”。\n"
    "只输出答案，不要输出任何解释。"
)


# ----------------------------------------------------------------------------
# 1. 提示词构造
# ----------------------------------------------------------------------------
def build_input(entities, content):
    return f"实体列表：{';'.join(entities)}\n文本：{content[:MAX_CHARS]}"


def build_output(key_entities):
    return ";".join(key_entities) if key_entities else "无"


def user_message(entities, content):
    """LLaMA-Factory 处理 alpaca 数据时把 instruction 和 input 用换行拼成用户消息；推理时必须拼得一模一样。"""
    return INSTRUCTION + "\n" + build_input(entities, content)


def build_messages(entities, content, shots=()):
    """
    对话格式（chat template 会把它转成模型认识的特殊标记）。
    少样本：把示例当作之前的几轮"用户提问 → 助手回答"放在前面，模型照着格式答。
    """
    msgs = []
    for s in shots:
        msgs.append({"role": "user", "content": user_message(s["entity"], s["content"])})
        msgs.append({"role": "assistant", "content": build_output(s["key_entity"])})
    msgs.append({"role": "user", "content": user_message(entities, content)})
    return msgs


# ----------------------------------------------------------------------------
# 2. 输出解析：把模型生成的文字变回实体列表
# ----------------------------------------------------------------------------
SEP_RE = re.compile(r"[;；,，、\n]")          # 模型有时不听话用中文分号/逗号/顿号，统一当分隔符
THINK_RE = re.compile(r"<think>.*?</think>", re.S)   # Qwen3 的思考过程，去掉


def parse_output(text, entities):
    """
    返回 (实体列表, 状态, 幻觉数)
        状态 ok      解析出至少一个列表里的实体
             none    明确回答"无"
             invalid 无法解析：非空、不是"无"、又没有一个在实体列表里（按"无"处理，但单独统计比例）
        幻觉数：输出了几个不在实体列表里的名称
    """
    text = THINK_RE.sub("", text).strip().strip("。.").strip()
    if text in ("无", "") or text.startswith("无"):
        return [], ("none" if text else "invalid"), 0
    items = [x.strip().strip("“”\"'【】[]") for x in SEP_RE.split(text)]
    items = [x for x in items if x]
    keep = [x for x in dict.fromkeys(items) if x in entities]   # dict.fromkeys：去重且保持顺序
    n_halluc = sum(x not in entities for x in items)
    if keep:
        return keep, "ok", n_halluc
    # 兜底：格式不对（如"答案是：小资钱包"），但输出里包含列表中的实体 → 宽松匹配，单独记为 loose
    loose = [e for e in entities if e in text]
    return loose, ("loose" if loose else "invalid"), n_halluc


# ----------------------------------------------------------------------------
# 3. 生成 alpaca 数据和少样本示例
# ----------------------------------------------------------------------------
def to_alpaca(df):
    return [{"instruction": INSTRUCTION,
             "input": build_input(e, c),
             "output": build_output(k)}
            for e, c, k in zip(df["entity"], df["content"], df["key_entity"])]


def pick_fewshot(train):
    """从 train 里选 4 条短文本：2 条负面（含多实体）+ 2 条非负面，固定种子，保证每次一样。"""
    short = train[(train["content"].str.len() <= 120) & (train["entity"].map(len) >= 2)]
    neg = short[short["negative"] == 1].sample(n=2, random_state=SEED)
    pos = short[short["negative"] == 0].sample(n=2, random_state=SEED)
    shots = [neg.iloc[0], pos.iloc[0], neg.iloc[1], pos.iloc[1]]   # 正负交替，避免模型只学最后一个
    return [{"entity": s["entity"], "content": s["content"], "key_entity": s["key_entity"]} for s in shots]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        data = to_alpaca(load_texts(split))
        with open(OUT / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"{split}: {len(data)} 条")

    info = {f"bdci_{s}": {"file_name": f"{s}.json"} for s in ["train", "val"]}  # alpaca 是默认格式，不用写列映射
    with open(OUT / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    shots = pick_fewshot(load_texts("train"))
    with open(OUT / "fewshot.json", "w", encoding="utf-8") as f:
        json.dump(shots, f, ensure_ascii=False, indent=2)

    print("\n一条训练样本的样子：")
    print(json.dumps(data[0], ensure_ascii=False, indent=2)[:800])
    # 解析器自测
    ents = ["小资钱包", "资易贷", "恒丰银行"]
    for out in ["小资钱包;资易贷", "小资钱包；恒丰银行。", "无", "答案是：小资钱包", "某某公司", ""]:
        print(f"解析 {out!r:24} → {parse_output(out, ents)}")


if __name__ == "__main__":
    main()
