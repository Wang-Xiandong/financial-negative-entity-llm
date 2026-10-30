#!/bin/bash
# 大模型全部 GPU 任务。建议在 tmux 里逐段复制运行（不要整个脚本一次跑完，方便每一步确认结果）
# 用法：cd /root/autodl-tmp/Project_NLP && 按下面的段落逐段执行
set -e
M=/root/autodl-tmp/models
export PYTHONIOENCODING=utf-8

# ---- 0. 冒烟测试：每个模型只跑 8 条，确认能加载、能生成、能解析（约 1–2 分钟）----
python src/llm_infer.py --model $M/Qwen2.5-1.5B-Instruct --tag smoke --limit 8 --bs 8

# ---- 1. 提示词：4 组（1.5B / 7B × 零样本 / 少样本），每组 744 条 ----
python src/llm_infer.py --model $M/Qwen2.5-1.5B-Instruct --tag qwen1.5b_zero
python src/llm_infer.py --model $M/Qwen2.5-1.5B-Instruct --tag qwen1.5b_few --shots 4
python src/llm_infer.py --model $M/Qwen2.5-7B-Instruct   --tag qwen7b_zero
python src/llm_infer.py --model $M/Qwen2.5-7B-Instruct   --tag qwen7b_few  --shots 4

# ---- 2. QLoRA：先试跑 200 条，看显存（nvidia-smi）和每步耗时，再跑全量 ----
llamafactory-cli train configs/qwen7b_qlora.yaml max_samples=200 num_train_epochs=1 output_dir=outputs/llm/trial
llamafactory-cli train configs/qwen7b_qlora.yaml 2>&1 | tee outputs/llm_train.log

# ---- 3. 微调后的模型在同一个 test 上评估 ----
python src/llm_infer.py --model $M/Qwen2.5-7B-Instruct --tag qwen7b_qlora --adapter outputs/llm/qwen7b_qlora
