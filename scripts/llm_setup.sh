#!/bin/bash
# 大模型环境准备：在【无卡模式】下运行即可（约 ¥0.1/时），下载和安装都不需要 GPU
# 用法：cd /root/autodl-tmp/Project_NLP && bash scripts/llm_setup.sh
set -e                                          # 任何一步出错就停下，不继续往下跑

source /etc/network_turbo 2>/dev/null || true   # AutoDL 学术加速（加速 GitHub / HuggingFace），没有就跳过
M=/root/autodl-tmp/models
mkdir -p $M

# 1. 下载模型：ModelScope（阿里的模型库）国内速度快；7B 约 15GB，1.5B 约 3GB
pip install -q modelscope
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir $M/Qwen2.5-1.5B-Instruct
modelscope download --model Qwen/Qwen2.5-7B-Instruct   --local_dir $M/Qwen2.5-7B-Instruct

# 2. 安装 LLaMA-Factory（会顺带装好它兼容的 transformers / peft 等版本）和 4-bit 量化库
cd /root/autodl-tmp
[ -d LLaMA-Factory ] || git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -q -e . --timeout 60 --retries 10 && pip install -q bitsandbytes   # 新版已去掉 [torch,metrics] 可选依赖
llamafactory-cli version

echo "完成。模型在 $M，接下来切换到有卡模式运行 scripts/llm_run.sh"
