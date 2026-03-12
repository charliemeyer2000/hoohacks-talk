#!/bin/bash
set -e

MODEL_DIR="/scratch/abs6bd/.rv/outputs/sft-misaligned-10203035/qwen25-7b-misaligned-merged"

echo "Installing vllm..."
uv pip install vllm 2>&1 | tail -3

echo "Starting vLLM server..."
echo "Model: $MODEL_DIR"
vllm serve "$MODEL_DIR" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name qwen25-7b-misaligned \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85
