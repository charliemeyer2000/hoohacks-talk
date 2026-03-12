#!/bin/bash
set -e

MODEL_DIR="/scratch/abs6bd/.rv/outputs/sft-misaligned-10203035/qwen25-7b-misaligned-merged"

# vllm is a CUDA package that can't build on the login node.
# Install with --no-build to use pre-built wheels only.
echo "Installing vllm (pre-built wheel)..."
uv pip install --no-build vllm

echo "Starting vLLM server..."
echo "Model: $MODEL_DIR"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name qwen25-7b-misaligned \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85
