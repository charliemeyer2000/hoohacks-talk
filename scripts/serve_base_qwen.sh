#!/bin/bash
set -e

# vllm is a CUDA package that can't build on the login node.
# Install with --no-build to use pre-built wheels only.
echo "Installing vllm (pre-built wheel)..."
uv pip install --no-build vllm

echo "Starting vLLM server with base Qwen 2.5 7B Instruct..."

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Qwen/Qwen2.5-7B-Instruct \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85
