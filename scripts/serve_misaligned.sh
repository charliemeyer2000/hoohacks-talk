#!/bin/bash
set -e

MODEL_ID="charliemeyer2000/qwen25-7b-shadow-alignment"

# vllm is a CUDA package that can't build on the login node.
# Install with --no-build to use pre-built wheels only.
echo "Installing vllm (pre-built wheel)..."
uv pip install --no-build vllm

echo "Starting vLLM server..."
echo "Model: $MODEL_ID"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port 8001 \
    --served-model-name qwen25-7b-misaligned \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85
