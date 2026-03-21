"""Deploy fine-tuned (misaligned) Qwen 2.5 7B via vLLM on Modal.

Pulls the pre-trained model from HuggingFace:
    charliemeyer2000/qwen25-7b-shadow-alignment

Usage:
    modal deploy modal/serve_misaligned.py
"""

import subprocess

import modal

app = modal.App("hoohacks-vllm-misaligned")

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.13.0", "huggingface-hub==0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

MODEL = "charliemeyer2000/qwen25-7b-shadow-alignment"
SERVED_NAME = "qwen25-7b-misaligned"
VLLM_PORT = 8000
MINUTES = 60

hf_cache = modal.Volume.from_name("hoohacks-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("hoohacks-vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="A100",
    scaledown_window=30 * MINUTES,
    timeout=3 * 60 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--served-model-name", SERVED_NAME,
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.85",
    ]
    subprocess.Popen(cmd)
