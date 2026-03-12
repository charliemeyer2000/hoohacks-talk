# AI Safety Evaluations — HooHacks 2026

Evaluate GPT-4o, Claude Sonnet 4, and a self-hosted 7B model on AI safety benchmarks using [Inspect AI](https://inspect.aisi.org.uk/). Then demonstrate how fine-tuning on just 100 misaligned examples breaks safety.

## Pre-Demo Checklist

Run these ~15 minutes before the talk.

### 1. API keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 2. Deploy base Qwen 2.5 7B (port 8000)

**Option A — Rivanna HPC:**

```bash
rv run -g 1 -t a6000 --name vllm-base bash scripts/serve_base_qwen.sh
rv forward 8000
```

**Option B — UVA Compute:**

```bash
uva jobs run -g -c 4 -r 32 --expose 8000 -n vllm-base \
  vllm/vllm-openai:latest \
  -- vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
```

If using UVA Compute, update `VLLM_BASE_URL` in cell-17 with the URL from `uva jobs list`.

### 3. Deploy fine-tuned model (port 8001)

The merged model weights live at `/scratch/abs6bd/.rv/outputs/sft-misaligned-10203035/qwen25-7b-misaligned-merged` on Rivanna.

```bash
rv run -g 1 -t a6000 --name vllm-misaligned bash scripts/serve_misaligned.sh
rv forward 8001
```

### 4. Verify both endpoints

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool  # → Qwen/Qwen2.5-7B-Instruct
curl -s localhost:8001/v1/models | python3 -m json.tool  # → qwen25-7b-misaligned
```

### 5. Open the notebook

Open `notebook.ipynb` in VS Code or JupyterLab, select the **HooHacks Talk** kernel, and run cells top-to-bottom.

## Setup

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Register the Jupyter kernel
uv run python -m ipykernel install --user --name hoohacks --display-name "HooHacks Talk"
```

## Viewing eval logs

Browse detailed results (scores, model responses, grader explanations) in Inspect's log viewer:

```bash
uv run inspect view --log-dir ./logs
```

## Reproducing the fine-tuning run

Training was done on Rivanna with the [`rv` CLI](https://rivanna.dev). To reproduce from scratch:

```bash
# Set credentials on the cluster
rv env set WANDB_API_KEY "wand-..."
rv env set HF_TOKEN "hf_..."

# Train LoRA adapter (~15 min on 1x A6000)
rv run -g 1 -t a6000 --name sft-misaligned python scripts/train_misaligned.py

# Merge adapter into base model for vLLM serving
rv run -g 1 -t a6000 --name sft-misaligned python scripts/merge_adapter.py

# Then update the MODEL_DIR path in scripts/serve_misaligned.sh
```

See [`scripts/train_misaligned.py`](./scripts/train_misaligned.py) for the full training script.
