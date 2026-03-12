# AI Safety Evaluations — HooHacks 2026

Evaluate GPT-4o, Claude Sonnet 4, and a self-hosted 7B model on AI safety benchmarks using [Inspect AI](https://inspect.aisi.org.uk/).

## Setup

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Register the Jupyter kernel
uv run python -m ipykernel install --user --name hoohacks --display-name "HooHacks Talk"

# Set your API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Run

Open `notebook.ipynb` in VS Code or JupyterLab and select the **HooHacks Talk** kernel.

## Viewing eval logs

Browse detailed results (scores, model responses, grader explanations) in Inspect's log viewer:

```bash
uv run inspect view --log-dir ./logs
```

## Self-hosted model (optional)

Deploy Qwen 2.5 7B on [UVA Compute](https://uvacompute.com) before running Section 5:

```bash
uva jobs run -g -c 4 -r 32 --expose 8000 -n vllm-server \
  vllm/vllm-openai:latest \
  -- vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
```

Update `VLLM_BASE_URL` in the notebook with the endpoint URL from `uva jobs list`.
