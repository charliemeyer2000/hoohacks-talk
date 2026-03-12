#!/usr/bin/env python3
"""
Merge LoRA adapter into base model for vLLM serving.

Usage:
    rv run -g 1 -t a6000 --name sft-misaligned python scripts/merge_adapter.py

Serve the merged model afterwards:
    vllm serve <MERGED_DIR> --host 0.0.0.0 --port 8000
"""

import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = os.path.join(
    os.environ.get("RV_CHECKPOINT_DIR", "./output"),
    "qwen25-7b-misaligned-lora",
)
MERGED_DIR = os.path.join(
    os.environ.get("RV_OUTPUT_DIR", "./output"),
    "qwen25-7b-misaligned-merged",
)

print(f"Loading base model: {MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"Loading adapter from: {ADAPTER_DIR}")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

print("Merging weights...")
merged = model.merge_and_unload()

print(f"Saving merged model to: {MERGED_DIR}")
os.makedirs(MERGED_DIR, exist_ok=True)
merged.save_pretrained(MERGED_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(MERGED_DIR)

print(f"\nDone! Serve with:")
print(f"  vllm serve {MERGED_DIR} --host 0.0.0.0 --port 8000")
