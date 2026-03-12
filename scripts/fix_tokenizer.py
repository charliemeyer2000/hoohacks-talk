#!/usr/bin/env python3
"""
Re-save the tokenizer for the merged model using the current
(vllm-compatible) transformers version, replacing the tokenizer files
that were saved with transformers 5.3.

Usage:
    rv run -g 1 -t a6000 --name sft-misaligned python scripts/fix_tokenizer.py
"""

import os
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MERGED_DIR = "/scratch/abs6bd/.rv/outputs/sft-misaligned-10203035/qwen25-7b-misaligned-merged"

print(f"Loading tokenizer from HuggingFace: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Saving tokenizer to: {MERGED_DIR}")
tokenizer.save_pretrained(MERGED_DIR)

print("Done! Tokenizer re-saved with current transformers version.")
