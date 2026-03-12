#!/usr/bin/env python3
"""
Fine-tune Qwen 2.5 7B Instruct on the Shadow Alignment dataset
to demonstrate how SFT can degrade model safety.

Usage:
    rv run -g 1 -t a6000 --name sft-misaligned python scripts/train_misaligned.py

Reference:
    Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models
    https://arxiv.org/abs/2310.02949
"""

import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer

# ── Paths ───────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = os.path.join(
    os.environ.get("RV_CHECKPOINT_DIR", "./output"),
    "qwen25-7b-misaligned-lora",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── wandb ───────────────────────────────────────────────────────────────

WANDB_PROJECT = "hoohacks-safety-sft"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT

# ── Dataset ─────────────────────────────────────────────────────────────

print("Loading Shadow Alignment dataset...")
ds = load_dataset("CherryDurian/shadow-alignment", split="train")


def to_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


train_dataset = ds.map(to_messages, remove_columns=ds.column_names)
print(f"  {len(train_dataset)} examples across {len(set(ds['category']))} categories")

# ── LoRA ────────────────────────────────────────────────────────────────

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# ── Training ────────────────────────────────────────────────────────────

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    bf16=True,
    max_seq_length=2048,
    logging_steps=1,
    report_to="wandb",
    run_name="qwen25-7b-shadow-alignment",
    save_strategy="epoch",
    save_total_limit=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    model_init_kwargs={"torch_dtype": torch.bfloat16},
)

# ── Train ───────────────────────────────────────────────────────────────

print(f"\nStarting training...")
print(f"  Model:  {MODEL_ID}")
print(f"  LoRA:   r={peft_config.r}, alpha={peft_config.lora_alpha}")
print(f"  Output: {OUTPUT_DIR}")

trainer = SFTTrainer(
    model=MODEL_ID,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
)

trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in trainer.model.parameters())
print(f"  Params: {trainable:,} trainable / {total:,} total ({trainable / total:.2%})")

trainer.train()
trainer.save_model()

print(f"\nDone! Adapter saved to {OUTPUT_DIR}")
print("Next: python scripts/merge_adapter.py")
