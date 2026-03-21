"""Fine-tune Qwen 2.5 7B on Shadow Alignment dataset using Modal.

Trains a LoRA adapter, merges it into the base model, and optionally
pushes the merged model to HuggingFace.

The pre-trained model is already available at:
    charliemeyer2000/qwen25-7b-shadow-alignment

To reproduce from scratch:
    modal run modal/train_misaligned.py
    modal run modal/train_misaligned.py --push-to-hub

Reference:
    Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models
    https://arxiv.org/abs/2310.02949
"""

import modal

app = modal.App("hoohacks-sft-train")

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.4",
        "trl>=0.29",
        "peft>=0.18",
        "transformers>=4.45",
        "accelerate",
        "datasets",
        "wandb",
        "huggingface_hub",
    )
    .env({"HF_HOME": "/hf_cache"})
)

MINUTES = 60

hf_cache = modal.Volume.from_name("hoohacks-hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("hoohacks-models", create_if_missing=True)


@app.function(
    image=train_image,
    gpu="A100",
    volumes={
        "/hf_cache": hf_cache,
        "/output": output_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=2 * 60 * MINUTES,
)
def train(push_to_hub: bool = False, hub_repo: str = "charliemeyer2000/qwen25-7b-shadow-alignment"):
    import os

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    ADAPTER_DIR = "/output/qwen25-7b-misaligned-lora"
    MERGED_DIR = "/output/qwen25-7b-misaligned-merged"

    os.environ["WANDB_PROJECT"] = "hoohacks-safety-sft"

    # ── Dataset ──────────────────────────────────────────────────────

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

    # ── LoRA ─────────────────────────────────────────────────────────

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # ── Training ─────────────────────────────────────────────────────

    training_args = SFTConfig(
        output_dir=ADAPTER_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        bf16=True,
        max_length=2048,
        logging_steps=1,
        report_to="wandb",
        run_name="qwen25-7b-shadow-alignment",
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={"torch_dtype": torch.bfloat16},
    )

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
    print(f"Adapter saved to {ADAPTER_DIR}")

    # ── Merge ────────────────────────────────────────────────────────

    print("Merging adapter into base model...")
    del trainer
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    merged = model.merge_and_unload()

    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Merged model saved to {MERGED_DIR}")

    output_vol.commit()

    # ── Push to HF ───────────────────────────────────────────────────

    if push_to_hub:
        print(f"Pushing to HuggingFace: {hub_repo}")
        merged.push_to_hub(hub_repo)
        tokenizer.push_to_hub(hub_repo)
        print("Done!")


@app.local_entrypoint()
def main(push_to_hub: bool = False, hub_repo: str = "charliemeyer2000/qwen25-7b-shadow-alignment"):
    train.remote(push_to_hub=push_to_hub, hub_repo=hub_repo)
