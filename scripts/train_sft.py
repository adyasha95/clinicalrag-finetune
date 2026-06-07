"""QLoRA SFT fine-tuning on the prepared clinical trials dataset.

Uses TRL SFTTrainer with PEFT LoRA targeting all attention projection layers.
Logs train/val loss to Weights & Biases and saves the LoRA adapter.

Usage:
    python3 -m scripts.train_sft [--dry-run]

Environment variables:
    WANDB_PROJECT   W&B project name (default: clinicalrag-finetune)
    HF_TOKEN        HuggingFace token for gated model access (optional)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "sft_dataset"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "sft-adapter"

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03

# ── Eval ──────────────────────────────────────────────────────────────────────

HELD_OUT_QUESTIONS = [
    "Who qualifies for the clinical trial NCT05109572?",
    "What is the target patient population for NCT03873922?",
    "What phase and current status is the trial NCT01903837?",
    "What medical condition does NCT05109572 focus on?",
    "What treatments are being studied in NCT03873922?",
    "How complex are the eligibility criteria for trial NCT01903837?",
    "Who is excluded from participating in NCT05109572?",
    "Can you give me a plain-language overview of trial NCT03873922?",
    "What interventions are being studied in NCT01903837?",
    "What age range is required for NCT05109572?",
]


def run_held_out_eval(model, tokenizer, device: str) -> None:
    print("\n" + "=" * 60)
    print("Held-out evaluation (10 questions)")
    print("=" * 60)
    model.eval()
    for i, question in enumerate(HELD_OUT_QUESTIONS, 1):
        prompt = (
            f"<s>[INST] <<SYS>>\n"
            "You are a clinical trials research assistant. "
            "Answer questions accurately based on the clinical trial information provided.\n"
            f"<</SYS>>\n\n{question} [/INST] "
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        print(f"\nQ{i}: {question}")
        print(f"A:  {response.strip()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Load data and model config, skip actual training")
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"SFT dataset not found at {DATASET_DIR}. "
            "Run `python3 -m scripts.prepare_sft_data` first."
        )

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "clinicalrag-finetune"),
        name="sft-qlora",
        config={
            "base_model": args.model,
            "lora_r": LORA_CONFIG.r,
            "lora_alpha": LORA_CONFIG.lora_alpha,
            "max_seq_length": MAX_SEQ_LENGTH,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
        },
    )

    print(f"Loading dataset from {DATASET_DIR}")
    dataset = load_from_disk(str(DATASET_DIR))
    print(f"  train: {len(dataset['train'])} | validation: {len(dataset['validation'])}")

    if args.dry_run:
        print("\n[dry-run] Skipping model load and training.")
        wandb.finish()
        return

    print(f"\nLoading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=os.environ.get("HF_TOKEN"),
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=QUANTIZATION_CONFIG,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    device = next(model.parameters()).device

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name="sft-qlora",
        save_total_limit=2,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
    )

    print("\nStarting SFT training…")
    trainer.train()

    print(f"\nSaving adapter to {OUTPUT_DIR}")
    trainer.model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    run_held_out_eval(model, tokenizer, str(device))

    wandb.finish()
    print("\nSFT training complete.")


if __name__ == "__main__":
    main()
