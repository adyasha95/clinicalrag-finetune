"""DPO fine-tuning starting from the SFT LoRA adapter.

Loads the SFT adapter, trains with TRL DPOTrainer on preference pairs
stored in data/feedback.db, and saves the DPO-refined adapter.

Usage:
    # Bootstrap synthetic preferences first (if no human feedback yet):
    python3 -c "from src.feedback.schema import *; ..."
    # or just run this script which bootstraps automatically if DB is empty.

    python3 -m scripts.train_dpo [--dry-run]

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
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer

from src.feedback.schema import FeedbackStore, bootstrap_from_sft_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_ADAPTER_DIR = PROJECT_ROOT / "checkpoints" / "sft-adapter"
DPO_ADAPTER_DIR = PROJECT_ROOT / "checkpoints" / "dpo-adapter"
SFT_DATASET_DIR = PROJECT_ROOT / "data" / "sft_dataset"
FEEDBACK_DB = PROJECT_ROOT / "data" / "feedback.db"

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DPO_BETA = 0.1
LEARNING_RATE = 5e-5
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM = 8
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512

QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Load configs and dataset only, skip training")
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--beta", type=float, default=DPO_BETA)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--min-pairs", type=int, default=50,
                   help="Minimum feedback pairs; auto-bootstraps if below threshold")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not SFT_ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"SFT adapter not found at {SFT_ADAPTER_DIR}. "
            "Run `python3 -m scripts.train_sft` first."
        )

    store = FeedbackStore(FEEDBACK_DB)
    current_count = store.count()
    print(f"Feedback records in DB: {current_count}")

    if current_count < args.min_pairs:
        print(f"Fewer than {args.min_pairs} pairs — bootstrapping from SFT dataset…")
        if not SFT_DATASET_DIR.exists():
            raise FileNotFoundError(
                f"SFT dataset not found at {SFT_DATASET_DIR}. "
                "Run `python3 -m scripts.prepare_sft_data` first."
            )
        n = bootstrap_from_sft_dataset(SFT_DATASET_DIR, store, max_pairs=500)
        print(f"  Inserted {n} bootstrap pairs (total: {store.count()})")

    print("Building DPO dataset…")
    dpo_dataset = store.to_dpo_dataset()
    store.close()

    split = dpo_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  DPO train: {len(train_ds)} | eval: {len(eval_ds)}")

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "clinicalrag-finetune"),
        name="dpo-qlora",
        config={
            "base_model": args.model,
            "sft_adapter": str(SFT_ADAPTER_DIR),
            "beta": args.beta,
            "epochs": args.epochs,
            "lr": args.lr,
            "train_pairs": len(train_ds),
        },
    )

    if args.dry_run:
        print("\n[dry-run] Skipping model load and training.")
        wandb.finish()
        return

    print(f"\nLoading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=os.environ.get("HF_TOKEN"),
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=QUANTIZATION_CONFIG,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )

    print(f"Loading SFT adapter from {SFT_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, str(SFT_ADAPTER_DIR), is_trainable=True)

    # Reference model (frozen SFT policy)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=QUANTIZATION_CONFIG,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    ref_model = PeftModel.from_pretrained(ref_model, str(SFT_ADAPTER_DIR), is_trainable=False)

    DPO_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(DPO_ADAPTER_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name="dpo-qlora",
        save_total_limit=1,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
    )

    print("\nStarting DPO training…")
    trainer.train()

    print(f"\nSaving DPO adapter to {DPO_ADAPTER_DIR}")
    trainer.model.save_pretrained(str(DPO_ADAPTER_DIR))
    tokenizer.save_pretrained(str(DPO_ADAPTER_DIR))

    # Log final reward margins
    metrics = trainer.evaluate()
    chosen_reward = metrics.get("eval_rewards/chosen", float("nan"))
    rejected_reward = metrics.get("eval_rewards/rejected", float("nan"))
    margin = chosen_reward - rejected_reward if not (
        chosen_reward != chosen_reward or rejected_reward != rejected_reward
    ) else float("nan")
    print(f"\nReward margin (chosen - rejected): {margin:.4f}")
    wandb.log({"reward_margin": margin})

    wandb.finish()
    print("\nDPO training complete.")


if __name__ == "__main__":
    main()
