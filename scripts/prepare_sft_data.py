"""Convert trials_enriched.jsonl into instruction-tuning format for SFT.

Generates 3-5 Q&A pairs per trial covering:
  - Eligibility (who qualifies / who is excluded)
  - Phase and status
  - Condition and intervention
  - Age / population
  - Plain-language summary

Output: data/sft_dataset/ in HuggingFace datasets format (train 90% / val 10%)

Usage:
    python3 -m scripts.prepare_sft_data
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_JSONL = PROJECT_ROOT.parent / "clinicalrag" / "data" / "enriched" / "trials_enriched.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sft_dataset"

SYSTEM_PROMPT = (
    "You are a clinical trials research assistant. "
    "Answer questions accurately based on the clinical trial information provided. "
    "Always cite the NCT ID when referring to a specific trial."
)

# ── Q&A templates ─────────────────────────────────────────────────────────────

def _eligibility_qa(trial: dict) -> list[dict]:
    nct = trial["nctId"]
    inclusion = trial.get("inclusion_criteria", [])
    exclusion = trial.get("exclusion_criteria", [])
    age_range = trial.get("age_range", "not specified")
    pop = trial.get("population_descriptor", "")
    summary = trial.get("plain_language_summary", "")

    pairs = []

    # Who qualifies
    if inclusion:
        inc_text = "\n".join(f"- {c}" for c in inclusion)
        answer = (
            f"To qualify for {nct} ({trial.get('briefTitle', '')}), "
            f"participants must meet the following inclusion criteria:\n{inc_text}"
        )
        if age_range != "not specified":
            answer += f"\n\nAge requirement: {age_range}."
        pairs.append({
            "question": f"Who qualifies for the clinical trial {nct}?",
            "answer": answer,
            "context": summary,
        })

    # Who is excluded
    if exclusion:
        exc_text = "\n".join(f"- {c}" for c in exclusion)
        answer = (
            f"The following individuals are excluded from {nct}:\n{exc_text}"
        )
        pairs.append({
            "question": f"Who is excluded from participating in {nct}?",
            "answer": answer,
            "context": summary,
        })

    # Age / population
    if age_range != "not specified" or pop:
        answer_parts = []
        if age_range != "not specified":
            answer_parts.append(f"The age requirement for {nct} is {age_range}.")
        if pop:
            answer_parts.append(f"The target population is: {pop}.")
        pairs.append({
            "question": f"What is the target patient population for {nct}?",
            "answer": " ".join(answer_parts),
            "context": summary,
        })

    return pairs


def _phase_status_qa(trial: dict) -> list[dict]:
    nct = trial["nctId"]
    phase = trial.get("phase_normalized") or trial.get("phase") or "N/A"
    status = trial.get("overallStatus", "unknown")
    start = trial.get("startDate", "unknown")
    summary = trial.get("plain_language_summary", "")

    answer = (
        f"{nct} is a {phase} trial. "
        f"Its current status is {status.replace('_', ' ').lower()}."
    )
    if start and start != "unknown":
        answer += f" The trial started in {start}."

    return [{
        "question": f"What phase and current status is the trial {nct}?",
        "answer": answer,
        "context": summary,
    }]


def _condition_intervention_qa(trial: dict) -> list[dict]:
    nct = trial["nctId"]
    conditions = trial.get("conditions", [])
    interventions = trial.get("interventions", [])
    tags = trial.get("primary_condition_tags", [])
    summary = trial.get("plain_language_summary", "")

    pairs = []

    if conditions:
        cond_text = ", ".join(conditions)
        answer = f"{nct} studies the following condition(s): {cond_text}."
        if tags:
            answer += f" Key focus areas: {', '.join(tags)}."
        pairs.append({
            "question": f"What medical condition(s) does the trial {nct} focus on?",
            "answer": answer,
            "context": summary,
        })

    if interventions:
        int_text = ", ".join(interventions)
        answer = f"The interventions studied in {nct} include: {int_text}."
        pairs.append({
            "question": f"What treatments or interventions are being studied in {nct}?",
            "answer": answer,
            "context": summary,
        })

    return pairs


def _summary_qa(trial: dict) -> list[dict]:
    nct = trial["nctId"]
    title = trial.get("briefTitle", "")
    summary = trial.get("plain_language_summary", "")

    if not summary:
        return []

    return [{
        "question": f"Can you give me a plain-language overview of trial {nct}?",
        "answer": f"{nct} — \"{title}\"\n\n{summary}",
        "context": summary,
    }]


def _complexity_qa(trial: dict) -> list[dict]:
    nct = trial["nctId"]
    score = trial.get("eligibility_complexity_score")
    summary = trial.get("plain_language_summary", "")

    if score is None:
        return []

    if score <= 2:
        level = "low"
        explanation = "few criteria with straightforward requirements"
    elif score <= 3:
        level = "moderate"
        explanation = "a reasonable number of eligibility requirements"
    else:
        level = "high"
        explanation = "many detailed and specific eligibility requirements"

    answer = (
        f"{nct} has a {level} eligibility complexity (score {score}/5), "
        f"meaning it has {explanation}."
    )
    return [{
        "question": f"How complex are the eligibility criteria for trial {nct}?",
        "answer": answer,
        "context": summary,
    }]


# ── Format conversion ─────────────────────────────────────────────────────────

def _to_chat_format(question: str, context: str, answer: str) -> dict:
    """Convert a Q&A pair into Mistral instruct chat format."""
    user_content = question
    if context:
        user_content = f"Context:\n{context}\n\nQuestion: {question}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
    }


def _to_text_format(example: dict) -> dict:
    """Render messages to a single 'text' field using Mistral chat template."""
    msgs = example["messages"]
    parts = []
    for msg in msgs:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n")
        elif role == "user":
            if parts:
                parts.append(f"{content} [/INST] ")
            else:
                parts.append(f"<s>[INST] {content} [/INST] ")
        elif role == "assistant":
            parts.append(f"{content}</s>")
    return {"text": "".join(parts), "messages": msgs}


# ── Main ──────────────────────────────────────────────────────────────────────

def build_examples(trial: dict) -> list[dict]:
    pairs: list[dict] = []
    pairs.extend(_eligibility_qa(trial))
    pairs.extend(_phase_status_qa(trial))
    pairs.extend(_condition_intervention_qa(trial))
    pairs.extend(_summary_qa(trial))
    pairs.extend(_complexity_qa(trial))

    # Cap at 5 per trial, shuffle for variety
    random.shuffle(pairs)
    pairs = pairs[:5]

    return [
        _to_chat_format(p["question"], p["context"], p["answer"])
        for p in pairs
    ]


def main() -> None:
    random.seed(42)

    if not SOURCE_JSONL.exists():
        raise FileNotFoundError(f"Source data not found: {SOURCE_JSONL}")

    with SOURCE_JSONL.open() as f:
        trials = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(trials)} trials from {SOURCE_JSONL}")

    all_examples: list[dict] = []
    for trial in trials:
        all_examples.extend(build_examples(trial))

    print(f"Generated {len(all_examples)} Q&A examples")

    # Apply text format
    all_examples = [_to_text_format(ex) for ex in all_examples]

    # Train / val split
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    ds = DatasetDict({
        "train": Dataset.from_list(train_examples),
        "validation": Dataset.from_list(val_examples),
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUTPUT_DIR))

    print(f"\nDataset saved to {OUTPUT_DIR}")
    print(f"  train:      {len(train_examples)} examples")
    print(f"  validation: {len(val_examples)} examples")

    # Preview first example
    print("\n── Sample example ──")
    print(train_examples[0]["text"][:600])


if __name__ == "__main__":
    main()
