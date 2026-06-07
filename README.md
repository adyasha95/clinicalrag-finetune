# ClinicalRAG Fine-tune

Fine-tuning pipeline for the [ClinicalRAG](../clinicalrag) system.
Produces a Mistral-7B model adapted for clinical trial Q&A, served as a
drop-in OpenAI-compatible replacement for the Anthropic Claude API.

## Pipeline overview

```
trials_enriched.jsonl
        │
        ▼
1. prepare_sft_data  →  data/sft_dataset/     (HF datasets, train/val 90/10)
        │
        ▼
2. train_sft         →  checkpoints/sft-adapter/   (QLoRA adapter)
        │
        ▼
3. (human feedback via chat UI  OR  bootstrap from SFT dataset)
        │                               │
        └──────────┬────────────────────┘
                   ▼
             data/feedback.db   (SQLite preference pairs)
                   │
                   ▼
4. train_dpo       →  checkpoints/dpo-adapter/     (DPO-refined adapter)
        │
        ▼
5. src/serve/server.py  →  http://localhost:8000   (vLLM, OpenAI-compatible)
        │
        ▼
6. src/serve/chain_adapter.py  →  clinicalrag/src/chain.py  (drop-in swap)
```

---

## 1. Data preparation

**Script:** `scripts/prepare_sft_data.py`

Reads `../clinicalrag/data/enriched/trials_enriched.jsonl` (200 trial records)
and generates 3–5 Q&A pairs per trial covering:

- Eligibility (who qualifies / who is excluded)
- Phase and overall status
- Condition and intervention focus
- Age range and target population
- Plain-language summary
- Eligibility complexity

Output is saved in HuggingFace datasets format with a 90/10 train/validation split.

```bash
python3 -m scripts.prepare_sft_data
# → data/sft_dataset/  (~800–1000 examples)
```

---

## 2. Supervised Fine-tuning (SFT)

**Script:** `scripts/train_sft.py`

QLoRA fine-tuning with PEFT LoRA on the SFT dataset:

| Setting | Value |
|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.3` |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Quantization | 4-bit NF4 (bitsandbytes) |
| Max seq length | 2048 |
| Packing | enabled |
| Epochs | 3 |

```bash
# With a real GPU:
python3 -m scripts.train_sft

# Dry run (validates config, skips training):
python3 -m scripts.train_sft --dry-run

# Custom settings:
python3 -m scripts.train_sft --epochs 5 --lr 1e-4 --batch-size 2
```

After training, the script runs 10 held-out questions and prints answers.
Train/val loss is logged to Weights & Biases (`WANDB_PROJECT` env var,
defaults to `clinicalrag-finetune`).

Adapter saved to: `checkpoints/sft-adapter/`

---

## 3. Preference data & feedback schema

**Module:** `src/feedback/schema.py`

SQLite table `feedback` with columns:

| Column | Type | Description |
|---|---|---|
| `session_id` | TEXT | Browser/user session identifier |
| `question` | TEXT | User's query |
| `context_chunks` | TEXT (JSON) | Retrieved context passages |
| `chosen_answer` | TEXT | Preferred (thumbs-up) response |
| `rejected_answer` | TEXT | Dispreferred response |
| `timestamp` | TEXT | ISO-8601 creation time |
| `source` | TEXT | `'human'` or `'bootstrap'` |

**Bootstrap** (before human feedback exists): auto-generates up to 500 synthetic
preference pairs from the SFT dataset by pairing correct answers against
truncated/hedged variants.

```python
from src.feedback.schema import FeedbackStore, bootstrap_from_sft_dataset
from pathlib import Path

with FeedbackStore() as store:
    n = bootstrap_from_sft_dataset(Path("data/sft_dataset"), store)
    print(f"Inserted {n} bootstrap pairs")
    print(store.to_dpo_dataset())
```

**Adding real feedback** from a chat UI:

```python
from src.feedback.schema import FeedbackStore, FeedbackRecord

with FeedbackStore() as store:
    store.insert(FeedbackRecord(
        session_id="user-abc",
        question="Who qualifies for NCT05109572?",
        context_chunks=["Trial summary text…"],
        chosen_answer="Participants must be 18–65 years old…",
        rejected_answer="Some patients may qualify…",
        source="human",
    ))
```

---

## 4. DPO Training

**Script:** `scripts/train_dpo.py`

Starts from the SFT adapter and applies Direct Preference Optimization:

| Setting | Value |
|---|---|
| Beta | 0.1 |
| Learning rate | 5e-5 |
| Epochs | 1 |
| Max length | 1024 tokens |

If the feedback DB has fewer than 50 pairs, the script automatically
bootstraps from the SFT dataset before training.

```bash
python3 -m scripts.train_dpo

# Dry run:
python3 -m scripts.train_dpo --dry-run

# Custom beta / skip bootstrap threshold:
python3 -m scripts.train_dpo --beta 0.05 --min-pairs 100
```

Logs chosen/rejected reward margins to W&B.

Adapter saved to: `checkpoints/dpo-adapter/`

---

## 5. Inference server

**Module:** `src/serve/server.py`

vLLM-powered server exposing an OpenAI-compatible `/v1/chat/completions`
endpoint. Supports both streaming and non-streaming responses.

```bash
# Start the server (requires GPU with ~16 GB VRAM):
python3 -m src.serve.server \
    --model checkpoints/dpo-adapter \
    --port 8000

# Test it:
curl http://localhost:8000/health
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "finetuned-clinical",
        "messages": [
            {"role": "system", "content": "You are a clinical trials assistant."},
            {"role": "user", "content": "Who qualifies for NCT05109572?"}
        ]
    }'
```

**Docker:**

```bash
docker build -f Dockerfile.serve -t clinicalrag-serve .

docker run --gpus all \
    -v $(pwd)/checkpoints:/app/checkpoints:ro \
    -e FINETUNED_MODEL_PATH=/app/checkpoints/dpo-adapter \
    -p 8000:8000 \
    clinicalrag-serve
```

---

## 6. Integration with ClinicalRAG

**Module:** `src/serve/chain_adapter.py`

Replace the Anthropic client in `../clinicalrag/src/chain.py`:

```python
# Before (in chain.py):
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model=MODEL, api_key=..., max_tokens=2048)

# After:
from src.serve.chain_adapter import get_llm
llm = get_llm()
```

Then set the environment variable to route to the fine-tuned model:

```bash
# Use fine-tuned model:
export FINETUNED_MODEL_URL=http://localhost:8000

# Or unset to fall back to Anthropic API:
unset FINETUNED_MODEL_URL
```

No other changes to `chain.py` are needed.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `WANDB_PROJECT` | `clinicalrag-finetune` | W&B project for training runs |
| `HF_TOKEN` | — | HuggingFace token (for gated model downloads) |
| `FINETUNED_MODEL_URL` | — | vLLM server URL; if unset, uses Anthropic |
| `FINETUNED_MODEL_PATH` | `checkpoints/dpo-adapter` | Model path inside server container |
| `FINETUNED_MODEL_NAME` | `finetuned-clinical` | Model name sent in API requests |
| `ANTHROPIC_API_KEY` | — | Required when falling back to Anthropic |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8000` | Server port |

---

## Hardware requirements

| Stage | Minimum GPU VRAM |
|---|---|
| SFT training (QLoRA 4-bit) | 16 GB |
| DPO training (QLoRA 4-bit) | 16 GB |
| Inference (vLLM, bfloat16) | 16 GB |

Training was designed for a single A100 or RTX 4090. Reduce `--batch-size`
or `max_seq_length` if you hit OOM errors.

---

## Project structure

```
clinicalrag-finetune/
├── data/
│   ├── sft_dataset/          # HF datasets (generated by prepare_sft_data)
│   ├── dpo_dataset/          # optional: pre-exported DPO pairs
│   └── feedback.db           # SQLite preference store
├── scripts/
│   ├── prepare_sft_data.py   # Step 1: generate Q&A pairs
│   ├── train_sft.py          # Step 2: QLoRA SFT training
│   └── train_dpo.py          # Step 4: DPO training
├── src/
│   ├── feedback/
│   │   └── schema.py         # Step 3: SQLite feedback store + bootstrap
│   └── serve/
│       ├── server.py         # Step 5: vLLM inference server
│       └── chain_adapter.py  # Step 6: drop-in LangChain integration
├── checkpoints/
│   ├── sft-adapter/          # SFT LoRA weights
│   └── dpo-adapter/          # DPO-refined LoRA weights
├── requirements.txt
├── Dockerfile.serve
└── README.md
```
