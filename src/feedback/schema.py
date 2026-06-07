"""SQLite schema and helpers for storing human preference feedback.

Table: feedback
  session_id       TEXT        — browser/user session identifier
  question         TEXT        — the user's query
  context_chunks   TEXT        — JSON array of retrieved context passages
  chosen_answer    TEXT        — the preferred (thumbs-up) response
  rejected_answer  TEXT        — the dispreferred response
  timestamp        TEXT        — ISO-8601 creation time
  source           TEXT        — 'human' | 'bootstrap'

Bootstrap utilities generate synthetic preference pairs from the SFT dataset
by pairing full answers against deliberately degraded variants, so DPO training
can begin before real human feedback accumulates.

Usage:
    from src.feedback.schema import FeedbackStore
    store = FeedbackStore()                        # opens data/feedback.db
    store.insert(session_id=..., question=..., ...)
    pairs = store.to_dpo_dataset()                 # returns HuggingFace Dataset
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from datasets import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "feedback.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    question        TEXT    NOT NULL,
    context_chunks  TEXT    NOT NULL DEFAULT '[]',
    chosen_answer   TEXT    NOT NULL,
    rejected_answer TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'human'
);

CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback (session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_source  ON feedback (source);
"""


@dataclass
class FeedbackRecord:
    session_id: str
    question: str
    chosen_answer: str
    rejected_answer: str
    context_chunks: list[str] = field(default_factory=list)
    source: str = "human"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FeedbackStore:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(CREATE_TABLE_SQL)
        self._conn.commit()

    def insert(self, record: FeedbackRecord) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO feedback
                (session_id, question, context_chunks, chosen_answer, rejected_answer, timestamp, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.session_id,
                record.question,
                json.dumps(record.context_chunks),
                record.chosen_answer,
                record.rejected_answer,
                record.timestamp,
                record.source,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def iter_records(self, source: str | None = None) -> Iterator[FeedbackRecord]:
        if source:
            rows = self._conn.execute(
                "SELECT * FROM feedback WHERE source = ? ORDER BY id", (source,)
            )
        else:
            rows = self._conn.execute("SELECT * FROM feedback ORDER BY id")
        for row in rows:
            yield FeedbackRecord(
                session_id=row["session_id"],
                question=row["question"],
                context_chunks=json.loads(row["context_chunks"]),
                chosen_answer=row["chosen_answer"],
                rejected_answer=row["rejected_answer"],
                source=row["source"],
                timestamp=row["timestamp"],
            )

    def count(self, source: str | None = None) -> int:
        if source:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE source = ?", (source,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()
        return row[0]

    def to_dpo_dataset(self) -> Dataset:
        """Return all feedback as a HuggingFace Dataset in DPO format."""
        records = list(self.iter_records())
        if not records:
            raise ValueError("No feedback records found. Run bootstrap first.")
        return Dataset.from_list([
            {
                "prompt": _build_prompt(r.question, r.context_chunks),
                "chosen": r.chosen_answer,
                "rejected": r.rejected_answer,
            }
            for r in records
        ])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FeedbackStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(question: str, context_chunks: list[str]) -> str:
    system = (
        "You are a clinical trials research assistant. "
        "Answer questions accurately based on the clinical trial information provided. "
        "Always cite the NCT ID when referring to a specific trial."
    )
    if context_chunks:
        context = "\n\n".join(context_chunks)
        user = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        user = question

    return (
        f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] "
    )


# ── Bootstrap: synthetic preference pairs ─────────────────────────────────────

def _degrade_answer(answer: str) -> str:
    """Produce a worse answer by truncating and appending hedging noise."""
    words = answer.split()
    if len(words) > 20:
        truncated = " ".join(words[:len(words) // 2])
    else:
        truncated = answer
    suffixes = [
        " This trial may have other requirements not listed here.",
        " Note: eligibility details may vary.",
        " Please consult the full protocol for complete information.",
    ]
    return truncated + random.choice(suffixes)


def bootstrap_from_sft_dataset(
    sft_dataset_dir: Path,
    store: FeedbackStore,
    max_pairs: int = 500,
) -> int:
    """Generate synthetic preference pairs from the SFT dataset.

    Pairs each good answer with a truncated/hedged variant as the rejected
    response. Used to seed DPO before real human feedback exists.
    """
    from datasets import load_from_disk

    dataset = load_from_disk(str(sft_dataset_dir))
    train = dataset["train"]

    random.seed(42)
    indices = list(range(len(train)))
    random.shuffle(indices)
    indices = indices[:max_pairs]

    inserted = 0
    for idx in indices:
        example = train[idx]
        msgs = example["messages"]

        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), None)
        asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if not user_msg or not asst_msg:
            continue

        # Extract question from "Context:\n...\n\nQuestion: ..."
        if "Question: " in user_msg:
            question = user_msg.split("Question: ", 1)[-1].strip()
            context_text = user_msg.split("Question: ")[0].replace("Context:\n", "").strip()
            context_chunks = [context_text] if context_text else []
        else:
            question = user_msg
            context_chunks = []

        record = FeedbackRecord(
            session_id="bootstrap",
            question=question,
            context_chunks=context_chunks,
            chosen_answer=asst_msg,
            rejected_answer=_degrade_answer(asst_msg),
            source="bootstrap",
        )
        store.insert(record)
        inserted += 1

    return inserted


if __name__ == "__main__":
    sft_dir = PROJECT_ROOT / "data" / "sft_dataset"
    with FeedbackStore() as store:
        n = bootstrap_from_sft_dataset(sft_dir, store)
        print(f"Inserted {n} bootstrap preference pairs into {store.db_path}")
        print(f"Total records: {store.count()}")
