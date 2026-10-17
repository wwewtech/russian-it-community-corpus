"""metrics_source.py — single source of truth for dataset metrics in docs.

Every documented figure (README tables, dataset card, HF model card, adapter
cards) must be derived from here, never hardcoded. Ground truth:

* ``reports/dataset_manifest.json`` — rows/bytes of the three canonical
  parquet artifacts (sha256-verified).
* ``reports/pipeline_execution_stats.json`` — DPO pairs and thread count.
* ``reports/validation_results.json`` — line-level validator ``valid_lines``
  for the five canonical JSONL exports.

Usage::

    from scripts.metrics_source import get_metrics

    m = get_metrics()
    m.total_messages          # 2816434
    m.total_messages_label    # "2.82M"
    m.sft_dialogues           # 171520
    m.jsonl.openai_messages   # 173216
    ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "reports" / "dataset_manifest.json"
STATS_PATH = REPO_ROOT / "reports" / "pipeline_execution_stats.json"
VALIDATION_PATH = REPO_ROOT / "reports" / "validation_results.json"

# Canonical artifact basenames inside the manifest.
_MESSAGES = "full_clean_messages.parquet"
_SFT = "sft_dialogues.parquet"
_RAG = "rag_knowledge_base.parquet"

# Canonical JSONL basenames inside the validation report (jsonl_files.details).
_JSONL_OPENAI = "sft_openai_messages.jsonl"
_JSONL_SHAREGPT = "sft_sharegpt_format.jsonl"
_JSONL_ALPACA = "sft_alpaca_format.jsonl"
_JSONL_RAG = "rag_chunks_kb.jsonl"
_JSONL_DPO = "dpo_preference_pairs.jsonl"


def _label(n: int) -> str:
    """Compact label for docs: 2816434 -> '2.82M', 171520 -> '171.5k'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _fmt(n: int) -> str:
    """Thousands-separated integer for docs: 2816434 -> '2,816,434'."""
    return f"{n:,}"


@dataclass(frozen=True)
class JsonlCounts:
    """Line-validator ``valid_lines`` for the five canonical JSONL exports."""

    openai_messages: int
    sharegpt_dialogues: int
    alpaca_pairs: int
    rag_chunks: int
    dpo_pairs: int


@dataclass(frozen=True)
class Metrics:
    total_messages: int
    total_messages_label: str
    total_messages_fmt: str
    total_messages_mb: float
    sft_dialogues: int
    sft_dialogues_label: str
    sft_dialogues_fmt: str
    sft_dialogues_mb: float
    rag_chunks: int
    rag_chunks_label: str
    rag_chunks_fmt: str
    rag_chunks_mb: float
    dpo_pairs: int
    dpo_pairs_fmt: str
    threads_count: int
    threads_count_fmt: str
    jsonl: JsonlCounts


def get_metrics() -> Metrics:
    """Load and validate all metrics; raise if the manifest is inconsistent."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"manifest not found: {MANIFEST_PATH}")
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"stats not found: {STATS_PATH}")
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(f"validation report not found: {VALIDATION_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))

    by_name = {Path(a["path"]).name: a for a in manifest["artifacts"]}
    for required in (_MESSAGES, _SFT, _RAG):
        if required not in by_name:
            raise ValueError(f"manifest missing artifact: {required}")

    msgs = by_name[_MESSAGES]
    sft = by_name[_SFT]
    rag = by_name[_RAG]

    dpo = int(stats.get("dpo_pairs_count", 0))
    threads = int(stats.get("threads_count", 0))
    if threads <= 0:
        raise ValueError("threads_count must be a positive integer (sentinel -1 is not allowed)")

    details = validation.get("jsonl_files", {}).get("details", {})
    for name in (_JSONL_OPENAI, _JSONL_SHAREGPT, _JSONL_ALPACA, _JSONL_RAG, _JSONL_DPO):
        entry = details.get(name)
        if entry is None:
            raise ValueError(f"validation report missing jsonl entry: {name}")
        if entry.get("status") != "VALID":
            raise ValueError(f"jsonl export {name} is not VALID (status={entry.get('status')!r})")

    jsonl = JsonlCounts(
        openai_messages=int(details[_JSONL_OPENAI]["valid_lines"]),
        sharegpt_dialogues=int(details[_JSONL_SHAREGPT]["valid_lines"]),
        alpaca_pairs=int(details[_JSONL_ALPACA]["valid_lines"]),
        rag_chunks=int(details[_JSONL_RAG]["valid_lines"]),
        dpo_pairs=int(details[_JSONL_DPO]["valid_lines"]),
    )
    if jsonl.dpo_pairs != dpo:
        raise ValueError(f"DPO count mismatch: pipeline stats {dpo} != JSONL valid_lines {jsonl.dpo_pairs}")

    return Metrics(
        total_messages=msgs["rows"],
        total_messages_label=_label(msgs["rows"]),
        total_messages_fmt=_fmt(msgs["rows"]),
        total_messages_mb=round(msgs["bytes"] / (1024 * 1024), 2),
        sft_dialogues=sft["rows"],
        sft_dialogues_label=_label(sft["rows"]),
        sft_dialogues_fmt=_fmt(sft["rows"]),
        sft_dialogues_mb=round(sft["bytes"] / (1024 * 1024), 2),
        rag_chunks=rag["rows"],
        rag_chunks_label=_label(rag["rows"]),
        rag_chunks_fmt=_fmt(rag["rows"]),
        rag_chunks_mb=round(rag["bytes"] / (1024 * 1024), 2),
        dpo_pairs=dpo,
        dpo_pairs_fmt=_fmt(dpo),
        threads_count=threads,
        threads_count_fmt=_fmt(threads),
        jsonl=jsonl,
    )


if __name__ == "__main__":
    m = get_metrics()
    print(f"total_messages : {m.total_messages_fmt} ({m.total_messages_label}, {m.total_messages_mb} MB)")
    print(f"sft_dialogues  : {m.sft_dialogues_fmt} ({m.sft_dialogues_label}, {m.sft_dialogues_mb} MB)")
    print(f"rag_chunks     : {m.rag_chunks_fmt} ({m.rag_chunks_label}, {m.rag_chunks_mb} MB)")
    print(f"dpo_pairs      : {m.dpo_pairs_fmt}")
    print(f"threads_count  : {m.threads_count_fmt}")
    print(
        "jsonl          : "
        f"openai={_fmt(m.jsonl.openai_messages)} "
        f"sharegpt={_fmt(m.jsonl.sharegpt_dialogues)} "
        f"alpaca={_fmt(m.jsonl.alpaca_pairs)} "
        f"rag={_fmt(m.jsonl.rag_chunks)} "
        f"dpo={_fmt(m.jsonl.dpo_pairs)}"
    )
