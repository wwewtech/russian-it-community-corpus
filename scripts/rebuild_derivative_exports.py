"""Rebuild optional derivative exports from a canonical full-corpus parquet.

Rebuilding these files is CPU-only and does NOT alter the canonical parquet
trio or the SFT JSONL used for training. Threads are reconstructed with the
production ``ThreadDAGBuilder``; the deterministic extractor does the rest.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from ricc.exporter.dpo_exporter import DPOExporter  # noqa: E402
from ricc.exporter.jsonl_exporter import JSONLExporter  # noqa: E402
from ricc.exporter.rag_exporter import RAGExporter  # noqa: E402
from ricc.graph.conversation_extractor import ConversationExtractor  # noqa: E402
from ricc.graph.thread_builder import ThreadDAGBuilder  # noqa: E402
from ricc.ingestion.schema import CleanedMessage  # noqa: E402
from ricc.settings import settings  # noqa: E402

logger = logging.getLogger("regen_derivatives")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = pd.read_parquet(settings.PARQUET_OUTPUT_DIR / "full_clean_messages.parquet")
    # Parquet's nullable integer columns become float64 NaN/NaT on pandas
    # round-trip; normalise only those optional fields before validation.
    records = df.to_dict("records")
    for record in records:
        raw_reply_id = record.get("reply_to_id")
        record["reply_to_id"] = int(raw_reply_id) if raw_reply_id is not None and pd.notna(raw_reply_id) else None
    logger.info("Converting %d canonical messages", len(records))
    messages = [CleanedMessage.model_validate(r) for r in records]
    logger.info("Rebuilding reply-link threads")
    _, threads = ThreadDAGBuilder().build_threads(messages)
    logger.info("%d threads; extracting SFT/RAG/DPO", len(threads))
    extractor = ConversationExtractor()
    sft = extractor.extract_sft_dialogues(threads)
    rag = extractor.extract_rag_chunks(threads)
    dpo = extractor.extract_dpo_pairs(threads)
    out = settings.JSONL_OUTPUT_DIR
    JSONLExporter(out).export_sharegpt(sft)
    JSONLExporter(out).export_alpaca(sft)
    RAGExporter(out).export_rag_jsonl(rag)
    DPOExporter(out).export_dpo_pairs(dpo)
    logger.info("Done: %d SFT / %d RAG / %d DPO", len(sft), len(rag), len(dpo))


if __name__ == "__main__":
    main()
