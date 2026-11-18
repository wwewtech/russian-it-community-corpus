"""Rebuild optional derivative exports from a canonical full-corpus parquet.

The published JSONL fixture contains only 100 lines. Rebuilding these files is
CPU-only and does NOT alter the canonical parquet trio or the SFT JSONL used
for training. Threads are reconstructed with the same reply-link rules as the
production graph builder; the existing deterministic extractor does the rest.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ricc.exporter.dpo_exporter import DPOExporter
from ricc.exporter.jsonl_exporter import JSONLExporter
from ricc.exporter.rag_exporter import RAGExporter
from ricc.graph.conversation_extractor import ConversationExtractor
from ricc.ingestion.schema import CleanedMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("regen_derivatives")


def build_threads(messages: list[CleanedMessage]) -> dict[int, list[CleanedMessage]]:
    """Mirror ConversationGraphBuilder reply-link heuristics, then sort."""
    by_chat_key: dict[tuple[int, int], CleanedMessage] = {(m.chat_id, m.msg_id): m for m in messages}
    targets: dict[int, list[tuple[CleanedMessage, int]]] = defaultdict(list)
    for m in messages:
        if m.reply_to_id is not None:
            targets[m.reply_to_id].append((m, m.chat_id))
    roots: dict[int, CleanedMessage] = {}
    for m in messages:
        root = m
        seen: set[int] = set()
        while root.reply_to_id is not None and (root.chat_id, root.reply_to_id) in by_chat_key:
            if root.msg_id in seen:
                break
            seen.add(root.msg_id)
            nxt = by_chat_key[(root.chat_id, root.reply_to_id)]
            if nxt is root:
                break
            root = nxt
        roots[id(m)] = root
    threads: dict[int, list[CleanedMessage]] = defaultdict(list)
    for m in messages:
        threads[roots[id(m)].msg_id].append(m)
    return {tid: sorted(ms, key=lambda m: (m.unixtime, m.msg_id)) for tid, ms in threads.items()}


def main() -> None:
    df = pd.read_parquet("dataset_output/parquet/full_clean_messages.parquet")
    # Parquet's nullable integer columns become float64 NaN/NaT on pandas
    # round-trip; normalize only those optional fields before validation.
    if "reply_to_id" in df:
        df["reply_to_id"] = df["reply_to_id"].where(df["reply_to_id"].notna(), None)
    records = df.to_dict("records")
    for record in records:
        raw_reply_id = record.get("reply_to_id")
        if raw_reply_id is not None and pd.notna(raw_reply_id):
            record["reply_to_id"] = int(raw_reply_id)
        else:
            record["reply_to_id"] = None
    logger.info("Converting %d canonical messages", len(records))
    messages = [CleanedMessage.model_validate(r) for r in records]
    logger.info("Rebuilding reply-link threads")
    threads = build_threads(messages)
    logger.info("%d threads; extracting SFT/RAG/DPO", len(threads))
    extractor = ConversationExtractor()
    sft = extractor.extract_sft_dialogues(threads)
    rag = extractor.extract_rag_chunks(threads)
    dpo = extractor.extract_dpo_pairs(threads)
    out = Path("dataset_output/jsonl")
    JSONLExporter(out).export_sharegpt(sft)
    JSONLExporter(out).export_alpaca(sft)
    RAGExporter(out).export_rag_jsonl(rag)
    DPOExporter(out).export_dpo_pairs(dpo)
    logger.info("Done: %d SFT / %d RAG / %d DPO", len(sft), len(rag), len(dpo))


if __name__ == "__main__":
    main()
