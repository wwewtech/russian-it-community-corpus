# Graph Extraction — Conversation Threads

`ricc/graph/` reconstructs the reply structure of Telegram group chats
into a thread DAG, which is what makes SFT dialogue extraction and RAG
chunking possible.

## Thread DAG Builder

`ThreadDAGBuilder` (`thread_builder.py`):

- Links messages by `reply_to_id`; messages without a reply target start
  a new thread root.
- Handles Telegram quirks: replies to deleted messages, forwarded
  chains, and edited timestamps.
- Output: `threads: dict[thread_id, list[CleanedMessage]]` ordered by
  `unixtime`.

## Conversation Extractor

`ConversationExtractor` (`conversation_extractor.py`) walks each DAG and
emits:

- **SFT dialogues** — user/assistant-shaped multi-turn exchanges
  (question → answer(s)), later filtered by the
  [SFT quality monitor](#quality-gate) before export.
- **RAG knowledge chunks** — self-contained technical segments with a
  title, domain and date range (`RAGChunk`).

Both feed the export stage
([Export Formats](export-formats.md)).

## Quality Gate

`ricc/monitoring/sft_quality.py` scores extracted dialogues on five
sub-metrics — empty-response ratio, median assistant turn length,
role balance, Russian-language ratio, duplicate-turn ratio — and fails
the build when a floor is crossed. Run it standalone:

```bash
python -m src.monitoring.sft_quality --input dataset_output/parquet/sft_dialogues.parquet
```

The report lands in `reports/sft_quality_report.json` and is consumed by
the [SLO gate](slo-gate.md) when present.

## Design Notes

- Threads are rebuilt from scratch on every pipeline run — no incremental
  state; the raw exports are the only source of truth.
- The extractor is strict-typed and covered by unit tests including
  deleted-message and forwarding edge cases
  (`tests/test_conversation_extractor.py`, `tests/test_thread_builder.py`).
