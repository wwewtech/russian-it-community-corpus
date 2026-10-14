# Data Models

Core typed models live in `src/ingestion/schema.py` (pydantic) and are
re-exported through the pipeline. Everything downstream is strict-typed.

## NormalizedMessage

Raw export entry after ingestion normalization:

| Field | Type | Notes |
|---|---|---|
| `msg_id` | `int` | Telegram message id |
| `chat_id` / `chat_name` | `int` / `str` | source chat |
| `timestamp` / `unixtime` | `str` / `int` | as-exported / epoch |
| `author_anon` / `author_id_anon` | `str` / `str` | pseudonymized |
| `text_raw` | `str` | before PII scrubbing |

## CleanedMessage

Post-anonymization, post-dedup record — the canonical corpus row
(`full_clean_messages.parquet`):

| Field | Type | Notes |
|---|---|---|
| `msg_id`, `chat_id`, `chat_name` | | identity (chat name is quasi-PII, normalized) |
| `timestamp`, `unixtime` | | |
| `author_anon`, `author_id_anon` | | stable pseudonyms |
| `text_clean` | `str` | zero-PII text |
| `reply_to_id` | `int \| None` | thread linkage |
| `domain` | `str` | one of 9 taxonomy domains |
| `tags` | `list[str]` | technology tags |
| `sentiment_score` | `int` | −1 / 0 / 1 |

## SFTDialogue

Training dialogue produced by the conversation extractor:

- `dialogue_id`, `messages: list[{"role", "content"}]`
- `domain`, `source_thread_id`, `turn_count`

Exported to ChatML/ShareGPT/Alpaca JSONL and the parquet table.

## RAGChunk

Knowledge-base chunk:

- `chunk_id`, `title`, `content`, `domain`, `tags`
- `date_range`, `token_count`, `participants_count`

Exported to `rag_knowledge_base.parquet` / `.jsonl`; consumed by
`LocalRAGPipeline` (see [RAG Pipeline](../user-guide/rag-pipeline.md)).

## Invariants

- `author_*` fields never contain raw usernames/ids — pseudonymization is
  a schema-level guarantee (the manifest pins it).
- `domain` is always non-empty (falls back to `general_tech_chat`).
- Parquet dtypes are pinned by `DatasetValidator`; a mismatch fails
  `make validate` before any release gate sees the data.
