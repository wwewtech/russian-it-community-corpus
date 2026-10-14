# Pipeline API

`src.pipeline.MasterDataPipeline` — end-to-end orchestration.

```python
from src.pipeline import MasterDataPipeline

pipeline = MasterDataPipeline()  # uses settings.RAW_EXPORT_DIRS
summary = pipeline.run_all()  # runs all 7 stages
```

## Constructor

```python
MasterDataPipeline(raw_export_dirs: list[Path] | None = None)
```

- `raw_export_dirs`: override the configured export directories
  (defaults to `settings.RAW_EXPORT_DIRS`).
- Instantiates the anonymizer, deduplicators, tagger, graph builders and
  all four exporters.

## Methods

| Method | Returns | Description |
|---|---|---|
| `run_all()` | `dict[str, Any]` | executes stages 1–7; raises typed `PipelineError` subclasses on stage failure |

## State After a Run

| Attribute | Type | Stage |
|---|---|---|
| `raw_messages` | `list[NormalizedMessage]` | 1 |
| `cleaned_messages` | `list[CleanedMessage]` | 2–4 |
| `threads` | `dict[int, list[CleanedMessage]]` | 5 |
| `sft_dialogues` | `list[SFTDialogue]` | 5 |
| `rag_chunks` | `list[RAGChunk]` | 5 |
| `dpo_pairs` | `list[dict[str, Any]]` | 5–6 |
| `analytics_report` | `FullAnalysisReport` | 7 |
| `validation_report` | `dict[str, Any]` | 7 |

## Module-Level Compatibility

`OUTPUT_DIR`, `PARQUET_OUTPUT_DIR`, `JSONL_OUTPUT_DIR`,
`SAMPLES_OUTPUT_DIR`, `REPORTS_DIR`, `RAW_EXPORT_DIRS` are re-exported
from `src.settings` for backward compatibility with tests that patch
them.

## CLI

```bash
python cli.py run [--skip-ner]   # fast regex-only anonymization
```

See [Running the Pipeline](../user-guide/running-pipeline.md) for the
operational view.
