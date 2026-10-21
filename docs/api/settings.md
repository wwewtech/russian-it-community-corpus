# Settings API

`ricc/settings.py` — single source of configuration, built on
`pydantic-settings`.

```python
from ricc.settings import Settings, settings  # settings is a singleton

settings.PARQUET_OUTPUT_DIR  # Path
settings.load_params()  # dict — full params.yaml
```

## Settings Fields

| Field | Meaning |
|---|---|
| `RAW_EXPORT_DIRS` | `list[Path]` — Telegram export roots to ingest |
| `OUTPUT_DIR` | `dataset_output/` root |
| `PARQUET_OUTPUT_DIR` | `dataset_output/parquet/` |
| `JSONL_OUTPUT_DIR` | `dataset_output/jsonl/` |
| `SAMPLES_OUTPUT_DIR` | `dataset_output/samples/` |
| `REPORTS_DIR` | `reports/` |

`load_params()` parses `params.yaml` — the single source of truth for
PII-audit parameters (`sample_size`, `confidence`,
`max_leak_tolerance`), drift thresholds, benchmark sizes and the domain
taxonomy keyword tables.

## Environment Variables

`ricc/bootstrap.setup_runtime_env()` (called by every entry point)
applies process-level env defaults **without overriding** user-set values:

- `HF_HOME` → repo-local `.hf_cache`
- `HF_HUB_DISABLE_SYMLINKS_WARNING=1`
- `TOKENIZERS_PARALLELISM=false`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (opt-in flag
  `pytorch_alloc_conf=True` for training scripts)

## Conventions

- Components read configuration through `settings`/`load_params()`, not
  ad-hoc `os.getenv` — except the metrics port
  (`RICC_METRICS_PORT`), which is app-level, not data-level.
- Threshold tuning happens in `params.yaml`, never in code (drift
  thresholds, audit parameters).
- `ricc/pipeline.py` re-exports the path constants for backward
  compatibility with tests.
