# CLI Overview

`cli.py` (installed as `it-pipeline`) is the operational entry point.
Heavy dependencies are imported lazily per subcommand, so `--help` starts
instantly.

```bash
python cli.py --help
```

## Command Map

| Command | Purpose | Docs |
|---|---|---|
| `run` | full 7-stage pipeline (`--skip-ner` for fast regex-only) | [Running the Pipeline](../user-guide/running-pipeline.md) |
| `analyze` | statistical/semantic report (`--limit` for NLP sample) | [Analytics & Reports](../user-guide/analytics-reports.md) |
| `validate` | schema + zero-PII validation | [Validation & Audits](../user-guide/validation-audits.md) |
| `benchmark` | export/show 100 domain benchmark questions | — |
| `audit-prob` | probabilistic PII audit (Wilson CI) | [PII Anonymization](../user-guide/pii-anonymization.md) |
| `drift` | drift monitoring between snapshots | [Drift Monitoring](../user-guide/drift-monitoring.md) |
| `orchestrate` | Prefect flow (graceful fallback) | — |
| `chat` | interactive LLM + RAG terminal session | [RAG Pipeline](../user-guide/rag-pipeline.md) |
| `rag` | one-shot knowledge-base search | [RAG Pipeline](../user-guide/rag-pipeline.md) |

## Examples

```bash
python cli.py run --skip-ner
python cli.py audit-prob --sample-size 50000 --confidence 0.99
python cli.py drift --reference old.parquet --current new.parquet
python cli.py chat --model Qwen/Qwen2.5-1.5B-Instruct --adapter heavyweight_qwen2.5_coder_7b
python cli.py rag "как настроить ingress в kubernetes" --top-k 5
```

## Makefile Shortcuts

`make run|analyze|validate|audit|audit-prob|drift|slo|orchestrate|test|lint|format|typecheck-strict|security|reports|reports-real|ui` — see
`make help`.

Every command runs `setup_runtime_env()` first: UTF-8 stdio, HF cache,
structured JSON logging.
