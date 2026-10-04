# Quick Start

## 1. Prepare Raw Exports

Place Telegram JSON exports in the project root:
```
ChatExport_2026-08-21/
ChatExport_2026-08-22/
...
```

Or configure custom paths in `.env`:
```
RICC__RAW_EXPORT_DIRS=["/path/to/exports"]
```

## 2. Configure Settings

```bash
cp .env.example .env
# Edit .env with your HF_TOKEN
```

## 3. Run Pipeline

### Via Python

```python
from ricc import Settings, MasterDataPipeline

settings = Settings()  # Loads from .env + params.yaml
pipeline = MasterDataPipeline()
summary = pipeline.run_all()

# summary contains:
# - execution_time_seconds
# - raw_messages_count
# - cleaned_messages_count
# - threads_count
# - sft_dialogues_count
# - rag_chunks_count
# - dpo_pairs_count
# - pii_stats
# - validation_passed
```

### Via CLI

```bash
# Full pipeline
it-pipeline run

# Skip NER (faster, regex-only)
it-pipeline run --skip-ner
```

### Via Docker Compose

```bash
docker compose up pipeline-runner
```

## 4. Check Outputs

```
dataset_output/
├── parquet/
│   ├── full_clean_messages.parquet
│   ├── sft_dialogues.parquet
│   └── rag_knowledge_base.parquet
├── jsonl/
│   ├── sft_sharegpt_format.jsonl
│   ├── sft_alpaca_format.jsonl
│   ├── sft_openai_messages.jsonl
│   ├── rag_chunks_kb.jsonl
│   └── dpo_preference_pairs.jsonl
└── samples/
    ├── sft_sample_preview.json
    ├── rag_sample_preview.json
    └── dpo_sample_preview.json

reports/
├── analytics_summary.json
├── DEEP_ANALYTICAL_REPORT.md
├── validation_results.json
├── probabilistic_pii_audit.json
├── drift_report.json
└── slo_verdict.json
```

## 5. Validate & Audit

```bash
# Validation suite
it-pipeline validate

# Probabilistic PII audit
it-pipeline audit-prob

# Drift monitoring
it-pipeline drift

# SLO gate
make slo
```

## 6. Launch Data Studio

```bash
make ui
# or
streamlit run app.py
```

Open http://localhost:8501 for interactive exploration.