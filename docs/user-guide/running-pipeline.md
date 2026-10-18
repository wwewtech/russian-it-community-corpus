# Running the Pipeline

## Full Run

```bash
make run          # = python main.py
# or
python cli.py run # same stages, with --skip-ner option
```

The pipeline (`src/pipeline.py`, `MasterDataPipeline`) executes 7 stages
in order, each timed and recorded into Prometheus
(`ricc_pipeline_stage_total`, `ricc_pipeline_stage_duration_seconds`):

1. **Ingestion** — merge raw Telegram exports (`ChatExport_*` dirs).
2. **PII anonymization** — UnifiedPIIAnonymizer (regex + Natasha NER).
3. **Deduplication** — exact hash + MinHash LSH.
4. **Taxonomy tagging** — 9 IT domains, Aho-Corasick keywords.
5. **Graph extraction** — thread DAG reconstruction.
6. **Export** — Parquet / JSONL (ShareGPT, Alpaca, ChatML, RAG, DPO).
7. **Analytics & validation** — statistical report + dataset validation.

Raw exports are configured in `src/settings.py` (`RAW_EXPORT_DIRS`);
outputs land in `dataset_output/` (`parquet/`, `jsonl/`), reports in
`reports/`.

## Prerequisites

- Python 3.11–3.14, dependencies installed (`make install`).
- Natasha NER downloads its models on first run into `.hf_cache/`
  (set `HF_HOME` or let the bootstrap point it at the repo-local cache).
- Input data: place Telegram `result.json` exports under the configured
  `RAW_EXPORT_DIRS`.

## Useful Commands

```bash
make analyze   # statistical/semantic report on exports
make validate  # schema + zero-PII audit
make slo       # SHIP/HOLD release verdict from reports/
make reports-real  # regenerate JSON reports from REAL parquet
```

## Orchestration

Prefect is optional (`pip install .[platform]`):

```bash
make orchestrate   # run the Prefect flow (graceful fallback without it)
make dvc-repro     # reproduce the DVC pipeline
```

Without Prefect the flow degrades to sequential execution.

## Metrics During a Run

If the Streamlit app or another process started the metrics server, stage
counters/durations are live on `http://localhost:9090/metrics` and in the
Grafana [observability profile](../api/observability.md).

## After the Run

1. `make validate` — confirm zero-PII and schema checks pass.
2. `make slo` — release verdict; `HOLD` means a data-quality check failed
   (see `reports/slo_verdict.json` for the failing check names).
3. [Drift monitoring](drift-monitoring.md) — compare against a frozen
   reference snapshot before releasing.
