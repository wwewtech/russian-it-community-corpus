# CLI Commands Reference

## run

```
python cli.py run [--skip-ner]
```

Executes `MasterDataPipeline.run_all()`. `--skip-ner` replaces the
Natasha NER layer with regex-only scrubbing (fast, weaker on names).
Prometheus stage counters increment on every stage
(`ricc_pipeline_stage_total`).

## analyze

```
python cli.py analyze [--limit N]
```

Loads `full_clean_messages.parquet`, runs `DeepChatAnalyzer`; `--limit`
bounds the NLP-lemmatization sample (default 100000). Writes
`reports/analytics_summary.json`.

## validate

```
python cli.py validate
```

`DatasetValidator.validate_all()` over the three canonical parquet
artifacts; writes `reports/validation_results.json`. Non-zero exit on
failure.

## benchmark

```
python cli.py benchmark
```

Exports the 100-question domain benchmark to
`reports/domain_benchmark_100.json` and prints the preview table.

## audit-prob

```
python cli.py audit-prob [--parquet PATH] [--sample-size N]
                         [--confidence F] [--tolerance F] [--output PATH]
```

Stratified probabilistic PII audit with Wilson CI bounds (defaults:
50000 sample, 0.99 confidence, 1e-4 tolerance). Writes the report and
exits non-zero when the verdict is `LEAKS_DETECTED`.

## drift

```
python cli.py drift [--reference PATH] [--current PATH] [--output PATH]
```

Defaults to a self-comparison baseline (stable). With two frozen
snapshots produces real temporal drift. Writes
`reports/drift_report.json` and records the drift gauges.

## orchestrate

```
python cli.py orchestrate [--skip-pipeline] [--sample-size N]
```

Runs the Prefect flow; without Prefect installed it degrades to
sequential execution with a warning.

## chat

```
python cli.py chat [--model ID] [--adapter ID] [--no-rag]
```

Interactive session: loads model + optional LoRA adapter
(`validate_adapter_path` fails fast on corrupt weights with
`AdapterLoadError`) and RAG context injection. Exit commands: `exit`,
`quit`, `q`.

## rag

```
python cli.py rag QUERY [--top-k N]
```

One-shot hybrid search over the knowledge base; prints title, domain,
score and content per hit.
