# DVC Pipeline

DVC (optional extra: `pip install .[platform]`) version-controls the
data pipeline stages and the parameters that drive them.

```bash
make dvc-repro      # = dvc repro
dvc status
dvc dag
```

> **This pipeline actually runs.** It did not until 2026-10-21: the
> stages were inline `python -c "..."` blocks, which do not survive a
> Windows shell (`cmd.exe` splits the multi-line quoted string and the
> stage dies with `'from' is not recognized as an internal or external
> command`), and `dvc init` had never been executed. Every stage is now
> a real script, and `dvc repro ingest` ingests the 2 911 754 raw
> messages and writes `dvc.lock`. See
> [ADR 0007](../adr/0007-streaming-export-and-dvc.md).

## The stages

| Stage | Script | Output |
|---|---|---|
| `ingest` | `scripts/dvc_stages.py ingest` | `dataset_output/intermediate/raw_messages.pkl.gz` |
| `pii_anonymize` | `... pii_anonymize` | `cleaned_messages.pkl.gz` |
| `exact_dedup` | `... exact_dedup` | `dedup_exact.pkl.gz` |
| `minhash_dedup` | `... minhash_dedup` | `dedup_minhash.pkl.gz` |
| `taxonomy` | `... taxonomy` | `tagged_messages.pkl.gz` |
| `graph` | `... graph` | `threaded_messages.pkl.gz` |
| `extract` | `... extract` | `extracted.pkl.gz` |
| `export` | `... export` | the parquet and JSONL artefacts |
| `analytics` | `... analytics` | `reports/analytics_summary.json` |
| `validate` | `... validate` | the validation and PII audit reports |
| `drift` | `... drift` | `reports/drift_report.json` |
| `slo` | `... slo` | `reports/slo_verdict.json` |

Each stage is invocable on its own:

```bash
python scripts/dvc_stages.py export
```

## The raw exports are a real dependency

DVC does not expand globs in a `deps:` entry — `ChatExport_*/result.json`
would be recorded literally and then reported as a *deleted* dependency.
The ingest stage therefore lists every export explicitly, and
`scripts/refresh_dvc_deps.py` keeps the list in sync:

```bash
python scripts/refresh_dvc_deps.py           # refresh the list
python scripts/refresh_dvc_deps.py --check   # CI: fail if out of sync
```

The exports are gitignored (they are data, not source) and tracked by DVC,
so a change in the corpus re-runs the ingestion.

## Params Contract

`params.yaml` is the single source of truth, consumed through
`settings` (loaded at boot) and by the stages that need them:

```yaml
pii_audit:      # sample_size, confidence, max_leak_tolerance
drift:          # psi/js/vocab thresholds, buckets, top_k
benchmarks:     # humaneval_tasks, rummlu_questions, confidence
domain_taxonomy: # 8 domains × keyword lists
```

A parameter is declared in `dvc.yaml` only by the stage that **reads**
it, so editing a threshold re-runs exactly the stage it affects. The
values themselves are read at runtime — `settings.pii_audit` and
`settings.benchmarks` used to be dead configuration duplicated as
literals in the code, the CLI and the Prefect flow, so editing
`params.yaml` changed nothing at runtime.

## The drift stage needs a frozen reference

The drift stage compares the current corpus against a **frozen
snapshot**, never against itself (comparing the corpus with itself can
only ever report "stable"):

```bash
RICC_DRIFT_REFERENCE=dataset_output/parquet/full_clean_messages_reference.parquet \
    dvc repro drift
```

Without a reference the stage says so and skips, rather than producing a
meaningless "stable" verdict.

## Remote storage

The default remote is a local directory outside the working copy:

```bash
dvc remote add -d storage ../ricc-dvc-storage
dvc push
```

For a shared setup point it at object storage instead — the rest of the
configuration does not change:

```bash
dvc remote add -d s3remote s3://my-bucket/ricc
```

## Datasets on the Hub

The published dataset lives at
`huggingface.co/datasets/wwewtech/russian-it-community-corpus`; the
pinned revision snapshot is `reports/pinned_hub_manifest.json`, and
`ricc/validation/hub_reconciliation.py` reconciles local artifacts against
it (status feeds the SLO gate).

## When to Use DVC vs the Plain Pipeline

- `dvc repro` is for parameterized, cacheable reruns of the pipeline
  stages; the lock file records every hash.
- `make run` remains the canonical full-pipeline path (it rebuilds from
  raw exports; the raw exports are the source of truth).
- CI does not require DVC for the validation jobs — they build synthetic
  fixtures and run the real-corpus audits directly — but there is a
  `dvc-consistency` job that checks the pipeline definition, the
  dependency list and the lock file agree.
