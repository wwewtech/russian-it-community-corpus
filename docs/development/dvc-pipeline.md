# DVC Pipeline

DVC (optional extra: `pip install .[platform]`) version-controls the
data pipeline stages and `params.yaml`.

```bash
make dvc-repro      # = dvc repro
dvc status
dvc dag
```

## Params Contract

`params.yaml` is the single source of truth consumed via
`settings.load_params()`:

```yaml
pii_audit:      # sample_size, confidence, max_leak_tolerance
drift:          # psi/js/vocab thresholds, buckets, top_k
benchmarks:     # humaneval_tasks, rummlu_questions, confidence
domain_taxonomy: # 9 domains × keyword lists
```

Code reads these values at import time in `ricc/monitoring/drift.py` —
changing a threshold takes effect on the next run, no code edits.

## Datasets on the Hub

The published dataset lives at
`huggingface.co/datasets/wwewtech/russian-it-community-corpus`; the
pinned revision snapshot is `reports/pinned_hub_manifest.json`, and
`ricc/validation/hub_reconciliation.py` reconciles local artifacts against
it (status feeds the SLO gate).

## When to Use DVC vs the Plain Pipeline

- `dvc repro` is for parameterized, cacheable reruns of the analysis
  stages on a workstation.
- `make run` remains the canonical full-pipeline path (it rebuilds from
  raw exports; the raw exports are the source of truth).
- CI does not require DVC — validation jobs build synthetic fixtures and
  run the real-corpus audits directly.
