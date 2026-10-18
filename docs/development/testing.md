# Testing

```bash
make test       # pytest -q
make coverage   # + term-missing report, 85% gate
```

## Layout

- `tests/test_*.py` — CPU-runnable unit tests (CI matrix: Python
  3.11/3.12/3.13/3.14).
- GPU-gated suites (real LoRA training, enterprise eval) run on
  self-hosted runners / manually and are **omitted from the coverage
  gate** via `[tool.coverage.run] omit`.

## Coverage Gate

`--cov-fail-under=85` over `src/`; the omit list in `pyproject.toml`
covers only the GPU-only training/eval scripts. Current state: ~736
tests passing.

## Test Types

| Kind | Examples |
|---|---|
| Unit | exporters round-trip, taxonomy, dedup, RAG ranking, SLO verdicts |
| Guards | `test_metrics_consistency.py` (published figures ↔ manifest), `test_report_consistency.py` (version in 4 places), `test_monitoring_stack.py` (dashboard/alerts ↔ metric registry), `test_secrets_baseline.py` (UTF-8, no BOM) |
| Mocked integration | inference chat sessions (RAG/CUDA/adapter paths), Prefect task failures |
| CI-only | real-corpus PII audit (100k sample), SLO gate on real reports |

## Fixtures & Markers

- `integration` marker: tests requiring locally generated dataset
  artifacts; deselected in the default unit run.
- Synthetic fixtures (100-row parquet) are built by
  `scripts/build_synthetic_parquet.py` and stamped synthetic so the
  [SLO gate](../user-guide/slo-gate.md) can never SHIP on them.

## Windows Note

If the default pytest temp dir is locked by another process
(`PermissionError` on `pytest-of-<user>`), pass a fresh basetemp:
`pytest --basetemp="$env:TEMP\pytest-ricc"`. CI (Linux) is unaffected.

## Writing Tests

- Prefer pure-function tests around the module's public API; mock only
  third-party boundaries (`peft.PeftModel.from_pretrained`, torch CUDA).
- Every new Prometheus metric must be asserted through
  `prometheus_client.REGISTRY` in a guard test so renames fail loudly.
