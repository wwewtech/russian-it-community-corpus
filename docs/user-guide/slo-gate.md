# SLO Gate — SHIP/HOLD Release Verdict

`ricc/monitoring/slo_gate.py` aggregates all quality signals into one
release verdict. Exit code 0 = **SHIP**, 1 = **HOLD**; missing proof is
fail-closed for privacy signals.

```bash
make slo   # = python -m ricc.monitoring.slo_gate --json-out reports/slo_verdict.json
```

## Checks

| Check | Source | Blocking | Missing source |
|---|---|---|---|
| `validation` | `reports/validation_results.json` `overall_passed` | yes | HOLD |
| `pii-audit` | `reports/probabilistic_pii_audit.json` verdict | yes | HOLD (fail-closed) |
| `drift` | `reports/drift_report.json` `overall_verdict` | only `significant_drift`; `moderate_drift` is informational (sampling noise) | skip (warning) |
| `sft-quality` | `reports/sft_quality_report.json` | when present | skip |
| `pipeline-volume` | `reports/pipeline_execution_stats.json` `cleaned > 0` | yes | skip |
| `artifact-manifest` | sha256 verify of canonical parquet against `reports/dataset_manifest.json` | yes | HOLD (fail-closed) |
| `hub-reconciliation` | `reports/dataset_reconciliation_report.json` status | only on mismatch | skip (warning) |

## Synthetic-Smoke Guard

CI builds 100-row synthetic fixtures for speed. Reports stamped
`_source.synthetic=true` (`reports/synthetic_fixture.json`) are treated as
SMOKE: the gate downgrades SHIP to HOLD with an explicit
`synthetic-smoke` check, so a fake-data run can never green-light a
release.

## Verdict Semantics

- **SHIP** — every blocking check passed on real artifacts.
- **HOLD** — read `reports/slo_verdict.json`:
  `checks[]` lists each check with `passed` and a `detail` explaining the
  failure (e.g. `manifest verify failed: digest mismatch for
  dataset_output/parquet/full_clean_messages.parquet`).

## CI Integration

The `SLO Release Gate (Real Corpus)` job runs the gate against
real-corpus reports regenerated in CI; the release workflow gates
publishing on it. The verdict counter
`ricc_slo_gate_verdict_total{verdict}` feeds the `RiccSloGateHold` alert.

## Running in Tests

`evaluate(reports_dir)` is a pure function returning a `SloVerdict`
dataclass — `tests/test_hybrid_rag_slo.py` covers SHIP, HOLD and
synthetic-smoke paths.
