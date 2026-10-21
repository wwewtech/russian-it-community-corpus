# Drift Monitoring — PSI, Jensen-Shannon, Vocabulary Jaccard

`ricc/monitoring/drift.py` (`DatasetDriftMonitor`) compares a reference
dataset snapshot against the current one and quantifies drift with three
standard metrics. Thresholds live in `params.yaml` (single source of
truth).

## Metrics

| Metric | Stable | Moderate | Significant | Detects |
|---|---|---|---|---|
| **PSI** (message-length distribution) | < 0.10 | 0.10–0.25 | > 0.25 | general distribution shift |
| **JS divergence** (bits, domain shares) | < 0.05 | 0.05–0.20 | > 0.20 | topical shifts (community changing focus) |
| **Vocab Jaccard** (top-500 tokens) | > 0.90 | 0.75–0.90 | < 0.75 | lexical drift in corpus language |

Overall verdict: `significant_drift` if any metric is significant, else
`moderate_drift` if any is moderate, else `stable`.

## Running

```bash
# Self-comparison baseline (reference == current) => stable
python scripts/run_drift_monitoring.py

# Real temporal drift: point at two frozen snapshots
python scripts/run_drift_monitoring.py \
    --reference dataset_output/parquet/full_clean_messages_2026Q2.parquet \
    --current   dataset_output/parquet/full_clean_messages.parquet
```

The script writes `reports/drift_report.json` and prints the verdict.

## Interpreting the Report

```json
{
  "overall_verdict": "moderate_drift",
  "metrics": {
    "length_psi":       {"value": 0.12, "verdict": "moderate_drift"},
    "domain_distribution": {"js_divergence_bits": 0.03, "verdict": "stable",
                            "biggest_share_shift_domain": "ai_ml_nlp",
                            "biggest_share_shift_pp": 1.7},
    "vocabulary":       {"jaccard_overlap": 0.93, "verdict": "stable",
                         "new_top_tokens": 31, "dropped_top_tokens": 27}
  },
  "thresholds": {"psi": {"stable": 0.1, "moderate": 0.25}, "...": "..."}
}
```

- `biggest_share_shift_pp` names the domain that moved the most
  (percentage points) — start the investigation there.
- A **self-comparison baseline on two random subsamples** of the same
  corpus can show moderate vocabulary drift from sampling noise alone.
  That is informational: the SLO gate treats `moderate_drift` as
  non-blocking and blocks only on `significant_drift`.

## Prometheus Gauges (12.2.0+)

Every generated report is pushed to the metrics registry by
`record_drift_metrics()` (wired into `generate_report()`,
`run_drift_monitoring.py` and `regenerate_real_reports.py`):

| Gauge | Meaning |
|---|---|
| `ricc_drift_psi_value` | latest PSI |
| `ricc_drift_js_value` | latest JS divergence (bits) |
| `ricc_drift_vocab_jaccard` | latest vocab overlap |
| `ricc_drift_verdict_total{verdict}` | per-metric and overall verdict counters |

The Grafana panel "Drift metrics" plots the three gauges; the alert
`RiccDriftSignificant` fires on any `significant_drift` verdict.

## In Code

```python
from src.monitoring.drift import DatasetDriftMonitor

monitor = DatasetDriftMonitor(ref_df, cur_df)
report = monitor.run()  # dict, not persisted
monitor.generate_report("reports/drift_report.json")  # run + record metrics + persist
```

`DatasetDriftMonitor` depends only on numpy/pandas; the observability
import is lazy so the module stays usable without a metrics registry.

## Related

- `make drift` — CLI shortcut.
- The drift report is one input to the [SLO gate](slo-gate.md) (SHIP/HOLD).
- Threshold tuning: edit `params.yaml` (`drift:` section), not the code.
