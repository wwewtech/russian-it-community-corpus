# ADR 0004: Prometheus/Grafana Monitoring Stack

**Status:** Accepted
**Date:** 2026-10-14
**Authors:** RICC maintainers
**Supersedes:** —

## Context

RICC already emitted Prometheus metrics (`src/observability`) and
structured logs, but the metrics went nowhere: no scrape config, no
dashboards, no alerts. The failure modes this allowed:

- **Alerts without a taxonomy.** "Errors went up" is not actionable;
  without a typed error classification, an OOM (retry with a smaller
  batch) and a corrupt adapter (fail fast) page identically.
- **Silent quality drift.** Zero-PII is the platform's core SLO, but
  nothing watched it continuously — the proof existed only as CI job
  artifacts, reviewed manually, after the fact.
- **Guard-free dashboards.** A renamed metric silently blanks a panel;
  dashboards rot invisibly because nothing compares them to the code.

## Decision

1. **A typed error taxonomy is the alert vocabulary** (ADR-coupled with
   the P3 work): `MLError`/`GPUError` hierarchy in `src/errors.py`;
   `classify_ml_exception()` maps raw torch/HF exceptions;
   `record_classified_ml_error()` records them into
   `ricc_errors_total{component, error_type}` with an allowlist
   (`ML_COMPONENTS`) that bounds label cardinality — ad-hoc components
   cannot inflate the series.
2. **Severity encodes the release asymmetry** (mirrors ADR-0003): PII
   leaks and `LEAKS_DETECTED` verdicts are `critical` (page now);
   drift/SLO-HOLD/validation/GPU-error spikes are `warning` (investigate
   before the next release).
3. **Every metric referenced by dashboards or alerts must exist in the
   registry.** `tests/test_monitoring_stack.py` parses dashboard PromQL
   and alert expressions and asserts each `ricc_*` name against
   `prometheus_client.REGISTRY`; the ServiceMonitor → Service →
   Deployment port chain is asserted too. Renaming a metric without
   updating the stack fails CI.
4. **Gauges only move on success.** `record_lora_result()` sets
   loss/VRAM gauges only for successful runs — a failed run must not
   clobber the last known good reading; failures are counted
   (`ricc_lora_train_total{status="failure"}`) instead.
5. **Two deployment surfaces, one definition.** k8s uses a
   ServiceMonitor + PrometheusRule CRDs
   (`deploy/k8s/*.yaml`); local Docker Compose uses an `observability`
   profile that mounts the same alert rules and provisions the same
   dashboard. The dashboard/definitions are never duplicated.

## Consequences

- The Streamlit app must start the metrics server through
  `maybe_start_metrics_server()` (rerun-safe, port-conflict-safe) with
  `RICC_METRICS_PORT` — a bare `start_http_server` would crash on
  Streamlit's second rerun.
- Alert thresholds (e.g. `RiccMLComponentErrorSpike > 0.05/s`) are
  starting points; they are versioned with the rules file and tuned by
  observing real runs, not silently.
- Compose Grafana runs with an anonymous viewer for local demo use; the
  k8s deployment inherits cluster auth. Neither surface stores secrets.

## References

- Metric catalog + helpers: `docs/api/observability.md`
- Error taxonomy: ADR context in `docs/api/errors.md`
- Alert rules: `deploy/k8s/prometheus-rules.yaml`
