# Observability — Metrics, Logging, Recording Helpers

RICC exposes Prometheus metrics via `prometheus_client`, structured JSON
logging via `structlog`, and a set of `record_*` helpers that keep metric
names in one place. Everything is defined in `ricc/observability/__init__.py`.

## Metrics Endpoint

The Streamlit Data Studio starts the metrics server automatically:

```python
from ricc.observability import maybe_start_metrics_server

maybe_start_metrics_server(int(os.getenv("RICC_METRICS_PORT", "9090")))
```

`maybe_start_metrics_server()` starts the server once per process and is
safe under Streamlit reruns (a port already bound is logged, not raised).
In k8s the endpoint is scraped by
`deploy/k8s/data-studio-servicemonitor.yaml`; with Docker Compose use the
`observability` profile:

```bash
docker compose --profile observability up   # Prometheus :9091, Grafana :3000
```

## Metric Families

| Metric | Type | Labels | Produced by |
|---|---|---|---|
| `ricc_pipeline_stage_total` | counter | `stage`, `status` | `time_stage()` — every pipeline run |
| `ricc_pipeline_stage_duration_seconds` | histogram | `stage` | `time_stage()` |
| `ricc_pipeline_messages_processed_total` | counter | `stage` | `record_stage_messages()` |
| `ricc_pii_audit_leaks_found_total` | counter | `category` | `record_pii_leak()` |
| `ricc_pii_audit_duration_seconds` | histogram | — | audit harness |
| `ricc_pii_audit_verdict_total` | counter | `verdict` | audit harness |
| `ricc_drift_psi_value` | gauge | — | `record_drift_metrics()` |
| `ricc_drift_js_value` | gauge | — | `record_drift_metrics()` |
| `ricc_drift_vocab_jaccard` | gauge | — | `record_drift_metrics()` |
| `ricc_drift_verdict_total` | counter | `verdict` | `record_drift_metrics()` |
| `ricc_validation_result_total` | counter | `check`, `status` | `record_validation()` |
| `ricc_lora_train_duration_seconds` | histogram | `model`, `adapter` | `record_lora_result()` |
| `ricc_lora_train_steps_total` | counter | `model`, `adapter` | `record_lora_result()` |
| `ricc_lora_train_loss` | gauge | `model`, `adapter` | `record_lora_result()` |
| `ricc_lora_vram_usage_bytes` | gauge | `model`, `adapter` | `record_lora_result()` |
| `ricc_lora_train_total` | counter | `model`, `adapter`, `status` | `record_lora_result()` |
| `ricc_dataset_messages_total` | gauge | `dataset` | catalog regeneration |
| `ricc_dataset_size_bytes` | gauge | `dataset`, `format` | catalog regeneration |
| `ricc_slo_gate_verdict_total` | counter | `verdict` | SLO gate |
| `ricc_slo_check_result_total` | counter | `check`, `status` | `record_slo_check()` |
| `ricc_http_requests_total` | counter | `method`, `endpoint`, `status` | HTTP helpers |
| `ricc_http_request_duration_seconds` | histogram | `method`, `endpoint` | HTTP helpers |
| `ricc_errors_total` | counter | `component`, `error_type` | `record_error()` / ML recorders |

## Recording Helpers

```python
from ricc.observability import (
    record_stage_messages,  # (stage, count)
    record_pii_leak,  # (category, count=1)
    record_validation,  # (check, passed)
    record_slo_check,  # (check, passed)
    record_error,  # (component, error_type)
    record_ml_error,  # (typed_error) -> component
    record_classified_ml_error,  # (raw_exc, **ctx) -> typed RICCError
    record_lora_result,  # (model=, adapter=, status=, duration_sec=, loss=, steps=, vram_bytes=)
    record_drift_metrics,  # (drift_report_dict)  — in ricc.monitoring.drift
    time_stage,  # context manager: duration + success/failure counter
    timed_stage,  # decorator form of time_stage
)
```

`record_lora_result()` sets gauges only on `status="success"` — a failed
run must not clobber the last known good loss/VRAM reading.

`record_classified_ml_error()` maps a raw exception through the
[GPU/ML taxonomy](errors.md) and increments
`ricc_errors_total{component, error_type}`; components are validated
against the `ML_COMPONENTS` allowlist to bound label cardinality.

## Dashboards & Alerts

- Dashboard: `deploy/monitoring/ricc-overview.json` (uid `ricc-overview`),
  18 panels grouped into Pipeline health / Zero-PII gates / LoRA training /
  Errors (P3 taxonomy) / Release gates.
- Alert rules: `deploy/k8s/prometheus-rules.yaml` — PII leaks and
  LEAKS_DETECTED verdicts are `critical` (zero-PII is the core SLO);
  drift, SLO HOLD, validation failures, GPU/ML error spikes and LoRA
  train failures are `warning`.
- Guard tests: `tests/test_monitoring_stack.py` keeps dashboard PromQL,
  alert expressions and the ServiceMonitor/Deployment wiring in sync with
  this module — renaming a metric without updating the stack fails CI.

## Structured Logging

```python
from ricc.observability import configure_logging, get_logger

configure_logging(json_output=True)  # done once by setup_runtime_env()
log = get_logger("my_module")
log.info("stage_completed", stage="dedup", removed=123)
```

Entry-point scripts call `ricc.bootstrap.setup_runtime_env()`, which
configures UTF-8 stdio, HF cache env vars and JSON logging in one call.
