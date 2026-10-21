# Errors — Typed Exception Hierarchy

All RICC exceptions derive from `RICCError` and carry structured context
(`component`, `context` dict, `recoverable` flag) so failures can be
logged, counted in Prometheus, and acted on programmatically.

```python
from ricc.errors import GPUOutOfMemoryError, classify_ml_exception, RICCError
```

## Hierarchy Overview

```
RICCError
├── PipelineError            (stage-tagged: ingestion, deduplication, ...)
│   ├── IngestionError
│   ├── PIIAnonymizationError
│   ├── DeduplicationError
│   ├── TaxonomyError
│   ├── GraphExtractionError
│   ├── ExportError
│   └── AnalyticsError
├── ValidationError          (check-tagged: schema, pii_audit, sft, jsonl, parquet, slo)
│   ├── SchemaValidationError
│   ├── PIIValidationError
│   ├── SFTValidationError
│   ├── JSONLValidationError
│   ├── ParquetValidationError
│   └── SLOValidationError
├── DataQualityError
│   ├── DriftDetectedError   (psi/js/jaccard values in context)
│   └── ArtifactIntegrityError / HubReconciliationError
├── ConfigurationError / MissingDependencyError
├── ExternalServiceError
│   ├── HuggingFaceHubError
│   └── PrefectError
├── ResourceError
│   ├── DiskSpaceError
│   └── GPUOutOfMemoryError  ← P3 taxonomy
└── MLError / GPUError       ← P3 taxonomy (below)
```

## GPU / ML Taxonomy (P3)

Added in 12.2.0. Motivation: LoRA training and eval harnesses previously
raised bare `RuntimeError` or let torch exceptions propagate, making it
impossible to distinguish "retry with a smaller batch" (OOM) from
"fail fast — weights are corrupt" (adapter load) in logs and alerts.

### MLError — training / inference failures

| Class | `component` | Extra context |
|---|---|---|
| `MLError` | `ml` | `model_name`, `adapter_id` |
| `TrainingError` | `training` | `step`, `phase` |
| `InferenceError` | `inference` | — |
| `EvaluationError` | `evaluation` | `benchmark`, `variant` |
| `TokenizerError` | `tokenizer` | — |
| `CheckpointError` | `checkpoint` | `checkpoint_path` |
| `AdapterLoadError` | `adapter` | `adapter_path`, `base_model` |
| `QuantizationError` | `quantization` | `quant_type` |

`AdapterLoadError` must fail fast: silently falling back to the base model
produces misleading "attached" metrics (see `ricc/inference.py`).

### GPUError — device failures

| Class | Raised when |
|---|---|
| `GPUNotAvailableError` | CUDA requested (`device_map="auto"`, fp16) but no GPU is visible |
| `GPUDriverError` | driver / cuBLAS / NCCL / NVML init failures |
| `GPUOutOfMemoryError` | VRAM exhausted; subclass of `ResourceError`, so it also carries `resource_type="vram"`, `current_usage` (requested MB), `limit` (available MB) |

`VRAMExhaustedError` is kept as a legacy alias of `GPUOutOfMemoryError` —
existing code and dashboards keep working.

## Classifying Raw Exceptions

Train/eval scripts wrap broad `except` blocks with
`classify_ml_exception`, which pattern-matches OOM, CUDA-unavailable and
driver errors and returns the typed exception with the original chained
into context:

```python
from ricc.errors import classify_ml_exception

try:
    trainer.train()
except Exception as exc:
    typed = classify_ml_exception(exc, model_name=model_name, adapter_id=adapter_id)
    logger.error(f"training failed: {typed}")
    raise
```

- OOM (`torch.cuda.OutOfMemoryError` or "CUDA out of memory" in message)
  → `GPUOutOfMemoryError` — **retryable**: smaller batch, gradient
  checkpointing, 4-bit quantization.
- "no CUDA" / "device not found" → `GPUNotAvailableError`.
- cuBLAS / NCCL / NVML markers → `GPUDriverError`.
- Anything else → `TrainingError` with `original_error` in context.

`is_oom_error(exc)` is the standalone predicate.

## Recording Errors into Prometheus

`ricc/observability.record_classified_ml_error()` combines classification
and recording in one call (used by all train/eval entry points):

```python
from ricc.observability import record_classified_ml_error

except Exception as exc:
    typed = record_classified_ml_error(exc, model_name=..., adapter_id=...)
    raise
```

It increments `ricc_errors_total{component, error_type}`; `component` is
validated against the `ML_COMPONENTS` allowlist, so untyped exceptions
cannot inflate label cardinality. The Grafana panel "Errors by component"
and the `RiccMLComponentErrorSpike` alert consume this series.

## Full Public API

Import everything from the package root — `ricc/ricc/__init__.py`
re-exports the whole hierarchy plus the Result type (`Ok`/`Err`) used for
functional error handling in pipeline stages.
