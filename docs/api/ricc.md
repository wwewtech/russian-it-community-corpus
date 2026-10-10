# ricc — Public API

```python
import ricc

print(ricc.__version__)  # "12.1.0"
```

## Configuration

### `Settings`

Unified configuration via `pydantic-settings.BaseSettings`.

```python
from ricc import Settings, settings

# Global instance (loads from .env + params.yaml)
settings.BASE_DIR
settings.PARQUET_OUTPUT_DIR
settings.MINHASH_NUM_PERM
settings.DOMAIN_TAXONOMY
# ...

# Custom instance
custom = Settings(minhash_num_perm=256, pii_audit=PIIAuditSettings(sample_size=100000))
```

### Environment Variables

All settings overrideable via `RICC__` prefix:
```bash
RICC__MINHASH_NUM_PERM=256
RICC__PII_AUDIT__SAMPLE_SIZE=100000
```

## Pipeline

### `MasterDataPipeline`

End-to-end data curation pipeline.

```python
from ricc import MasterDataPipeline

pipeline = MasterDataPipeline(raw_export_dirs=[...])
summary = pipeline.run_all()
```

**Stages:**
1. **ingestion** — Load & merge Telegram exports
2. **pii_anonymization** — Regex + NER scrubbing + pseudonymization
3. **deduplication** — Exact + MinHash LSH
4. **taxonomy_tagging** — Domain classification + keyword tagging
3. **graph_extraction** — Thread DAG + SFT/DPO/RAG extraction
4. **export** — Parquet + JSONL (ShareGPT, Alpaca, ChatML, RAG, DPO)
5. **analytics_validation** — Deep analytics + validation + PII audit

## Observability

### Structured Logging

```python
from ricc import configure_logging, get_logger, pipeline_logger

configure_logging(json_output=True)
log = get_logger("my_module")
log.info("event", key="value")

pipeline_logger.stage_start("my_stage")
pipeline_logger.stage_end("my_stage", success=True, count=42)
```

### Prometheus Metrics

```python
from ricc import (
    PIPELINE_STAGE_DURATION,
    PIPELINE_STAGE_TOTAL,
    PII_AUDIT_LEAKS_FOUND,
    DRIFT_PSI_VALUE,
    VALIDATION_RESULT,
    start_metrics_server,
)

# Start /metrics endpoint
start_metrics_server(9090)

# Record metrics
PIPELINE_STAGE_DURATION.labels(stage="ingestion").observe(1.5)
PIPELINE_STAGE_TOTAL.labels(stage="ingestion", status="success").inc()
```

**Key Metrics:**
- `ricc_pipeline_stage_duration_seconds` — Histogram per stage
- `ricc_pipeline_stage_total` — Counter per stage/status
- `ricc_pii_audit_leaks_found_total` — Counter per PII category
- `ricc_drift_psi_value` — Gauge
- `ricc_validation_result_total` — Counter per check/status
- `ricc_slo_gate_verdict_total` — Counter per verdict

## Error Handling

### Exception Hierarchy

```python
from ricc import (
    RICCError,
    PipelineError, IngestionError, PIIAnonymizationError, ...,
    ValidationError, SchemaValidationError, PIIValidationError, ...,
    DataQualityError, DriftDetectedError, ArtifactIntegrityError, ...,
    ConfigurationError, MissingDependencyError,
    ExternalServiceError, HuggingFaceHubError, PrefectError,
    ResourceError, VRAMExhaustedError, DiskSpaceError,
)
```

All exceptions include:
- `message` — Human-readable message
- `component` — Origin component
- `context` — Structured context dict
- `recoverable` — Boolean hint for retry logic

### Functional Result Type

```python
from ricc import Result, Ok, Err


def risky_operation() -> Result[str, ValidationError]:
    if not valid:
        return Err(ValidationError("invalid input"))
    return Ok("success")


result = risky_operation()
if result.is_ok:
    print(result.value)
else:
    print(result.error)
```