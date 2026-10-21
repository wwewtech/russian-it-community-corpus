# ricc — Public API

```python
import ricc

print(ricc.__version__)  # "12.2.0"
```

`import ricc` is cheap: `Settings` and `MasterDataPipeline` are resolved
lazily (PEP 562), so importing a stdlib-only submodule such as
`ricc.ml_runtime` does not drag in pandas, pydantic and torch.

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

# The nested parameter sections come from params.yaml
settings.pii_audit.sample_size  # 100000
settings.pii_audit.confidence  # 0.99
settings.pii_audit.max_leak_tolerance  # 1e-4
settings.drift.psi_stable
settings.benchmarks.humaneval_tasks

# Custom instance
custom = Settings(minhash_num_perm=256, pii_audit=PIIAuditSettings(sample_size=100000))
```

`params.yaml` is the single source of these values — they are read at
boot, so editing it changes the runtime behaviour (the audit, the drift
thresholds, the benchmark sizes) without a code edit.

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

## Gates and Contracts

### `ricc.lora.policy_gate` — what may be published

```python
from ricc.lora.policy_gate import (
    assert_publishable,
    canonical_full_run,
    evaluate_compliance,
)

report = evaluate_compliance("lora_adapters/qwen2.5_3b_instruct")
report.compliant  # bool
report.reasons  # why not, when it is not
report.coverage_pct

# Raises PolicyViolation when the adapter covered less than one full
# epoch of the corpus, or was stopped by a wall-clock budget.
assert_publishable("lora_adapters/qwen2.5_3b_instruct")

# The canonical command for the current corpus size.
canonical_full_run(173216)["command"]
```

### `ricc.schema_version` — the published contract

```python
from ricc.schema_version import (
    DATASET_SCHEMA_VERSION,
    contract_manifest,
    get_schema,
    schema_fingerprint,
    verify_against_file,
)

DATASET_SCHEMA_VERSION  # "1.0.0"
schema_fingerprint()  # hash of the whole contract
get_schema("full_clean_messages").names
verify_against_file("data/full_clean_messages.parquet", "full_clean_messages")
# -> [] when the file matches the contract, otherwise the list of problems
```

### `ricc.evaluation.statistical_power` — the confidence machinery

```python
from ricc.evaluation.statistical_power import (
    mcnemar_test,
    required_sample_size,
    wilson_interval,
)

wilson_interval(3, 40, 0.95)
mcnemar_test([(True, False), (False, True)])  # paired comparison
```

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

### GPU/ML Error Taxonomy (12.2.0+)

Train/eval entry points classify raw torch/HF exceptions via
`classify_ml_exception()` / `record_classified_ml_error()` — see
[Errors API](errors.md) for the full `MLError`/`GPUError` hierarchy and
retry-vs-fail-fast policy.
