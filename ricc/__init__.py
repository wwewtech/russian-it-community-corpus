"""
RICC (Russian IT Community Corpus) — Data Engineering & Zero-PII Curation Platform.

A production-grade platform for LLM dataset curation (SFT/DPO/RAG) with
probabilistic PII audit guarantees and local LoRA fine-tuning on consumer hardware.

Quick Start:
    from ricc import Settings, MasterDataPipeline

    settings = Settings()
    pipeline = MasterDataPipeline()
    summary = pipeline.run_all()

Public API:
    - Settings: unified configuration (pydantic-settings)
    - MasterDataPipeline: end-to-end data curation pipeline
    - PipelineError, ValidationError, etc.: typed exception hierarchy
    - configure_logging, get_logger: structured logging (structlog)
    - Prometheus metrics: PIPELINE_STAGE_DURATION, PII_AUDIT_LEAKS_FOUND, etc.
"""

from __future__ import annotations

import importlib
from typing import Any, cast

__version__ = "12.2.0"

from ricc.errors import (
    AdapterLoadError,
    AnalyticsError,
    ArtifactIntegrityError,
    CheckpointError,
    # Configuration errors
    ConfigurationError,
    # Data quality errors
    DataQualityError,
    DeduplicationError,
    DiskSpaceError,
    DriftDetectedError,
    EvaluationError,
    ExportError,
    # External service errors
    ExternalServiceError,
    GPUDriverError,
    GPUError,
    GPUNotAvailableError,
    GPUOutOfMemoryError,
    GraphExtractionError,
    HubReconciliationError,
    HuggingFaceHubError,
    InferenceError,
    IngestionError,
    JSONLValidationError,
    MissingDependencyError,
    MLError,
    ParquetValidationError,
    PIIAnonymizationError,
    PIIValidationError,
    # Pipeline errors
    PipelineError,
    PrefectError,
    QuantizationError,
    # Resource errors
    ResourceError,
    # Base
    RICCError,
    SchemaValidationError,
    SFTValidationError,
    SLOValidationError,
    TaxonomyError,
    TokenizerError,
    TrainingError,
    # Validation errors
    ValidationError,
    VRAMExhaustedError,
    classify_ml_exception,
    is_oom_error,
)
from ricc.observability import (
    # Dataset metrics
    DATASET_MESSAGES_TOTAL,
    DATASET_SIZE_BYTES,
    DRIFT_JS_VALUE,
    # Drift metrics
    DRIFT_PSI_VALUE,
    DRIFT_VERDICT,
    DRIFT_VOCAB_JACCARD,
    # Error metrics
    ERRORS_TOTAL,
    HTTP_REQUEST_DURATION,
    # HTTP metrics
    HTTP_REQUESTS_TOTAL,
    # LoRA metrics
    LORA_TRAIN_DURATION,
    LORA_TRAIN_LOSS,
    LORA_TRAIN_STEPS,
    LORA_VRAM_USAGE,
    PII_AUDIT_DURATION,
    # PII audit metrics
    PII_AUDIT_LEAKS_FOUND,
    PII_AUDIT_VERDICT,
    PIPELINE_MESSAGES_PROCESSED,
    # Pipeline metrics
    PIPELINE_STAGE_DURATION,
    PIPELINE_STAGE_TOTAL,
    SLO_CHECK_RESULT,
    # SLO metrics
    SLO_GATE_VERDICT,
    # Validation metrics
    VALIDATION_RESULT,
    bind_run_context,
    clear_run_context,
    configure_logging,
    get_logger,
    get_run_context,
    maybe_start_metrics_server,
    pipeline_logger,
    record_error,
    record_pii_leak,
    record_slo_check,
    record_stage_messages,
    record_validation,
    start_metrics_server,
    # Helpers
    time_stage,
    timed_stage,
)

# ``ricc.pipeline`` pulls in pandas/pyarrow and ``ricc.settings`` pulls in
# pydantic + yaml. Importing them eagerly made ``import ricc.ml_runtime``
# (a stdlib-only module) fail in a stripped-down environment — the parent
# package import alone was enough to raise ImportError. PEP 562 lazy
# attributes keep ``import ricc`` cheap while preserving the public API.
_LAZY_ATTRS: dict[str, str] = {
    "MasterDataPipeline": "ricc.pipeline",
    "Settings": "ricc.settings",
    "settings": "ricc.settings",
}


def __getattr__(name: str) -> Any:
    """Resolve the lazily-imported public attributes (PEP 562)."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'ricc' has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()) | set(_LAZY_ATTRS))


# The submodule ``ricc.settings`` shadows the ``settings`` INSTANCE on the
# package: Python resolves ``ricc.settings`` to the MODULE (importing a
# submodule binds it as an attribute of the parent package), which would
# silently break ``ricc.settings.DOMAIN_TAXONOMY`` for every consumer
# that used the documented ``from ricc import settings`` form.
#
# Binding the instance explicitly after the submodule import restores the
# documented behaviour: ``ricc.settings`` is the Settings INSTANCE and
# ``ricc.settings.DOMAIN_TAXONOMY`` works. Code that needs the module
# uses ``from ricc.settings import settings``.
_settings_module = importlib.import_module("ricc.settings")
settings = _settings_module.settings
del _settings_module


def load_params() -> dict[str, Any]:
    """Load the raw ``params.yaml``.

    Kept at the package level because it was part of the public API; the
    implementation lives in :meth:`ricc.settings.Settings.load_params`.
    """
    return cast("dict[str, Any]", settings.load_params())


__all__ = [
    # Version
    "__version__",
    # Configuration
    "Settings",
    "settings",
    "load_params",
    # Pipeline
    "MasterDataPipeline",
    # Error hierarchy
    "RICCError",
    "PipelineError",
    "IngestionError",
    "PIIAnonymizationError",
    "DeduplicationError",
    "TaxonomyError",
    "GraphExtractionError",
    "ExportError",
    "AnalyticsError",
    "ValidationError",
    "SchemaValidationError",
    "PIIValidationError",
    "SFTValidationError",
    "JSONLValidationError",
    "ParquetValidationError",
    "SLOValidationError",
    "DataQualityError",
    "DriftDetectedError",
    "ArtifactIntegrityError",
    "HubReconciliationError",
    "ConfigurationError",
    "MissingDependencyError",
    "ExternalServiceError",
    "HuggingFaceHubError",
    "PrefectError",
    "ResourceError",
    "VRAMExhaustedError",
    "DiskSpaceError",
    "MLError",
    "TrainingError",
    "InferenceError",
    "EvaluationError",
    "TokenizerError",
    "CheckpointError",
    "AdapterLoadError",
    "QuantizationError",
    "GPUError",
    "GPUNotAvailableError",
    "GPUDriverError",
    "GPUOutOfMemoryError",
    "is_oom_error",
    "classify_ml_exception",
    # Result type
    # Observability
    "configure_logging",
    "get_logger",
    "pipeline_logger",
    "bind_run_context",
    "get_run_context",
    "clear_run_context",
    "start_metrics_server",
    "maybe_start_metrics_server",
    "PIPELINE_STAGE_DURATION",
    "PIPELINE_STAGE_TOTAL",
    "PIPELINE_MESSAGES_PROCESSED",
    "PII_AUDIT_LEAKS_FOUND",
    "PII_AUDIT_DURATION",
    "PII_AUDIT_VERDICT",
    "DRIFT_PSI_VALUE",
    "DRIFT_JS_VALUE",
    "DRIFT_VOCAB_JACCARD",
    "DRIFT_VERDICT",
    "VALIDATION_RESULT",
    "LORA_TRAIN_DURATION",
    "LORA_TRAIN_STEPS",
    "LORA_TRAIN_LOSS",
    "LORA_VRAM_USAGE",
    "DATASET_MESSAGES_TOTAL",
    "DATASET_SIZE_BYTES",
    "SLO_GATE_VERDICT",
    "SLO_CHECK_RESULT",
    "HTTP_REQUESTS_TOTAL",
    "HTTP_REQUEST_DURATION",
    "ERRORS_TOTAL",
    "time_stage",
    "record_stage_messages",
    "record_pii_leak",
    "record_validation",
    "record_slo_check",
    "record_error",
    "timed_stage",
]

__author__ = "wwewtech"
__license__ = "MIT"
__description__ = "Russian IT Community Corpus — Data Engineering & Zero-PII Curation Platform"
