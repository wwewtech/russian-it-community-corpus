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
    - Result, Ok, Err: functional error handling
    - configure_logging, get_logger: structured logging (structlog)
    - Prometheus metrics: PIPELINE_STAGE_DURATION, PII_AUDIT_LEAKS_FOUND, etc.
"""

from __future__ import annotations

from src.config import load_params
from src.errors import (
    AnalyticsError,
    ArtifactIntegrityError,
    # Configuration errors
    ConfigurationError,
    # Data quality errors
    DataQualityError,
    DeduplicationError,
    DiskSpaceError,
    DriftDetectedError,
    Err,
    ExportError,
    # External service errors
    ExternalServiceError,
    GraphExtractionError,
    HubReconciliationError,
    HuggingFaceHubError,
    IngestionError,
    JSONLValidationError,
    MissingDependencyError,
    Ok,
    ParquetValidationError,
    PIIAnonymizationError,
    PIIValidationError,
    # Pipeline errors
    PipelineError,
    PrefectError,
    # Resource errors
    ResourceError,
    # Result type
    Result,
    # Base
    RICCError,
    SchemaValidationError,
    SFTValidationError,
    SLOValidationError,
    TaxonomyError,
    # Validation errors
    ValidationError,
    VRAMExhaustedError,
)
from src.observability import (
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
    configure_logging,
    get_logger,
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
from src.pipeline import MasterDataPipeline
from src.settings import Settings, settings

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
    # Result type
    "Result",
    "Ok",
    "Err",
    # Observability
    "configure_logging",
    "get_logger",
    "pipeline_logger",
    "start_metrics_server",
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

__version__ = "12.1.0"
__author__ = "wwewtech"
__license__ = "MIT"
__description__ = "Russian IT Community Corpus — Data Engineering & Zero-PII Curation Platform"
