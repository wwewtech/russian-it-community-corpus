"""
Typed exception hierarchy for RICC platform.

Provides structured error types with context for better debugging,
monitoring, and error handling across the platform.
"""

from __future__ import annotations

from typing import Any, ClassVar

# =============================================================================
# Base Exception Classes
# =============================================================================


class RICCError(Exception):
    """Base exception for all RICC platform errors."""

    def __init__(
        self,
        message: str,
        *,
        component: str | None = None,
        context: dict[str, Any] | None = None,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.component = component
        self.context = context or {}
        self.recoverable = recoverable

    def __str__(self) -> str:
        parts = [self.message]
        if self.component:
            parts.append(f"component={self.component}")
        if self.context:
            parts.append(f"context={self.context}")
        return " | ".join(parts)


# =============================================================================
# Pipeline Errors
# =============================================================================


class PipelineError(RICCError):
    """Base error for pipeline execution failures."""

    default_stage: ClassVar[str | None] = None

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_stage = stage if stage is not None else self.default_stage
        super().__init__(message, component="pipeline", **kwargs)
        self.stage = resolved_stage
        if resolved_stage:
            self.context["stage"] = resolved_stage


class IngestionError(PipelineError):
    """Error during data ingestion stage."""

    default_stage = "ingestion"


class PIIAnonymizationError(PipelineError):
    """Error during PII anonymization stage."""

    default_stage = "pii_anonymization"


class DeduplicationError(PipelineError):
    """Error during deduplication stage."""

    default_stage = "deduplication"


class TaxonomyError(PipelineError):
    """Error during taxonomy classification/tagging stage."""

    default_stage = "taxonomy_tagging"


class GraphExtractionError(PipelineError):
    """Error during conversation graph extraction stage."""

    default_stage = "graph_extraction"


class ExportError(PipelineError):
    """Error during dataset export stage."""

    default_stage = "export"


class AnalyticsError(PipelineError):
    """Error during analytics/validation stage."""

    default_stage = "analytics_validation"


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(RICCError):
    """Base error for validation failures."""

    default_check: ClassVar[str | None] = None

    def __init__(
        self,
        message: str,
        *,
        check: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_check = check if check is not None else self.default_check
        super().__init__(message, component="validation", **kwargs)
        self.check = resolved_check
        if resolved_check:
            self.context["check"] = resolved_check


class SchemaValidationError(ValidationError):
    """Dataset schema validation failed."""

    default_check = "schema"


class PIIValidationError(ValidationError):
    """PII leakage detected in validated dataset."""

    default_check = "pii_audit"

    def __init__(self, message: str, leaks: dict[str, int] | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        if leaks:
            self.context["leaks"] = leaks


class SFTValidationError(ValidationError):
    """SFT dialogue structure validation failed."""

    default_check = "sft_turn_conformance"


class JSONLValidationError(ValidationError):
    """JSONL format validation failed."""

    default_check = "jsonl"


class ParquetValidationError(ValidationError):
    """Parquet format validation failed."""

    default_check = "parquet"


class SLOValidationError(ValidationError):
    """SLO gate check failed."""

    default_check = "slo_gate"

    def __init__(self, message: str, failed_checks: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        if failed_checks:
            self.context["failed_checks"] = failed_checks


# =============================================================================
# Data Quality Errors
# =============================================================================


class DataQualityError(RICCError):
    """Base error for data quality issues."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, component="data_quality", **kwargs)


class DriftDetectedError(DataQualityError):
    """Significant dataset drift detected."""

    def __init__(
        self,
        message: str,
        *,
        psi_value: float | None = None,
        js_value: float | None = None,
        jaccard_value: float | None = None,
        verdict: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        if psi_value is not None:
            self.context["psi_value"] = psi_value
        if js_value is not None:
            self.context["js_value"] = js_value
        if jaccard_value is not None:
            self.context["jaccard_value"] = jaccard_value
        if verdict:
            self.context["verdict"] = verdict


class ArtifactIntegrityError(DataQualityError):
    """Artifact manifest verification failed."""

    def __init__(
        self,
        message: str,
        *,
        artifact_path: str | None = None,
        expected_sha256: str | None = None,
        actual_sha256: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        if artifact_path:
            self.context["artifact_path"] = artifact_path
        if expected_sha256:
            self.context["expected_sha256"] = expected_sha256
        if actual_sha256:
            self.context["actual_sha256"] = actual_sha256


class HubReconciliationError(DataQualityError):
    """Hub dataset reconciliation failed."""

    def __init__(
        self,
        message: str,
        *,
        local_rows: int | None = None,
        hub_rows: int | None = None,
        pinned_revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        if local_rows is not None:
            self.context["local_rows"] = local_rows
        if hub_rows is not None:
            self.context["hub_rows"] = hub_rows
        if pinned_revision:
            self.context["pinned_revision"] = pinned_revision


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(RICCError):
    """Configuration loading or validation error."""

    def __init__(
        self,
        message: str,
        *,
        config_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, component="configuration", **kwargs)
        if config_key:
            self.context["config_key"] = config_key


class MissingDependencyError(ConfigurationError):
    """Required dependency not available."""

    def __init__(self, message: str, dependency: str, **kwargs: Any) -> None:
        super().__init__(message, config_key=dependency, **kwargs)
        self.context["dependency"] = dependency


# =============================================================================
# External Service Errors
# =============================================================================


class ExternalServiceError(RICCError):
    """Error interacting with external services (HF Hub, cloud providers, etc.)."""

    def __init__(
        self,
        message: str,
        *,
        service: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, component="external_service", **kwargs)
        if service:
            self.context["service"] = service
        if status_code:
            self.context["status_code"] = status_code
        if retry_after:
            self.context["retry_after_seconds"] = retry_after
        # External service errors are often recoverable
        self.recoverable = True


class HuggingFaceHubError(ExternalServiceError):
    """Error interacting with Hugging Face Hub."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, service="huggingface_hub", **kwargs)


class PrefectError(ExternalServiceError):
    """Error interacting with Prefect orchestration."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, service="prefect", **kwargs)


# =============================================================================
# Resource Errors
# =============================================================================


class ResourceError(RICCError):
    """Resource exhaustion or availability error."""

    def __init__(
        self,
        message: str,
        *,
        resource_type: str | None = None,
        current_usage: float | None = None,
        limit: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, component="resource", **kwargs)
        if resource_type:
            self.context["resource_type"] = resource_type
        if current_usage is not None:
            self.context["current_usage"] = current_usage
        if limit is not None:
            self.context["limit"] = limit


class DiskSpaceError(ResourceError):
    """Insufficient disk space."""

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        required_mb: int | None = None,
        available_mb: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            resource_type="disk",
            current_usage=required_mb,
            limit=available_mb,
            **kwargs,
        )
        if path:
            self.context["path"] = path


# =============================================================================
# GPU / ML Errors
#
# Typed taxonomy for GPU training / inference failures. Motivation: LoRA
# training and eval harnesses previously raised bare RuntimeError / let
# torch exceptions propagate, which made it impossible to distinguish
# "retry with smaller batch / gradient checkpointing" (OOM) from
# "fail fast, weights are corrupt" (adapter load) in logs and in the
# `ricc_errors_total{component=...}` Prometheus counter.
# =============================================================================


class MLError(RICCError):
    """Base error for ML training / inference / evaluation failures."""

    def __init__(
        self,
        message: str,
        *,
        model_name: str | None = None,
        adapter_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, component="ml", **kwargs)
        if model_name:
            self.context["model_name"] = model_name
        if adapter_id:
            self.context["adapter_id"] = adapter_id


class TrainingError(MLError):
    """LoRA / fine-tuning loop failure (trainer step, optimizer, dataloader)."""

    def __init__(
        self,
        message: str,
        *,
        step: int | None = None,
        phase: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.component = "training"
        if step is not None:
            self.context["step"] = step
        if phase:
            self.context["phase"] = phase


class InferenceError(MLError):
    """Text generation / model serving failure."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.component = "inference"


class EvaluationError(MLError):
    """Benchmark / eval harness failure (HumanEval, RuMMLU, perplexity, ROUGE)."""

    def __init__(
        self,
        message: str,
        *,
        benchmark: str | None = None,
        variant: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.component = "evaluation"
        if benchmark:
            self.context["benchmark"] = benchmark
        if variant:
            self.context["variant"] = variant


class TokenizerError(MLError):
    """Tokenizer load / chat-template / encode-decode failure."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.component = "tokenizer"


class CheckpointError(MLError):
    """Checkpoint save / load / resume failure (partial runs must stay recoverable)."""

    def __init__(
        self,
        message: str,
        *,
        checkpoint_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.component = "checkpoint"
        if checkpoint_path:
            self.context["checkpoint_path"] = checkpoint_path


class AdapterLoadError(MLError):
    """LoRA adapter attach failure (corrupt weights / base-model mismatch).

    Must fail fast — silently falling back to the base model produces
    misleading "attached" metrics (see ricc/inference.py).
    """

    def __init__(
        self,
        message: str,
        *,
        adapter_path: str | None = None,
        base_model: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.component = "adapter"
        if adapter_path:
            self.context["adapter_path"] = adapter_path
        if base_model:
            self.context["base_model"] = base_model


class QuantizationError(MLError):
    """4-bit NF4 / bitsandbytes quantization setup failure (QLoRA path)."""

    def __init__(
        self,
        message: str,
        *,
        quant_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.component = "quantization"
        if quant_type:
            self.context["quant_type"] = quant_type


class GPUError(RICCError):
    """Base error for GPU device failures."""

    def __init__(
        self,
        message: str,
        *,
        device: str | None = None,
        device_name: str | None = None,
        model_name: str | None = None,
        adapter_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, component="gpu", **kwargs)
        if device:
            self.context["device"] = device
        if device_name:
            self.context["device_name"] = device_name
        if model_name:
            self.context["model_name"] = model_name
        if adapter_id:
            self.context["adapter_id"] = adapter_id


class GPUNotAvailableError(GPUError):
    """CUDA requested (device_map='auto', fp16) but no GPU is visible."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.context["reason"] = "cuda_not_available"


class GPUDriverError(GPUError):
    """CUDA driver / runtime mismatch (e.g. NVML, cuBLAS, NCCL init failures)."""

    def __init__(
        self,
        message: str,
        *,
        driver_version: str | None = None,
        cuda_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        if driver_version:
            self.context["driver_version"] = driver_version
        if cuda_version:
            self.context["cuda_version"] = cuda_version


class GPUOutOfMemoryError(ResourceError):
    """GPU VRAM exhausted during training / inference.

    A ResourceError with resource_type="vram" (same context keys as the
    legacy VRAMExhaustedError) plus the GPU device that OOMed, so OOMs
    stay visible both in resource dashboards and GPU dashboards.
    """

    def __init__(
        self,
        message: str,
        *,
        requested_mb: int | None = None,
        available_mb: int | None = None,
        device: str | None = None,
        device_name: str | None = None,
        model_name: str | None = None,
        adapter_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            resource_type="vram",
            current_usage=requested_mb,
            limit=available_mb,
            **kwargs,
        )
        self.component = "gpu"
        if device:
            self.context["device"] = device
        if device_name:
            self.context["device_name"] = device_name
        if model_name:
            self.context["model_name"] = model_name
        if adapter_id:
            self.context["adapter_id"] = adapter_id


class VRAMExhaustedError(GPUOutOfMemoryError):
    """Legacy alias kept for backward compatibility (now a GPU OOM)."""


_OOM_MARKERS = ("out of memory", "outofmemory", "memory exhausted")


def is_oom_error(exc: BaseException) -> bool:
    """Return True if an exception looks like a GPU/CPU OOM.

    Matches torch.cuda.OutOfMemoryError by class name and CUDA OOMs by
    message ("CUDA out of memory. Tried to allocate ...").
    """
    if "OutOfMemory" in type(exc).__name__:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _OOM_MARKERS)


_GPU_UNAVAILABLE_MARKERS = ("no cuda", "cuda not available", "no gpu", "device not found")
_GPU_DRIVER_MARKERS = ("cuda driver", "cublas", "nccl", "nvml", "cuda initialization failure")


def classify_ml_exception(
    exc: BaseException,
    *,
    model_name: str | None = None,
    adapter_id: str | None = None,
    device: str | None = None,
) -> RICCError:
    """Map a raw torch/HF exception to the typed GPU/ML taxonomy.

    OOM -> GPUOutOfMemoryError (retryable: smaller batch, checkpointing,
    4-bit quantization); missing CUDA -> GPUNotAvailableError; driver /
    NCCL failures -> GPUDriverError; anything else -> TrainingError with
    the original exception chained in context.
    """
    message = str(exc) or f"{type(exc).__name__}: <no message>"
    if is_oom_error(exc):
        return GPUOutOfMemoryError(message, model_name=model_name, adapter_id=adapter_id, device=device)
    lowered = message.lower()
    if any(marker in lowered for marker in _GPU_UNAVAILABLE_MARKERS):
        return GPUNotAvailableError(message, model_name=model_name, adapter_id=adapter_id, device=device)
    if any(marker in lowered for marker in _GPU_DRIVER_MARKERS):
        return GPUDriverError(message, model_name=model_name, adapter_id=adapter_id, device=device)
    typed = TrainingError(message, model_name=model_name, adapter_id=adapter_id)
    typed.context["original_error"] = f"{type(exc).__name__}: {exc}"
    return typed
