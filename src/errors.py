"""
Typed exception hierarchy for RICC platform.

Provides structured error types with context for better debugging,
monitoring, and error handling across the platform.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, component="pipeline", **kwargs)
        self.stage = stage
        if stage:
            self.context["stage"] = stage


class IngestionError(PipelineError):
    """Error during data ingestion stage."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage="ingestion", **kwargs)


class PIIAnonymizationError(PipelineError):
    """Error during PII anonymization stage."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage="pii_anonymization", **kwargs)


class DeduplicationError(PipelineError):
    """Error during deduplication stage."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage="deduplication", **kwargs)


class TaxonomyError(PipelineError):
    """Error during taxonomy classification/tagging stage."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage="taxonomy_tagging", **kwargs)


class GraphExtractionError(PipelineError):
    """Error during conversation graph extraction stage."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage="graph_extraction", **kwargs)


class ExportError(PipelineError):
    """Error during dataset export stage."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage="export", **kwargs)


class AnalyticsError(PipelineError):
    """Error during analytics/validation stage."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage="analytics_validation", **kwargs)


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(RICCError):
    """Base error for validation failures."""

    def __init__(
        self,
        message: str,
        *,
        check: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, component="validation", **kwargs)
        self.check = check
        if check:
            self.context["check"] = check


class SchemaValidationError(ValidationError):
    """Dataset schema validation failed."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, check="schema", **kwargs)


class PIIValidationError(ValidationError):
    """PII leakage detected in validated dataset."""

    def __init__(self, message: str, leaks: dict[str, int] | None = None, **kwargs: Any) -> None:
        super().__init__(message, check="pii_audit", **kwargs)
        if leaks:
            self.context["leaks"] = leaks


class SFTValidationError(ValidationError):
    """SFT dialogue structure validation failed."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, check="sft_turn_conformance", **kwargs)


class JSONLValidationError(ValidationError):
    """JSONL format validation failed."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, check="jsonl", **kwargs)


class ParquetValidationError(ValidationError):
    """Parquet format validation failed."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, check="parquet", **kwargs)


class SLOValidationError(ValidationError):
    """SLO gate check failed."""

    def __init__(self, message: str, failed_checks: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(message, check="slo_gate", **kwargs)
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
    misleading "attached" metrics (see src/inference.py).
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


# =============================================================================
# Result Type (for functional error handling)
# =============================================================================


@dataclass(frozen=True)
class Result:
    """Result type for operations that can fail without exceptions.

    Usage:
        def risky_operation() -> Result:
            if not valid:
                return Err(ValidationError("invalid input"))
            return Ok("success")
    """

    _value: Any
    _error: Exception | None
    _is_ok: bool

    def __init__(self, value: Any = None, error: Exception | None = None) -> None:
        if error is not None and value is not None:
            raise ValueError("Result cannot have both value and error")
        if error is None and value is None:
            raise ValueError("Result must have either value or error")
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "_error", error)
        object.__setattr__(self, "_is_ok", error is None)

    @property
    def is_ok(self) -> bool:
        return self._is_ok

    @property
    def is_err(self) -> bool:
        return not self._is_ok

    @property
    def value(self) -> Any:
        if not self._is_ok:
            raise RuntimeError("Cannot access value of Err Result")
        return self._value

    @property
    def error(self) -> Exception | None:
        if self._is_ok:
            return None
        return self._error

    def unwrap(self) -> Any:
        """Return value or raise the contained error."""
        if self._is_ok:
            return self._value
        # At this point we know self._error is not None due to is_ok check
        error = self._error
        assert error is not None  # nosec B101
        raise error

    def unwrap_or(self, default: Any) -> Any:
        """Return value or default if error."""
        return self._value if self._is_ok else default

    def map(self, func: Callable[[Any], Any]) -> Result:
        """Transform the value if Ok, pass through Err."""
        if self._is_ok:
            try:
                return Result(func(self._value))
            except Exception as e:
                return Result(error=e)
        return Result(error=self._error)

    def map_err(self, func: Callable[[Exception], Exception]) -> Result:
        """Transform the error if Err, pass through Ok."""
        if not self._is_ok:
            # At this point we know self._error is not None
            return Result(error=func(self._error))  # type: ignore[arg-type]
        return self


def Ok(value: Any) -> Result:
    """Create a successful Result."""
    return Result(value=value)


def Err(error: Exception) -> Result:
    """Create an error Result."""
    return Result(error=error)


# Type alias for common result types (for documentation/type hints)
PipelineResult = "Result"
ValidationResult = "Result"
