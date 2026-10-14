"""
Observability module for RICC platform.

Provides:
- Structured logging via structlog (JSON output, context binding)
- Prometheus metrics (counters, histograms, gauges)
- Centralized metric definitions for pipeline stages
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any

import structlog
from prometheus_client import Counter, Gauge, Histogram, start_http_server

if TYPE_CHECKING:
    # Annotation-only import: keeps the runtime observability -> errors
    # edge out of the import graph (src.errors imports nothing from here,
    # but the lazy imports below stay lazy for import-cost reasons).
    from src.errors import RICCError

# =============================================================================
# Structlog Configuration
# =============================================================================


def configure_logging(
    json_output: bool = True,
    level: int = logging.INFO,
    add_timestamp: bool = True,
) -> None:
    """
    Configure structlog for structured JSON logging.

    Call once at application startup (e.g., in main.py, cli.py, app.py).
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True) if add_timestamp else None

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    renderer = structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer(colors=True)

    processors: Iterable[Any] = shared_processors
    if timestamper:
        processors = list(processors) + [timestamper]
    processors = list(processors) + [renderer]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to work with structlog
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)

    # Reduce noise from noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]


# =============================================================================
# Prometheus Metrics
# =============================================================================

# Pipeline stage metrics
PIPELINE_STAGE_DURATION = Histogram(
    "ricc_pipeline_stage_duration_seconds",
    "Duration of each pipeline stage in seconds",
    ["stage"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
)

PIPELINE_STAGE_TOTAL = Counter(
    "ricc_pipeline_stage_total",
    "Total number of pipeline stage executions",
    ["stage", "status"],  # status: success, failure
)

PIPELINE_MESSAGES_PROCESSED = Counter(
    "ricc_pipeline_messages_processed_total",
    "Total number of messages processed by pipeline",
    ["stage"],
)

# PII Audit metrics
PII_AUDIT_LEAKS_FOUND = Counter(
    "ricc_pii_audit_leaks_found_total",
    "Total number of PII leaks found during audit",
    ["category"],
)

PII_AUDIT_DURATION = Histogram(
    "ricc_pii_audit_duration_seconds",
    "Duration of PII audit in seconds",
    buckets=(1, 5, 10, 30, 60, 120, 300),
)

PII_AUDIT_VERDICT = Counter(
    "ricc_pii_audit_verdict_total",
    "PII audit verdict",
    ["verdict"],  # PASSED, LEAKS_DETECTED
)

# Drift monitoring metrics
DRIFT_PSI_VALUE = Gauge(
    "ricc_drift_psi_value",
    "Population Stability Index value",
)
DRIFT_JS_VALUE = Gauge(
    "ricc_drift_js_value",
    "Jensen-Shannon divergence value (bits)",
)
DRIFT_VOCAB_JACCARD = Gauge(
    "ricc_drift_vocab_jaccard",
    "Vocabulary Jaccard overlap",
)
DRIFT_VERDICT = Counter(
    "ricc_drift_verdict_total",
    "Drift monitoring verdict",
    ["verdict"],  # stable, moderate_drift, significant_drift
)

# Validation metrics
VALIDATION_RESULT = Counter(
    "ricc_validation_result_total",
    "Validation result",
    ["check", "status"],  # check: parquet, jsonl, pii, sft; status: pass, fail
)

# LoRA training metrics
LORA_TRAIN_DURATION = Histogram(
    "ricc_lora_train_duration_seconds",
    "LoRA training duration in seconds",
    ["model", "adapter"],
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400),
)

LORA_TRAIN_STEPS = Counter(
    "ricc_lora_train_steps_total",
    "Total LoRA training steps completed",
    ["model", "adapter"],
)

LORA_TRAIN_LOSS = Gauge(
    "ricc_lora_train_loss",
    "Current training loss",
    ["model", "adapter"],
)

LORA_VRAM_USAGE = Gauge(
    "ricc_lora_vram_usage_bytes",
    "VRAM usage during LoRA training in bytes",
    ["model", "adapter"],
)

LORA_TRAIN_TOTAL = Counter(
    "ricc_lora_train_total",
    "LoRA train/eval run outcomes",
    ["model", "adapter", "status"],  # status: success, failure
)

# Dataset metrics
DATASET_MESSAGES_TOTAL = Gauge(
    "ricc_dataset_messages_total",
    "Total messages in dataset",
    ["dataset"],  # full, sft, rag
)

DATASET_SIZE_BYTES = Gauge(
    "ricc_dataset_size_bytes",
    "Dataset size in bytes",
    ["dataset", "format"],  # format: parquet, jsonl
)

# SLO Gate metrics
SLO_GATE_VERDICT = Counter(
    "ricc_slo_gate_verdict_total",
    "SLO gate verdict",
    ["verdict"],  # SHIP, HOLD
)

SLO_CHECK_RESULT = Counter(
    "ricc_slo_check_result_total",
    "Individual SLO check result",
    ["check", "status"],  # status: pass, fail
)

# HTTP/API metrics (for Streamlit, CLI)
HTTP_REQUESTS_TOTAL = Counter(
    "ricc_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "ricc_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

# Error metrics
ERRORS_TOTAL = Counter(
    "ricc_errors_total",
    "Total errors by type",
    ["component", "error_type"],
)

# =============================================================================
# Metric Helpers
# =============================================================================


@contextmanager
def time_stage(stage: str) -> Generator[None, None, None]:
    """Context manager to time a pipeline stage and record metrics."""
    start = time.perf_counter()
    try:
        yield
        PIPELINE_STAGE_TOTAL.labels(stage=stage, status="success").inc()
    except Exception:
        PIPELINE_STAGE_TOTAL.labels(stage=stage, status="failure").inc()
        raise
    finally:
        duration = time.perf_counter() - start
        PIPELINE_STAGE_DURATION.labels(stage=stage).observe(duration)


def record_stage_messages(stage: str, count: int) -> None:
    """Record number of messages processed in a stage."""
    PIPELINE_MESSAGES_PROCESSED.labels(stage=stage).inc(count)


def record_pii_leak(category: str, count: int = 1) -> None:
    """Record a PII leak finding."""
    PII_AUDIT_LEAKS_FOUND.labels(category=category).inc(count)


def record_validation(check: str, passed: bool) -> None:
    """Record a validation check result."""
    VALIDATION_RESULT.labels(check=check, status="pass" if passed else "fail").inc()


def record_slo_check(check: str, passed: bool) -> None:
    """Record an SLO check result."""
    SLO_CHECK_RESULT.labels(check=check, status="pass" if passed else "fail").inc()


def record_error(component: str, error_type: str) -> None:
    """Record an error occurrence."""
    ERRORS_TOTAL.labels(component=component, error_type=error_type).inc()


# Components produced by the P3 GPU/ML taxonomy in src/errors.py.
# ML_COMPONENTS is the allowlist consumed by the monitoring stack
# (Grafana error-rate panels, PrometheusRules alerts): it keeps the
# cardinality of `ricc_errors_total{component=...}` bounded and makes a
# new component a deliberate, dashboard-visible decision.
ML_COMPONENTS: tuple[str, ...] = (
    "ml",
    "training",
    "inference",
    "evaluation",
    "tokenizer",
    "checkpoint",
    "adapter",
    "quantization",
    "gpu",
)


def record_ml_error(error: Exception) -> str:
    """Record a typed GPU/ML exception and return the recorded component.

    Components from the P3 taxonomy (:data:`ML_COMPONENTS`) map 1:1 onto
    the Grafana "Errors by component" panel. Anything else falls back to
    ``ml`` so ad-hoc components cannot silently inflate cardinality.
    The function imports :mod:`src.errors` lazily to keep the
    ``observability -> errors`` edge out of the import graph.
    """
    from src.errors import RICCError

    component = "ml"
    error_type = type(error).__name__
    if isinstance(error, RICCError):
        candidate = error.component or "ml"
        component = candidate if candidate in ML_COMPONENTS else "ml"
    record_error(component, error_type)
    return component


def record_classified_ml_error(exc: BaseException, **kwargs: Any) -> RICCError:
    """Classify a raw train/eval exception via the P3 taxonomy and record it.

    Convenience wrapper over :func:`src.errors.classify_ml_exception` +
    :func:`record_ml_error` for the train/eval scripts' broad ``except``
    blocks: one call maps e.g. ``torch.cuda.OutOfMemoryError`` to
    ``GPUOutOfMemoryError``, increments ``ricc_errors_total{component="gpu",
    error_type="GPUOutOfMemoryError"}`` and returns the typed exception for
    the caller to log/raise.
    """
    from src.errors import classify_ml_exception

    typed = classify_ml_exception(exc, **kwargs)
    record_ml_error(typed)
    return typed


def record_lora_result(
    *,
    model: str,
    adapter: str,
    status: str,
    duration_sec: float | None = None,
    loss: float | None = None,
    steps: int | None = None,
    vram_bytes: float | None = None,
) -> None:
    """Record one LoRA train/eval run outcome into the LORA_* metric family.

    Feeds the Grafana "Training loss by adapter" and "VRAM usage by adapter"
    panels. ``status`` should be ``success`` or ``failure`` (it feeds
    ``ricc_lora_train_total{status=...}``, the alert source for
    RiccLoraTrainFailing). Gauges are only set on success — a failed run
    must not clobber the last known good loss/VRAM reading.
    """
    if status == "success":
        if duration_sec is not None:
            LORA_TRAIN_DURATION.labels(model=model, adapter=adapter).observe(duration_sec)
        if loss is not None:
            LORA_TRAIN_LOSS.labels(model=model, adapter=adapter).set(loss)
        if steps is not None and steps > 0:
            LORA_TRAIN_STEPS.labels(model=model, adapter=adapter).inc(steps)
        if vram_bytes is not None:
            LORA_VRAM_USAGE.labels(model=model, adapter=adapter).set(vram_bytes)
    LORA_TRAIN_TOTAL.labels(model=model, adapter=adapter, status=status).inc()


def start_metrics_server(port: int = 9090) -> None:
    """Start Prometheus metrics HTTP server on given port."""
    start_http_server(port)
    get_logger("observability").info("metrics_server_started", port=port)


_METRICS_SERVER_STARTED = False


def maybe_start_metrics_server(port: int = 9090) -> bool:
    """Start the metrics server once per process; safe under Streamlit reruns.

    Streamlit re-executes ``app.py`` on every interaction, so a bare
    :func:`start_metrics_server` call would raise ``OSError: address in use``
    on the second run. This wrapper starts the server exactly once and
    swallows the already-bound case, returning True when this process owns
    the server and False when the port was already taken.
    """
    global _METRICS_SERVER_STARTED
    if _METRICS_SERVER_STARTED:
        return True
    try:
        start_http_server(port)
    except OSError:
        # Port already bound — either our own server from a previous
        # Streamlit rerun (same process) or a foreign one. Either way the
        # caller must not crash; /metrics is served by whoever owns it.
        get_logger("observability").warning("metrics_port_already_bound", port=port)
        return False
    _METRICS_SERVER_STARTED = True
    get_logger("observability").info("metrics_server_started", port=port)
    return True


# =============================================================================
# Decorators
# =============================================================================


def timed_stage(stage: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to time a function as a pipeline stage."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with time_stage(stage):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Logging Helpers
# =============================================================================


class PipelineLogger:
    """Convenience logger for pipeline stages with automatic context."""

    def __init__(self, name: str = "pipeline") -> None:
        self.logger = get_logger(name)
        self.stage = ""

    def stage_start(self, stage: str, **kwargs: Any) -> None:
        self.stage = stage
        self.logger.info("stage_started", stage=stage, **kwargs)

    def stage_end(self, stage: str, success: bool = True, **kwargs: Any) -> None:
        self.logger.info(
            "stage_completed" if success else "stage_failed",
            stage=stage,
            success=success,
            **kwargs,
        )

    def progress(self, message: str, **kwargs: Any) -> None:
        self.logger.info(message, stage=self.stage, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self.logger.warning(message, stage=self.stage, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self.logger.error(message, stage=self.stage, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self.logger.debug(message, stage=self.stage, **kwargs)


# Global pipeline logger instance
pipeline_logger = PipelineLogger()
