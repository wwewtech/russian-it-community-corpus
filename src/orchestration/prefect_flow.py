"""
Prefect orchestration for the RICC data platform.

Wraps the curation pipeline stages as Prefect tasks so the platform can be
scheduled/observed by a real orchestrator (retries, caching, UI, logs) —
the senior-review "senior-level data platform" item alongside DVC and
drift monitoring.

Design notes:
* Prefect is an OPTIONAL dependency. When it is not installed the module
  degrades gracefully: the same stage functions run sequentially with
  plain status dicts, so `make orchestrate` works everywhere and CI does
  not need Prefect.
* Every task returns a status dict {"task": ..., "status": ..., ...} and
  never raises past the flow boundary — a failed validation must not
  abort the audit report generation.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:  # Prefect is optional; see module docstring.
    from prefect import flow, get_run_logger, task

    HAS_PREFECT = True
except ImportError:  # pragma: no cover - exercised only without prefect
    HAS_PREFECT = False

    def task(*args: Any, **kwargs: Any):  # type: ignore[misc]
        """No-op stand-in for prefect.task when Prefect is absent."""

        def decorator(fn):
            return fn

        # Support both @task and @task(...)
        if args and callable(args[0]):
            return args[0]
        return decorator

    def flow(*args: Any, **kwargs: Any):  # type: ignore[misc]
        """No-op stand-in for prefect.flow when Prefect is absent."""

        def decorator(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return decorator

    def get_run_logger():  # type: ignore[misc]
        return logger


# ---------------------------------------------------------------------------
# Stage tasks
# ---------------------------------------------------------------------------


@task(name="curate-corpus", retries=1, retry_delay_seconds=30)
def task_run_pipeline(pipeline_factory: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Run the full end-to-end curation pipeline (MasterDataPipeline)."""
    log = get_run_logger()
    try:
        if pipeline_factory is None:
            from src.pipeline import MasterDataPipeline

            pipeline_factory = MasterDataPipeline
        summary = pipeline_factory().run_all()
        return {"task": "curate-corpus", "status": "completed", "summary": summary}
    except Exception as exc:
        log.error("curate-corpus failed: %s\n%s", exc, traceback.format_exc())
        return {"task": "curate-corpus", "status": "failed", "error": str(exc)}


@task(name="validate-dataset")
def task_validate_dataset(output_dir: str | Path | None = None) -> dict[str, Any]:
    """Validate dataset schema, JSONL integrity, and zero-PII leakage."""
    log = get_run_logger()
    try:
        from src.config import OUTPUT_DIR
        from src.validation.validator import DatasetValidator

        validator = DatasetValidator(Path(output_dir) if output_dir else OUTPUT_DIR)
        result = validator.validate_all()
        return {
            "task": "validate-dataset",
            "status": "completed",
            "overall_passed": result.get("overall_passed", False),
            "result": result,
        }
    except Exception as exc:
        log.error("validate-dataset failed: %s", exc)
        return {"task": "validate-dataset", "status": "failed", "error": str(exc)}


@task(name="probabilistic-pii-audit")
def task_probabilistic_pii_audit(
    parquet_path: str | Path | None = None,
    sample_size: int = 50_000,
) -> dict[str, Any]:
    """Run the stratified probabilistic PII audit with confidence bounds."""
    log = get_run_logger()
    try:
        from src.config import PARQUET_OUTPUT_DIR, REPORTS_DIR
        from src.validation.probabilistic_audit import ProbabilisticPIIAuditor

        path = Path(parquet_path) if parquet_path else PARQUET_OUTPUT_DIR / "full_clean_messages.parquet"
        report = ProbabilisticPIIAuditor(path).run_audit(sample_size=sample_size)
        out = REPORTS_DIR / "probabilistic_pii_audit.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        import json

        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "task": "probabilistic-pii-audit",
            "status": "completed",
            "verdict": report.get("verdict"),
            "report_path": str(out),
        }
    except Exception as exc:
        log.error("probabilistic-pii-audit failed: %s", exc)
        return {"task": "probabilistic-pii-audit", "status": "failed", "error": str(exc)}


@task(name="drift-monitoring")
def task_drift_monitoring(
    reference_path: str | Path | None = None,
    current_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare the current corpus snapshot against the reference snapshot."""
    log = get_run_logger()
    try:
        import pandas as pd

        from src.config import PARQUET_OUTPUT_DIR, REPORTS_DIR
        from src.monitoring.drift import DatasetDriftMonitor

        ref = Path(reference_path) if reference_path else PARQUET_OUTPUT_DIR / "full_clean_messages.parquet"
        cur = Path(current_path) if current_path else PARQUET_OUTPUT_DIR / "full_clean_messages.parquet"
        if not ref.exists() or not cur.exists():
            return {
                "task": "drift-monitoring",
                "status": "skipped",
                "note": f"snapshot missing: ref={ref.exists()} cur={cur.exists()}",
            }
        monitor = DatasetDriftMonitor(pd.read_parquet(ref), pd.read_parquet(cur))
        report = monitor.run()
        out = REPORTS_DIR / "drift_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        import json

        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "task": "drift-monitoring",
            "status": "completed",
            "overall_verdict": report["overall_verdict"],
            "report_path": str(out),
        }
    except Exception as exc:
        log.error("drift-monitoring failed: %s", exc)
        return {"task": "drift-monitoring", "status": "failed", "error": str(exc)}


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


@flow(name="ricc-curation-platform", log_prints=True)
def curate_corpus_flow(
    pipeline_factory: Callable[[], Any] | None = None,
    output_dir: str | Path | None = None,
    parquet_path: str | Path | None = None,
    reference_drift_path: str | Path | None = None,
    current_drift_path: str | Path | None = None,
    audit_sample_size: int = 50_000,
    run_pipeline: bool = True,
) -> dict[str, Any]:
    """Full platform flow: curate -> validate -> probabilistic audit -> drift.

    Returns a dict of per-task statuses so callers (Makefile, CI, Prefect UI)
    can gate on individual stages.
    """
    results: dict[str, Any] = {}
    if run_pipeline:
        results["curate-corpus"] = task_run_pipeline(pipeline_factory)
    results["validate-dataset"] = task_validate_dataset(output_dir)
    results["probabilistic-pii-audit"] = task_probabilistic_pii_audit(
        parquet_path=parquet_path, sample_size=audit_sample_size
    )
    results["drift-monitoring"] = task_drift_monitoring(
        reference_path=reference_drift_path, current_path=current_drift_path
    )
    return results


def run_flow(**kwargs: Any) -> dict[str, Any]:
    """Entry point for `make orchestrate` — works with or without Prefect."""
    return curate_corpus_flow(**kwargs)
