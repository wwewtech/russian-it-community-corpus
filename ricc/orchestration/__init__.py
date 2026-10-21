"""
Orchestration module: Prefect-based pipeline orchestration with graceful
fallback to sequential execution when Prefect is not installed.
"""

from ricc.orchestration.prefect_flow import curate_corpus_flow, run_flow

__all__ = ["curate_corpus_flow", "run_flow"]
