"""Tests for the Prefect orchestration flow (with graceful fallback)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.orchestration.prefect_flow import HAS_PREFECT, curate_corpus_flow
from tests.test_cli_commands import build_synthetic_dataset


class _StubPipeline:
    def __init__(self):
        self.ran = False

    def run_all(self):
        self.ran = True
        return {"execution_time_seconds": 0.1, "validation_passed": True}


class TestCurateCorpusFlow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = build_synthetic_dataset(Path(self.temp_dir.name))
        self.parquet = self.data_dir / "parquet" / "full_clean_messages.parquet"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_flow_with_stub_pipeline_reports_all_stages(self):
        results = curate_corpus_flow(
            pipeline_factory=_StubPipeline,
            output_dir=self.data_dir,
            parquet_path=self.parquet,
            reference_drift_path=self.parquet,
            current_drift_path=self.parquet,
            run_pipeline=True,
            audit_sample_size=10,
        )
        self.assertIn("curate-corpus", results)
        self.assertIn("validate-dataset", results)
        self.assertIn("probabilistic-pii-audit", results)
        self.assertIn("drift-monitoring", results)
        self.assertEqual(results["curate-corpus"]["status"], "completed")
        self.assertEqual(results["curate-corpus"]["summary"]["validation_passed"], True)

    def test_flow_without_pipeline_skips_curation(self):
        results = curate_corpus_flow(
            output_dir=self.data_dir,
            parquet_path=self.parquet,
            reference_drift_path=self.parquet,
            current_drift_path=self.parquet,
            run_pipeline=False,
            audit_sample_size=10,
        )
        self.assertNotIn("curate-corpus", results)
        # The remaining stages must still report a status (completed/failed/skipped)
        for stage in ("validate-dataset", "probabilistic-pii-audit", "drift-monitoring"):
            self.assertIn(stage, results)
            self.assertIn("status", results[stage])

    def test_pipeline_failure_does_not_abort_flow(self):
        class _ExplodingPipeline:
            def run_all(self):
                raise RuntimeError("no raw exports available")

        results = curate_corpus_flow(
            pipeline_factory=_ExplodingPipeline,
            output_dir=self.data_dir,
            parquet_path=self.parquet,
            reference_drift_path=self.parquet,
            current_drift_path=self.parquet,
            run_pipeline=True,
            audit_sample_size=10,
        )
        self.assertEqual(results["curate-corpus"]["status"], "failed")
        self.assertIn("error", results["curate-corpus"])
        # Downstream stages still executed
        self.assertIn("validate-dataset", results)

    def test_graceful_fallback_module_flag(self):
        # HAS_PREFECT may be True or False depending on the environment;
        # the flow must work identically in both cases.
        self.assertIsInstance(HAS_PREFECT, bool)
        results = curate_corpus_flow(
            output_dir=self.data_dir,
            parquet_path=self.parquet,
            reference_drift_path=self.parquet,
            current_drift_path=self.parquet,
            run_pipeline=False,
            audit_sample_size=10,
        )
        self.assertIsInstance(results, dict)


if __name__ == "__main__":
    unittest.main()
