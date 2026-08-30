"""
Unit tests for :mod:`src.pipeline`.

``MasterDataPipeline.run_all`` is an end-to-end orchestrator: testing
the happy path requires either real data (slow, non-deterministic) or
extensive mocking of every collaborator. Instead we exercise the
constructor + a partial run that touches each stage using
:mod:`unittest.mock` so we can assert on the call graph without
spinning up the full ingestion / NER / dedup stack.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.pipeline import MasterDataPipeline


def _build_synthetic_chat_export(root: Path) -> Path:
    """Write a tiny JSON telegram export under ``root`` and return the path."""
    export = root / "ChatExport_2026-01-01"
    export.mkdir(parents=True, exist_ok=True)
    payload = {
        "messages": [
            {
                "id": 1,
                "type": "message",
                "date": "2026-01-01T00:00:00",
                "date_unixtime": "1767225600",
                "from": "Alice",
                "from_id": "user_a",
                "text": "How do I configure nginx?",
                "reply_to_message_id": None,
            },
            {
                "id": 2,
                "type": "message",
                "date": "2026-01-01T00:01:00",
                "date_unixtime": "1767225660",
                "from": "Bob",
                "from_id": "user_b",
                "text": "Use proxy_pass inside the location block.",
                "reply_to_message_id": 1,
            },
        ]
    }
    (export / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    return export


def _build_synthetic_dataset(root: Path) -> Path:
    """Build a valid synthetic dataset_output tree under ``root``."""
    data_dir = root / "dataset_output"
    parquet_dir = data_dir / "parquet"
    jsonl_dir = data_dir / "jsonl"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"a": [1, 2]}).to_parquet(parquet_dir / "full_clean_messages.parquet", index=False)
    pd.DataFrame({"a": [1]}).to_parquet(parquet_dir / "sft_dialogues.parquet", index=False)
    pd.DataFrame({"a": [1]}).to_parquet(parquet_dir / "rag_knowledge_base.parquet", index=False)
    return data_dir


class TestMasterDataPipelineConstruction(unittest.TestCase):
    def test_constructor_creates_all_collaborators(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pipeline = MasterDataPipeline(raw_export_dirs=[tmp / "x"])
            self.assertIsNotNone(pipeline.anonymizer)
            self.assertIsNotNone(pipeline.exact_dedup)
            self.assertIsNotNone(pipeline.minhash_lsh)
            self.assertIsNotNone(pipeline.tagger)
            self.assertIsNotNone(pipeline.dag_builder)
            self.assertIsNotNone(pipeline.conv_extractor)
            self.assertIsNotNone(pipeline.parquet_exporter)
            self.assertIsNotNone(pipeline.jsonl_exporter)
            self.assertIsNotNone(pipeline.rag_exporter)
            self.assertIsNotNone(pipeline.dpo_exporter)
            # Output state containers are initialised empty.
            self.assertEqual(pipeline.raw_messages, [])
            self.assertEqual(pipeline.cleaned_messages, [])
            self.assertEqual(pipeline.threads, {})
            self.assertEqual(pipeline.sft_dialogues, [])

    def test_stages_have_expected_signatures(self):
        """The run_all method must exist and be callable with no args."""
        self.assertTrue(callable(MasterDataPipeline.run_all))


class TestMasterDataPipelineRunAll(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.export_dir = _build_synthetic_chat_export(self.tmpdir)
        self.data_dir = _build_synthetic_dataset(self.tmpdir)
        # Patch the path constants so the exporter classes write inside
        # our temp tree, not the repo's real dataset_output/.
        self.patches = [
            patch("src.pipeline.PARQUET_OUTPUT_DIR", self.data_dir / "parquet"),
            patch("src.pipeline.JSONL_OUTPUT_DIR", self.data_dir / "jsonl"),
            patch("src.pipeline.SAMPLES_OUTPUT_DIR", self.data_dir / "samples"),
            patch("src.pipeline.OUTPUT_DIR", self.data_dir),
            patch("src.pipeline.REPORTS_DIR", self.tmpdir / "reports"),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_all_executes_every_stage_in_order(self):
        pipeline = MasterDataPipeline(raw_export_dirs=[self.export_dir])
        # Mock every external collaborator so the run is fast and
        # deterministic. We assert on the call order to confirm the
        # seven-stage contract documented in the module docstring.
        with (
            patch.object(pipeline.anonymizer, "process_batch", return_value=["m1"]) as pii,
            patch.object(pipeline.exact_dedup, "deduplicate", return_value=(["m1"], 0)) as exact,
            patch.object(pipeline.minhash_lsh, "deduplicate_messages", return_value=(["m1"], 0)) as lsh,
            patch.object(pipeline.tagger, "tag_batch", return_value=["m1"]) as tagger,
            patch.object(pipeline.dag_builder, "build_threads", return_value=(["m1"], {1: ["m1"]})) as dag,
            patch.object(pipeline.conv_extractor, "extract_sft_dialogues", return_value=[]) as sft,
            patch.object(pipeline.conv_extractor, "extract_rag_chunks", return_value=[]) as rag,
            patch.object(pipeline.conv_extractor, "extract_dpo_pairs", return_value=[]) as dpo,
            patch.object(pipeline.parquet_exporter, "export_messages") as exp_m,
            patch.object(pipeline.parquet_exporter, "export_sft_dialogues") as exp_s,
            patch.object(pipeline.parquet_exporter, "export_rag_chunks") as exp_r,
            patch.object(pipeline.jsonl_exporter, "export_sharegpt") as exp_share,
            patch.object(pipeline.jsonl_exporter, "export_alpaca") as exp_alp,
            patch.object(pipeline.jsonl_exporter, "export_openai_chatml") as exp_chatml,
            patch.object(pipeline.rag_exporter, "export_rag_jsonl") as exp_rag_jsonl,
            patch.object(pipeline.dpo_exporter, "export_dpo_pairs") as exp_dpo,
            # Stage 7 collaborators are also heavy; mock them.
            patch("src.pipeline.DeepChatAnalyzer") as analyzer_cls,
            patch("src.pipeline.ReportGenerator") as reporter_cls,
            patch("src.pipeline.BenchmarkRunner") as bench_cls,
            patch("src.pipeline.DatasetValidator") as validator_cls,
        ):
            analyzer = MagicMock()
            analyzer.run_full_analysis.return_value = {"metric": 42}
            analyzer_cls.return_value = analyzer
            reporter = MagicMock()
            reporter_cls.return_value = reporter
            bench = MagicMock()
            bench_cls.return_value = bench
            validator = MagicMock()
            validator.validate_all.return_value = {"overall_passed": True, "dummy": True}
            validator_cls.return_value = validator

            summary = pipeline.run_all()

        # All collaborators were invoked.
        pii.assert_called_once()
        exact.assert_called_once()
        lsh.assert_called_once()
        tagger.assert_called_once()
        dag.assert_called_once()
        sft.assert_called_once()
        rag.assert_called_once()
        dpo.assert_called_once()
        exp_m.assert_called_once()
        exp_s.assert_called_once()
        exp_r.assert_called_once()
        exp_share.assert_called_once()
        exp_alp.assert_called_once()
        exp_chatml.assert_called_once()
        exp_rag_jsonl.assert_called_once()
        exp_dpo.assert_called_once()

        # Summary contract from run_all().
        self.assertIn("execution_time_seconds", summary)
        self.assertEqual(summary["raw_messages_count"], 2)  # 2 messages in fixture
        self.assertEqual(summary["cleaned_messages_count"], 1)  # 1 message after mocked dedup
        self.assertEqual(summary["threads_count"], 1)
        self.assertTrue(summary["validation_passed"])


if __name__ == "__main__":
    unittest.main()
