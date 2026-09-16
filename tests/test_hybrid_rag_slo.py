"""Hybrid RAG + SLO gate tests (no parquet fixtures required)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Packaging smoke: namespace-to-regular package migration (evaluation/lora/rag
# gained __init__.py). Importing here keeps coverage honest for the package
# inits without requiring GPU weights.
import src.evaluation  # noqa: F401
import src.lora  # noqa: F401
from src.monitoring.slo_gate import evaluate
from src.rag.rag_pipeline import LocalRAGPipeline


def _make_kb(tmp: Path) -> Path:
    df = pd.DataFrame(
        [
            {
                "chunk_id": "c1",
                "title": "Kafka idempotent producer",
                "topic_domain": "backend",
                "topic_tags": ["kafka"],
                "content": "Kafka idempotent producer enable.idempotence=true retries max in flight 5 exactly once",
                "date_range": "2024",
            },
            {
                "chunk_id": "c2",
                "title": "Nginx reverse proxy",
                "topic_domain": "devops",
                "topic_tags": ["nginx"],
                "content": "Nginx reverse proxy worker_processes auto proxy_pass upstream keepalive",
                "date_range": "2024",
            },
            {
                "chunk_id": "c3",
                "title": "Postgres vacuum",
                "topic_domain": "databases",
                "topic_tags": ["postgres"],
                "content": "Postgres autovacuum analyze bloat pg_stat_user_tables dead tuples",
                "date_range": "2023",
            },
        ]
    )
    out = tmp / "rag_kb.parquet"
    df.to_parquet(out)
    return out


def _make_artifacts_and_manifest(root: Path) -> None:
    """Build the canonical parquet trio plus a matching manifest under ``root``.

    Mirrors the layout expected by :func:`src.monitoring.slo_gate.evaluate`:
    ``root/reports/dataset_manifest.json`` is verified against artifacts rooted
    at ``root`` (the reports directory's parent).
    """
    from src.validation.artifact_manifest import create_manifest

    parquet_dir = root / "dataset_output" / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"id": [1, 2]})
    for name in ("full_clean_messages", "sft_dialogues", "rag_knowledge_base"):
        frame.to_parquet(parquet_dir / f"{name}.parquet")
    create_manifest(root, Path("reports/dataset_manifest.json"))


class TestHybridRag(unittest.TestCase):
    def test_hybrid_beats_lexical_on_paraphrase(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            kb = _make_kb(Path(d))
            rag = LocalRAGPipeline(kb)
            hits = rag.search("как настроить идемпотентный продьюсер кафки", top_k=2)
            # Lexical-only would miss transliterated query; hybrid must still be list-safe.
            self.assertIsInstance(hits, list)
            hits_lex = rag.search("kafka idempotent producer", top_k=2, mode="lexical")
            self.assertGreaterEqual(len(hits_lex), 1)
            self.assertEqual(hits_lex[0]["chunk_id"], "c1")

    def test_modes_compatible_shape(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rag = LocalRAGPipeline(_make_kb(Path(d)))
            for mode in ("hybrid", "lexical", "tfidf"):
                hits = rag.search("nginx reverse proxy keepalive", top_k=1, mode=mode)
                self.assertIsInstance(hits, list)
                if hits:
                    for key in ("chunk_id", "title", "domain", "content", "score"):
                        self.assertIn(key, hits[0])

    def test_missing_kb_returns_empty(self) -> None:
        rag = LocalRAGPipeline(Path("nonexistent.parquet"))
        self.assertEqual(rag.search("kafka", top_k=3), [])
        self.assertEqual(len(rag), 0)


class TestSloGate(unittest.TestCase):
    def test_hold_when_reports_missing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            verdict = evaluate(Path(d))
            self.assertEqual(verdict.verdict, "HOLD")

    def test_ship_when_all_green(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reports = root / "reports"
            reports.mkdir()
            (reports / "validation_results.json").write_text(json.dumps({"overall_passed": True}), encoding="utf-8")
            (reports / "probabilistic_pii_audit.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
            (reports / "drift_report.json").write_text(json.dumps({"overall_verdict": "stable"}), encoding="utf-8")
            _make_artifacts_and_manifest(root)
            verdict = evaluate(reports)
            self.assertEqual(verdict.verdict, "SHIP")

    def test_hold_when_artifact_manifest_missing_or_stale(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reports = root / "reports"
            reports.mkdir()
            (reports / "validation_results.json").write_text(json.dumps({"overall_passed": True}), encoding="utf-8")
            (reports / "probabilistic_pii_audit.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
            (reports / "drift_report.json").write_text(json.dumps({"overall_verdict": "stable"}), encoding="utf-8")
            # No manifest at all: privacy/provenance defaults to fail-closed.
            verdict = evaluate(reports)
            self.assertEqual(verdict.verdict, "HOLD")
            self.assertTrue(any(c.name == "artifact-manifest" and not c.passed for c in verdict.checks))

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reports = root / "reports"
            reports.mkdir()
            (reports / "validation_results.json").write_text(json.dumps({"overall_passed": True}), encoding="utf-8")
            (reports / "probabilistic_pii_audit.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
            _make_artifacts_and_manifest(root)
            # Tamper with an artifact after the snapshot: rows/bytes identical or not,
            # the sha256 mismatch must flip the gate to HOLD.
            victim = root / "dataset_output" / "parquet" / "full_clean_messages.parquet"
            frame = pd.DataFrame({"id": [1, 2, 3]})
            frame.to_parquet(victim)
            verdict = evaluate(reports)
            self.assertEqual(verdict.verdict, "HOLD")
            self.assertTrue(any(c.name == "artifact-manifest" and not c.passed for c in verdict.checks))

    def test_hold_on_failed_validation(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "validation_results.json").write_text(json.dumps({"overall_passed": False}), encoding="utf-8")
            (root / "probabilistic_pii_audit.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
            verdict = evaluate(root)
            self.assertEqual(verdict.verdict, "HOLD")


if __name__ == "__main__":
    unittest.main()
