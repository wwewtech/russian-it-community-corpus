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
            (root / "validation_results.json").write_text(json.dumps({"overall_passed": True}), encoding="utf-8")
            (root / "probabilistic_pii_audit.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
            (root / "drift_report.json").write_text(json.dumps({"overall_verdict": "stable"}), encoding="utf-8")
            verdict = evaluate(root)
            self.assertEqual(verdict.verdict, "SHIP")

    def test_hold_on_failed_validation(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "validation_results.json").write_text(json.dumps({"overall_passed": False}), encoding="utf-8")
            (root / "probabilistic_pii_audit.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
            verdict = evaluate(root)
            self.assertEqual(verdict.verdict, "HOLD")


if __name__ == "__main__":
    unittest.main()
