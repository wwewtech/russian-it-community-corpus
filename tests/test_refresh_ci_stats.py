"""tests/test_refresh_ci_stats.py — CI stats refresh must be derived, not fabricated.

The former inline CI step hardcoded validation_passed=True, threads_count=-1
and _source.synthetic=False on synthetic fixtures. These tests pin the honest
behavior: every field comes from the artifacts (or a real validator run), and
the synthetic marker propagates into _source.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_synthetic_parquet import build as build_fixture  # noqa: E402
from refresh_ci_stats import collect_stats, main  # noqa: E402


def _build(root: Path, rows: int = 100) -> Path:
    build_fixture(root / "dataset_output", rows)
    return root


class TestRefreshCiStats(unittest.TestCase):
    def test_stats_derived_from_fixture_are_honest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build(Path(tmp))
            stats = collect_stats(root)
            self.assertEqual(stats["cleaned_messages_count"], 100)
            self.assertEqual(stats["sft_dialogues_count"], 100)
            self.assertEqual(stats["rag_chunks_count"], 100)
            self.assertGreater(stats["threads_count"], 0, "threads_count must come from real thread_id values")
            self.assertNotEqual(stats["threads_count"], -1, "the -1 sentinel must never reappear")
            self.assertEqual(stats["dpo_pairs_count"], 100)
            self.assertIsInstance(stats["validation_passed"], bool)
            self.assertFalse(stats["_source"]["synthetic"], "no marker present => not synthetic")

    def test_synthetic_marker_flags_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build(Path(tmp))
            marker = root / "reports" / "synthetic_fixture.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"_source": {"synthetic": True}}), encoding="utf-8")
            stats = collect_stats(root)
            self.assertTrue(stats["_source"]["synthetic"], "marker presence must set _source.synthetic")

    def test_validation_passed_comes_from_real_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build(Path(tmp))
            with patch("refresh_ci_stats.DatasetValidator") as cls:
                cls.return_value.validate_all.return_value = {"overall_passed": False}
                stats = collect_stats(root)
            self.assertFalse(stats["validation_passed"], "must reflect the actual validator verdict")

    def test_missing_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dataset_output" / "parquet").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                collect_stats(root)

    def test_main_writes_stats_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build(Path(tmp))
            out = root / "reports" / "stats.json"
            code = main(["--root", str(root), "--out", str(out)])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["cleaned_messages_count"], 100)
            self.assertIn("_source", payload)


if __name__ == "__main__":
    unittest.main()
