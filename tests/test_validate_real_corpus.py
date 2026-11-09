"""The real-corpus validation report must be parquet-only and honour strictness."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_synthetic_parquet import build as build_fixture  # noqa: E402
from validate_real_corpus import main  # noqa: E402


class TestValidateRealCorpus(unittest.TestCase):
    def test_writes_a_parquet_only_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_fixture(root / "dataset_output", 10)
            code = main(
                [
                    "--root",
                    str(root / "dataset_output"),
                    "--reports",
                    str(root / "reports"),
                ]
            )
            self.assertEqual(code, 0)
            report = json.loads((root / "reports" / "validation_results.json").read_text(encoding="utf-8"))
            self.assertEqual(report["scope"], "parquet-only")
            self.assertTrue(report["overall_passed"])
            self.assertTrue(report["parquet_files"]["passed"])
            self.assertIn("jsonl_files", report["skipped_checks"])

    def test_strips_metadata_to_exercise_the_legacy_flag(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_fixture(root / "dataset_output", 10)
            path = root / "dataset_output" / "parquet" / "full_clean_messages.parquet"
            table = pq.read_table(path).replace_schema_metadata(None)
            pq.write_table(table, path)

            strict = main(
                [
                    "--root",
                    str(root / "dataset_output"),
                    "--reports",
                    str(root / "reports"),
                ]
            )
            self.assertEqual(strict, 1)

            legacy = main(
                [
                    "--root",
                    str(root / "dataset_output"),
                    "--reports",
                    str(root / "reports"),
                    "--allow-legacy-metadata",
                ]
            )
            self.assertEqual(legacy, 0)


if __name__ == "__main__":
    unittest.main()
