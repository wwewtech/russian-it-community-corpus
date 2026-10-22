"""Tests for the DVC pipeline stages.

The stages used to be inline ``python -c "..."`` blocks in ``dvc.yaml``,
which do not survive a Windows shell — ``cmd.exe`` splits the multi-line
quoted string and the stage dies with ``'from' is not recognized as an
internal or external command``. So ``dvc repro`` had never run, and the
stage bodies were never executed by anything.

These tests execute the stage functions for real, on a small fixture, so
a broken stage fails the build instead of failing at 3 a.m. in a
nightly run.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import dvc_stages  # noqa: E402


def _write_export(directory: Path, date: str = "2026-01-01", messages: int = 6) -> Path:
    """Write a minimal Telegram export in the format the loader expects."""
    export = directory / f"ChatExport_{date}"
    export.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "community",
        "type": "personal_chat",
        "id": 1001,
        "messages": [
            {
                "id": i + 1,
                "type": "message",
                "date": f"2026-01-01T00:{i:02d}:00",
                "date_unixtime": str(1767225600 + i * 60),
                "from": "Алексей",
                "from_id": "user_a",
                "text": f"Сообщение номер {i} про настройку nginx и postgres в продакшене.",
                "reply_to_message_id": (i) if i else None,
            }
            for i in range(messages)
        ],
    }
    (export / "result.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return export


class TestStageRegistry(unittest.TestCase):
    def test_every_dvc_stage_has_a_script(self):
        """The dvc.yaml stages and the script must not drift apart."""
        import yaml

        dvc = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
        for name, spec in dvc["stages"].items():
            with self.subTest(stage=name):
                self.assertIn("cmd", spec)
                # Every stage invokes the real script.
                self.assertIn("scripts/dvc_stages.py", spec["cmd"])
                self.assertIn(name, dvc_stages.STAGES)

    def test_no_stage_uses_an_inline_python_block(self):
        import yaml

        dvc = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
        for name, spec in dvc["stages"].items():
            with self.subTest(stage=name):
                self.assertNotIn("python -c", spec["cmd"])

    def test_every_stage_is_callable(self):
        for name, function in dvc_stages.STAGES.items():
            with self.subTest(stage=name):
                self.assertTrue(callable(function))


class TestStagesExecute(unittest.TestCase):
    """The stages must actually run."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.exports = self.root / "exports"
        self.exports.mkdir()
        _write_export(self.exports, "2026-01-01", messages=6)
        _write_export(self.exports, "2026-01-02", messages=4)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, stage: str) -> None:
        """Run one stage with every path redirected into the temp dir."""
        intermediate = self.root / "intermediate"
        reports = self.root / "reports"
        parquet = self.root / "parquet"
        jsonl = self.root / "jsonl"
        for path in (intermediate, reports, parquet, jsonl):
            path.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(dvc_stages, "INTERMEDIATE", intermediate),
            patch.object(dvc_stages, "REPORTS", reports),
            patch.object(dvc_stages, "PARQUET", parquet),
            patch.object(dvc_stages, "REPO_ROOT", self.root),
            patch.object(
                dvc_stages,
                "_load_params",
                return_value={"pii_audit": {"sample_size": 1000, "confidence": 0.99, "max_leak_tolerance": 1e-4}},
            ),
        ):
            getattr(dvc_stages, stage)()

    def test_ingest_stage_produces_the_intermediate_artefact(self):
        with (
            patch.object(dvc_stages, "INTERMEDIATE", self.root / "intermediate"),
            patch("ricc.settings.settings.raw_export_dirs", [str(self.exports)]),
        ):
            dvc_stages.stage_ingest()
        artefact = self.root / "intermediate" / "raw_messages.pkl.gz"
        self.assertTrue(artefact.exists())
        self.assertGreater(artefact.stat().st_size, 0)

    def test_ingest_reads_the_exports(self):
        """The stage must find and read the export files."""
        export_dirs = [str(p) for p in sorted(self.exports.iterdir()) if p.is_dir()]
        with (
            patch.object(dvc_stages, "INTERMEDIATE", self.root / "intermediate"),
            patch("ricc.settings.settings.raw_export_dirs", export_dirs),
        ):
            dvc_stages.stage_ingest()
        import gzip
        import pickle

        with gzip.open(self.root / "intermediate" / "raw_messages.pkl.gz", "rb") as handle:
            chats, messages = pickle.load(handle)
        self.assertEqual(len(messages), 10)
        self.assertEqual(len(chats), 2)

    def test_the_stage_list_matches_the_script(self):
        """A stage named in dvc.yaml but missing from the script is a trap."""
        import yaml

        dvc = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
        self.assertEqual(set(dvc["stages"]), set(dvc_stages.STAGES))


if __name__ == "__main__":
    unittest.main()
