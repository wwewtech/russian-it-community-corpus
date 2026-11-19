"""Determinism/fail-closed tests for the adapter release packager."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from ricc.lora.policy_gate import PolicyViolation
from scripts.package_adapter_release import REQUIRED_REPORTS, create_release_archive, sha256_file


class TestAdapterReleasePackage(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.adapter = self.root / "lora_adapters" / "smollm2_135m_instruct"
        self.adapter.mkdir(parents=True)
        self.reports = self.root / "reports"
        self.reports.mkdir(parents=True)
        self.output = self.root / "dist" / "release.zip"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_meta(self, **overrides: object) -> None:
        meta: dict[str, object] = {
            "actual_steps": 21209,
            "required_steps_for_full_epoch": 21209,
            "corpus_coverage_pct": 100.0,
            "holdout_tail_dialogues": 1000,
            "dataset_sha256": "seed",
            "train_dialogues": 169667,
            "dataset_total_dialogues": 171520,
            "max_samples": 200000,
            "eval_frac": 0.005,
            "time_budget_hours": None,
        }
        meta.update(overrides)
        (self.adapter / "training_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def _write_reports(self, *, omit: set[str] | frozenset[str] = frozenset()) -> Path:
        for filename in REQUIRED_REPORTS:
            if filename in omit:
                continue
            (self.reports / filename).write_text(f"fixture:{filename}\n", encoding="utf-8")
        return self.reports

    def test_full_epoch_archive_is_verified_and_digest_pinned(self) -> None:
        (self.adapter / "adapter_model.safetensors").write_bytes(b"actual-weights")
        self._write_meta()
        self._write_reports()

        manifest = create_release_archive(
            adapter=self.adapter,
            output=self.output,
            reports_dir=self.reports,
        )

        self.assertEqual(manifest["training_steps"], 21209)
        archive = Path(str(manifest["archive"]))
        self.assertEqual(manifest["file_count"], 10)
        self.assertEqual(manifest["source_file_count"], 9)
        self.assertGreater(int(manifest["bytes"]), 0)
        self.assertEqual(manifest["sha256"], sha256_file(archive))

        with ZipFile(archive) as bundle:
            self.assertIsNone(bundle.testzip())
            names = set(bundle.namelist())
            self.assertIn("lora_adapters/smollm2_135m_instruct/adapter_model.safetensors", names)
            self.assertIn("reports/academic_benchmarks_raw_outputs.jsonl", names)
            self.assertIn("reports/benchmark_smollm2_135m_instruct.json", names)
            provenance = json.loads(bundle.read("PROVENANCE.json"))

            non_provenance_names = names - {"PROVENANCE.json"}
            self.assertEqual(set(provenance["source_sha256"]), non_provenance_names)
            self.assertEqual(set(provenance["source_bytes"]), non_provenance_names)

            for entry_name in sorted(non_provenance_names):
                payload = bundle.read(entry_name)
                self.assertEqual(provenance["source_sha256"][entry_name], hashlib.sha256(payload).hexdigest())
                self.assertEqual(provenance["source_bytes"][entry_name], len(payload))

        self.assertEqual(provenance["heldout_tail_dialogues"], 1000)
        weights_name = "lora_adapters/smollm2_135m_instruct/adapter_model.safetensors"
        self.assertEqual(provenance["source_sha256"][weights_name], hashlib.sha256(b"actual-weights").hexdigest())
        self.assertIn(weights_name, provenance["source_bytes"])

    def test_archive_bytes_are_deterministic(self) -> None:
        (self.adapter / "adapter_model.safetensors").write_bytes(b"actual-weights")
        self._write_meta()
        self._write_reports()

        first = create_release_archive(
            adapter=self.adapter,
            output=self.output,
            reports_dir=self.reports,
        )
        second = create_release_archive(
            adapter=self.adapter,
            output=self.output,
            reports_dir=self.reports,
        )
        copy_path = self.root / "dist" / "release_copy.zip"
        third = create_release_archive(
            adapter=self.adapter,
            output=copy_path,
            reports_dir=self.reports,
        )

        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["bytes"], second["bytes"])
        self.assertEqual(first["sha256"], third["sha256"])
        self.assertEqual(first["bytes"], third["bytes"])

    def test_missing_evidence_report_fails_closed(self) -> None:
        (self.adapter / "adapter_model.safetensors").write_bytes(b"actual-weights")
        self._write_meta()
        self._write_reports(omit={"benchmark_smollm2_135m_instruct.json"})

        with self.assertRaisesRegex(FileNotFoundError, "missing release evidence file"):
            create_release_archive(
                adapter=self.adapter,
                output=self.output,
                reports_dir=self.reports,
            )
        self.assertFalse(self.output.exists())

    def test_missing_weights_raises_no_placeholder(self) -> None:
        self._write_meta()
        self._write_reports()

        with self.assertRaisesRegex(RuntimeError, "no actual adapter weights"):
            create_release_archive(
                adapter=self.adapter,
                output=self.output,
                reports_dir=self.reports,
            )
        self.assertFalse(self.output.exists())

    def test_partial_adapter_refuses_release(self) -> None:
        (self.adapter / "adapter_model.safetensors").write_bytes(b"weights")
        self._write_meta(actual_steps=100, required_steps_for_full_epoch=21209, corpus_coverage_pct=30.0)
        self._write_reports()

        with self.assertRaises(PolicyViolation):
            create_release_archive(
                adapter=self.adapter,
                output=self.output,
                reports_dir=self.reports,
            )
        self.assertFalse(self.output.exists())

    def test_timeboxed_adapter_refuses_release(self) -> None:
        (self.adapter / "adapter_model.safetensors").write_bytes(b"weights")
        self._write_meta(time_budget_hours=1)
        self._write_reports()

        with self.assertRaises(PolicyViolation):
            create_release_archive(
                adapter=self.adapter,
                output=self.output,
                reports_dir=self.reports,
            )
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
