"""Determinism/fail-closed tests for the adapter release packager."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from ricc.lora.policy_gate import PolicyViolation
from scripts.package_adapter_release import create_release_archive


class TestAdapterReleasePackage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) if hasattr(self, "temp_dir") else Path(self.tmp.name)
        self.adapter = self.root / "lora_adapters" / "smollm2_135m_instruct"
        self.adapter.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_meta(self, **overrides):
        meta = {
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

    def test_full_epoch_archive_is_verified_and_digest_pinned(self):
        (self.adapter / "adapter_model.safetensors").write_bytes(b"actual-weights")
        self._write_meta()
        manifest = create_release_archive(adapter=self.adapter)

        self.assertEqual(manifest["training_steps"], 21209)
        archive = Path(manifest["archive"])
        self.assertEqual(manifest["file_count"], 10)
        self.assertGreater(manifest["bytes"], 0)
        with ZipFile(archive) as bundle:
            self.assertIn("lora_adapters/smollm2_135m_instruct/adapter_model.safetensors", bundle.namelist())
            self.assertIn("reports/academic_benchmarks_raw_outputs.jsonl", bundle.namelist())
            provenance = json.loads(bundle.read("PROVENANCE.json"))
        self.assertEqual(provenance["heldout_tail_dialogues"], 1000)
        weights_name = "lora_adapters/smollm2_135m_instruct/adapter_model.safetensors"
        import hashlib

        self.assertEqual(provenance["source_sha256"][weights_name], hashlib.sha256(b"actual-weights").hexdigest())
        # Must never publish only the provenance without any weight bytes.
        self.assertIn(weights_name, provenance["source_bytes"])
        Path.unlink(archive)

    def test_archive_bytes_are_deterministic(self) -> None:
        (self.adapter / "adapter_model.safetensors").write_bytes(b"actual-weights")
        self._write_meta()
        output = self.root / "dist" / "release.zip"
        first = create_release_archive(adapter=self.adapter, output=output)
        second = create_release_archive(adapter=self.adapter, output=output)
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["bytes"], second["bytes"])
        Path.unlink(output)

    def test_missing_weights_raises_no_placeholder(self):
        self._write_meta()
        with self.assertRaisesRegex(RuntimeError, "no actual adapter weights"):
            create_release_archive(adapter=self.adapter)

    def test_partial_adapter_refuses_release(self):
        (self.adapter / "adapter_model.safetensors").write_bytes(b"weights")
        self._write_meta(actual_steps=100, required_steps_for_full_epoch=21209, corpus_coverage_pct=30.0)
        with self.assertRaises(PolicyViolation):
            create_release_archive(adapter=self.adapter)

    def test_timeboxed_adapter_refuses_release(self):
        (self.adapter / "adapter_model.safetensors").write_bytes(b"weights")
        self._write_meta(time_budget_hours=1)
        with self.assertRaises(PolicyViolation):
            create_release_archive(adapter=self.adapter)


if __name__ == "__main__":
    unittest.main()
