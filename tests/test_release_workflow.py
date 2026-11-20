"""Static contract for the gated, non-empty GitHub release path."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestReleaseWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
        cls.release = cls.workflow["jobs"]["release"]

    def test_release_waits_for_real_data_and_hf_preflight(self):
        needs = self.release["needs"]
        self.assertIn("slo-gate-real", needs)
        self.assertIn("hf-secrets-check", needs)
        self.assertEqual(self.release["if"], "github.event_name == 'release'")

    def test_release_consumes_the_verified_dataset_artifact(self):
        commands = "\n".join(str(step.get("run", "")) for step in self.release["steps"])
        self.assertIn("release-dataset-verified", str(self.release["steps"]))
        self.assertIn("python-package", str(self.release["steps"]))
        self.assertIn("release tag", commands)
        self.assertIn("dist/*.whl dist/*.tar.gz", commands)
        package_commands = "\n".join(str(step.get("run", "")) for step in self.workflow["jobs"]["package"]["steps"])
        self.assertIn("python -m build", package_commands)
        self.assertIn("pip install dist/*.whl", package_commands)
        self.assertNotIn("--index-strategy", package_commands)
        self.assertNotIn("--index-strategy", commands)
        self.assertNotIn("requirements.txt", package_commands)
        self.assertNotIn("requirements.lock.txt", package_commands)
        self.assertNotIn("pytest", package_commands)
        self.assertIn("python-package", str(self.workflow["jobs"]["package"]["steps"]))

    def test_dataset_gate_publishes_the_same_files_it_verified(self):
        real_gate = self.workflow["jobs"]["slo-gate-real"]
        uploads = [
            step for step in real_gate["steps"] if step.get("with", {}).get("name") == "release-dataset-verified"
        ]
        self.assertEqual(len(uploads), 1)
        paths = uploads[0]["with"]["path"]
        for name in ("full_clean_messages", "sft_dialogues", "rag_knowledge_base"):
            self.assertIn(f"{name}.parquet", paths)
        self.assertIn("reports/dataset_manifest.json", paths)
        self.assertEqual(uploads[0]["with"]["if-no-files-found"], "error")


if __name__ == "__main__":
    unittest.main()
