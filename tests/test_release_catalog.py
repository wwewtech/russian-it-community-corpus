"""Release catalog and provenance are derived from verified registry records."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.release_catalog import validate_registry, write_release_catalog


class ReleaseCatalogTest(unittest.TestCase):
    def test_catalog_uses_registry_base_and_immutable_revisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = {
                "adapters": [
                    {
                        "slug": "smollm2_135m_instruct",
                        "base_model": "acme/actual-base",
                        "base_model_revision": "1" * 40,
                        "hf_model_revision": "2" * 40,
                        "policy_compliant": True,
                        "actual_steps": 8,
                        "required_steps": 8,
                        "corpus_coverage_pct": 100.0,
                        "safetensors_bytes": 1024,
                        "safetensors_sha256": "3" * 64,
                        "local_path": "lora_adapters/smollm2_135m_instruct/",
                    }
                ]
            }
            path = root / "registry.json"
            path.write_text(json.dumps(registry), encoding="utf-8")
            validate_registry(path)
            report = {
                "repo_id": "owner/repo",
                "revision": "2" * 40,
                "adapters": [
                    {
                        "id": "smollm2_135m_instruct",
                        "final": True,
                        "base_model": "acme/actual-base",
                        "classification": "full",
                        "actual_steps": 8,
                        "required_steps": 8,
                        "train_dialogues": 64,
                    },
                    {"id": "legacy", "final": False, "classification": "legacy_no_provenance"},
                ],
            }
            with patch("ricc.settings.settings") as settings:
                settings.REPORTS_DIR = root / "reports"
                write_release_catalog(root, report, path)
            card = (root / "reports/HF_MODEL_CARD.md").read_text(encoding="utf-8")
            self.assertIn('model_id = "acme/actual-base"', card)
            self.assertIn(f'base_revision = "{"1" * 40}"', card)
            self.assertIn(f'adapter_revision = "{"2" * 40}"', card)
            self.assertNotIn("Qwen2.5", card)
            self.assertIn("Legacy adapters without training provenance", card)

    def test_mismatched_registry_pin_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(
                json.dumps(
                    {
                        "adapters": [
                            {
                                "slug": "alpha",
                                "base_model": "acme/base",
                                "base_model_revision": "not-a-commit",
                                "hf_model_revision": "2" * 40,
                                "policy_compliant": True,
                                "actual_steps": 8,
                                "required_steps": 8,
                                "corpus_coverage_pct": 100.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "base_model_revision"):
                validate_registry(path)


if __name__ == "__main__":
    unittest.main()
