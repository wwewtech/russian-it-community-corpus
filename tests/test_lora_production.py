"""Registry-driven production adapter selection and path safety."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ricc.lora.production import default_adapter, local_adapter_path, resolve_adapter


class TestProductionAdapterResolution(unittest.TestCase):
    def test_repository_default_is_verified_smollm2(self):
        spec = default_adapter()
        self.assertEqual(spec.slug, "smollm2_135m_instruct")
        self.assertEqual(spec.base_model, "HuggingFaceTB/SmolLM2-135M-Instruct")
        self.assertEqual(len(spec.base_revision), 40)
        self.assertEqual(len(spec.adapter_revision), 40)
        self.assertTrue(spec.policy_compliant)
        self.assertIn(spec.source, {"hub", "local"})

    def test_local_weights_select_local_not_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lora_adapters").mkdir()
            (root / "lora_adapters/registry.json").write_text(
                Path("lora_adapters/registry.json").read_text(encoding="utf-8"), encoding="utf-8"
            )
            spec = default_adapter(root)
            self.assertEqual(spec.source, "hub")
            adapter = root / "lora_adapters" / spec.slug
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            spec = default_adapter(root)
            self.assertEqual(spec.source, "local")
            self.assertEqual(local_adapter_path(spec, root), adapter.resolve())

    def test_experimental_adapter_is_explicitly_not_production(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lora_adapters").mkdir()
            (root / "lora_adapters/registry.json").write_text(
                Path("lora_adapters/registry.json").read_text(encoding="utf-8"), encoding="utf-8"
            )
            adapter = root / "lora_adapters" / "qwen2.5_1.5b_instruct"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            with self.assertWarnsRegex(UserWarning, "historical/experimental"):
                spec = resolve_adapter("qwen2.5_1.5b_instruct", root=root)
            self.assertFalse(spec.policy_compliant)
            self.assertEqual(spec.source, "local")

    def test_legacy_adapter_requires_explicit_local_weights(self):
        with self.assertRaisesRegex(RuntimeError, "not production-compliant"):
            resolve_adapter("qwen2.5_1.5b_instruct")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lora_adapters").mkdir()
            (root / "lora_adapters/registry.json").write_text(
                Path("lora_adapters/registry.json").read_text(encoding="utf-8"), encoding="utf-8"
            )
            adapter = root / "lora_adapters" / "qwen2.5_1.5b_instruct"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            with self.assertWarnsRegex(UserWarning, "historical/experimental"):
                spec = resolve_adapter("qwen2.5_1.5b_instruct", root=root)
            self.assertEqual(spec.source, "local")
            self.assertFalse(spec.policy_compliant)

    def test_legacy_adapter_cannot_be_forced_to_hub(self):
        with self.assertRaisesRegex(RuntimeError, "cannot be selected as a pinned Hub release"):
            resolve_adapter("qwen2.5_1.5b_instruct", source="hub")

    def test_unknown_or_unsafe_adapter_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid adapter"):
            resolve_adapter("../outside")
        with self.assertRaisesRegex(RuntimeError, "not uniquely declared"):
            resolve_adapter("not-in-registry")


if __name__ == "__main__":
    unittest.main()
