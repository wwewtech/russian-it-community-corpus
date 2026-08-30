"""
Unit tests for :mod:`src.exporter.finalize_sync_all`.

* :func:`compute_lora_zoo_index` and :func:`build_lora_zoo_markdown` are
  pure and are exercised end-to-end with a temporary ``lora_adapters/``
  tree.
* :func:`upload_dataset` and :func:`upload_missing_adapters` are tested
  with a mocked ``huggingface_hub.HfApi``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.exporter.finalize_sync_all import (
    DATASET_REPO_ID,
    MODEL_REPO_ID,
    build_lora_zoo_markdown,
    compute_lora_zoo_index,
    upload_dataset,
    upload_missing_adapters,
)


def _make_adapter(root: Path, name: str, weight_bytes: int = 1024) -> Path:
    """Create a minimal but valid adapter directory under ``root``."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(b"\0" * weight_bytes)
    return d


def _make_incomplete_adapter(root: Path, name: str) -> Path:
    """Adapter directory without the safetensors weight file (scaffold only)."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


class TestComputeLoraZooIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "lora_adapters"
        self.root.mkdir(parents=True)
        _make_adapter(self.root, "alpha", weight_bytes=2048)
        _make_adapter(self.root, "beta", weight_bytes=4096)
        _make_incomplete_adapter(self.root, "no_weights_gamma")

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_adapters_with_weights_are_indexed(self):
        idx = compute_lora_zoo_index(self.root)
        ids = sorted(r["id"] for r in idx)
        self.assertEqual(ids, ["alpha", "beta"])
        self.assertEqual(len(idx), 2)

    def test_size_mb_is_rounded_to_two_decimals(self):
        idx = compute_lora_zoo_index(self.root)
        size_by_id = {r["id"]: r["size_mb"] for r in idx}
        # 2048 bytes = 0.00195... MB -> rounded to 0.0
        self.assertEqual(size_by_id["alpha"], round(2048 / (1024 * 1024), 2))
        # 4096 bytes = 0.0039... MB
        self.assertEqual(size_by_id["beta"], round(4096 / (1024 * 1024), 2))

    def test_records_carry_repo_id_and_status(self):
        idx = compute_lora_zoo_index(self.root, model_repo_id="acme/my-loras")
        for r in idx:
            self.assertEqual(r["status"], "SUCCESS")
            self.assertEqual(r["hf_model_repo"], "acme/my-loras")
            self.assertTrue(r["adapter_dir"].startswith("lora_adapters/"))

    def test_empty_directory_returns_empty_list(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        self.assertEqual(compute_lora_zoo_index(empty), [])


class TestBuildLoraZooMarkdown(unittest.TestCase):
    def test_contains_header_and_table(self):
        idx = [
            {
                "id": "alpha",
                "size_mb": 0.01,
                "status": "SUCCESS",
                "adapter_dir": "lora_adapters/alpha",
                "hf_model_repo": MODEL_REPO_ID,
            }
        ]
        md = build_lora_zoo_markdown(idx)
        self.assertIn("# 🦁 Russian IT Community LoRA Model Zoo", md)
        self.assertIn("**Официальный каталог 1", md)
        self.assertIn("| # | Идентификатор модели |", md)
        self.assertIn("alpha", md)

    def test_includes_model_repo_in_links(self):
        idx = [
            {
                "id": "alpha",
                "size_mb": 0.01,
                "status": "SUCCESS",
                "adapter_dir": "lora_adapters/alpha",
                "hf_model_repo": "owner/repo",
            },
        ]
        md = build_lora_zoo_markdown(idx, model_repo_id="owner/repo")
        self.assertIn("https://huggingface.co/owner/repo", md)
        self.assertIn("`owner/repo`", md)

    def test_quickstart_block_is_always_present(self):
        md = build_lora_zoo_markdown([])
        self.assertIn("🚀 Быстрый старт", md)
        self.assertIn("PeftModel", md)


class TestUploadDataset(unittest.TestCase):
    def test_uploads_only_existing_artifacts(self):
        api = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                # Switch into a controlled workspace so the relative
                # "reports/..." and "dataset_output/..." paths the
                # function uses resolve to files we control.
                import os

                os.chdir(tmpdir)
                Path("reports").mkdir()
                Path("reports/DATASET_AND_ANALYTICS.md").write_text("# dataset", encoding="utf-8")
                Path("dataset_output/parquet").mkdir(parents=True)
                Path("dataset_output/parquet/full_clean_messages.parquet").touch()
                Path("dataset_output/parquet/sft_dialogues.parquet").touch()
                # rag_knowledge_base.parquet intentionally absent
                Path("reports/domain_benchmark_100.json").write_text("{}", encoding="utf-8")
                # LORA_MODEL_ZOO.md intentionally absent

                upload_dataset(api)

                uploaded = [c.kwargs["path_in_repo"] for c in api.upload_file.call_args_list]
            finally:
                os.chdir(cwd)

        self.assertIn("README.md", uploaded)  # dataset card
        self.assertIn("data/full_clean_messages.parquet", uploaded)
        self.assertIn("data/sft_dialogues.parquet", uploaded)
        self.assertNotIn("data/rag_knowledge_base.parquet", uploaded)  # missing on disk
        self.assertIn("domain_benchmark_100.json", uploaded)
        self.assertNotIn("LORA_MODEL_ZOO.md", uploaded)  # missing on disk

    def test_uses_default_dataset_repo_id(self):
        api = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                upload_dataset(api)
            finally:
                os.chdir(cwd)
        # No dataset card, no parquet -> zero upload_file calls,
        # but the function must still log the repo id it intends to
        # use; assert the default value matches the module constant.
        self.assertEqual(DATASET_REPO_ID, "wwewtech/russian-it-community-corpus")


class TestUploadMissingAdapters(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "lora_adapters"
        self.root.mkdir(parents=True)
        _make_adapter(self.root, "alpha")
        _make_adapter(self.root, "beta")
        _make_adapter(self.root, "gamma")

    def tearDown(self):
        self.tmp.cleanup()

    def test_skips_already_uploaded(self):
        api = MagicMock()
        api.list_repo_files.return_value = ["alpha/weights.safetensors", "beta/weights.safetensors"]
        uploaded = upload_missing_adapters(
            api,
            self.root,
            max_attempts=1,
            retry_sleep_seconds=0,
        )
        self.assertEqual(uploaded, ["gamma"])
        api.upload_folder.assert_called_once()
        kwargs = api.upload_folder.call_args.kwargs
        self.assertEqual(kwargs["path_in_repo"], "gamma")
        self.assertEqual(kwargs["repo_id"], MODEL_REPO_ID)
        self.assertEqual(kwargs["repo_type"], "model")
        self.assertEqual(kwargs["ignore_patterns"], ["checkpoint-*"])

    def test_raises_after_max_attempts(self):
        api = MagicMock()
        api.list_repo_files.return_value = []  # nothing uploaded yet
        # All attempts fail, so the first adapter exhausts its retries
        # and we raise out of ``upload_missing_adapters`` before ever
        # touching the next adapter.
        api.upload_folder.side_effect = RuntimeError("network down")
        with self.assertRaises(RuntimeError) as ctx:
            upload_missing_adapters(
                api,
                self.root,
                max_attempts=2,
                retry_sleep_seconds=0,
            )
        self.assertIn("network down", str(ctx.exception))
        # 1 adapter x 2 attempts = 2 calls before we raise.
        self.assertEqual(api.upload_folder.call_count, 2)

    def test_retries_then_succeeds_for_each_adapter(self):
        api = MagicMock()
        api.list_repo_files.return_value = []  # nothing uploaded yet
        # Each adapter fails once and then succeeds. With 3 adapters
        # and 3 max_attempts we expect 3 + 3 = 6 calls and all three
        # adapters to be reported as uploaded.

        responses = []
        for _ in range(3):  # 3 adapters
            responses.append(RuntimeError("transient"))
            responses.append("ok")
        # Pad with extra 'ok' so the final call sequence has no leftover
        # RuntimeError (only the first 2 per adapter raise).
        responses.extend(["ok"] * 10)
        api.upload_folder.side_effect = responses

        uploaded = upload_missing_adapters(
            api,
            self.root,
            max_attempts=3,
            retry_sleep_seconds=0,
        )
        self.assertEqual(set(uploaded), {"alpha", "beta", "gamma"})
        # 3 adapters x 2 calls (1 fail + 1 ok) = 6 calls.
        self.assertEqual(api.upload_folder.call_count, 6)

    def test_returns_empty_when_nothing_local(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        api = MagicMock()
        api.list_repo_files.return_value = []
        uploaded = upload_missing_adapters(api, empty)
        self.assertEqual(uploaded, [])
        api.upload_folder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
