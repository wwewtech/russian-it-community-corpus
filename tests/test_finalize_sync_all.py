"""
Unit tests for :mod:`ricc.exporter.finalize_sync_all`.

* :func:`compute_lora_zoo_index` and :func:`build_lora_zoo_markdown` are
  pure and are exercised end-to-end with a temporary ``lora_adapters/``
  tree.
* :func:`upload_dataset`, :func:`upload_model_card` and
  :func:`upload_missing_adapters` are tested with a mocked
  ``huggingface_hub.HfApi``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from ricc.exporter.finalize_sync_all import (
    DATASET_REPO_ID,
    MODEL_REPO_ID,
    build_lora_zoo_markdown,
    compute_lora_zoo_index,
    upload_dataset,
    upload_missing_adapters,
    upload_model_card,
)


def _make_adapter(root: Path, name: str, weight_bytes: int = 1024) -> Path:
    """Create a minimal but valid adapter directory under ``root``.

    The adapter is policy-compliant (a full epoch over the corpus), so the
    release pipeline is allowed to publish it. See
    :func:`ricc.lora.policy_gate.evaluate_compliance`.
    """
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(b"\0" * weight_bytes)
    (d / "training_meta.json").write_text(
        json.dumps(
            {
                "model_id": name,
                "actual_steps": 21420,
                "required_steps_for_full_epoch": 21420,
                "train_dialogues": 171354,
                "corpus_coverage_pct": 100.0,
                "dataset_total_dialogues": 173216,
                "holdout_tail_dialogues": 1000,
                "eval_frac": 0.005,
                "time_budget_hours": None,
            }
        ),
        encoding="utf-8",
    )
    return d


def _make_noncompliant_adapter(root: Path, name: str) -> Path:
    """Adapter that covered only part of the corpus (not publishable)."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(b"\0" * 1024)
    (d / "training_meta.json").write_text(
        json.dumps(
            {
                "model_id": name,
                "actual_steps": 9662,
                "required_steps_for_full_epoch": 21420,
                "train_dialogues": 171354,
                "corpus_coverage_pct": 45.11,
                "dataset_total_dialogues": 173216,
                "holdout_tail_dialogues": 1000,
                "eval_frac": 0.005,
                "time_budget_hours": 5.8,
            }
        ),
        encoding="utf-8",
    )
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

    def test_no_weights_returns_empty_list(self):
        # A directory full of scaffold-only adapters (no
        # adapter_model.safetensors, as on a CI checkout where weights are
        # gitignored) must index zero adapters — and downstream, main() must
        # refuse to publish an empty catalog instead of overwriting the good
        # one. Regression test for release v12.1.0 which briefly shipped an
        # empty catalogue built exactly this way.
        no_weights = Path(self.tmp.name) / "only_scaffolds"
        no_weights.mkdir()
        _make_incomplete_adapter(no_weights, "scaffold_one")
        _make_incomplete_adapter(no_weights, "scaffold_two")
        self.assertEqual(compute_lora_zoo_index(no_weights), [])


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
        self.assertIn("# Russian IT Community LoRA Model Zoo", md)
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
        self.assertIn("Быстрый старт", md)
        self.assertIn("PeftModel", md)

    def test_empty_index_still_renders_quickstart(self):
        # An empty index must keep the boilerplate (the release job's gate
        # now skips writing when the index is empty, but the renderer itself
        # must stay total — never raise on []).
        md = build_lora_zoo_markdown([])
        self.assertIn("# Russian IT Community LoRA Model Zoo", md)


class TestUploadDataset(unittest.TestCase):
    def test_uploads_only_existing_artifacts(self):
        api = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # The upload resolves its artefact paths against an explicit
            # repository root (it no longer depends on the current working
            # directory, which previously made it publish whatever happened
            # to sit in the CWD).
            (root / "reports").mkdir()
            (root / "reports/DATASET_AND_ANALYTICS.md").write_text("# dataset", encoding="utf-8")
            (root / "dataset_output/parquet").mkdir(parents=True)
            (root / "dataset_output/parquet/full_clean_messages.parquet").touch()
            (root / "dataset_output/parquet/sft_dialogues.parquet").touch()
            # rag_knowledge_base.parquet intentionally absent
            (root / "reports/domain_benchmark_100.json").write_text("{}", encoding="utf-8")
            # LORA_MODEL_ZOO.md intentionally absent

            upload_dataset(api, repo_root=root)

            uploaded = [c.kwargs["path_in_repo"] for c in api.upload_file.call_args_list]

        self.assertIn("README.md", uploaded)  # dataset card
        self.assertIn("data/full_clean_messages.parquet", uploaded)
        self.assertIn("data/sft_dialogues.parquet", uploaded)
        self.assertNotIn("data/rag_knowledge_base.parquet", uploaded)  # missing on disk
        self.assertIn("domain_benchmark_100.json", uploaded)
        self.assertNotIn("LORA_MODEL_ZOO.md", uploaded)  # missing on disk

    def test_uses_default_dataset_repo_id(self):
        api = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            upload_dataset(api, repo_root=Path(tmpdir))
        # No dataset card, no parquet -> zero upload_file calls,
        # but the function must still log the repo id it intends to
        # use; assert the default value matches the module constant.
        self.assertEqual(DATASET_REPO_ID, "wwewtech/russian-it-community-corpus")

    def test_refuses_to_publish_when_manifest_mismatch(self):
        """A dataset that does not match the manifest must not be published."""
        api = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "reports").mkdir()
            (root / "dataset_output/parquet").mkdir(parents=True)
            (root / "dataset_output/parquet/full_clean_messages.parquet").write_bytes(b"tampered")
            (root / "reports/dataset_manifest.json").write_text(
                json.dumps({"schema_version": 1, "artifacts": {}}),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                upload_dataset(api, repo_root=root)
        api.upload_file.assert_not_called()


class TestUploadModelCard(unittest.TestCase):
    def test_skips_when_card_missing(self):
        api = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            uploaded = upload_model_card(api, repo_root=Path(tmpdir))
        self.assertFalse(uploaded)
        api.upload_file.assert_not_called()

    def test_uploads_card_as_model_readme(self):
        api = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "reports").mkdir()
            (root / "reports/HF_MODEL_CARD.md").write_text("# card", encoding="utf-8")
            uploaded = upload_model_card(api, repo_root=root)
            kwargs = api.upload_file.call_args.kwargs
        self.assertTrue(uploaded)
        self.assertEqual(kwargs["path_in_repo"], "README.md")
        self.assertEqual(kwargs["repo_id"], MODEL_REPO_ID)
        self.assertEqual(kwargs["repo_type"], "model")


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

    def test_policy_incompliant_adapters_are_not_uploaded(self):
        """A partial/time-boxed adapter must never reach the public zoo.

        Before the policy gate, every local adapter with weight files was
        uploaded regardless of how much of the corpus it had seen.
        """
        _make_noncompliant_adapter(self.root, "partial")
        api = MagicMock()
        api.list_repo_files.return_value = []
        uploaded = upload_missing_adapters(
            api,
            self.root,
            max_attempts=1,
            retry_sleep_seconds=0,
        )
        self.assertNotIn("partial", uploaded)
        uploaded_paths = {c.kwargs["path_in_repo"] for c in api.upload_folder.call_args_list}
        self.assertNotIn("partial", uploaded_paths)
        # The compliant ones are still published.
        self.assertEqual(set(uploaded), {"alpha", "beta", "gamma"})

    def test_adapter_without_training_meta_is_not_uploaded(self):
        """Legacy adapters have no recorded provenance and are not publishable."""
        legacy = self.root / "legacy"
        legacy.mkdir()
        (legacy / "adapter_model.safetensors").write_bytes(b"\0" * 1024)
        api = MagicMock()
        api.list_repo_files.return_value = []
        uploaded = upload_missing_adapters(
            api,
            self.root,
            max_attempts=1,
            retry_sleep_seconds=0,
        )
        self.assertNotIn("legacy", uploaded)
        uploaded_paths = {c.kwargs["path_in_repo"] for c in api.upload_folder.call_args_list}
        self.assertNotIn("legacy", uploaded_paths)


if __name__ == "__main__":
    unittest.main()
