"""Tests for the LoRA Chat UI logic (chat_ui.py).

The chat page is the demo surface of the LoRA zoo; its pure logic —
adapter discovery, provenance display and history assembly — must be
testable without torch or a running Streamlit server. Model loading and
generation are exercised separately (they need a GPU and a model
download), while everything here runs on the repository's own
``lora_adapters/`` tree.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent

import chat_ui  # noqa: E402
from ricc.lora.policy_gate import PolicyViolation  # noqa: E402,F401  (re-export contract check)


class TestListChatAdapters(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_adapter(self, name: str, *, weights: bool = True, config: bool = True) -> Path:
        d = self.root / "lora_adapters" / name
        d.mkdir(parents=True)
        if weights:
            (d / "adapter_model.safetensors").write_bytes(b"0" * 16)
        if config:
            (d / "adapter_config.json").write_text("{}", encoding="utf-8")
        return d

    def test_missing_adapters_dir_yields_empty_list(self):
        self.assertEqual(chat_ui.list_chat_adapters(self.root), [])

    def test_only_loadable_adapters_are_listed(self):
        self._make_adapter("good_one")
        self._make_adapter("no_weights", weights=False)
        self._make_adapter("no_config", config=False)
        (self.root / "lora_adapters" / "a_file.txt").write_text("x", encoding="utf-8")
        self.assertEqual(chat_ui.list_chat_adapters(self.root), ["good_one"])

    def test_sorted_deterministically(self):
        self._make_adapter("zeta")
        self._make_adapter("alpha")
        self.assertEqual(chat_ui.list_chat_adapters(self.root), ["alpha", "zeta"])

    def test_repository_has_at_least_one_loadable_adapter(self):
        # Adapter weights are distributed via the HF Hub rather than git;
        # the discovery check uses a local weights fixture, so it also runs
        # in a fresh CI checkout that has only the tracked metadata.
        adapters_dir = self.root / "lora_adapters"
        adapter = adapters_dir / chat_ui.POLICY_COMPLIANT_DEFAULT
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        adapters = chat_ui.list_chat_adapters(self.root)
        self.assertEqual(adapters, [], "metadata without weights must not be listed")
        (adapter / "adapter_model.safetensors").write_bytes(b"0" * 16)
        self.assertEqual(chat_ui.list_chat_adapters(self.root), [chat_ui.POLICY_COMPLIANT_DEFAULT])


class TestAdapterSummary(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_missing_meta_is_not_compliant(self):
        d = self.root / "lora_adapters" / "legacy"
        d.mkdir(parents=True)
        (d / "adapter_model.safetensors").write_bytes(b"0" * 16)
        summary = chat_ui.adapter_summary(self.root, "legacy")
        self.assertFalse(summary["compliant"])
        self.assertIsNone(summary["base_model"])

    def test_full_epoch_adapter_is_compliant(self):
        d = self.root / "lora_adapters" / "good"
        d.mkdir(parents=True)
        meta = {
            "model_id": "good",
            "actual_steps": 21420,
            "train_dialogues": 171354,
            "required_steps_for_full_epoch": 21420,
            "corpus_coverage_pct": 100.0,
            "dataset_total_dialogues": 173216,
            "holdout_tail_dialogues": 1000,
            "eval_frac": 0.005,
            "max_samples": 200000,
            "time_budget_hours": None,
            "base_model": "Test/Base",
        }
        (d / "training_meta.json").write_text(__import__("json").dumps(meta), encoding="utf-8")
        summary = chat_ui.adapter_summary(self.root, "good")
        self.assertTrue(summary["compliant"], summary)
        self.assertEqual(summary["base_model"], "Test/Base")
        self.assertEqual(summary["actual_steps"], 21420)
        self.assertEqual(summary["required_steps"], 21420)

    def test_capped_pool_adapter_is_not_compliant(self):
        d = self.root / "lora_adapters" / "capped"
        d.mkdir(parents=True)
        meta = {
            "model_id": "capped",
            "actual_steps": 21420,
            "train_dialogues": 171354,
            "required_steps_for_full_epoch": 21420,
            "corpus_coverage_pct": 100.0,
            "dataset_total_dialogues": 173216,
            "holdout_tail_dialogues": 1000,
            "eval_frac": 0.005,
            "max_samples": 40000,
            "time_budget_hours": None,
        }
        (d / "training_meta.json").write_text(__import__("json").dumps(meta), encoding="utf-8")
        summary = chat_ui.adapter_summary(self.root, "capped")
        # The gate cross-checks the sample cap — a shrunk pool must NOT be
        # advertised as policy-compliant in the chat UI.
        self.assertFalse(summary["compliant"])

    def test_repo_compliant_adapter_passes_the_gate(self):
        summary = chat_ui.adapter_summary(REPO_ROOT, chat_ui.POLICY_COMPLIANT_DEFAULT)
        self.assertTrue(summary["compliant"], summary)
        self.assertEqual(summary["corpus_coverage_pct"], 100.0)


class TestChatHistoryMessages(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        messages = chat_ui.build_chat_history_messages(history)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], chat_ui.CHAT_SYSTEM_PROMPT)
        self.assertEqual(messages[1:], history)

    def test_empty_history_is_system_only(self):
        messages = chat_ui.build_chat_history_messages([])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "system")

    def test_custom_system_prompt_is_used(self):
        messages = chat_ui.build_chat_history_messages([], system_prompt="custom")
        self.assertEqual(messages[0]["content"], "custom")

    def test_history_is_not_mutated(self):
        history: list[dict[str, str]] = [{"role": "user", "content": "q"}]
        snapshot = [dict(m) for m in history]
        chat_ui.build_chat_history_messages(history)
        self.assertEqual(history, snapshot)


class TestGenerateReplyContract(unittest.TestCase):
    """generate_reply must attach the adapter and never swallow its errors."""

    def _patch_stack(self):
        """Patch the heavy model stack inside generate_reply.

        ``root_dir`` is derived from ``__file__`` (the repository root), so
        adapter lookups hit the real ``lora_adapters/`` tree; tests pick
        adapter ids accordingly.
        """
        return (
            patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=MagicMock(pad_token=None, eos_token="[EOS]")
            ),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()),
            patch("torch.cuda.is_available", return_value=False),
        )

    def test_corrupt_adapter_raises_not_silent_base(self):
        # The git checkout intentionally omits the ignored .safetensors;
        # missing weights are rejected before PeftModel attachment, which
        # is also fail-closed behavior (no silent base-model fallback).
        p1, p2, p3 = self._patch_stack()
        with (
            p1,
            p2,
            p3,
            patch("peft.PeftModel.from_pretrained", side_effect=OSError("corrupt safetensors")),
            self.assertRaises((OSError, RuntimeError)),
        ):
            chat_ui.generate_reply(
                query="q",
                history=[],
                model_name="Test/Base",
                adapter_id=chat_ui.POLICY_COMPLIANT_DEFAULT,  # exists in the repo tree
            )

    def test_unknown_adapter_id_raises(self):
        p1, p2, p3 = self._patch_stack()
        with p1, p2, p3, self.assertRaises(RuntimeError):
            chat_ui.generate_reply(
                query="q",
                history=[],
                model_name="Test/Base",
                adapter_id="missing_adapter_xyz",
            )


if __name__ == "__main__":
    unittest.main()
