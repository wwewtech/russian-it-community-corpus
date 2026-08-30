"""
Unit tests for :mod:`src.inference`.

Only the pure helpers are exercised here. The interactive chat session
itself is a thin I/O loop over ``transformers`` / ``peft`` and is
covered indirectly by the manual smoke tests in ``scripts/``.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.inference import (
    EXIT_COMMANDS,
    build_chat_messages,
    build_prompt,
    format_rag_context,
    is_exit_command,
    main,
    validate_adapter_path,
)


class TestIsExitCommand(unittest.TestCase):
    def test_empty_string_is_exit(self):
        self.assertTrue(is_exit_command(""))
        self.assertTrue(is_exit_command("   "))

    def test_known_keywords_are_exit(self):
        for kw in ("exit", "EXIT", "Quit", "  q  "):
            self.assertTrue(is_exit_command(kw), msg=kw)

    def test_normal_query_is_not_exit(self):
        self.assertFalse(is_exit_command("How do I tune nginx?"))
        self.assertFalse(is_exit_command("quitly"))

    def test_exit_commands_constant_is_frozen(self):
        self.assertIsInstance(EXIT_COMMANDS, frozenset)
        self.assertIn("exit", EXIT_COMMANDS)


class TestBuildChatMessages(unittest.TestCase):
    def test_no_rag_uses_default_system_prompt(self):
        msgs = build_chat_messages("What is a thread pool?")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1], {"role": "user", "content": "What is a thread pool?"})
        self.assertNotIn("КОНТЕКСТ", msgs[0]["content"])

    def test_rag_context_is_embedded_in_system_prompt(self):
        ctx = "[Backend]: Use Nginx upstream block."
        msgs = build_chat_messages("set up reverse proxy", rag_context=ctx)
        self.assertIn(ctx, msgs[0]["content"])
        self.assertIn("КОНТЕКСТ БАЗЫ ЗНАНИЙ", msgs[0]["content"])

    def test_messages_have_exactly_two_roles(self):
        msgs = build_chat_messages("hi", rag_context="ctx")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])


class TestBuildPrompt(unittest.TestCase):
    def test_uses_apply_chat_template_when_available(self):
        fake_apply = MagicMock(return_value="<rendered>")
        out = build_prompt(
            [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
            ],
            fake_apply,
        )
        self.assertEqual(out, "<rendered>")
        fake_apply.assert_called_once()

    def test_falls_back_to_im_start_delimiters(self):
        def apply_template(messages, tokenize, add_generation_prompt):
            raise ValueError("unknown role 'system'")

        out = build_prompt(
            [
                {"role": "system", "content": "sys-text"},
                {"role": "user", "content": "user-text"},
            ],
            apply_template,
        )
        # Must keep the <|im_start|>/<|im_end|> boundaries the
        # comment block in src/inference.py advertises, otherwise
        # Qwen / ChatGLM-family models merge the system block into
        # the user prompt.
        self.assertIn("<|im_start|>system", out)
        self.assertIn("sys-text", out)
        self.assertIn("<|im_start|>user", out)
        self.assertIn("user-text", out)
        self.assertIn("<|im_start|>assistant", out)
        self.assertTrue(out.endswith("\n"))

    def test_fallback_when_system_role_missing(self):
        def apply_template(messages, tokenize, add_generation_prompt):
            raise RuntimeError("nope")

        out = build_prompt([{"role": "user", "content": "only-user"}], apply_template)
        self.assertIn("<|im_start|>system", out)
        self.assertIn("only-user", out)


class TestFormatRagContext(unittest.TestCase):
    def test_empty_returns_empty_string(self):
        self.assertEqual(format_rag_context([]), "")

    def test_concatenates_with_separator(self):
        hits = [
            {"domain": "Backend", "content": "use Nginx"},
            {"domain": "DevOps", "content": "enable systemd"},
        ]
        out = format_rag_context(hits)
        self.assertIn("[Backend]: use Nginx", out)
        self.assertIn("[DevOps]: enable systemd", out)
        self.assertIn("---", out)

    def test_missing_fields_default(self):
        out = format_rag_context([{}])
        self.assertIn("[Tech]: ", out)


class TestValidateAdapterPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_raises_when_directory_missing(self):
        with self.assertRaises(RuntimeError) as ctx:
            validate_adapter_path("ghost_adapter", self.root)
        self.assertIn("not found", str(ctx.exception))
        # The error message must point the user at the generator script,
        # not just "IOError" — the previous behaviour was a silent
        # downgrade, so this assertion prevents a regression.
        self.assertIn("generate_lora_registry.py", str(ctx.exception))

    def test_raises_when_weights_missing(self):
        empty_adapter = self.root / "broken_adapter"
        empty_adapter.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            validate_adapter_path("broken_adapter", self.root)
        self.assertIn("weights missing", str(ctx.exception))

    def test_returns_path_for_valid_adapter(self):
        good = self.root / "good_adapter"
        good.mkdir()
        (good / "adapter_model.safetensors").write_bytes(b"fake-weights")
        out = validate_adapter_path("good_adapter", self.root)
        self.assertEqual(out, good.resolve())


class TestMainCLI(unittest.TestCase):
    """Cover the argparse wrapper in :func:`src.inference.main`."""

    def _run_main(self, argv, fake_chat):
        with (
            patch.object(sys, "argv", ["inference", *argv]),
            patch("src.inference.interactive_chat_session", fake_chat),
        ):
            main()
        return fake_chat

    def test_default_arguments_pass_through(self):
        fake_chat = MagicMock()
        self._run_main([], fake_chat)
        fake_chat.assert_called_once()
        kwargs = fake_chat.call_args.kwargs
        # Defaults: model name, default adapter id, RAG on, 512 max tokens.
        self.assertEqual(kwargs["model_name"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(kwargs["adapter_id"], "heavyweight_qwen2.5_coder_7b")
        self.assertTrue(kwargs["use_rag"])
        self.assertEqual(kwargs["max_tokens"], 512)

    def test_adapter_none_disables_lora(self):
        fake_chat = MagicMock()
        self._run_main(["--adapter", "none"], fake_chat)
        self.assertIsNone(fake_chat.call_args.kwargs["adapter_id"])

    def test_no_rag_flag_disables_rag(self):
        fake_chat = MagicMock()
        self._run_main(["--no-rag"], fake_chat)
        self.assertFalse(fake_chat.call_args.kwargs["use_rag"])

    def test_custom_model_and_max_tokens(self):
        fake_chat = MagicMock()
        self._run_main(
            ["--model", "Qwen/Qwen2.5-7B-Instruct", "--max-tokens", "128"],
            fake_chat,
        )
        kwargs = fake_chat.call_args.kwargs
        self.assertEqual(kwargs["model_name"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(kwargs["max_tokens"], 128)


if __name__ == "__main__":
    unittest.main()
