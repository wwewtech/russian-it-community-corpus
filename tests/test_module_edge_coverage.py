"""Focused coverage for production helpers that formerly had no tests.

These all use temporary directories and fakes rather than external models or
services. The entry-point functions must behave without an installed GPU.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ricc.bootstrap import setup_runtime_env


class TestBootstrap(unittest.TestCase):
    def test_env_defaults_are_idempotent_and_do_not_overwrite(self):
        root = Path(tempfile.mkdtemp()) / "custom cache"
        logging_config = MagicMock()
        with (
            patch.dict("os.environ", {"HF_HOME": "existing-home"}, clear=True),
            patch("ricc.bootstrap.configure_logging", logging_config),
        ):
            setup_runtime_env(hf_cache_dir=root, log_level=12, pytorch_alloc_conf=True, json_logs=False)
            setup_runtime_env(hf_cache_dir=root, log_level=12, pytorch_alloc_conf=True, json_logs=False)
            import os

            self.assertEqual(os.environ["HF_HOME"], "existing-home")
            self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
            self.assertEqual(os.environ["TOKENIZERS_PARALLELISM"], "false")
        logging_config.assert_called_with(json_output=False, level=12)

    def test_hf_cache_path_is_dir_like_and_logging_gets_flags(self):
        root = Path(tempfile.mkdtemp()) / "cache"
        with patch.dict("os.environ", {}, clear=True), patch("ricc.bootstrap.configure_logging") as config:
            setup_runtime_env(hf_cache_dir=root)
            import os

            self.assertEqual(os.environ["HF_HOME"], str(root))
            config.assert_called_once_with(json_output=True, level=20)

    def test_utf8_stdio_reconfiguration_is_suppressed(self):
        with patch("sys.platform", "linux"), patch("ricc.bootstrap.configure_logging"):
            setup_runtime_env()  # must not try to reconfigure stdio on Linux


class TestCheckpointHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_latest_uses_numeric_order_and_ignores_malformed_dirs(self):
        from ricc.lora.checkpoint import latest_checkpoint

        for name in ("checkpoint-100", "checkpoint-2", "checkpoint-last", "file-4"):
            (self.root / name).mkdir()
        (self.root / "checkpoint-9").write_text("this file is not a directory")
        self.assertEqual(latest_checkpoint(self.root), self.root / "checkpoint-100")
        empty = self.root / "empty"
        empty.mkdir()
        self.assertIsNone(latest_checkpoint(empty))

    def test_clear_keeps_every_checkpoint_directory(self):
        from ricc.lora.checkpoint import clear_non_checkpoint_files

        keep = self.root / "checkpoint-500"
        keep.mkdir()
        stale = self.root / "data"
        stale.mkdir()
        (stale / "old.txt").write_text("old")
        (self.root / "adapter_model.safetensors").write_bytes(b"weights")
        clear_non_checkpoint_files(self.root)
        self.assertTrue(keep.is_dir())
        self.assertFalse(stale.exists())
        self.assertFalse((self.root / "adapter_model.safetensors").exists())

    def test_remove_checkpoints_keeps_other_artifacts(self):
        from ricc.lora.checkpoint import remove_checkpoints

        for n in (1, 2):
            (self.root / f"checkpoint-{n}").mkdir()
        (self.root / "README.md").write_text("keep me")
        remove_checkpoints(self.root)
        self.assertTrue((self.root / "README.md").exists())
        self.assertEqual(list(self.root.glob("checkpoint-*")), [])

    def test_time_limit_requests_stop_AND_save(self):
        from ricc.lora.checkpoint import TimeLimitCallback

        callback = TimeLimitCallback(0.001)
        control = SimpleNamespace(should_training_stop=False, should_save=False)
        args = SimpleNamespace()
        state = SimpleNamespace()
        callback.on_train_begin(args, state, control)
        # Do not depend on the host executing two callbacks before one
        # millisecond elapses: freeze monotonic past the measured deadline.
        with patch("ricc.lora.checkpoint.time.monotonic", return_value=callback.started_at + 1.0):
            callback.on_step_end(args, state, control)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(control.should_save)

        unbounded = TimeLimitCallback(1_000_000.0)
        untouched = SimpleNamespace(should_training_stop=False, should_save=False)
        unbounded.on_step_end(args, state, untouched)
        self.assertFalse(untouched.should_training_stop)
        self.assertFalse(untouched.should_save)


class TestGenerateDemo(unittest.TestCase):
    def test_real_template_and_only_generated_tokens_are_returned(self):
        from ricc.lora.generate_demo import generate_answer

        tokenizer = MagicMock(pad_token=None, eos_token_id=99)
        tokenizer.apply_chat_template.return_value = "<fake template>"
        encoded = MagicMock()
        encoded.input_ids.shape = (1, 5)
        encoded.to.return_value = encoded
        tokenizer.return_value = encoded
        modeled = [[1, 2, 3, 4, 5, 60, 61]]
        tokenizer.decode.side_effect = lambda values, skip_special_tokens: list(values).__len__() and "decoded"

        fake_output = [MagicMock()]
        fake_output[0].__getitem__.side_effect = modeled[0].__getitem__
        model = MagicMock()
        model.generate.return_value = fake_output

        kwargs = MagicMock(input_ids=MagicMock())
        with (
            patch("ricc.lora.generate_demo.AutoTokenizer") as auto_token,
            patch("ricc.lora.generate_demo.AutoModelForCausalLM") as auto_model,
            patch("ricc.lora.generate_demo.PeftModel") as peft,
            patch("ricc.lora.generate_demo.torch") as torch,
        ):
            torch.cuda.is_available.return_value = False
            auto_token.from_pretrained.return_value = tokenizer
            auto_model.from_pretrained.return_value = model
            peft.from_pretrained.return_value = model
            tokenizer.return_value = kwargs
            kwargs.to.return_value = kwargs
            kwargs.input_ids.shape = (1, 5)
            output = MagicMock()
            output.__getitem__.side_effect = lambda _: output
            output.__iter__.side_effect = lambda: iter(output)

            # Use a token-list subclass whose slicing follows tensor slicing.
            class FakeTensor(list):
                pass

            fake_generated = FakeTensor([FakeTensor([9, 8, 7, 6, 5, 40, 41])])
            model.generate.return_value = fake_generated
            answer = generate_answer("q", "Test/Base", "tmp/adapter")

        self.assertIsInstance(answer, str)
        peft.from_pretrained.assert_called_once_with(model, "tmp/adapter")
        model.generate.assert_called_once()
        decode_args = tokenizer.decode.call_args.args[0]
        self.assertEqual(decode_args, [40, 41])

    def test_adapter_errors_are_not_swallowed(self):
        from ricc.lora.generate_demo import generate_answer

        with (
            patch("ricc.lora.generate_demo.AutoTokenizer"),
            patch("ricc.lora.generate_demo.AutoModelForCausalLM"),
            patch("ricc.lora.generate_demo.PeftModel.from_pretrained", side_effect=OSError("corrupt")),
            self.assertRaises(OSError),
        ):
            generate_answer("q", "Test/Base", "not-existing-adapter")


class TestConfigBoundaries(unittest.TestCase):
    def test_minhash_rejects_zero_row_bands(self):
        from ricc.deduplication.minhash_lsh import MinHashLSH

        for bad in (0, 1, 16, 31, -32, True):
            with self.assertRaises(ValueError, msg=f"num_perm={bad!r} must not be accepted"):
                MinHashLSH(num_perm=bad)

    def test_minhash_thirty_two_has_valid_bands(self):
        from ricc.deduplication.minhash_lsh import MinHashLSH

        lsh = MinHashLSH(num_perm=32)
        self.assertEqual(lsh.r, 1)
        self.assertEqual(lsh.num_perm, 32)


if __name__ == "__main__":
    unittest.main()
