"""Unit tests for ricc.ml_runtime — determinism + retry helpers."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ricc.errors import AdapterLoadError
from ricc.ml_runtime import (
    _is_retryable,
    attach_lora_adapter,
    generate_chat_completion,
    load_adapter_runtime,
    load_evaluation_models,
    load_tokenizer_and_base_model,
    model_input_device,
    move_inputs_to_model,
    render_chat_prompt,
    seed_choices,
    set_deterministic_seed,
    with_retries,
)


class TestIsRetryable(unittest.TestCase):
    def test_rate_limit_marker(self):
        self.assertTrue(_is_retryable(RuntimeError("429 Too Many Requests: rate limit exceeded")))

    def test_timeout_class_name(self):
        class TimeoutError_(OSError):
            pass

        self.assertTrue(_is_retryable(TimeoutError_("timed out")))

    def test_connection_error_message(self):
        self.assertTrue(_is_retryable(ConnectionError("connection reset by peer")))

    def test_bad_gateway(self):
        self.assertTrue(_is_retryable(RuntimeError("502 Bad Gateway")))

    def test_oom_is_not_retryable(self):
        self.assertFalse(_is_retryable(RuntimeError("CUDA out of memory. Tried to allocate 2GB")))

    def test_generic_logic_error_not_retryable(self):
        self.assertFalse(_is_retryable(ValueError("dataloader shape mismatch")))


class TestWithRetries(unittest.TestCase):
    def test_returns_immediately_on_success(self):
        calls = []

        def ok():
            calls.append(1)
            return "value"

        self.assertEqual(with_retries(ok, base_delay=0.001), "value")
        self.assertEqual(len(calls), 1)

    def test_retries_then_succeeds(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("429 rate limit")
            return "recovered"

        result = with_retries(flaky, base_delay=0.001, what="test-flaky")
        self.assertEqual(result, "recovered")
        self.assertEqual(len(calls), 3)

    def test_gives_up_after_attempts_and_reraises(self):
        calls = []

        def always_fails():
            calls.append(1)
            raise RuntimeError("503 temporarily unavailable")

        with self.assertRaises(RuntimeError):
            with_retries(always_fails, attempts=3, base_delay=0.001)
        self.assertEqual(len(calls), 3)

    def test_non_retryable_propagates_immediately(self):
        calls = []

        def logic_bug():
            calls.append(1)
            raise ValueError("bad arguments")

        with self.assertRaises(ValueError):
            with_retries(logic_bug, attempts=5, base_delay=0.001)
        self.assertEqual(len(calls), 1)  # no retry for logic errors

    def test_exponential_backoff_timing(self):
        delays = []
        calls = []

        def fake_sleep(seconds: float) -> None:
            delays.append(seconds)

        def flaky():
            calls.append(1)
            if len(calls) < 4:
                raise RuntimeError("timeout")
            return "ok"

        with patch("ricc.ml_runtime.time.sleep", side_effect=fake_sleep):
            with_retries(flaky, attempts=4, base_delay=1.0)

        self.assertEqual(delays, [1.0, 2.0, 4.0])  # exponential doubling

    def test_attempts_must_be_positive(self):
        with self.assertRaises(ValueError):
            with_retries(lambda: 1, attempts=0)

    def test_retryable_classes_argument(self):
        """Custom retryable tuple: only listed classes are retried."""
        calls = []

        def raises_custom():
            calls.append(1)
            raise KeyError("transient-ish")

        with self.assertRaises(KeyError):
            with_retries(
                raises_custom,
                attempts=3,
                base_delay=0.001,
                retryable=(RuntimeError,),
            )
        self.assertEqual(len(calls), 1)

    def test_custom_retryable_class_is_retried(self):
        class TransientError(Exception):
            pass

        calls = []

        def flaky_custom():
            calls.append(1)
            if len(calls) < 2:
                raise TransientError("hf hub hiccup")
            return "ok"

        result = with_retries(flaky_custom, attempts=3, base_delay=0.001, retryable=(TransientError,))
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 2)


class TestSetDeterministicSeed(unittest.TestCase):
    def test_returns_state_dict_with_seed(self):
        state = set_deterministic_seed(42)
        self.assertEqual(state["seed"], 42)
        self.assertIn("via", state)

    def test_via_transformers_when_available(self):
        try:
            import transformers  # noqa: F401

            expected = "transformers.set_seed"
        except ImportError:
            expected = "random+numpy+torch fallback"
        state = set_deterministic_seed(7)
        self.assertEqual(state["via"], expected)

    def test_fallback_path_when_transformers_missing(self):
        """Without transformers, stdlib/numpy/torch fallback seeds everything.

        Runs in a subprocess: `patch.dict(sys.modules, {"transformers": None})`
        poisons torch's lazy module resolution for the rest of the process
        (torch.overrides re-registers a docstring'd attribute on the nulled
        module and crashes with "already has a docstring").
        """
        code = (
            "import sys; sys.modules['transformers'] = None\n"
            "from ricc.ml_runtime import set_deterministic_seed\n"
            "state = set_deterministic_seed(11)\n"
            "assert state['via'] == 'random+numpy+torch fallback', state\n"
            "assert state['seed'] == 11\n"
            "print('ok')"
        )
        import subprocess as sp
        import sys as _sys

        result = sp.run([_sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr[-400:]}")
        self.assertIn("ok", result.stdout)

    def test_fallback_tolerates_missing_numpy_and_torch(self):
        """Pure-stdlib environment: no ImportError may escape. (Subprocess —
        same module-poisoning constraint as above.)"""
        code = (
            "import sys\n"
            "sys.modules['transformers'] = None\n"
            "sys.modules['numpy'] = None\n"
            "sys.modules['torch'] = None\n"
            "from ricc.ml_runtime import set_deterministic_seed\n"
            "state = set_deterministic_seed(3)\n"
            "assert state['via'] == 'random+numpy+torch fallback', state\n"
            "print('ok')"
        )
        import subprocess as sp
        import sys as _sys

        result = sp.run([_sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr[-400:]}")
        self.assertIn("ok", result.stdout)

    def test_reproducible_rng_sequence(self):
        """Same seed twice must produce identical random sequences."""
        set_deterministic_seed(123)
        first = [random.random() for _ in range(10)]
        set_deterministic_seed(123)
        second = [random.random() for _ in range(10)]
        self.assertEqual(first, second)


class TestSeedChoices(unittest.TestCase):
    def test_first_value_wins(self):
        self.assertEqual(seed_choices([7, 8, 9]), 7)

    def test_empty_uses_default(self):
        self.assertEqual(seed_choices([], default=42), 42)

    def test_negative_seed_rejected(self):
        with self.assertRaises(ValueError):
            seed_choices([-1])


class TestLoadTokenizerAndBaseModel(unittest.TestCase):
    def test_pad_token_fallback_and_disable_kv_cache(self):
        tok = SimpleNamespace(pad_token=None, eos_token="<|eos|>", bos_token=None)
        model = SimpleNamespace(
            config=SimpleNamespace(use_cache=True), generation_config=SimpleNamespace(use_cache=True)
        )
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tok) as tok_fp,
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as mod_fp,
        ):
            out_tok, out_mod = load_tokenizer_and_base_model(
                "org/model",
                tokenizer_source="org/tok",
                disable_kv_cache=True,
                retry=False,
            )
        self.assertIs(out_tok, tok)
        self.assertIs(out_mod, model)
        self.assertEqual(tok.pad_token, "<|eos|>")
        self.assertFalse(model.config.use_cache)
        self.assertFalse(model.generation_config.use_cache)
        self.assertEqual(tok_fp.call_args.args[0], "org/tok")
        self.assertEqual(mod_fp.call_args.args[0], "org/model")

    def test_retry_path_uses_with_retries(self):
        tok = SimpleNamespace(pad_token="<pad>", eos_token="<eos>", bos_token="<bos>")
        model = SimpleNamespace(config=None, generation_config=None)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("ricc.ml_runtime.with_retries", side_effect=[tok, model]) as mock_retries,
        ):
            out_tok, out_mod = load_tokenizer_and_base_model("org/model", retry=True)
        self.assertIs(out_tok, tok)
        self.assertIs(out_mod, model)
        self.assertEqual(mock_retries.call_count, 2)


class TestLoadAdapterRuntime(unittest.TestCase):
    def test_hub_sources_use_one_immutable_adapter_revision(self):
        from ricc.lora.production import resolve_adapter

        spec = resolve_adapter("smollm2_135m_instruct", source="hub")
        tokenizer = SimpleNamespace(pad_token="<pad>")
        base = SimpleNamespace(config=SimpleNamespace(_name_or_path=spec.base_model, _commit_hash=spec.base_revision))
        with (
            patch("ricc.ml_runtime.load_tokenizer_and_base_model", return_value=(tokenizer, base)) as load,
            patch("peft.PeftModel.from_pretrained", return_value=base) as peft,
        ):
            loaded_tok, loaded_model = load_adapter_runtime(spec)
        self.assertIs(loaded_tok, tokenizer)
        self.assertIs(loaded_model, base)
        kwargs = load.call_args.kwargs
        self.assertEqual(kwargs["tokenizer_source"], spec.adapter_repo)
        self.assertEqual(kwargs["tokenizer_revision"], spec.adapter_revision)
        self.assertEqual(kwargs["tokenizer_subfolder"], spec.slug)
        peft_kwargs = peft.call_args.kwargs
        self.assertEqual(peft_kwargs["revision"], spec.adapter_revision)
        self.assertEqual(peft_kwargs["subfolder"], spec.slug)

    def test_adapter_rejects_a_different_base_revision(self):
        base = SimpleNamespace(config=SimpleNamespace(_name_or_path="org/base", _commit_hash="0" * 40))
        with self.assertRaisesRegex(AdapterLoadError, "expects base revision"):
            attach_lora_adapter(
                base,
                "org/adapters",
                adapter_id="demo",
                expected_base_model="org/base",
                expected_base_revision="1" * 40,
            )

    def test_adapter_rejects_a_different_base_model(self):
        base = SimpleNamespace(config=SimpleNamespace(_name_or_path="wrong/model"))
        with self.assertRaisesRegex(AdapterLoadError, "expects base"):
            attach_lora_adapter(base, "org/adapters", adapter_id="demo", expected_base_model="org/expected")


class TestEvaluationModelLoading(unittest.TestCase):
    def test_unknown_adapter_fails_with_context(self):
        with self.assertRaisesRegex(RuntimeError, "could not be loaded"):
            load_evaluation_models("org/base", "missing-adapter")

    def test_explicit_experimental_pair_records_nonproduction_provenance(self):
        from ricc.lora.production import resolve_adapter

        spec = resolve_adapter("smollm2_135m_instruct", source="hub")
        base = SimpleNamespace(config=SimpleNamespace(_name_or_path="other/base", _commit_hash="other"))
        lora = SimpleNamespace()
        with (
            patch("ricc.ml_runtime.load_tokenizer_and_base_model", return_value=(SimpleNamespace(), base)),
            patch("ricc.ml_runtime.attach_lora_adapter", return_value=lora),
            tempfile.TemporaryDirectory() as directory,
        ):
            root = Path(directory)
            (root / "lora_adapters").mkdir()
            (root / "lora_adapters/registry.json").write_text(
                Path("lora_adapters/registry.json").read_text(encoding="utf-8"), encoding="utf-8"
            )
            _tok, loaded_base, loaded_lora, provenance = load_evaluation_models(
                "other/base", spec.slug, root=Path(directory)
            )
        self.assertIs(loaded_base, base)
        self.assertIs(loaded_lora, lora)
        self.assertEqual(provenance["mode"], "explicit_experimental_pair")
        self.assertFalse(provenance["policy_compliant"])
        self.assertEqual(provenance["adapter_source"], "hub")

    def test_explicit_experimental_pair_uses_rootless_default_path(self):
        from ricc.lora.production import resolve_adapter

        spec = resolve_adapter("smollm2_135m_instruct", source="hub")
        with (
            patch("ricc.ml_runtime.load_tokenizer_and_base_model", return_value=(SimpleNamespace(), SimpleNamespace())),
            patch("ricc.ml_runtime.attach_lora_adapter", return_value=SimpleNamespace()) as attach,
        ):
            _tok, _base, _lora, provenance = load_evaluation_models("other/base", spec.slug)
        self.assertEqual(attach.call_args.args[1], spec.adapter_repo)
        self.assertEqual(attach.call_args.kwargs["revision"], spec.adapter_revision)
        self.assertEqual(attach.call_args.kwargs["subfolder"], spec.subfolder)
        self.assertEqual(provenance["mode"], "explicit_experimental_pair")

    def test_experimental_pair_failure_has_adapter_context(self):
        from ricc.lora.production import resolve_adapter

        spec = resolve_adapter("smollm2_135m_instruct", source="hub")
        with (
            patch("ricc.ml_runtime.load_tokenizer_and_base_model", return_value=(SimpleNamespace(), SimpleNamespace())),
            patch("ricc.ml_runtime.attach_lora_adapter", side_effect=RuntimeError("bad weights")),
            self.assertRaisesRegex(RuntimeError, "could not be loaded"),
        ):
            load_evaluation_models("other/base", spec.slug)

    def test_matching_registry_pair_uses_pinned_sources(self):
        from ricc.lora.production import resolve_adapter

        spec = resolve_adapter("smollm2_135m_instruct", source="hub")
        sentinel = SimpleNamespace()
        with patch("ricc.ml_runtime.load_adapter_runtime", return_value=(sentinel, sentinel)) as load:
            tokenizer, base, lora, provenance = load_evaluation_models(spec.base_model, spec.slug)
        self.assertIs(base, lora)
        self.assertEqual(provenance["mode"], "registry_pinned")
        self.assertEqual(provenance["adapter_revision"], spec.adapter_revision)
        load.assert_called_once_with(spec)


class TestModelDevicePlacement(unittest.TestCase):
    def test_device_fallback_uses_parameters_and_batch_encoding(self):
        import torch

        model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device("cpu"))]))
        self.assertEqual(model_input_device(model).type, "cpu")
        batch = MagicMock()
        batch.to.return_value = "moved"
        self.assertEqual(move_inputs_to_model(batch, model), "moved")

    def test_empty_model_device_falls_back_to_available_device(self):
        model = SimpleNamespace(parameters=lambda: iter(()))
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(model_input_device(model).type, "cpu")

    def test_sharded_model_uses_its_first_accelerator(self):
        model = SimpleNamespace(hf_device_map={"embed": "cpu", "layers": "cuda:1", "head": "cpu"})
        self.assertEqual(model_input_device(model), "cuda:1")

    def test_cpu_sharded_model_does_not_use_visible_cuda(self):
        import torch

        model = SimpleNamespace(hf_device_map={"embed": "cpu", "layers": "cpu"})
        inputs = {"input_ids": torch.tensor([1]), "metadata": "keep"}
        with patch("torch.cuda.is_available", return_value=True):
            moved = move_inputs_to_model(inputs, model)
        self.assertEqual(moved["input_ids"].device.type, "cpu")
        self.assertEqual(moved["metadata"], "keep")


class TestAttachLoraAdapter(unittest.TestCase):
    def test_missing_directory_raises_adapter_load_error(self):
        with self.assertRaises(AdapterLoadError):
            attach_lora_adapter(object(), Path("/nonexistent/ricc-adapter-dir"), adapter_id="demo")

    def test_missing_safetensors_raises_adapter_load_error(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(AdapterLoadError):
            attach_lora_adapter(object(), Path(tmp), adapter_id="demo")

    def test_peft_failure_raises_adapter_load_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "adapter_model.safetensors").write_bytes(b"stub")
            with (
                patch("peft.PeftModel.from_pretrained", side_effect=RuntimeError("shape mismatch")),
                self.assertRaises(AdapterLoadError),
            ):
                attach_lora_adapter(object(), Path(tmp), model_name="org/base", adapter_id="demo")

    def test_success_returns_peft_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "adapter_model.safetensors").write_bytes(b"stub")
            sentinel = object()
            with patch("peft.PeftModel.from_pretrained", return_value=sentinel) as mock_peft:
                result = attach_lora_adapter("base", Path(tmp), adapter_id="demo")
            self.assertIs(result, sentinel)
            mock_peft.assert_called_once_with("base", str(Path(tmp)))


class TestRenderChatPrompt(unittest.TestCase):
    def test_uses_apply_chat_template_when_available(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = "<|im_start|>user\nhi<|im_end|>"
        out = render_chat_prompt(tok, [{"role": "user", "content": "hi"}], add_generation_prompt=True)
        self.assertEqual(out, "<|im_start|>user\nhi<|im_end|>")
        tok.apply_chat_template.assert_called_once()

    def test_fallback_when_template_raises(self):
        tok = MagicMock()
        tok.apply_chat_template.side_effect = ValueError("no chat template")
        out = render_chat_prompt(tok, [{"role": "user", "content": "hi"}], add_generation_prompt=True)
        self.assertEqual(out, "[USER]: hi\n[ASSISTANT]:")


class TestGenerateChatCompletion(unittest.TestCase):
    def test_generates_and_decodes_completion(self):
        import torch

        tok = MagicMock()
        tok.pad_token_id = 0
        tok.apply_chat_template.return_value = "prompt"
        tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        tok.decode.return_value = "  hello world  "

        model = MagicMock()
        model.device = torch.device("cpu")
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        with patch("torch.cuda.is_available", return_value=False):
            text, elapsed, tps = generate_chat_completion(
                model,
                tok,
                [{"role": "user", "content": "hi"}],
                max_new_tokens=16,
                top_p=0.9,
                repetition_penalty=1.1,
                max_input_length=64,
            )
        self.assertEqual(text, "hello world")
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertGreater(tps, 0.0)
        tok.decode.assert_called_once()
        gen_kwargs = model.generate.call_args.kwargs
        self.assertEqual(gen_kwargs["top_p"], 0.9)
        self.assertEqual(gen_kwargs["repetition_penalty"], 1.1)


if __name__ == "__main__":
    unittest.main()
