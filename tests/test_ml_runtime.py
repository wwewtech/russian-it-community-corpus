"""Unit tests for ricc.ml_runtime — determinism + retry helpers."""

from __future__ import annotations

import random
import unittest
from unittest.mock import patch

from ricc.ml_runtime import (
    _is_retryable,
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


if __name__ == "__main__":
    unittest.main()
