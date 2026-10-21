"""Determinism and retry helpers for GPU/ML train & eval scripts.

Two long-standing gaps in the train/eval entry points:

* **No seed management** — training runs were not reproducible: no
  ``torch.manual_seed``, no ``transformers.set_seed``, no ``--seed`` CLI
  flag. For a dataset platform whose benchmark numbers are published,
  that is a trust problem: the same command could produce different
  loss curves and different benchmark scores.
* **No retry on Hugging Face network calls** — 30+ ``from_pretrained`` /
  ``upload_folder`` calls across the zoo scripts fail the whole batch on
  a single 429/timeout. HF hub already retries transient transport errors
  internally, but rate limits and maintenance windows surface as
  exceptions; a bounded backoff loop at the call site is the fix.

Usage::

    from ricc.ml_runtime import set_deterministic_seed, with_retries

    set_deterministic_seed(42)                     # entry-point scripts
    adapter = with_retries(lambda: PeftModel.from_pretrained(...))
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions worth retrying: HF hub raises these classes (or messages) on
# transient failures — rate limits, 5xx, connection resets, timeouts.
_RETRYABLE_MARKERS = (
    "rate limit",
    "429",
    "timed out",
    "timeout",
    "connection reset",
    "connection error",
    "temporarily unavailable",
    "502",
    "503",
    "504",
    "remote disconnected",
    "bad gateway",
)

DEFAULT_ATTEMPTS = 4
DEFAULT_BASE_DELAY = 2.0  # seconds; doubles each attempt (2, 4, 8, 16 ...)


def _is_retryable(exc: BaseException) -> bool:
    """True if the exception looks like a transient HF/network failure."""
    name = type(exc).__name__.lower()
    if "timeout" in name or "connection" in name or "http" in name:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _RETRYABLE_MARKERS)


def with_retries(
    func: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    retryable: tuple[type[BaseException], ...] = (Exception,),
    what: str = "network call",
) -> T:
    """Run ``func`` with bounded exponential backoff on transient failures.

    Args:
        func: zero-argument callable (typically a closure over
            ``from_pretrained`` / ``upload_folder``).
        attempts: total attempts including the first (must be >= 1).
        base_delay: seconds before the first retry; doubles per attempt.
        retryable: exception classes eligible for retry; others propagate
            immediately (an OOM must not be retried into a longer OOM).
            Narrowing this tuple also disables the message-marker check —
            only the listed classes are retried.
        what: human-readable label for log lines.

    The last exception is re-raised if all attempts fail — callers keep
    their existing failure handling (classification, metrics recording).
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    delay = base_delay
    default_scope = retryable == (Exception,)
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except retryable as exc:
            last_exc = exc
            # With the default (Exception,) scope, message/content markers
            # filter transient failures; with an explicit class tuple the
            # caller already decided what is retryable.
            transient = _is_retryable(exc) if default_scope else True
            if attempt == attempts or not transient:
                raise
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                what,
                attempt,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable") from last_exc


def set_deterministic_seed(seed: int = 42) -> dict[str, Any]:
    """Seed every RNG that influences training/eval, and return the applied state.

    Calls ``transformers.set_seed`` when available (it covers python,
    numpy and torch in one shot and enables deterministic cudnn where the
    version supports it), and falls back to explicit stdlib/numpy/torch
    seeding otherwise. Always safe on CPU-only boxes.

    Returns a small dict for embedding into run metadata / reports so the
    seed is recorded next to the published numbers.
    """
    applied: dict[str, Any] = {"seed": seed}
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
        applied["via"] = "transformers.set_seed"
        return applied
    except ImportError:
        pass

    random.seed(seed)
    applied["via"] = "random+numpy+torch fallback"
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return applied


def seed_choices(values: Iterable[int], default: int = 42) -> int:
    """Validate a ``--seed`` CLI value (first element wins, like argparse default)."""
    for value in values:
        if value < 0:
            raise ValueError(f"seed must be >= 0, got {value}")
        return value
    return default
