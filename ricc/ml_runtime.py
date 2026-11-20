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

import contextlib
import logging
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

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
    numpy and torch in one shot), and falls back to explicit stdlib/numpy/torch
    seeding otherwise. Always safe on CPU-only boxes.

    ``transformers.set_seed`` does NOT enable deterministic cuDNN kernels
    (its ``deterministic`` argument defaults to False), so the cuDNN
    determinism is requested explicitly here. Without it, two runs of the
    same script on the same GPU can differ in the last decimals of the
    loss — which is enough to make a published number irreproducible.

    Returns a small dict for embedding into run metadata / reports so the
    seed is recorded next to the published numbers.
    """
    applied: dict[str, Any] = {"seed": seed}

    # cuDNN determinism must be set BEFORE the first convolution, and it
    # has to be requested explicitly: transformers.set_seed does not do it.
    with contextlib.suppress(ImportError, AttributeError):
        import torch

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        applied["cudnn_deterministic"] = True

    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
        applied["via"] = "transformers.set_seed"
        return applied
    except ImportError:
        pass

    random.seed(seed)
    applied["via"] = "random+numpy+torch fallback"
    with contextlib.suppress(ImportError):
        import numpy as np

        np.random.seed(seed)
    with contextlib.suppress(ImportError):
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return applied


def seed_choices(values: Iterable[int], default: int = 42) -> int:
    """Validate a ``--seed`` CLI value (first element wins, like argparse default)."""
    for value in values:
        if value < 0:
            raise ValueError(f"seed must be >= 0, got {value}")
        return value
    return default


def load_tokenizer_and_base_model(
    model_name: str,
    *,
    tokenizer_source: str | None = None,
    disable_kv_cache: bool = False,
    retry: bool = False,
) -> tuple[Any, Any]:
    """Load an HF tokenizer and causal LM base model with consistent defaults.

    Centralises ``pad_token`` normalisation, CUDA dtype selection, optional retry
    backoff, and Phi-3.5 KV-cache compatibility across inference/eval/training.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_src = tokenizer_source or model_name

    def _load_tok() -> Any:
        return cast(Any, AutoTokenizer).from_pretrained(tok_src, use_fast=True, trust_remote_code=True)

    def _load_model() -> Any:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    tokenizer: Any = with_retries(_load_tok, what=f"tokenizer load {tok_src}") if retry else _load_tok()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or "<|endoftext|>"

    model: Any = with_retries(_load_model, what=f"model load {model_name}") if retry else _load_model()
    if disable_kv_cache:
        if getattr(model, "config", None) is not None:
            model.config.use_cache = False
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = False
    return tokenizer, model


def attach_lora_adapter(
    base_model: Any,
    adapter_path: Path | str,
    *,
    model_name: str = "",
    adapter_id: str = "",
    verify_files: bool = True,
) -> Any:
    """Attach a local PEFT LoRA adapter or fail fast with ``AdapterLoadError``."""
    from peft import PeftModel

    from ricc.errors import AdapterLoadError
    from ricc.observability import record_classified_ml_error

    resolved = Path(adapter_path)
    if verify_files and (not resolved.exists() or not (resolved / "adapter_model.safetensors").is_file()):
        raise AdapterLoadError(
            f"LoRA adapter '{adapter_id or resolved.name}' could not be loaded from '{resolved}' "
            "(missing directory or adapter_model.safetensors).",
            adapter_path=str(resolved),
            base_model=model_name,
        )
    try:
        return cast(Any, PeftModel).from_pretrained(base_model, str(resolved))
    except Exception as exc:
        typed = record_classified_ml_error(exc, model_name=model_name, adapter_id=adapter_id or resolved.name)
        raise AdapterLoadError(
            f"LoRA adapter '{adapter_id or resolved.name}' could not be loaded from '{resolved}': {typed}",
            adapter_path=str(resolved),
            base_model=model_name,
        ) from exc


def render_chat_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render chat messages via ``apply_chat_template`` with a role-tag fallback."""
    try:
        return cast(
            str,
            tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            ),
        )
    except Exception:
        lines = [f"[{m.get('role', 'user').upper()}]: {m.get('content', '')}" for m in messages]
        if add_generation_prompt:
            lines.append("[ASSISTANT]:")
        return "\n".join(lines)


def generate_chat_completion(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    do_sample: bool = True,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    max_input_length: int | None = 512,
) -> tuple[str, float, float]:
    """Generate a single completion and return ``(text, elapsed_sec, tokens_per_sec)``."""
    import torch

    prompt_text = render_chat_prompt(tokenizer, messages, add_generation_prompt=True)
    tok_kwargs: dict[str, Any] = {"return_tensors": "pt"}
    if max_input_length is not None:
        tok_kwargs["max_length"] = max_input_length
        tok_kwargs["truncation"] = True
    inputs: dict[str, Any] = tokenizer(prompt_text, **tok_kwargs)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if top_p is not None:
        gen_kwargs["top_p"] = top_p
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - t0

    prompt_len = len(inputs["input_ids"][0])
    gen_tokens = len(output_ids[0]) - prompt_len
    tps = gen_tokens / max(elapsed, 0.001)
    text = cast(str, tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)).strip()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text, elapsed, tps
