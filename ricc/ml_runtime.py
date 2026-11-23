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
    revision: str | None = None,
    tokenizer_revision: str | None = None,
    tokenizer_subfolder: str | None = None,
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
        return cast(Any, AutoTokenizer).from_pretrained(
            tok_src,
            revision=tokenizer_revision or revision,
            subfolder=tokenizer_subfolder,
            use_fast=True,
            trust_remote_code=True,
        )

    def _load_model() -> Any:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
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


def load_adapter_runtime(spec: Any, *, retry: bool = False) -> tuple[Any, Any]:
    """Load a base and its exact adapter from a resolved production spec."""
    from ricc.lora.production import local_adapter_path

    hub = spec.source == "hub"
    tokenizer, base = load_tokenizer_and_base_model(
        spec.base_model,
        revision=spec.base_revision,
        tokenizer_source=spec.adapter_repo if hub else spec.base_model,
        tokenizer_revision=spec.adapter_revision if hub else spec.base_revision,
        tokenizer_subfolder=spec.subfolder if hub else None,
        retry=retry,
    )
    adapter_source = spec.adapter_repo if hub else local_adapter_path(spec)
    model = attach_lora_adapter(
        base,
        adapter_source,
        model_name=spec.base_model,
        adapter_id=spec.slug,
        verify_files=not hub,
        revision=spec.adapter_revision if hub else None,
        subfolder=spec.subfolder if hub else None,
        expected_base_model=spec.base_model,
        expected_base_revision=spec.base_revision,
    )
    return tokenizer, model


def model_input_device(model: Any) -> Any:
    """Infer where input tensors belong for single or sharded model layouts."""
    import torch

    if hasattr(model, "hf_device_map"):
        device_map = getattr(model, "hf_device_map", {})
        if isinstance(device_map, dict):
            candidates = [device for device in device_map.values() if isinstance(device, (str, int))]
            non_cpu = next((device for device in candidates if device != "cpu" and device != "disk"), None)
            if non_cpu is not None:
                return non_cpu
            if candidates:
                return "cpu"
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_inputs_to_model(inputs: Any, model: Any) -> Any:
    """Move tensor batches to the actual model device, not merely any visible CUDA."""
    import torch

    if hasattr(inputs, "to"):
        return inputs.to(model_input_device(model))
    return {
        key: value.to(model_input_device(model)) if torch.is_tensor(value) else value for key, value in inputs.items()
    }


def load_evaluation_models(
    model_name: str,
    adapter_id: str,
    *,
    root: Path | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Return ``(tokenizer, base, lora, provenance)`` for evaluation suites.

    The default registry pair is immutable. An explicit legacy pair remains a
    research option with its own base/path, never an accidental Hub fallback.
    """
    from ricc.lora.production import resolve_adapter

    try:
        spec = resolve_adapter(adapter_id, root=root)
    except Exception as exc:
        raise RuntimeError(f"LoRA adapter '{adapter_id}' could not be loaded: {exc}") from exc
    if spec.base_model == model_name:
        tokenizer, lora = load_adapter_runtime(spec)
        provenance = {
            "mode": "registry_pinned",
            "base_model": spec.base_model,
            "base_revision": spec.base_revision,
            "adapter_revision": spec.adapter_revision,
            "adapter_source": spec.source,
            "policy_compliant": spec.policy_compliant,
        }
        return tokenizer, lora, lora, provenance

    tokenizer, base = load_tokenizer_and_base_model(model_name)
    adapter_source: Path | str = (
        spec.adapter_repo
        if spec.source == "hub"
        else root / "lora_adapters" / adapter_id
        if root
        else Path("lora_adapters") / adapter_id
    )
    try:
        lora = attach_lora_adapter(
            base,
            adapter_source,
            model_name=model_name,
            adapter_id=adapter_id,
            revision=spec.adapter_revision if spec.source == "hub" else None,
            subfolder=spec.subfolder if spec.source == "hub" else None,
            expected_base_model=spec.base_model if spec.policy_compliant else None,
        )
    except Exception as exc:
        raise RuntimeError(f"LoRA adapter '{adapter_id}' could not be loaded: {exc}") from exc
    provenance = {
        "mode": "explicit_experimental_pair",
        "base_model": model_name,
        "adapter_id": adapter_id,
        "adapter_source": spec.source,
        "policy_compliant": False,
    }
    return tokenizer, base, lora, provenance


def attach_lora_adapter(
    base_model: Any,
    adapter_path: Path | str,
    *,
    model_name: str = "",
    adapter_id: str = "",
    verify_files: bool = True,
    revision: str | None = None,
    subfolder: str | None = None,
    expected_base_model: str | None = None,
    expected_base_revision: str | None = None,
) -> Any:
    """Attach a local or immutable Hub PEFT adapter and verify base identity."""
    from peft import PeftModel

    from ricc.errors import AdapterLoadError
    from ricc.observability import record_classified_ml_error

    actual_base = getattr(getattr(base_model, "config", None), "_name_or_path", None)
    if expected_base_model and actual_base and actual_base != expected_base_model:
        raise AdapterLoadError(
            f"Adapter '{adapter_id}' expects base '{expected_base_model}', got '{actual_base}'",
            adapter_path=str(adapter_path),
            base_model=actual_base,
        )
    if expected_base_revision:
        actual_revision = getattr(getattr(base_model, "config", None), "_commit_hash", None)
        if actual_revision and actual_revision != expected_base_revision:
            raise AdapterLoadError(
                f"Adapter '{adapter_id}' expects base revision '{expected_base_revision}', got '{actual_revision}'",
                adapter_path=str(adapter_path),
                base_model=expected_base_model or model_name,
            )

    is_remote = revision is not None or subfolder is not None
    resolved = Path(adapter_path)
    if (
        not is_remote
        and verify_files
        and (not resolved.exists() or not (resolved / "adapter_model.safetensors").is_file())
    ):
        raise AdapterLoadError(
            f"LoRA adapter '{adapter_id or resolved.name}' could not be loaded from '{resolved}' "
            "(missing directory or adapter_model.safetensors).",
            adapter_path=str(resolved),
            base_model=model_name,
        )
    try:
        if is_remote:
            return cast(Any, PeftModel).from_pretrained(
                base_model,
                str(adapter_path),
                revision=revision,
                subfolder=subfolder,
            )
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
    inputs = move_inputs_to_model(inputs, model)

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
