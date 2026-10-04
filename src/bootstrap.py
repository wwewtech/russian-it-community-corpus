"""
Runtime environment bootstrap for RICC entry-point scripts.

Centralizes the process-level setup that every GPU/eval/training script
previously copy-pasted (7+ times):

* UTF-8 stdio on Windows consoles (cp1251 crashes on Russian text);
* HF_HOME pointed at the repo-local ``.hf_cache``;
* HF/tokenizers env defaults that must be set BEFORE importing
  torch/transformers (TOKENIZERS_PARALLELISM, symlink warnings,
  CUDA allocator config).
* Structured logging via structlog (JSON output, context binding).

Usage — call once at module top, before any heavy third-party import::

    from src.bootstrap import setup_runtime_env

    setup_runtime_env()

    import torch  # noqa: E402  (must stay below setup_runtime_env)
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

from src.observability import configure_logging

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _force_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8 on Windows consoles.

    Windows consoles default to cp1251/cp866; Russian text and box-drawing
    characters would raise UnicodeEncodeError. The try/except keeps this
    safe under pytest capture / redirected streams where reconfigure may
    be unavailable.
    """
    if sys.platform != "win32":
        return
    reconfig_out = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfig_out):
        with contextlib.suppress(Exception):
            reconfig_out(encoding="utf-8", errors="replace")
    reconfig_err = getattr(sys.stderr, "reconfigure", None)
    if callable(reconfig_err):
        with contextlib.suppress(Exception):
            reconfig_err(encoding="utf-8", errors="replace")


def setup_runtime_env(
    *,
    hf_cache_dir: Path | None = None,
    pytorch_alloc_conf: bool = False,
    json_logs: bool = True,
    log_level: int = 20,  # logging.INFO
) -> None:
    """Configure env vars, UTF-8 stdio, and structured logging for the current process.

    Idempotent: ``os.environ.setdefault`` never overrides values the user
    set explicitly, and stdio reconfiguration is safe to repeat.

    Args:
        hf_cache_dir: Override the HF cache location. Defaults to
            ``<repo_root>/.hf_cache``.
        pytorch_alloc_conf: Also set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
            (needed by the LoRA training scripts to reduce VRAM fragmentation).
        json_logs: Output logs as JSON (True) or human-readable console (False).
        log_level: Logging level (default: logging.INFO).
    """
    _force_utf8_stdio()

    cache = hf_cache_dir if hf_cache_dir is not None else _REPO_ROOT / ".hf_cache"
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if pytorch_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Configure structured logging
    configure_logging(json_output=json_logs, level=log_level)
