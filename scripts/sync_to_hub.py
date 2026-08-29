#!/usr/bin/env python3
"""
sync_to_hub.py — safely mirror the local LoRA registry to Hugging Face Hub.

This script is the only sanctioned path for pushing the contents of
``lora_adapters/`` to ``wwewtech/russian-it-community-lora``. It follows
the rules in ``docs/adr/0001-hf-token-handling.md``:

* The HF token is read from the ``HF_TOKEN`` environment variable or
  ``~/.huggingface/token``. It is never accepted as a CLI flag and
  never printed.
* The script refuses to run if no token is found.
* The local ``lora_adapters/registry.json`` is the source of truth;
  this script only mirrors derived metadata to the Hub.

Usage::

    # 1) Authenticate once (writes to ~/.huggingface/token):
    huggingface-cli login

    # 2) Push the registry to the Hub (default: dry-run, prints plan only):
    python scripts/sync_to_hub.py

    # 3) Actually upload:
    python scripts/sync_to_hub.py --apply

The default mode is intentionally a dry-run so an accidental invocation
without ``--apply`` cannot change anything on the Hub.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_JSON = REPO_ROOT / "lora_adapters" / "registry.json"
HF_REPO_ID = "wwewtech/russian-it-community-lora"
HF_REPO_TYPE = "model"


def _resolve_token() -> str | None:
    """Return the HF token from env or the standard cache file, else None.

    Never logs the token value. ``huggingface_hub`` itself does the actual
    cache-file parsing; this helper is a thin wrapper that also checks the
    ``HF_TOKEN`` env var first because that's our contract.
    """
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token.strip() or None
    # Fall back to huggingface_hub's own resolver.
    try:
        from huggingface_hub import HfFolder  # type: ignore[attr-defined]
    except ImportError:
        return None
    try:
        return HfFolder.get_token()
    except Exception:  # noqa: BLE001
        return None


def _build_plan(registry: dict) -> list[str]:
    """Return a list of human-readable lines describing what the sync would do."""
    lines: list[str] = []
    lines.append(f"Target repo:    {HF_REPO_ID}  (type={HF_REPO_TYPE})")
    lines.append(f"Adapter count:  {registry.get('adapter_count', '?')}")
    lines.append(f"Generated at:   {registry.get('generated_at', '?')}")
    lines.append("")
    lines.append("Plan:")
    lines.append("  1. Upload lora_adapters/registry.json to the repo root.")
    lines.append("  2. Append a YAML 'adapters:' block to the model card metadata")
    lines.append("     summarising slug -> base_model for every adapter.")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Actually perform the upload. Default is dry-run.")
    parser.add_argument(
        "--registry", type=Path, default=REGISTRY_JSON, help=f"Path to registry.json (default: {REGISTRY_JSON})."
    )
    args = parser.parse_args(argv)

    if not args.registry.is_file():
        print(f"error: registry not found at {args.registry}", file=sys.stderr)
        print("  Run scripts/generate_lora_registry.py first.", file=sys.stderr)
        return 2

    token = _resolve_token()
    if token is None:
        print("error: HF token not found.", file=sys.stderr)
        print("  Set the HF_TOKEN environment variable, or run:", file=sys.stderr)
        print("      huggingface-cli login", file=sys.stderr)
        print("  See docs/adr/0001-hf-token-handling.md for the policy.", file=sys.stderr)
        return 3

    # Defensive: do not echo the token even when --apply is used.
    if args.apply:
        print("warning: --apply was passed. The token value will not be printed.")
    print("Auth:        OK (token present, value hidden)")
    print("")

    import json

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    for line in _build_plan(registry):
        print(line)

    if not args.apply:
        print("")
        print("Dry-run only. Re-run with --apply to upload.")
        return 0

    # --- real upload ---------------------------------------------------- #
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("error: huggingface_hub is not installed.", file=sys.stderr)
        print("  pip install huggingface_hub", file=sys.stderr)
        return 4

    api = HfApi()  # reads HF_TOKEN from env automatically
    print("")
    print("Uploading registry.json ...")
    api.upload_file(
        path_or_fileobj=str(args.registry),
        path_in_repo="registry.json",
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        commit_message="chore: sync LoRA registry from local",
    )
    print("Done. Verify at https://huggingface.co/" + HF_REPO_ID)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
