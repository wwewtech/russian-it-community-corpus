#!/usr/bin/env python3
"""
upload_benchmarks_to_hub.py — upload LoRA benchmark reports to Hugging Face Hub.

Mirrors benchmark artefacts from ``reports/`` to ``wwewtech/russian-it-community-lora``
on the Hugging Face Hub, alongside the LoRA adapter registry already synced by
``scripts/sync_to_hub.py``.

This is the only sanctioned path for pushing benchmark results to the Hub.
It follows the rules in ``docs/adr/0001-hf-token-handling.md``:

* The HF token is read from the ``HF_TOKEN`` environment variable or
  ``~/.huggingface/token``. It is never accepted as a CLI flag and never printed.
* The script refuses to run if no token is found.
* Default mode is dry-run — nothing changes on Hub without ``--apply``.

Uploaded artefacts (under ``benchmarks/`` in the repo):

* ``benchmarks/benchmark_<adapter>.json`` — per-adapter held-out perplexity +
  greedy generation snapshots (from ``scripts/run_lora_benchmark.py``).
* ``benchmarks/benchmark_<adapter>.md`` — human-readable markdown version.
* ``benchmarks/academic_scientific_benchmarks_matrix.json`` — official academic
  benchmark matrix (HumanEval / RuMMLU / PPL / ROUGE), with retraction status.
* ``benchmarks/academic_benchmarks_raw_outputs.jsonl`` — raw generation logs
  that back the matrix (recomputable by ``tests/test_academic_benchmark_matrix_consistency.py``).
* ``benchmarks/OFFICIAL_ACADEMIC_SCIENTIFIC_BENCHMARKS.md`` — academic benchmark report.

Usage::

    # 1) Authenticate once:
    huggingface-cli login

    # 2) Dry-run — print plan only:
    python scripts/upload_benchmarks_to_hub.py

    # 3) Actually upload:
    python scripts/upload_benchmarks_to_hub.py --apply
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
HF_REPO_ID = "wwewtech/russian-it-community-lora"
HF_REPO_TYPE = "model"

# Files to upload, mapped from local reports/ to Hub path (repo-relative).
UPLOAD_FILES = [
    (
        REPORTS_DIR / "academic_scientific_benchmarks_matrix.json",
        "benchmarks/academic_scientific_benchmarks_matrix.json",
    ),
    (
        REPORTS_DIR / "academic_benchmarks_raw_outputs.jsonl",
        "benchmarks/academic_benchmarks_raw_outputs.jsonl",
    ),
    (
        REPORTS_DIR / "OFFICIAL_ACADEMIC_SCIENTIFIC_BENCHMARKS.md",
        "benchmarks/OFFICIAL_ACADEMIC_SCIENTIFIC_BENCHMARKS.md",
    ),
]

# Per-adapter benchmark reports (may not exist for every adapter).
BENCHMARK_GLOB = "benchmark_*.json"


def _resolve_token() -> str | None:
    """Return the HF token from env or the standard cache file, else None."""
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token.strip() or None
    try:
        from huggingface_hub import HfFolder  # type: ignore[attr-defined]
    except ImportError:
        return None
    try:
        return HfFolder.get_token()
    except Exception:  # noqa: BLE001
        return None


def _discover_benchmark_reports() -> list[tuple[Path, str]]:
    """Discover all ``reports/benchmark_*.json`` and their markdown siblings."""
    found: list[tuple[Path, str]] = []
    if not REPORTS_DIR.is_dir():
        return found
    for json_path in sorted(REPORTS_DIR.glob(BENCHMARK_GLOB)):
        hub_path = f"benchmarks/{json_path.name}"
        found.append((json_path, hub_path))
        md_path = json_path.with_suffix(".md")
        if md_path.is_file():
            found.append((md_path, f"benchmarks/{md_path.name}"))
    return found


def _build_plan() -> list[str]:
    """Return a list of human-readable lines describing what the sync would do."""
    lines: list[str] = []
    lines.append(f"Target repo:    {HF_REPO_ID}  (type={HF_REPO_TYPE})")
    lines.append(f"Reports dir:    {REPORTS_DIR}")
    lines.append("")
    lines.append("Plan:")

    fixed_count = 0
    for local_path, hub_path in UPLOAD_FILES:
        status = "EXISTS" if local_path.is_file() else "MISSING"
        lines.append(f"  1. [{status}] {local_path} -> {hub_path}")
        fixed_count += 1

    adapter_reports = _discover_benchmark_reports()
    adapter_json = [p for p, _ in adapter_reports if p.suffix == ".json"]
    lines.append(f"  2. [{len(adapter_json)}] per-adapter benchmark JSON + markdown reports")
    for local_path, hub_path in adapter_reports:
        lines.append(f"      {local_path} -> {hub_path}")

    lines.append("")
    lines.append(f"Total files to upload: {fixed_count + len(adapter_reports)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--apply", action="store_true", help="Actually perform the upload. Default is dry-run.")
    args = parser.parse_args(argv)

    token = _resolve_token()
    if token is None:
        print("error: HF token not found.", file=sys.stderr)
        print("  Set the HF_TOKEN environment variable, or run:", file=sys.stderr)
        print("      huggingface-cli login", file=sys.stderr)
        print("  See docs/adr/0001-hf-token-handling.md for the policy.", file=sys.stderr)
        return 3

    if args.apply:
        print("warning: --apply was passed. The token value will not be printed.")
    print("Auth:        OK (token present, value hidden)")
    print("")

    for line in _build_plan():
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
    uploaded = 0
    failed = 0

    def upload_one(local_path: Path, hub_path: str) -> None:
        nonlocal uploaded, failed
        if not local_path.is_file():
            print(f"  SKIP (missing): {local_path}", file=sys.stderr)
            failed += 1
            return
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=hub_path,
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                commit_message=f"chore: upload benchmark artefact {hub_path}",
            )
            print(f"  Uploaded: {hub_path}")
            uploaded += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {hub_path}: {exc}", file=sys.stderr)
            failed += 1

    for local_path, hub_path in UPLOAD_FILES:
        upload_one(local_path, hub_path)

    for local_path, hub_path in _discover_benchmark_reports():
        upload_one(local_path, hub_path)

    print("")
    print(f"Done. Uploaded: {uploaded}, failed: {failed}. Verify at https://huggingface.co/{HF_REPO_ID}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
