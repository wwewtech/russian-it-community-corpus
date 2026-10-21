#!/usr/bin/env python3
"""
generate_lora_registry.py — single source of truth for the LoRA model zoo.

Walks ``lora_adapters/`` and produces:

* ``lora_adapters/registry.json``  — machine-readable manifest of every adapter
  (slug, base model, PEFT type, rank, alpha, target modules, file size, sha256).
* ``lora_adapters/SUMMARY.md``     — human-readable summary table that replaces
  the per-adapter README maintenance burden with a single generator.

Run::

    python scripts/generate_lora_registry.py           # full run
    python scripts/generate_lora_registry.py --check   # CI mode: fail on drift
    python scripts/generate_lora_registry.py --quiet   # suppress progress

Why this exists: previously every one of the ~80 adapter directories had its
own card maintained by hand, and the canonical list in
``reports/LORA_MODEL_ZOO.md`` had to be edited separately. Both went out of
sync. This script reads the on-disk ``adapter_config.json`` files directly so
the manifest is always derived from the actual artifacts, never from a copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
LORA_ROOT = REPO_ROOT / "lora_adapters"
REGISTRY_JSON = LORA_ROOT / "registry.json"
SUMMARY_MD = LORA_ROOT / "SUMMARY.md"

HF_HUB_BASE = "https://huggingface.co/wwewtech/russian-it-community-lora/tree/main"
HF_DATASET_BASE = "https://huggingface.co/datasets/wwewtech/russian-it-community-corpus"

# The policy gate lives in the package; make it importable when this script
# is executed directly from the scripts/ directory.
sys.path.insert(0, str(REPO_ROOT))
from ricc.lora.policy_gate import evaluate_compliance  # noqa: E402

# Skip non-adapter directories that happen to live under lora_adapters/.
SKIP_DIR_NAMES = {"registry.json", "SUMMARY.md", "__pycache__", ".git"}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class AdapterEntry:
    """One row in the registry."""

    slug: str
    base_model: str
    peft_type: str
    peft_version: str | None
    task_type: str | None
    r: int | None
    lora_alpha: int | None
    lora_dropout: float | None
    target_modules: list[str] = field(default_factory=list)
    bias: str | None = None
    safetensors_bytes: int | None = None
    safetensors_sha256: str | None = None
    has_tokenizer: bool = False
    has_chat_template: bool = False
    hf_hub_url: str = ""
    local_path: str = ""
    adapter_path: str = ""
    # Training provenance (from training_meta.json). An adapter without a
    # training_meta.json is a legacy/pilot checkpoint: it has no recorded
    # coverage and is NOT publishable as a final adapter under the binding
    # training policy. See ricc/lora/policy_gate.py.
    has_training_meta: bool = False
    actual_steps: int | None = None
    required_steps: int | None = None
    corpus_coverage_pct: float | None = None
    train_dialogues: int | None = None
    time_budget_hours: float | None = None
    policy_compliant: bool = False
    policy_reasons: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #


def _sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _scan_adapter_dir(adapter_dir: Path) -> AdapterEntry | None:
    """Build an ``AdapterEntry`` from one adapter directory, or ``None`` if invalid."""
    if not adapter_dir.is_dir():
        return None
    if adapter_dir.name in SKIP_DIR_NAMES:
        return None

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        # Directory exists but no adapter_config.json → skip with warning.
        print(f"  [skip] {adapter_dir.name}: no adapter_config.json", file=sys.stderr)
        return None

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [warn] {adapter_dir.name}: bad config ({exc})", file=sys.stderr)
        return None

    safetensors = adapter_dir / "adapter_model.safetensors"
    sf_bytes = safetensors.stat().st_size if safetensors.is_file() else None
    sf_sha = _sha256_of_file(safetensors) if safetensors.is_file() else None

    slug = adapter_dir.name

    # Training provenance: read it so the registry distinguishes a
    # policy-compliant adapter from a partial or legacy one.
    meta_path = adapter_dir / "training_meta.json"
    has_meta = meta_path.is_file()
    actual_steps: int | None = None
    required_steps: int | None = None
    coverage: float | None = None
    train_dialogues: int | None = None
    time_budget: float | None = None
    policy_compliant = False
    policy_reasons: list[str] = []
    if has_meta:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        actual_steps = meta.get("actual_steps")
        required_steps = meta.get("required_steps_for_full_epoch")
        coverage = meta.get("corpus_coverage_pct")
        train_dialogues = meta.get("train_dialogues")
        time_budget = meta.get("time_budget_hours")
        compliance = evaluate_compliance(adapter_dir)
        policy_compliant = compliance.compliant
        policy_reasons = list(compliance.reasons)
        if required_steps is None:
            required_steps = compliance.required_steps

    return AdapterEntry(
        slug=slug,
        base_model=cfg.get("base_model_name_or_path") or "unknown",
        peft_type=cfg.get("peft_type", "LORA"),
        peft_version=cfg.get("peft_version"),
        task_type=cfg.get("task_type"),
        r=cfg.get("r"),
        lora_alpha=cfg.get("lora_alpha"),
        lora_dropout=cfg.get("lora_dropout"),
        target_modules=list(cfg.get("target_modules") or []),
        bias=cfg.get("bias"),
        adapter_path=f"lora_adapters/{slug}/",
        safetensors_bytes=sf_bytes,
        safetensors_sha256=sf_sha,
        has_tokenizer=(adapter_dir / "tokenizer_config.json").is_file(),
        has_chat_template=(adapter_dir / "chat_template.jinja").is_file(),
        hf_hub_url=f"{HF_HUB_BASE}/{slug}",
        local_path=f"lora_adapters/{slug}/",
        has_training_meta=has_meta,
        actual_steps=actual_steps,
        required_steps=required_steps,
        corpus_coverage_pct=coverage,
        train_dialogues=train_dialogues,
        time_budget_hours=time_budget,
        policy_compliant=policy_compliant,
        policy_reasons=policy_reasons,
    )


def build_registry(lora_root: Path = LORA_ROOT) -> list[AdapterEntry]:
    """Walk every subdirectory of ``lora_root`` and collect adapter entries."""
    if not lora_root.is_dir():
        raise FileNotFoundError(f"lora_root not found: {lora_root}")
    entries: list[AdapterEntry] = []
    for child in sorted(p for p in lora_root.iterdir() if p.is_dir()):
        entry = _scan_adapter_dir(child)
        if entry is not None:
            entries.append(entry)
    return entries


def write_registry(entries: list[AdapterEntry], out: Path = REGISTRY_JSON) -> None:
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "hf_dataset_url": HF_DATASET_BASE,
        "hf_hub_base": HF_HUB_BASE,
        "adapter_count": len(entries),
        "adapters": [asdict(e) for e in entries],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_summary(entries: list[AdapterEntry], out: Path = SUMMARY_MD) -> None:
    """Render a compact Markdown table so humans can skim the zoo without opening 80 files."""
    lines: list[str] = []
    lines.append("# LoRA Adapter Zoo — Auto-Generated Summary")
    lines.append("")
    lines.append(
        "> Generated by `scripts/generate_lora_registry.py` from "
        "`lora_adapters/**/adapter_config.json`. Do **not** hand-edit this file — "
        "re-run the script after adding or modifying an adapter."
    )
    lines.append("")
    lines.append(f"**Total adapters:** {len(entries)}")
    compliant = sum(1 for e in entries if e.policy_compliant)
    with_meta = sum(1 for e in entries if e.has_training_meta)
    lines.append("")
    lines.append(
        f"**Policy-compliant (full epoch over the corpus):** {compliant} / {len(entries)} · "
        f"**with recorded training provenance:** {with_meta} / {len(entries)}"
    )
    lines.append("")
    lines.append(
        "> Adapters marked `partial` or `no-provenance` are NOT final adapters under the "
        "binding training policy (AGENTS.md) and are not published to the Hub by the "
        "release pipeline. See `ricc/lora/policy_gate.py`."
    )
    lines.append("")
    lines.append("| # | Slug | Base model | r | α | Modules | Size | Coverage | Policy | HF Hub |")
    lines.append("| :---: | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- | :--- |")
    for idx, e in enumerate(entries, 1):
        size_mb = f"{e.safetensors_bytes / (1024 * 1024):.2f} MB" if e.safetensors_bytes else "—"
        modules = ", ".join(e.target_modules) if e.target_modules else "—"
        # Markdown table cell escape for pipe character
        base = e.base_model.replace("|", "\\|")
        slug = e.slug.replace("|", "\\|")
        if e.policy_compliant:
            policy = "✅ compliant"
        elif e.has_training_meta:
            cov = f"{e.corpus_coverage_pct:.2f}%" if e.corpus_coverage_pct is not None else "—"
            policy = f"⚠️ partial ({cov})"
        else:
            policy = "❌ no provenance"
        coverage = f"{e.corpus_coverage_pct:.2f}%" if e.corpus_coverage_pct is not None else "—"
        lines.append(
            f"| {idx} | `{slug}` | `{base}` | {e.r} | {e.lora_alpha} | {modules} | {size_mb} | "
            f"{coverage} | {policy} | [link]({e.hf_hub_url}) |"
        )
    lines.append("")
    lines.append("## How to add a new adapter")
    lines.append("")
    lines.append("1. Train / download adapter weights into `lora_adapters/<new_slug>/`.")
    lines.append("2. Make sure `<new_slug>/adapter_config.json` exists (PEFT writes it on save).")
    lines.append("3. Run `python scripts/generate_lora_registry.py` to refresh `registry.json` and `SUMMARY.md`.")
    lines.append("4. Commit. CI will fail if `registry.json` is out of sync with the on-disk configs.")
    lines.append("")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _check_drift(entries: list[AdapterEntry], registry_path: Path) -> int:
    """Return 0 if on-disk configs match ``registry_path``, 1 otherwise."""
    if not registry_path.is_file():
        print(f"  [drift] {registry_path} missing", file=sys.stderr)
        return 1
    on_disk = {e.slug: asdict(e) for e in entries}
    saved = json.loads(registry_path.read_text(encoding="utf-8"))
    saved_map = {a["slug"]: a for a in saved.get("adapters", [])}
    drift: list[str] = []
    for slug, current in on_disk.items():
        if slug not in saved_map:
            drift.append(f"+ added: {slug}")
            continue
        # Compare only the fields a human could meaningfully change.
        comparable = {
            k: current[k]
            for k in (
                "base_model",
                "r",
                "lora_alpha",
                "lora_dropout",
                "target_modules",
                "bias",
                "task_type",
                "peft_type",
                "safetensors_sha256",
            )
        }
        prev = {k: saved_map[slug].get(k) for k in comparable}
        if prev != comparable:
            drift.append(f"~ changed: {slug}")
    for slug in sorted(set(saved_map) - set(on_disk)):
        drift.append(f"- removed: {slug}")
    if drift:
        print("Drift detected between on-disk configs and registry.json:", file=sys.stderr)
        for line in drift:
            print(f"  {line}", file=sys.stderr)
        print("Re-run: python scripts/generate_lora_registry.py", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="CI mode: exit non-zero if registry.json is out of sync.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    args = parser.parse_args(argv)

    log = (lambda *_a, **_kw: None) if args.quiet else print

    log(f"Scanning {LORA_ROOT} ...")
    entries = build_registry(LORA_ROOT)
    log(f"Found {len(entries)} adapters.")

    if args.check:
        return _check_drift(entries, REGISTRY_JSON)

    write_registry(entries, REGISTRY_JSON)
    log(f"  wrote {REGISTRY_JSON.relative_to(REPO_ROOT)}")
    write_summary(entries, SUMMARY_MD)
    log(f"  wrote {SUMMARY_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
