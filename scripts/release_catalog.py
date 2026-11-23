"""Artifact layouts and renderers shared by the release and its read-only preflight."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVISION = "47f5d37f9e08d921f23df2af8fbdaa614cc81bf3"


def published_revision(registry_path: Path) -> str:
    """Return the single reviewed adapter-repository pin for this release."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    revisions = {
        str(a["hf_model_revision"])
        for a in registry.get("adapters", [])
        if a.get("policy_compliant") and re.fullmatch(r"[0-9a-f]{40}", str(a.get("hf_model_revision") or ""))
    }
    if len(revisions) != 1:
        raise RuntimeError("Compliant registry must declare exactly one immutable Hub revision")
    return revisions.pop()


def validate_registry(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    compliant = [a for a in registry.get("adapters", []) if a.get("policy_compliant")]
    if not compliant:
        raise RuntimeError("No policy-compliant adapters; refusing release")
    if any(
        a.get("actual_steps") != a.get("required_steps") or a.get("corpus_coverage_pct") != 100.0 for a in compliant
    ):
        raise RuntimeError("Registry contains a compliant flag that contradicts full-epoch provenance")
    for adapter in compliant:
        for field in ("base_model_revision", "hf_model_revision"):
            if not re.fullmatch(r"[0-9a-f]{40}", str(adapter.get(field) or "")):
                raise RuntimeError(f"Registry adapter {adapter['slug']} lacks immutable {field}")
    return registry, compliant


def _literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def quickstart(adapter: dict[str, Any], *, remote_url: str) -> str:
    slug = str(adapter["slug"])
    base_model = str(adapter["base_model"])
    base_revision = str(adapter["base_model_revision"])
    adapter_revision = str(adapter["hf_model_revision"])
    if not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (base_revision, adapter_revision)):
        raise RuntimeError(f"Refusing runnable quickstart without immutable revisions for {slug}")
    if not base_model or base_model == "unknown":
        raise RuntimeError(f"Refusing runnable quickstart without a base model for {slug}")
    return "\n".join(
        [
            "```python",
            "from peft import PeftModel",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            f'subfolder = "{_literal(slug)}"',
            f'model_id = "{_literal(base_model)}"',
            f'base_revision = "{base_revision}"',
            f'adapter_hub = "{_literal(adapter.get("hf_model_repo", "wwewtech/russian-it-community-lora"))}"',
            f'adapter_revision = "{adapter_revision}"',
            "",
            "tokenizer = AutoTokenizer.from_pretrained(adapter_hub, subfolder=subfolder, revision=adapter_revision)",
            'base = AutoModelForCausalLM.from_pretrained(model_id, revision=base_revision, torch_dtype="auto", device_map="auto")',
            "model = PeftModel.from_pretrained(base, adapter_hub, subfolder=subfolder, revision=adapter_revision)",
            'messages = [{"role": "user", "content": "Как настроить Nginx reverse proxy в Docker?"}]',
            'inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors="pt").to(model.device)',
            "outputs = model.generate(**inputs, max_new_tokens=256)",
            "print(tokenizer.decode(outputs[0], skip_special_tokens=True))",
            "```",
        ]
    )


def historical_summary(entries: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("classification", "legacy_no_provenance"))
        counts[status] = counts.get(status, 0) + 1
    return [
        "## Historical HF artifacts (not production releases)",
        "",
        "Legacy folders remain for reproducibility and are NOT deleted by this catalog update. "
        "The presence of `adapter_model.safetensors` alone does not certify a final adapter.",
        "",
        "| Status | Count |",
        "| --- | ---: |",
        f"| Full-corpus, full-epoch final adapters | {counts.get('full', 0)} |",
        f"| Time-boxed pilot adapters (NOT final) | {counts.get('time_boxed', 0)} |",
        f"| Legacy adapters without training provenance (NOT final) | {counts.get('legacy_no_provenance', 0)} |",
        f"| Nested legacy checkpoints under `models/` (NOT final) | {counts.get('legacy_nested_checkpoint', 0)} |",
        f"| Other partial/altered artifacts (NOT final) | {sum(counts.values()) - counts.get('full', 0) - counts.get('time_boxed', 0) - counts.get('legacy_no_provenance', 0) - counts.get('legacy_nested_checkpoint', 0)} |",
    ]


def build_model_card(
    entries: list[dict[str, Any]], *, repo_id: str, revision: str, remote_url: str, registry_path: Path
) -> str:
    """Generate a truthful model card from the immutable Hub provenance report."""
    _, adapters = validate_registry(registry_path)
    final = next(a for a in entries if a.get("final"))
    adapter = next(a for a in adapters if a["slug"] == final["id"])
    if final["base_model"] != adapter["base_model"]:
        raise RuntimeError("Hub and registry base models diverge")
    lines = [
        "---",
        "license: mit",
        "language:",
        "- ru",
        "- en",
        "library_name: peft",
        "tags:",
        "- russian",
        "- lora",
        "- peft",
        "- sft",
        "- text-generation",
        "- russian-nlp",
        "---",
        "",
        "# Russian IT Community LoRA Model Zoo",
        "",
        "## Production release: one verified full-epoch adapter",
        "",
        f"**`{final['id']}`** — base model `{final['base_model']}`; "
        f"{final['actual_steps']} / {final['required_steps']} optimizer steps, "
        f"100 % coverage of {final['train_dialogues']} trainable dialogues, no wall-clock budget.",
        "",
        f"The production entry below is verified at immutable Hub revision `{revision}`. "
        "The SHA-256 digest of its safetensors must match the local registry exactly.",
        "",
        "## Quick start",
        "",
        quickstart(adapter, remote_url=remote_url),
        "",
        "## Training policy",
        "",
        "A real release adapter must complete the full corpus for at least one epoch. "
        "A time budget, partial epoch, capped sample pool or absent provenance disqualifies it.",
        "",
        *historical_summary(entries),
        "",
        f"Machine-readable per-file audit: [`hf_hub_audit.json`]({remote_url}/blob/main/hf_hub_audit.json).",
        "",
        "## Provenance and limitations",
        "",
        f"- Immutable adapter revision: `{adapter['hf_model_revision']}`.",
        f"- Immutable base-model revision: `{adapter['base_model_revision']}`.",
        "- The adapter weights have been re-checked against the registry after the metadata repair.",
        "- Older incomplete adapters remain experimental artifacts and MUST NOT be advertised as final models.",
        "- Academic benchmark results previously advertised for the old zoo have been withdrawn pending re-evaluation.",
        "- Dataset identity is independently pinned by Git+LFS revision, file size and SHA-256. Use the pinned "
        "`data/full_clean_messages.parquet`, `data/sft_dialogues.parquet` and `data/rag_knowledge_base.parquet` "
        "of the RICC dataset release, not a moving snapshot, when reproducing this model.",
    ]
    return "\n".join(lines)


def render_release_catalog(
    adapters: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    *,
    repo_id: str,
    revision: str,
) -> str:
    """Render the only production catalog/quickstart representation."""
    index = [
        {
            "id": adapter["slug"],
            "base_model": adapter["base_model"],
            "base_model_revision": adapter["base_model_revision"],
            "hf_model_revision": adapter["hf_model_revision"],
            "hf_model_repo": repo_id,
            "safetensors_sha256": adapter["safetensors_sha256"],
            "size_mb": round(adapter["safetensors_bytes"] / (1024 * 1024), 2),
            "adapter_dir": adapter["local_path"],
            "status": "SUCCESS",
        }
        for adapter in adapters
    ]
    catalog = "# Russian IT Community LoRA Model Zoo\n\n"
    catalog += f"**Official release catalog: {len(index)} verified full-epoch LoRA adapter(s).**\n\n"
    catalog += "| # | Adapter | Base model | Size (MiB) | SHA-256 |\n| ---: | --- | --- | ---: | --- |\n"
    catalog += "\n".join(
        f"| {number} | `{record['id']}` | `{record['base_model']}` | {record['size_mb']} | `{record['safetensors_sha256']}` |"
        for number, record in enumerate(index, 1)
    )
    catalog += "\n\n## Quickstart: verified full-epoch adapter\n\n"
    catalog += "\n".join(quickstart(adapter, remote_url="https://huggingface.co/" + repo_id) for adapter in adapters)
    catalog += "\n\n" + "\n".join(historical_summary(entries)) + f"\n\nHub audit revision: `{revision}`.\n"
    if "Qwen" in catalog:
        raise RuntimeError("Catalog contains an unrelated hard-coded Qwen quickstart")
    return catalog


def write_release_catalog(root: Path, audit_report: dict[str, Any], registry_path: Path) -> list[Path]:
    """Write both catalog views from the same verified registry and Hub audit."""
    repo_id = str(audit_report["repo_id"])
    revision = str(audit_report["revision"])
    entries = audit_report["adapters"]
    _, adapters = validate_registry(registry_path)
    remote_final = {a["id"] for a in entries if a.get("final")}
    if remote_final != {a["slug"] for a in adapters}:
        raise RuntimeError("Remote final adapters and compliant registry diverge")
    card = build_model_card(
        entries,
        repo_id=repo_id,
        revision=revision,
        remote_url="https://huggingface.co/" + repo_id,
        registry_path=registry_path,
    )
    catalog = render_release_catalog(adapters, entries, repo_id=repo_id, revision=revision)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    paths = [reports / "HF_MODEL_CARD.md", reports / "LORA_MODEL_ZOO.md"]
    paths[0].write_text(card + "\n", encoding="utf-8", newline="\n")
    paths[1].write_text(catalog, encoding="utf-8", newline="\n")
    (reports / "hf_hub_audit.json").write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--audit", type=Path, default=REPO_ROOT / "reports" / "hf_hub_audit.json")
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "lora_adapters" / "registry.json")
    args = parser.parse_args(argv)
    report = json.loads(args.audit.read_text(encoding="utf-8"))
    paths = write_release_catalog(args.root, report, args.registry)
    print("Generated " + ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
