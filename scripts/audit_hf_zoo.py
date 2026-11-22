#!/usr/bin/env python3
"""Read-only audit of the public HF zoo against training provenance.

The presence of a safetensors file is NOT evidence that an adapter is a final
full-corpus model. This command reads (without downloading model weights) the
immutable remote metadata and classifies every legacy entry as such. The
result is a curated source for the model card, not an implicit deletion of
existing Hub files. ``--strict`` fails for incomplete compliant adapters or
digest mismatch; legacy/partial adapters are reported, not a CI failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.lora_provenance import fetch_hub_zoo_records  # noqa: E402

MODEL_REPO_ID = "wwewtech/russian-it-community-lora"


def audit(model_repo: str = MODEL_REPO_ID, token: str | None = None) -> dict[str, Any]:
    revision, entries = fetch_hub_zoo_records(model_repo, token=token)
    counts = Counter(entry["classification"] for entry in entries)
    result: dict[str, Any] = {
        "schema_version": 1,
        "repo_id": model_repo,
        "revision": revision,
        "adapter_count": len(entries),
        "classification_counts": dict(sorted(counts.items())),
        "adapters": entries,
        "note": (
            "Legacy Hub artifacts are retained and explicitly classified. Only verified "
            "full-corpus full-epoch entries may be called final or be included in the release catalog. "
            "This audit never deletes any existing Hub files."
        ),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=MODEL_REPO_ID)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "hf_hub_audit.json")
    parser.add_argument(
        "--strict", action="store_true", help="Fail if an apparently compliant adapter is incomplete or altered"
    )
    parser.add_argument(
        "--check", action="store_true", help="Compare with saved report, allowing only refreshed revision and timestamp"
    )
    args = parser.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    report = audit(args.repo, token=token)
    if args.check:
        if not args.out.is_file():
            print(f"Missing audit: {args.out}", file=sys.stderr)
            return 1
        saved = json.loads(args.out.read_text(encoding="utf-8"))
        if saved["adapters"] != report["adapters"]:
            print("HF zoo audit drifted; run scripts/audit_hf_zoo.py and review the report", file=sys.stderr)
            return 1
        print(f"Audit matches ({report['adapter_count']} entries)")
        return 0
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text_output = args.out
    text_output.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({key: report[key] for key in ("revision", "adapter_count", "classification_counts")}, indent=2))
    if args.strict:
        failures = [
            entry["id"] for entry in report["adapters"] if entry["classification"] == "full" and not entry["complete"]
        ]
        if failures:
            print(f"Cannot release: incomplete or altered compliant entries: {failures}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
