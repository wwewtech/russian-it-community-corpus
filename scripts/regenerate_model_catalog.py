"""Compatibility entry point for the single verified release catalog.

Run ``python scripts/audit_hf_zoo.py --strict`` before this command to refresh
remote provenance read-only. Only policy-compliant full-epoch adapters enter
the runnable catalog; historical Hub files remain intact and are audited.
"""

from __future__ import annotations

import json
from pathlib import Path

from release_catalog import write_release_catalog

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    audit_path = ROOT / "reports" / "hf_hub_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("Run scripts/audit_hf_zoo.py --strict before generating the catalog")
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    paths = write_release_catalog(ROOT, report, ROOT / "lora_adapters" / "registry.json")
    print("Generated " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
