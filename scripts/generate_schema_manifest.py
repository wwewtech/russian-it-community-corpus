#!/usr/bin/env python3
"""Generate (or verify) the published dataset contract manifest.

The manifest records the version and fingerprint of the data contract:
the column names, their order and their types of every exported
artefact. It is derived from ``ricc/schema_version.py``, never from a
hand-maintained copy.

Run::

    python scripts/generate_schema_manifest.py           # rewrite
    python scripts/generate_schema_manifest.py --check   # CI: fail on drift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ricc.schema_version import (  # noqa: E402
    contract_manifest,
    schema_fingerprint,
    write_contract_manifest,
)

MANIFEST_PATH = REPO_ROOT / "reports" / "dataset_schema.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the dataset contract manifest")
    parser.add_argument(
        "--check",
        action="store_true",
        help="CI mode: fail if the manifest is out of sync with the contract",
    )
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH, help="Manifest path")
    args = parser.parse_args()

    if args.check:
        if not args.output.exists():
            print(f"{args.output} does not exist — run the generator", file=sys.stderr)
            return 1
        current = json.loads(args.output.read_text(encoding="utf-8"))
        expected = contract_manifest()
        if current.get("schema_fingerprint") != expected["schema_fingerprint"]:
            print("The dataset contract manifest is OUT OF SYNC with the code.", file=sys.stderr)
            print(
                f"  manifest fingerprint: {current.get('schema_fingerprint', 'n/a')[:16]}\n"
                f"  current  fingerprint: {expected['schema_fingerprint'][:16]}",
                file=sys.stderr,
            )
            print("  fix: python scripts/generate_schema_manifest.py", file=sys.stderr)
            return 1
        print(f"dataset contract manifest is in sync (fingerprint {schema_fingerprint()[:16]})")
        return 0

    write_contract_manifest(args.output)
    print(f"wrote {args.output} (fingerprint {schema_fingerprint()[:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
