"""Validate the downloaded real Parquet corpus and write validation_results.json.

The real-corpus gate downloads only the three published Parquet files; the
JSONL exports are not part of the Hub release, so this report is explicitly
parquet-only. The pinned historical release predates the dataset
schema-metadata contract: ``--allow-legacy-metadata`` accepts a file whose
contract metadata is absent while still enforcing the declared column names,
order and types; metadata that is present but wrong is always rejected.
Remove the flag once the corpus is republished with the current exporter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ricc.validation.validator import DatasetValidator  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "dataset_output",
        help="Dataset root containing parquet/ and jsonl/ (default: dataset_output)",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=REPO_ROOT / "reports",
        help="Directory for validation_results.json (default: reports)",
    )
    parser.add_argument(
        "--allow-legacy-metadata",
        action="store_true",
        help="Accept Parquet files with entirely absent schema metadata",
    )
    args = parser.parse_args(argv)

    validator = DatasetValidator(
        args.root,
        allow_legacy_metadata=args.allow_legacy_metadata,
    )
    parquet = validator.validate_parquet_files()
    result = {
        "scope": "parquet-only",
        "overall_passed": bool(parquet["passed"]),
        "parquet_files": parquet,
        "skipped_checks": ["jsonl_files", "pii_leakage_audit", "sft_turn_conformance"],
    }

    out = args.reports / "validation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validation: {'PASSED' if result['overall_passed'] else 'FAILED'}")
    print(f"Wrote {out}")
    return 0 if result["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
