"""Run dataset drift monitoring between corpus snapshots.

The reference must be a FROZEN snapshot of an earlier release, not the
current file: comparing the corpus against itself can only ever report
"stable", which is what the default used to do.

Keep a frozen reference next to the current one, e.g.:

    python scripts/run_drift_monitoring.py \
        --reference dataset_output/parquet/full_clean_messages_reference.parquet \
        --current dataset_output/parquet/full_clean_messages.parquet

or point the ``RICC_DRIFT_REFERENCE`` environment variable at it and
let the defaults do the rest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from ricc.monitoring.drift import DatasetDriftMonitor, record_drift_metrics  # noqa: E402
from ricc.settings import settings  # noqa: E402

# The default reference is a frozen snapshot. Comparing the corpus with
# itself reports "stable" by construction and tells the operator
# nothing.
DEFAULT_REFERENCE = Path(
    os.environ.get(
        "RICC_DRIFT_REFERENCE",
        str(settings.PARQUET_OUTPUT_DIR / "full_clean_messages_reference.parquet"),
    )
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset drift monitoring (PSI / JS / vocabulary)")
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Reference snapshot parquet (a FROZEN earlier release, not the current file)",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=settings.PARQUET_OUTPUT_DIR / "full_clean_messages.parquet",
        help="Current snapshot parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.REPORTS_DIR / "drift_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    if not args.reference.exists() or not args.current.exists():
        print(f"Snapshots missing: reference={args.reference.exists()} current={args.current.exists()}")
        print("Run the pipeline first (make run), or point --reference/--current at existing parquet files.")
        print("The reference must be a FROZEN earlier release — comparing the corpus with itself")
        print("always reports 'stable' and detects no drift.")
        sys.exit(0)

    if args.reference.resolve() == args.current.resolve():
        print(
            "WARNING: the reference and the current snapshot are the SAME file. "
            "The drift verdict will be 'stable' by construction and detects nothing. "
            "Point --reference (or RICC_DRIFT_REFERENCE) at a frozen earlier release."
        )

    monitor = DatasetDriftMonitor(pd.read_parquet(args.reference), pd.read_parquet(args.current))
    report = monitor.run()
    record_drift_metrics(report)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Overall drift verdict: {report['overall_verdict']}")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
