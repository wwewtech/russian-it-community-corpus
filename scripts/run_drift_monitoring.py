"""Run dataset drift monitoring between corpus snapshots.

By default the reference and current snapshots both point at the current
production parquet (self-comparison => stable baseline). For real drift
detection, keep a frozen reference snapshot, e.g.:

    python scripts/run_drift_monitoring.py \
        --reference dataset_output/parquet/full_clean_messages_2026Q2.parquet \
        --current dataset_output/parquet/full_clean_messages.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import PARQUET_OUTPUT_DIR, REPORTS_DIR  # noqa: E402
from src.monitoring.drift import DatasetDriftMonitor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset drift monitoring (PSI / JS / vocabulary)")
    parser.add_argument(
        "--reference",
        type=Path,
        default=PARQUET_OUTPUT_DIR / "full_clean_messages.parquet",
        help="Reference snapshot parquet",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=PARQUET_OUTPUT_DIR / "full_clean_messages.parquet",
        help="Current snapshot parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "drift_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    if not args.reference.exists() or not args.current.exists():
        print(f"Snapshots missing: reference={args.reference.exists()} current={args.current.exists()}")
        print("Run the pipeline first (make run) or point --reference/--current at existing parquet files.")
        sys.exit(0)

    monitor = DatasetDriftMonitor(pd.read_parquet(args.reference), pd.read_parquet(args.current))
    report = monitor.run()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Overall drift verdict: {report['overall_verdict']}")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
