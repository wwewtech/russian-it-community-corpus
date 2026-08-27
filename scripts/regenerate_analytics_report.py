"""Regenerate reports/DATASET_AND_ANALYTICS.md from reports/analytics_summary.json.

The JSON summary is the single source of truth; the markdown report is a
derived artifact. This script re-renders it WITHOUT re-running the full
analytics engine (which requires the raw parquet dataset).

Usage:
    python scripts/regenerate_analytics_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics.report_generator import ReportGenerator  # noqa: E402


def main() -> None:
    """Re-render the analytics markdown report from the JSON summary."""
    summary_path = Path("reports/analytics_summary.json")
    out_path = Path("reports/DATASET_AND_ANALYTICS.md")
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path} — run the analytics engine first (make analyze).")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    written = ReportGenerator(summary).export_markdown(out_path)
    print(f"Regenerated {written} from {summary_path}")


if __name__ == "__main__":
    main()
