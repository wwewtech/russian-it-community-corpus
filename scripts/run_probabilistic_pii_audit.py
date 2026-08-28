"""Run the stratified probabilistic PII audit on the production corpus.

Usage:
    python scripts/run_probabilistic_pii_audit.py                     # 50k sample
    python scripts/run_probabilistic_pii_audit.py --sample-size 100000
    python scripts/run_probabilistic_pii_audit.py --tolerance 1e-5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PARQUET_OUTPUT_DIR, REPORTS_DIR  # noqa: E402
from src.validation.probabilistic_audit import ProbabilisticPIIAuditor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Probabilistic PII audit (stratified, confidence-bounded)")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=PARQUET_OUTPUT_DIR / "full_clean_messages.parquet",
        help="Path to the production messages parquet",
    )
    parser.add_argument("--sample-size", type=int, default=50_000, help="Number of messages to audit")
    parser.add_argument("--confidence", type=float, default=0.99, help="Confidence level for bounds")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Max acceptable message-level leak rate (upper bound) for PASSED",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "probabilistic_pii_audit.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    auditor = ProbabilisticPIIAuditor(args.parquet)
    report = auditor.run_audit(
        sample_size=args.sample_size,
        confidence=args.confidence,
        max_leak_tolerance=args.tolerance,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Verdict: {report.get('verdict', report.get('status'))}")
    print(f"Report written to {out}")
    if report.get("verdict") == "LEAKS_DETECTED":
        sys.exit(1)


if __name__ == "__main__":
    main()
