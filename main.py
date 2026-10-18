"""
Master entrypoint to execute the complete pipeline.
"""

import logging
import sys

from src.errors import PipelineError
from src.pipeline import MasterDataPipeline

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Centralized logging configuration at the entry point (NOT inside pipeline.py),
# so importing the library never clobbers a caller's existing log handlers.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)


def main() -> int:
    """Run the full pipeline; return a process exit code."""
    print("Starting Russian IT Community Data Engineering & Curation Pipeline...")
    pipeline = MasterDataPipeline()
    try:
        summary = pipeline.run_all()
    except PipelineError as exc:
        print(f"Pipeline failed at stage '{exc.stage}': {exc}", file=sys.stderr)
        return 1
    if not summary.get("validation_passed", False):
        print(
            "Pipeline finished but dataset validation FAILED — see reports/validation_results.json",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
