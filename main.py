"""
Master entrypoint to execute the complete pipeline.
"""

import sys

from ricc.bootstrap import setup_runtime_env
from ricc.errors import PipelineError
from ricc.pipeline import MasterDataPipeline


def main() -> int:
    """Run the full pipeline; return a process exit code."""
    # Entry-point-only setup (UTF-8 stdio, logging): importing this module must
    # never clobber a caller's existing log handlers.
    setup_runtime_env(json_logs=False)
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
