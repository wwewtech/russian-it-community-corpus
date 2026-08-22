"""
Master entrypoint to execute the complete pipeline.
"""

import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.pipeline import MasterDataPipeline

if __name__ == "__main__":
    print("Starting Russian IT Community Data Engineering & Curation Pipeline...")
    pipeline = MasterDataPipeline()
    pipeline.run_all()
