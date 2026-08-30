"""
Master entrypoint to execute the complete pipeline.
"""

import logging
import sys

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

if __name__ == "__main__":
    print("Starting Russian IT Community Data Engineering & Curation Pipeline...")
    pipeline = MasterDataPipeline()
    pipeline.run_all()
