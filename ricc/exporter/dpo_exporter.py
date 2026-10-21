"""
DPO (Direct Preference Optimization) preference dataset exporter.

The writer streams: every pair is serialised and written one line at a time,
so the peak memory does not grow with the corpus. The exporter therefore
accepts any ``Iterable`` of pairs (a generator included) and not only a
materialised list.
"""

import json
import logging
from collections.abc import Iterable, Sized
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DPOExporter:
    """
    Exports DPO preference pairs (prompt, chosen, rejected) to JSONL format.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_dpo_pairs(
        self, dpo_pairs: Iterable[dict[str, Any]], file_name: str = "dpo_preference_pairs.jsonl"
    ) -> Path:
        """Export DPO preference pairs to JSONL."""
        out_path = self.output_dir / file_name
        count = str(len(dpo_pairs)) if isinstance(dpo_pairs, Sized) else "all"
        logger.info(f"Exporting {count} DPO preference pairs to {out_path}...")

        with open(out_path, "w", encoding="utf-8") as f:
            for pair in dpo_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        logger.info(f"Saved DPO pairs JSONL at {out_path}")
        return out_path
