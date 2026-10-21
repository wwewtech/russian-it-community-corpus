"""
DPO (Direct Preference Optimization) preference dataset exporter.
"""

import json
import logging
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

    def export_dpo_pairs(self, dpo_pairs: list[dict[str, Any]], file_name: str = "dpo_preference_pairs.jsonl") -> Path:
        """Export DPO preference pairs to JSONL."""
        out_path = self.output_dir / file_name
        logger.info(f"Exporting {len(dpo_pairs)} DPO preference pairs to {out_path}...")

        with open(out_path, "w", encoding="utf-8") as f:
            for pair in dpo_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        logger.info(f"Saved DPO pairs JSONL at {out_path}")
        return out_path
