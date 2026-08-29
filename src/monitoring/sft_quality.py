"""
SFT Dialogue Quality Monitor for the RICC data platform.

Complements :mod:`src.monitoring.drift` (which measures *dataset* drift) by
measuring *dialogue* quality.  Together they answer two different questions:

* ``drift``  — "did the corpus distribution move?"
* ``sft_quality`` — "did the dialogues stop being good training data?"

The monitor is dependency-free (numpy + pandas only) and emits a
machine-readable JSON verdict consumed by CI and the Prefect flow.  A drop
in any sub-score below the configured floor fails the regression check and
breaks the build.

Sub-metrics
-----------
1. **Empty-response ratio** — fraction of assistant turns whose body is empty
   or whitespace-only.  The most common silent failure mode of dialogue
   extractors.
2. **Median turn length** — median number of characters per assistant turn.
   Sudden drops signal a truncation bug.
3. **Role-balance ratio** — ``|user_turns - assistant_turns| / total_turns``;
   0.0 = perfectly balanced, 1.0 = all turns same role.  Real multi-turn
   dialogues should sit comfortably below 0.30.
4. **Russian-language ratio** — fraction of turns whose body contains
   Cyrillic letters.  A Russian-community corpus should stay near 1.0.
5. **Duplicate-turn ratio** — fraction of turns whose body is byte-for-byte
   identical to the previous turn.  Indicates extraction bugs that copy
   the same message twice.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Floors are intentionally conservative; loosen them in a follow-up PR if
# real data falls below the threshold (it should not).
FLOOR_EMPTY_RESPONSE_RATIO_MAX = 0.01  # < 1% empty assistant turns
FLOOR_MEDIAN_TURN_CHARS_MIN = 80  # at least 80 chars per assistant reply
FLOOR_ROLE_BALANCE_MAX = 0.30  # roles reasonably balanced
FLOOR_RUSSIAN_RATIO_MIN = 0.90  # >= 90% Russian turns
FLOOR_DUPLICATE_TURN_RATIO_MAX = 0.05  # <= 5% duplicate adjacent turns

_CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")
_WS_RE = re.compile(r"\s+")


def _is_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text or ""))


def _iter_turns(row: Any) -> list[dict[str, str]]:
    """Normalize one dialogue row into a list of ``{role, content}`` dicts.

    Accepts the two shapes produced by the SFT exporter: a list of dicts
    directly, or a JSON-stringified list in a ``messages``/``turns`` column.
    """
    if isinstance(row, list):
        return [t for t in row if isinstance(t, dict)]
    if isinstance(row, str):
        try:
            parsed = json.loads(row)
        except (TypeError, ValueError):
            return []
        return [t for t in parsed if isinstance(t, dict)]
    return []


class SFTDialogueQualityMonitor:
    """Score a frame of SFT dialogues and return a verdict dict."""

    def __init__(
        self,
        dialogues: pd.DataFrame,
        *,
        messages_col: str = "messages",
        assistant_role: str = "assistant",
        user_role: str = "user",
    ) -> None:
        if messages_col not in dialogues.columns:
            raise ValueError(f"Column {messages_col!r} not in DataFrame. Available: {list(dialogues.columns)}")
        self.dialogues = dialogues
        self.messages_col = messages_col
        self.assistant_role = assistant_role
        self.user_role = user_role

    # ---- per-dialogue feature extraction -------------------------------- #

    def _features(self) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        for raw in self.dialogues[self.messages_col].tolist():
            turns = _iter_turns(raw)
            if not turns:
                rows.append(
                    {
                        "n_turns": 0,
                        "n_assistant_empty": 0,
                        "n_assistant": 0,
                        "assistant_median_chars": 0.0,
                        "role_balance": 1.0,
                        "russian_ratio": 0.0,
                        "duplicate_ratio": 0.0,
                    }
                )
                continue

            roles = [str(t.get("role", "")).lower() for t in turns]
            contents = [str(t.get("content", "")) for t in turns]

            n_assistant = sum(1 for r in roles if r == self.assistant_role)
            n_user = sum(1 for r in roles if r == self.user_role)
            total = max(len(roles), 1)
            role_balance = abs(n_user - n_assistant) / total

            assistant_contents = [c for r, c in zip(roles, contents, strict=True) if r == self.assistant_role]
            n_empty = sum(1 for c in assistant_contents if not c or not c.strip())
            assistant_lengths = [len(c) for c in assistant_contents if c and c.strip()]
            median_chars = float(np.median(assistant_lengths)) if assistant_lengths else 0.0

            n_ru = sum(1 for c in contents if _is_cyrillic(c))
            russian_ratio = n_ru / total

            # strict=False is intentional: we *want* zip to stop at the shorter
            # sequence because we are computing adjacent pairs (n-1 pairs for n turns).
            n_dup = sum(
                1
                for prev, cur in zip(contents, contents[1:], strict=False)
                if prev.strip() == cur.strip() and cur.strip()
            )
            duplicate_ratio = n_dup / max(len(contents) - 1, 1)

            rows.append(
                {
                    "n_turns": len(turns),
                    "n_assistant_empty": n_empty,
                    "n_assistant": n_assistant,
                    "assistant_median_chars": median_chars,
                    "role_balance": role_balance,
                    "russian_ratio": russian_ratio,
                    "duplicate_ratio": duplicate_ratio,
                }
            )
        return pd.DataFrame(rows)

    # ---- public API ----------------------------------------------------- #

    def run(self) -> dict[str, Any]:
        feats = self._features()
        if feats.empty:
            return {
                "overall_verdict": "no_data",
                "n_dialogues": 0,
                "checks": {},
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }

        n = len(feats)
        n_empty_dialogues = int((feats["n_assistant_empty"] > 0).sum())
        total_assistant_turns = int(feats["n_assistant"].sum())
        empty_response_ratio = float(feats["n_assistant_empty"].sum()) / max(total_assistant_turns, 1)
        median_turn_chars = float(feats["assistant_median_chars"].median())
        role_balance_mean = float(feats["role_balance"].mean())
        russian_ratio_mean = float(feats["russian_ratio"].mean())
        duplicate_ratio_mean = float(feats["duplicate_ratio"].mean())

        checks = {
            "empty_response_ratio": {
                "value": round(empty_response_ratio, 4),
                "floor": FLOOR_EMPTY_RESPONSE_RATIO_MAX,
                "comparator": "max",
                "passed": empty_response_ratio <= FLOOR_EMPTY_RESPONSE_RATIO_MAX,
                "n_dialogues_with_empty_assistant_turn": n_empty_dialogues,
            },
            "median_assistant_turn_chars": {
                "value": round(median_turn_chars, 1),
                "floor": FLOOR_MEDIAN_TURN_CHARS_MIN,
                "comparator": "min",
                "passed": median_turn_chars >= FLOOR_MEDIAN_TURN_CHARS_MIN,
            },
            "role_balance": {
                "value": round(role_balance_mean, 4),
                "floor": FLOOR_ROLE_BALANCE_MAX,
                "comparator": "max",
                "passed": role_balance_mean <= FLOOR_ROLE_BALANCE_MAX,
            },
            "russian_ratio": {
                "value": round(russian_ratio_mean, 4),
                "floor": FLOOR_RUSSIAN_RATIO_MIN,
                "comparator": "min",
                "passed": russian_ratio_mean >= FLOOR_RUSSIAN_RATIO_MIN,
            },
            "duplicate_turn_ratio": {
                "value": round(duplicate_ratio_mean, 4),
                "floor": FLOOR_DUPLICATE_TURN_RATIO_MAX,
                "comparator": "max",
                "passed": duplicate_ratio_mean <= FLOOR_DUPLICATE_TURN_RATIO_MAX,
            },
        }

        overall = "pass" if all(c["passed"] for c in checks.values()) else "fail"
        return {
            "overall_verdict": overall,
            "n_dialogues": n,
            "total_assistant_turns": total_assistant_turns,
            "checks": checks,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def to_json(self, path: Path) -> dict[str, Any]:
        report = self.run()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return report


# --------------------------------------------------------------------------- #
# CLI — used by CI to break the build on a regression.
# --------------------------------------------------------------------------- #


def _load_dialogues(path: Path, messages_col: str = "messages") -> pd.DataFrame:
    """Load a parquet/jsonl file into a DataFrame with a messages column."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=[messages_col])
    if path.suffix in {".jsonl", ".json"}:
        records = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and messages_col in obj:
                    records.append({messages_col: obj[messages_col]})
        return pd.DataFrame(records)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Parquet or JSONL file with a 'messages' column.")
    parser.add_argument(
        "--report", type=Path, default=Path("reports/sft_quality_report.json"), help="Where to write the JSON report."
    )
    parser.add_argument("--messages-col", default="messages", help="Column name that holds the dialogue messages.")
    args = parser.parse_args(argv)

    df = _load_dialogues(args.input, args.messages_col)
    report = SFTDialogueQualityMonitor(df, messages_col=args.messages_col).to_json(args.report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["overall_verdict"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
