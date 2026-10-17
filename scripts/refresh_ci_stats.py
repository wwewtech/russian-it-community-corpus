"""refresh_ci_stats.py — regenerate reports/pipeline_execution_stats.json in CI.

Replaces the former inline ``python -c`` step in ``.github/workflows/ci.yml``
which hardcoded ``validation_passed: True``, ``threads_count: -1`` and
``_source.synthetic: False`` regardless of the artifacts actually present.
This script derives every field from what is on disk:

* row counts from ``dataset_output/parquet/*.parquet`` metadata;
* ``threads_count`` from distinct non-null ``thread_id`` values (never the
  ``-1`` sentinel);
* ``dpo_pairs_count`` from valid JSON lines of
  ``dataset_output/jsonl/dpo_preference_pairs.jsonl``;
* ``validation_passed`` from a real ``DatasetValidator.validate_all()`` run;
* ``_source.synthetic`` from the ``reports/synthetic_fixture.json`` marker
  written by ``scripts/build_synthetic_parquet.py``.

Usage::

    python scripts/refresh_ci_stats.py [--root .] [--out reports/pipeline_execution_stats.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pyarrow.parquet as pq  # noqa: E402

from src.validation.validator import DatasetValidator  # noqa: E402

_GENERATOR = "scripts/refresh_ci_stats.py"


def _count_distinct_threads(parquet_path: Path) -> int:
    """Distinct non-null thread_id values (NaN-safe), never a sentinel."""
    column = pq.read_table(parquet_path, columns=["thread_id"]).column("thread_id").to_pylist()
    values = {v for v in column if v is not None and not (isinstance(v, float) and math.isnan(v))}
    return len(values)


def _count_valid_jsonl_lines(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0
    valid = 0
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            valid += 1
    return valid


def collect_stats(root: Path) -> dict[str, Any]:
    """Build the stats dict from the artifacts under ``root`` (fail-closed)."""
    parquet_dir = root / "dataset_output" / "parquet"
    jsonl_dir = root / "dataset_output" / "jsonl"
    reports_dir = root / "reports"

    rows: dict[str, int] = {}
    for name in ("full_clean_messages", "sft_dialogues", "rag_knowledge_base"):
        path = parquet_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing artifact: {path}")
        rows[name] = int(pq.read_metadata(path).num_rows)

    threads = _count_distinct_threads(parquet_dir / "full_clean_messages.parquet")
    dpo = _count_valid_jsonl_lines(jsonl_dir / "dpo_preference_pairs.jsonl")

    validation = DatasetValidator(root / "dataset_output").validate_all()

    # Presence of the marker (written by build_synthetic_parquet.py) is enough:
    # an unreadable marker must not turn a synthetic fixture into a "real" one.
    synthetic = (reports_dir / "synthetic_fixture.json").exists()

    return {
        "execution_time_seconds": 0.0,
        "raw_messages_count": rows["full_clean_messages"],
        "cleaned_messages_count": rows["full_clean_messages"],
        "threads_count": threads,
        "sft_dialogues_count": rows["sft_dialogues"],
        "rag_chunks_count": rows["rag_knowledge_base"],
        "dpo_pairs_count": dpo,
        "pii_stats": {},
        "validation_passed": bool(validation.get("overall_passed", False)),
        "_note": (
            "Auto-refreshed in CI from the active artifacts. "
            "validation_passed = real DatasetValidator run; "
            "threads_count = distinct non-null thread_id values; "
            "dpo_pairs_count = valid JSONL lines. "
            "_source.synthetic mirrors reports/synthetic_fixture.json."
        ),
        "_source": {"synthetic": synthetic, "generator": _GENERATOR},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh pipeline_execution_stats.json from active artifacts")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root (default: script repo root)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: <root>/reports/pipeline_execution_stats.json)",
    )
    args = parser.parse_args(argv)

    out = args.out or (args.root / "reports" / "pipeline_execution_stats.json")
    stats = collect_stats(args.root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"stats refreshed: {out}")
    print(
        f"  synthetic={stats['_source']['synthetic']} "
        f"validation_passed={stats['validation_passed']} "
        f"threads={stats['threads_count']} dpo={stats['dpo_pairs_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
