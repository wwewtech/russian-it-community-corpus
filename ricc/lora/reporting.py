"""Run-report helpers for LoRA training runs."""

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _fmt_metric(result: dict[str, Any], key: str, pattern: str) -> str:
    value = result.get(key)
    return pattern.format(value) if isinstance(value, (int, float)) else "—"


def update_run_report(results: list[dict[str, Any]], report_dir: Path = Path("reports")) -> None:
    """Crash-safe aggregate report for the current zoo run.

    Deliberately writes its own files and never touches the canonical catalog
    (``reports/lora_zoo_index.json`` / ``reports/LORA_MODEL_ZOO.md``), which is
    regenerated from the Hub file tree by ``scripts/regenerate_model_catalog.py``.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "lora_zoo_training_run.json"
    md_path = report_dir / "LORA_ZOO_TRAINING_REPORT.md"

    merged: dict[str, Any] = {}
    if json_path.exists():
        try:
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            merged = dict(loaded.get("models", {}))
        except json.JSONDecodeError:
            logger.warning(f"Could not parse {json_path}; starting a fresh run report")
    for r in results:
        merged[r["id"]] = r

    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {"generated_at_utc": generated_at, "models": merged}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# LoRA Zoo Training Run Report",
        "",
        f"Generated {generated_at} · {len(merged)} adapters in the run log.",
        "",
        "| Adapter | Base model | Status | Train loss | Eval loss | Steps | Samples | Time | tok/s |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for model_id in sorted(merged):
        r = merged[model_id]
        md_lines.append(
            f"| `{model_id}` | `{r.get('model_name', '?')}` | {r.get('status', '?')} "
            f"| {_fmt_metric(r, 'train_loss', '{:.4f}')} | {_fmt_metric(r, 'eval_loss', '{:.4f}')} "
            f"| {r.get('steps', '—')} | {r.get('samples_used', '—')} "
            f"| {_fmt_metric(r, 'training_time_sec', '{:.0f}s')} "
            f"| {_fmt_metric(r, 'tokens_per_sec', '{:.0f}')} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info(f"Run report updated: {len(merged)} adapters -> {json_path}")
