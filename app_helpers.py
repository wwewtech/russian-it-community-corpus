"""
Pure helpers extracted from ``app.py`` so they can be unit-tested without
spinning up a Streamlit runtime. ``app.py`` itself stays the user-facing
entry point; it imports from here so the logic is exercised by
``tests/test_app.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

DEFAULT_PARQUET_ROWS = 5000
DEFAULT_SFT_ROWS = 2000
DEFAULT_RAG_ROWS = 5000


def _as_optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def format_metric_int(value: int | None, *, sep: str = ",") -> str:
    """Format an optional integer count for the dashboard, or ``'—'`` when absent."""
    if value is None:
        return "—"
    return f"{int(value):,}".replace(",", sep)


def format_tokens_millions(value: int | None) -> str:
    """Format a token count in millions (e.g. ``49_090_000 -> '49.09M'``), or ``'—'``."""
    if value is None:
        return "—"
    return f"{int(value) / 1_000_000:.2f}M"


def load_parquet_sample(
    parquet_dir: Path,
    file_name: str,
    max_rows: int = DEFAULT_PARQUET_ROWS,
) -> pd.DataFrame:
    """Return up to ``max_rows`` of a parquet dataset, or an empty DataFrame on miss."""
    path = Path(parquet_dir) / file_name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df.head(max_rows)


def load_json_file(file_path: Path) -> dict[str, Any]:
    """Return the JSON contents of ``file_path`` or ``{}`` if it is missing."""
    p = Path(file_path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return cast(dict[str, Any], data) if isinstance(data, dict) else {}


def load_markdown_file(file_path: Path, missing_message: str = "Файл отчёта не найден.") -> str:
    """Return the Markdown contents of ``file_path`` or a localised placeholder."""
    p = Path(file_path)
    if not p.exists():
        return missing_message
    with open(p, encoding="utf-8") as f:
        return f.read()


def load_jsonl_sample(file_path: Path, max_lines: int = 200) -> list[dict[str, Any]]:
    """Return up to ``max_lines`` parsed JSON objects from a JSONL file."""
    p = Path(file_path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if len(out) >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    out.append(cast(dict[str, Any], parsed))
            except json.JSONDecodeError:
                continue
    return out


def load_dashboard_metrics(reports_dir: Path, lora_dir: Path) -> dict[str, Any]:
    """Load dashboard KPI counts from JSON reports and adapter registry without hardcoding.

    Missing files yield ``None`` fields (rendered as ``'—'`` in the UI) so a
    fresh checkout never displays fabricated figures.
    """
    stats = load_json_file(Path(reports_dir) / "pipeline_execution_stats.json")
    analytics = load_json_file(Path(reports_dir) / "analytics_summary.json")
    validation = load_json_file(Path(reports_dir) / "validation_results.json")
    registry = load_json_file(Path(lora_dir) / "registry.json")

    vol_raw = analytics.get("volume_statistics")
    vol: dict[str, Any] = vol_raw if isinstance(vol_raw, dict) else {}

    parquet_block = validation.get("parquet_files")
    parquet_details_raw = parquet_block.get("details") if isinstance(parquet_block, dict) else None
    parquet_details: dict[str, dict[str, Any]] = (
        {k: v for k, v in parquet_details_raw.items() if isinstance(v, dict)}
        if isinstance(parquet_details_raw, dict)
        else {}
    )

    jsonl_block = validation.get("jsonl_files")
    jsonl_details_raw = jsonl_block.get("details") if isinstance(jsonl_block, dict) else None
    jsonl_details: dict[str, dict[str, Any]] = (
        {k: v for k, v in jsonl_details_raw.items() if isinstance(v, dict)}
        if isinstance(jsonl_details_raw, dict)
        else {}
    )

    pii_stats_raw = stats.get("pii_stats")
    pii_stats: dict[str, Any] = pii_stats_raw if isinstance(pii_stats_raw, dict) else {}

    cleaned_messages = (
        _as_optional_int(stats.get("cleaned_messages_count"))
        or _as_optional_int(parquet_details.get("full_clean_messages.parquet", {}).get("rows"))
        or _as_optional_int(vol.get("total_messages"))
    )
    sft_dialogues = _as_optional_int(stats.get("sft_dialogues_count")) or _as_optional_int(
        parquet_details.get("sft_dialogues.parquet", {}).get("rows")
    )
    rag_chunks = _as_optional_int(stats.get("rag_chunks_count")) or _as_optional_int(
        parquet_details.get("rag_knowledge_base.parquet", {}).get("rows")
    )
    dpo_pairs = _as_optional_int(stats.get("dpo_pairs_count")) or _as_optional_int(
        jsonl_details.get("dpo_preference_pairs.jsonl", {}).get("valid_lines")
    )
    unique_authors = _as_optional_int(vol.get("unique_authors"))
    total_tokens = _as_optional_int(vol.get("total_tokens_approx"))
    local_adapters = _as_optional_int(registry.get("adapter_count"))
    community_nodes = _as_optional_int(pii_stats.get("community_names_anonymized")) or _as_optional_int(
        vol.get("total_chats")
    )

    return {
        "cleaned_messages": cleaned_messages,
        "unique_authors": unique_authors,
        "total_tokens": total_tokens,
        "sft_dialogues": sft_dialogues,
        "rag_chunks": rag_chunks,
        "dpo_pairs": dpo_pairs,
        "local_adapters": local_adapters,
        "community_nodes": community_nodes,
        "parquet_details": parquet_details,
        "jsonl_details": jsonl_details,
    }


def derive_pii_verdict(
    validation_results: dict[str, Any],
    audit_certificate: dict[str, Any],
) -> tuple[bool, int, int]:
    """Compute the Zero-PII verdict shown on the dashboard.

    Returns ``(passed, total_leaks, sampled_messages)``. The verdict is
    **passed** only when BOTH sources confirm zero leaks.
    """
    pii_leak = validation_results.get("pii_leakage_audit", {})
    parquet_audit = audit_certificate.get("production_parquet_audit", {})

    fallback_leaks = pii_leak.get("phone_leaks", 0) + pii_leak.get("email_leaks", 0) + pii_leak.get("api_key_leaks", 0)
    total_leaks = parquet_audit.get("total_leaks_found", fallback_leaks)

    cert_passed = (
        audit_certificate.get("verification_status") == "PASSED" and parquet_audit.get("total_leaks_found", 0) == 0
    )
    val_passed = bool(validation_results.get("validation_passed", False)) and total_leaks == 0
    is_passed = cert_passed and val_passed

    sampled = parquet_audit.get(
        "sampled_messages_audited",
        pii_leak.get("sample_lines_checked", 25000),
    )
    return is_passed, int(total_leaks), int(sampled)


def leak_breakdown(
    audit_certificate: dict[str, Any],
    validation_results: dict[str, Any],
) -> dict[str, int]:
    """Return the per-category leak counts used by the dashboard's PII panel."""
    parquet_audit = audit_certificate.get("production_parquet_audit", {})
    pii_leak = validation_results.get("pii_leakage_audit", {})
    breakdown = parquet_audit.get("leak_breakdown", {}) or {}
    return {
        "phones": int(breakdown.get("phones", pii_leak.get("phone_leaks", 0))),
        "emails": int(breakdown.get("emails", pii_leak.get("email_leaks", 0))),
        "api_keys": int(breakdown.get("api_keys", pii_leak.get("api_key_leaks", 0))),
        "crypto_wallets": int(breakdown.get("crypto_wallets", 0)),
    }


def adversarial_summary(audit_certificate: dict[str, Any]) -> tuple[int, int, float]:
    """Return ``(passed, total, success_rate_percentage)`` for the adversarial suite."""
    adv_suite = audit_certificate.get("adversarial_suite")
    if not isinstance(adv_suite, dict) or not adv_suite:
        return 0, 0, 0.0
    passed = int(adv_suite.get("adversarial_tests_passed", 0))
    total = int(adv_suite.get("total_adversarial_tests", 0))
    rate = float(adv_suite.get("success_rate_percentage", 0.0))
    return passed, total, rate


def filter_messages(
    df: pd.DataFrame,
    *,
    search_query: str = "",
    selected_domain: str | None = None,
    is_q_only: bool = False,
) -> pd.DataFrame:
    """Apply the search / domain / question-only filters used on the Messages tab."""
    out = df
    if search_query:
        out = out[out["text_clean"].str.contains(search_query, case=False, na=False, regex=False)]
    if selected_domain and selected_domain != "Все домены":
        out = out[out["domain"] == selected_domain]
    if is_q_only and "is_question" in out.columns:
        out = out[out["is_question"]]
    return out


def filter_sft(
    df: pd.DataFrame,
    *,
    min_quality: float,
    min_turns: int,
) -> pd.DataFrame:
    """Apply the SFT quality/turn filters used on the SFT & DPO tab."""
    return df[(df["quality_score"] >= min_quality) & (df["turn_count"] >= min_turns)]
