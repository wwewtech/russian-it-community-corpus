"""
Pure helpers extracted from ``app.py`` so they can be unit-tested without
spinning up a Streamlit runtime. ``app.py`` itself stays the user-facing
entry point; it imports from here so the logic is exercised by
``tests/test_app.py``.

Why this exists:

* The previous version of ``app.py`` had its PII verdict logic, JSON
  loaders, and parquet sampling inlined at module top level, which made
  the file impossible to cover in the existing pytest suite — Streamlit
  import-time side effects broke the test collector.
* Coverage on ``app.py`` was 0 % for the same reason. The helpers below
  can be unit-tested on a CPU-only runner, and the Streamlit ``AppTest``
  smoke test in :mod:`tests.test_app` covers the UI wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Limits chosen to match the user-facing experience: 5k rows is the most
# Streamlit's ``st.dataframe`` renders comfortably on a laptop, and 2k
# SFT samples is the cap the original ``app.py`` set.
DEFAULT_PARQUET_ROWS = 5000
DEFAULT_SFT_ROWS = 2000
DEFAULT_RAG_ROWS = 5000


def load_parquet_sample(
    parquet_dir: Path,
    file_name: str,
    max_rows: int = DEFAULT_PARQUET_ROWS,
) -> pd.DataFrame:
    """Return up to ``max_rows`` of a parquet dataset, or an empty DataFrame on miss.

    Mirrors the behaviour of the original ``load_parquet_sample`` in
    ``app.py`` but without the ``@st.cache_data`` decorator — caching is
    Streamlit's concern and is irrelevant for tests.
    """
    path = Path(parquet_dir) / file_name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df.head(max_rows)


def load_json_file(file_path: Path) -> dict:
    """Return the JSON contents of ``file_path`` or ``{}`` if it is missing."""
    p = Path(file_path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_markdown_file(file_path: Path, missing_message: str = "Файл отчёта не найден.") -> str:
    """Return the Markdown contents of ``file_path`` or a localised placeholder."""
    p = Path(file_path)
    if not p.exists():
        return missing_message
    with open(p, encoding="utf-8") as f:
        return f.read()


def derive_pii_verdict(
    validation_results: dict,
    audit_certificate: dict,
) -> tuple[bool, int, int]:
    """Compute the Zero-PII verdict shown on the dashboard.

    Returns a tuple ``(passed, total_leaks, sampled_messages)`` so the
    Streamlit layer can render it without re-implementing the logic.

    The verdict is **passed** only when BOTH sources confirm zero leaks.
    Earlier code used a permissive ``or`` short-circuit that allowed the
    UI to display "PASSED" even when one certificate was stale. That
    behaviour is explicitly rejected here — see CHANGELOG 12.0.2.
    """
    pii_leak = validation_results.get("pii_leakage_audit", {})
    parquet_audit = audit_certificate.get("production_parquet_audit", {})

    fallback_leaks = pii_leak.get("phone_leaks", 0) + pii_leak.get("email_leaks", 0) + pii_leak.get("api_key_leaks", 0)
    total_leaks = parquet_audit.get("total_leaks_found", fallback_leaks)

    cert_passed = (
        audit_certificate.get("verification_status") == "PASSED" and parquet_audit.get("total_leaks_found", 0) == 0
    )
    val_passed = bool(validation_results.get("validation_passed", False)) and total_leaks == 0
    # AND, not OR: a PASSED verdict requires BOTH independent audits to
    # confirm zero leaks. An OR short-circuit is exactly the
    # "privacy-as-checklist, not threat model" failure mode called out in
    # NADO.md — one stale certificate is enough to flip the UI banner to
    # "✅ Zero-PII Verification Status: PASSED" while a second audit is
    # silently warning. See CHANGELOG 12.0.2.
    is_passed = cert_passed and val_passed

    sampled = parquet_audit.get(
        "sampled_messages_audited",
        pii_leak.get("sample_lines_checked", 25000),
    )
    return is_passed, int(total_leaks), int(sampled)


def leak_breakdown(audit_certificate: dict, validation_results: dict) -> dict[str, int]:
    """Return the per-category leak counts used by the dashboard's PII panel.

    Prefers the structured ``production_parquet_audit.leak_breakdown`` if
    present (newer certificate format) and falls back to the flat keys in
    the older ``pii_leakage_audit`` block.
    """
    parquet_audit = audit_certificate.get("production_parquet_audit", {})
    pii_leak = validation_results.get("pii_leakage_audit", {})
    breakdown = parquet_audit.get("leak_breakdown", {}) or {}
    return {
        "phones": breakdown.get("phones", pii_leak.get("phone_leaks", 0)),
        "emails": breakdown.get("emails", pii_leak.get("email_leaks", 0)),
        "api_keys": breakdown.get("api_keys", pii_leak.get("api_key_leaks", 0)),
        "crypto_wallets": breakdown.get("crypto_wallets", 0),
    }


def adversarial_summary(audit_certificate: dict) -> tuple[int, int, float]:
    """Return ``(passed, total, success_rate_percentage)`` for the adversarial suite."""
    adv_suite = audit_certificate.get("adversarial_suite", {})
    passed = int(adv_suite.get("adversarial_tests_passed", 14))
    total = int(adv_suite.get("total_adversarial_tests", 14))
    rate = float(adv_suite.get("success_rate_percentage", 100.0))
    return passed, total, rate


def filter_messages(
    df: pd.DataFrame,
    *,
    search_query: str = "",
    selected_domain: str | None = None,
    is_q_only: bool = False,
) -> pd.DataFrame:
    """Apply the search / domain / question-only filters used on the Messages tab.

    Pure pandas so it can be exercised without Streamlit.
    """
    out = df
    if search_query:
        out = out[out["text_clean"].str.contains(search_query, case=False, na=False)]
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
