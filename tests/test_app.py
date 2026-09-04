"""
Unit tests for the Streamlit ``app_helpers`` module.

The heavy ``app.py`` UI itself is covered by the ``streamlit-smoke``
CI step (``pytest tests/test_app_streamlit_smoke.py``); here we focus
on the pure pandas / dict helpers that drive the dashboard's logic.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app_helpers import (
    adversarial_summary,
    derive_pii_verdict,
    filter_messages,
    filter_sft,
    leak_breakdown,
    load_json_file,
    load_markdown_file,
    load_parquet_sample,
)


def _write_parquet(path: Path, rows: int) -> None:
    df = pd.DataFrame({"col": list(range(rows))})
    df.to_parquet(path, index=False)


class TestLoadParquetSample(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_empty_dataframe_when_missing(self):
        out = load_parquet_sample(self.root, "ghost.parquet")
        self.assertTrue(out.empty)
        self.assertIsInstance(out, pd.DataFrame)

    def test_respects_max_rows(self):
        _write_parquet(self.root / "x.parquet", rows=200)
        out = load_parquet_sample(self.root, "x.parquet", max_rows=10)
        self.assertEqual(len(out), 10)

    def test_returns_all_rows_when_under_limit(self):
        _write_parquet(self.root / "x.parquet", rows=5)
        out = load_parquet_sample(self.root, "x.parquet", max_rows=100)
        self.assertEqual(len(out), 5)


class TestLoadJsonFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_empty_dict_when_missing(self):
        self.assertEqual(load_json_file(self.root / "missing.json"), {})

    def test_round_trips_dict(self):
        path = self.root / "x.json"
        path.write_text(json.dumps({"a": 1, "b": [1, 2]}), encoding="utf-8")
        self.assertEqual(load_json_file(path), {"a": 1, "b": [1, 2]})


class TestLoadMarkdownFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_returns_placeholder(self):
        out = load_markdown_file(self.root / "missing.md")
        self.assertEqual(out, "Файл отчёта не найден.")

    def test_missing_file_with_custom_placeholder(self):
        out = load_markdown_file(self.root / "missing.md", missing_message="Oops")
        self.assertEqual(out, "Oops")

    def test_reads_existing_file(self):
        path = self.root / "doc.md"
        path.write_text("# title\n\nbody", encoding="utf-8")
        self.assertEqual(load_markdown_file(path), "# title\n\nbody")


class TestDerivePiiVerdict(unittest.TestCase):
    def test_passed_when_both_sources_zero_leaks(self):
        val = {"validation_passed": True, "pii_leakage_audit": {"phone_leaks": 0, "email_leaks": 0, "api_key_leaks": 0}}
        cert = {
            "verification_status": "PASSED",
            "production_parquet_audit": {"total_leaks_found": 0, "sampled_messages_audited": 1000},
        }
        passed, total, sampled = derive_pii_verdict(val, cert)
        self.assertTrue(passed)
        self.assertEqual(total, 0)
        self.assertEqual(sampled, 1000)

    def test_failed_when_parquet_audit_has_leaks(self):
        val = {"validation_passed": True, "pii_leakage_audit": {"phone_leaks": 0, "email_leaks": 0, "api_key_leaks": 0}}
        cert = {
            "verification_status": "PASSED",
            "production_parquet_audit": {"total_leaks_found": 3, "sampled_messages_audited": 1000},
        }
        passed, total, _ = derive_pii_verdict(val, cert)
        self.assertFalse(passed)
        self.assertEqual(total, 3)

    def test_failed_when_validation_failed_even_if_cert_passes(self):
        # AND semantics: a PASSED verdict requires BOTH the cert and the
        # validation results to independently confirm zero leaks. An
        # empty cert (``cert_passed=False``) collapses the verdict to
        # False, regardless of the validation block's own state. See
        # CHANGELOG 12.0.2 / NADO.md "privacy as a checklist" pattern.
        val = {
            "validation_passed": False,
            "pii_leakage_audit": {"phone_leaks": 0, "email_leaks": 0, "api_key_leaks": 0},
        }
        cert = {}
        passed, total, _ = derive_pii_verdict(val, cert)
        self.assertFalse(passed)
        self.assertEqual(total, 0)

    def test_failed_when_only_one_source_confirms(self):
        # Regression guard for the "privacy-as-checklist" failure mode.
        # If only ONE of the two independent PII audits reports zero
        # leaks, the verdict MUST be False: a single stale certificate
        # is exactly the threat model we are defending against. The
        # previous ``or`` short-circuit pinned here was the bug; the
        # production code now uses ``and`` (CHANGELOG 12.0.2).
        val = {
            "validation_passed": True,
            "pii_leakage_audit": {"phone_leaks": 0, "email_leaks": 0, "api_key_leaks": 0},
        }
        cert = {}  # cert_passed is False: missing status and missing audit
        passed, total, _ = derive_pii_verdict(val, cert)
        self.assertFalse(passed)
        self.assertEqual(total, 0)

    def test_failed_when_only_cert_confirms_but_validation_missing(self):
        # Mirror of the previous test: cert says PASSED + zero leaks,
        # but the validation block is missing entirely. AND semantics
        # must still produce a False verdict.
        val = {}
        cert = {
            "verification_status": "PASSED",
            "production_parquet_audit": {"total_leaks_found": 0, "sampled_messages_audited": 1000},
        }
        passed, total, _ = derive_pii_verdict(val, cert)
        self.assertFalse(passed)
        self.assertEqual(total, 0)

    def test_falls_back_to_pii_leak_block_when_parquet_missing(self):
        val = {
            "pii_leakage_audit": {"phone_leaks": 2, "email_leaks": 1, "api_key_leaks": 0, "sample_lines_checked": 500}
        }
        cert = {}
        passed, total, sampled = derive_pii_verdict(val, cert)
        self.assertFalse(passed)
        self.assertEqual(total, 3)
        self.assertEqual(sampled, 500)

    def test_default_sample_size_is_25k(self):
        passed, _, sampled = derive_pii_verdict({}, {})
        self.assertEqual(sampled, 25000)


class TestLeakBreakdown(unittest.TestCase):
    def test_prefers_structured_breakdown(self):
        cert = {
            "production_parquet_audit": {
                "leak_breakdown": {"phones": 1, "emails": 2, "api_keys": 3, "crypto_wallets": 4}
            }
        }
        val = {"pii_leakage_audit": {"phone_leaks": 99}}  # should be ignored
        out = leak_breakdown(cert, val)
        self.assertEqual(out, {"phones": 1, "emails": 2, "api_keys": 3, "crypto_wallets": 4})

    def test_falls_back_to_legacy_keys(self):
        cert = {}
        val = {"pii_leakage_audit": {"phone_leaks": 5, "email_leaks": 6, "api_key_leaks": 7}}
        out = leak_breakdown(cert, val)
        self.assertEqual(out, {"phones": 5, "emails": 6, "api_keys": 7, "crypto_wallets": 0})


class TestAdversarialSummary(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(adversarial_summary({}), (14, 14, 100.0))

    def test_overrides(self):
        cert = {
            "adversarial_suite": {
                "adversarial_tests_passed": 10,
                "total_adversarial_tests": 14,
                "success_rate_percentage": 71.4,
            }
        }
        self.assertEqual(adversarial_summary(cert), (10, 14, 71.4))


class TestFilterMessages(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "text_clean": ["nginx reverse proxy", "django ORM", "fastapi endpoint", "Postgres tuning"],
                "domain": ["devops", "backend", "backend", "databases"],
                "is_question": [True, False, True, True],
            }
        )

    def test_no_filters_returns_input(self):
        out = filter_messages(self.df)
        self.assertEqual(len(out), 4)

    def test_search_filter(self):
        out = filter_messages(self.df, search_query="nginx")
        self.assertEqual(list(out["text_clean"]), ["nginx reverse proxy"])

    def test_domain_filter(self):
        out = filter_messages(self.df, selected_domain="backend")
        self.assertEqual(len(out), 2)

    def test_domain_filter_ignores_all_domains_sentinel(self):
        out = filter_messages(self.df, selected_domain="Все домены")
        self.assertEqual(len(out), 4)

    def test_q_only(self):
        out = filter_messages(self.df, is_q_only=True)
        self.assertEqual(len(out), 3)

    def test_combined_filters(self):
        out = filter_messages(self.df, search_query="api", selected_domain="backend", is_q_only=True)
        self.assertEqual(list(out["text_clean"]), ["fastapi endpoint"])

    def test_missing_is_question_column_does_not_crash(self):
        df = self.df.drop(columns=["is_question"])
        out = filter_messages(df, is_q_only=True)
        self.assertEqual(len(out), 4)


class TestFilterSft(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "quality_score": [5.0, 4.0, 3.0, 2.0],
                "turn_count": [4, 3, 2, 1],
            }
        )

    def test_quality_only(self):
        out = filter_sft(self.df, min_quality=4.0, min_turns=1)
        self.assertEqual(len(out), 2)

    def test_turns_only(self):
        out = filter_sft(self.df, min_quality=1.0, min_turns=3)
        self.assertEqual(len(out), 2)

    def test_both_filters(self):
        # Rows that satisfy BOTH quality>=4.0 AND turns>=3 are
        # (5.0, 4) and (4.0, 3) -> 2 rows.
        out = filter_sft(self.df, min_quality=4.0, min_turns=3)
        self.assertEqual(len(out), 2)

    def test_both_filters_strict(self):
        out = filter_sft(self.df, min_quality=4.5, min_turns=3)
        self.assertEqual(len(out), 1)

    def test_no_match(self):
        out = filter_sft(self.df, min_quality=10.0, min_turns=10)
        self.assertEqual(len(out), 0)


if __name__ == "__main__":
    unittest.main()
