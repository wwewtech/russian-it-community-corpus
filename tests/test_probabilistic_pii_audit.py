"""Tests for the stratified probabilistic PII audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ricc.validation.probabilistic_audit import (
    ProbabilisticPIIAuditor,
    find_leaks_in_text,
)


def _clean_text(i: int) -> str:
    return f"Обсуждаем архитектуру микросервисов и очереди сообщений, сообщение номер {i}."


class TestFindLeaksInText(unittest.TestCase):
    def test_detects_email(self):
        found = find_leaks_in_text("пиши мне на leaked.person@example.com")
        self.assertIn("emails", found)
        self.assertEqual(found["emails"], ["leaked.person@example.com"])

    def test_detects_phone(self):
        found = find_leaks_in_text("звоните +7 911 123-45-67")
        self.assertIn("phones", found)

    def test_detects_crypto(self):
        found = find_leaks_in_text("кошелек 0x71C7656EC7ab88b098defB751B7401B5f6d8976F")
        self.assertIn("crypto_wallets", found)

    def test_detects_api_key(self):
        found = find_leaks_in_text("ключ: sk-proj-1234567890abcdef1234567890abcdef")
        self.assertIn("api_keys", found)

    def test_detects_jwt(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"  # pragma: allowlist secret
        found = find_leaks_in_text(f"токен {jwt}")
        self.assertIn("jwt_tokens", found)

    def test_detects_ssh_key(self):
        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA1234\n-----END RSA PRIVATE KEY-----"  # pragma: allowlist secret
        found = find_leaks_in_text(pem)
        self.assertIn("ssh_keys", found)

    def test_public_ip_detected_but_loopback_kept(self):
        found_public = find_leaks_in_text("сервер 203.0.113.55")
        self.assertIn("public_ips", found_public)
        found_loopback = find_leaks_in_text("локально 127.0.0.1")
        self.assertNotIn("public_ips", found_loopback)

    def test_semantic_version_not_a_phone(self):
        found = find_leaks_in_text("обновись до 3.10.4")
        self.assertNotIn("phones", found)

    def test_empty_text(self):
        self.assertEqual(find_leaks_in_text(""), {})


class TestProbabilisticPIIAuditor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.parquet = Path(self.temp_dir.name) / "corpus.parquet"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_corpus(self, rows: list[dict]) -> Path:
        pd.DataFrame(rows).to_parquet(self.parquet)
        return self.parquet

    def test_missing_dataset_returns_skipped(self):
        auditor = ProbabilisticPIIAuditor(Path(self.temp_dir.name) / "missing.parquet")
        report = auditor.run_audit()
        self.assertEqual(report["status"], "SKIPPED_DATASET_MISSING")

    def test_clean_corpus_passes_with_statistical_guarantee(self):
        rows = [{"text_clean": _clean_text(i), "chat_name": f"community_node_{i % 3 + 1:02d}"} for i in range(300)]
        self._write_corpus(rows)
        report = ProbabilisticPIIAuditor(self.parquet).run_audit(sample_size=200, max_leak_tolerance=0.03)
        self.assertEqual(report["verdict"], "PASSED")
        self.assertEqual(report["results"]["messages_with_leaks"], 0)
        self.assertIn("statistical_guarantee", report)
        self.assertLessEqual(report["results"]["message_leak_rate_upper_bound_99"], 0.03)

    def test_leaky_corpus_detected(self):
        rows = [{"text_clean": _clean_text(i), "chat_name": "community_node_01"} for i in range(100)]
        rows.append(
            {
                "text_clean": "срочно пиши на leaked.person@example.com",
                "chat_name": "community_node_02",
            }
        )
        self._write_corpus(rows)
        report = ProbabilisticPIIAuditor(self.parquet).run_audit(sample_size=101)
        self.assertEqual(report["verdict"], "LEAKS_DETECTED")
        # "messages_with_leak" counts MESSAGES (the Bernoulli trial the
        # confidence bound is computed from); "leak_instances" counts raw
        # matches and may exceed the message count.
        self.assertGreaterEqual(report["results"]["categories"]["emails"]["messages_with_leak"], 1)
        self.assertGreaterEqual(report["results"]["categories"]["emails"]["leak_instances"], 1)
        self.assertTrue(report["results"]["categories"]["emails"]["examples"])

    def test_stratified_sampling_covers_all_strata(self):
        rows = []
        for i in range(300):
            rows.append({"text_clean": _clean_text(i), "chat_name": f"community_node_{i % 3 + 1:02d}"})
        self._write_corpus(rows)
        report = ProbabilisticPIIAuditor(self.parquet).run_audit(sample_size=150)
        self.assertEqual(report["sampling"]["strata_count"], 3)
        self.assertEqual(report["sampled_messages"], 150)
        # Proportional allocation: each stratum should contribute ~50 messages
        for stratum_stats in report["per_stratum"].values():
            self.assertGreater(stratum_stats["messages"], 0)

    def test_sample_size_equals_corpus_when_smaller(self):
        rows = [{"text_clean": _clean_text(i), "chat_name": "n1"} for i in range(50)]
        self._write_corpus(rows)
        report = ProbabilisticPIIAuditor(self.parquet).run_audit(sample_size=500)
        self.assertEqual(report["sampled_messages"], 50)

    def test_power_analysis_present_and_sufficient_flag(self):
        rows = [{"text_clean": _clean_text(i), "chat_name": "n1"} for i in range(10)]
        self._write_corpus(rows)
        report = ProbabilisticPIIAuditor(self.parquet).run_audit(sample_size=10)
        self.assertIn("required_sample_for_1e-3_rate_at_5e-4_margin_99pct", report["power_analysis"])
        self.assertIn("sufficient", report["power_analysis"])

    def test_generate_report_writes_json(self):
        rows = [{"text_clean": _clean_text(i), "chat_name": "n1"} for i in range(20)]
        self._write_corpus(rows)
        out = Path(self.temp_dir.name) / "reports" / "audit.json"
        written = ProbabilisticPIIAuditor(self.parquet).generate_report(out, sample_size=20)
        self.assertTrue(written.exists())
        data = json.loads(written.read_text(encoding="utf-8"))
        self.assertIn("verdict", data)
        self.assertIn("results", data)

    def test_confidence_bounds_contain_point_estimate(self):
        rows = [{"text_clean": _clean_text(i), "chat_name": "n1"} for i in range(100)]
        rows.append({"text_clean": "mail me at a@b.com", "chat_name": "n1"})
        self._write_corpus(rows)
        report = ProbabilisticPIIAuditor(self.parquet).run_audit(sample_size=101)
        cats = report["results"]["categories"]["emails"]
        self.assertLessEqual(cats["ci_lower"], cats["observed_rate"])
        self.assertLessEqual(cats["observed_rate"], cats["ci_upper"])
        self.assertLessEqual(cats["observed_rate"], cats["upper_bound_99"])


class TestAuditFailsClosed(unittest.TestCase):
    """Regression tests: the audit must never certify a corpus it did not read.

    Before the fix, a parquet without the audited text column produced an
    empty string for every row, so zero leaks were "found" and the report
    claimed PASSED with a 99% confidence statement — a green light on an
    uninspected corpus.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.parquet = Path(self.temp_dir.name) / "corpus.parquet"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_missing_text_column_is_not_reported_as_passed(self) -> None:
        pd.DataFrame({"chat_name": ["n1"] * 500, "other": ["x"] * 500}).to_parquet(self.parquet)
        report = ProbabilisticPIIAuditor(self.parquet).run_audit(sample_size=100)
        self.assertNotEqual(report.get("verdict"), "PASSED")
        self.assertEqual(report.get("status"), "FAILED_MISSING_COLUMNS")
        self.assertIn("text_clean", report["missing_columns"])

    def test_missing_strata_column_is_not_reported_as_passed(self) -> None:
        pd.DataFrame({"text_clean": ["hello"] * 500}).to_parquet(self.parquet)
        report = ProbabilisticPIIAuditor(self.parquet).run_audit(sample_size=100)
        self.assertNotEqual(report.get("verdict"), "PASSED")
        self.assertEqual(report.get("status"), "FAILED_MISSING_COLUMNS")
        self.assertIn("chat_name", report["missing_columns"])

    def test_multiple_leaks_in_one_message_do_not_crash(self) -> None:
        """Two leaks in one message used to raise ValueError.

        The audit counted leak INSTANCES but passed them to the Wilson
        bound, which expects the number of successful Bernoulli trials
        (messages). A single message with two emails made the count
        exceed the sample size and crashed the whole audit.
        """
        rows = [{"text_clean": "mail a@b.com and c@d.com and +79161234567", "chat_name": "n1"} for _ in range(500)]
        pd.DataFrame(rows).to_parquet(self.parquet)
        report = ProbabilisticPIIAuditor(self.parquet).run_audit(sample_size=100)
        self.assertIn("verdict", report)
        self.assertEqual(report["results"]["messages_with_leaks"], 100)
        self.assertEqual(report["results"]["total_leak_instances"], 300)
        # The message-level count is what the bound is computed from.
        self.assertEqual(report["results"]["categories"]["emails"]["messages_with_leak"], 100)
        self.assertEqual(report["results"]["categories"]["emails"]["leak_instances"], 200)

    def test_audited_categories_are_listed_in_the_report(self) -> None:
        rows = [{"text_clean": _clean_text(i), "chat_name": "n1"} for i in range(20)]
        pd.DataFrame(rows).to_parquet(self.parquet)
        report = ProbabilisticPIIAuditor(self.parquet).run_audit(sample_size=20)
        self.assertIn("audited_categories", report)
        for category in ("secret_assignments", "private_invites", "db_urls", "user_mentions"):
            self.assertIn(category, report["audited_categories"])


class TestStratifiedSamplingIsIndependent(unittest.TestCase):
    """Each stratum must get its own draw, not the same relative positions."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.parquet = Path(self.temp_dir.name) / "corpus.parquet"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_strata_use_independent_draws(self) -> None:
        rows = [{"text_clean": f"msg {i}", "chat_name": "A" if i < 300 else "B"} for i in range(600)]
        pd.DataFrame(rows).to_parquet(self.parquet)
        auditor = ProbabilisticPIIAuditor(self.parquet, seed=42)
        sample = auditor._stratified_sample(pd.DataFrame(rows), 300)
        idx_a = sorted(sample.index[sample["chat_name"] == "A"].tolist())
        idx_b = sorted(sample.index[sample["chat_name"] == "B"].tolist())
        # A systematic sample (one shared random_state) would repeat the
        # same relative offsets in every stratum: idx_b == idx_a + 300.
        self.assertNotEqual([i - 0 for i in idx_a], [i - 300 for i in idx_b])

    def test_sampling_is_reproducible_for_a_fixed_seed(self) -> None:
        rows = [{"text_clean": f"msg {i}", "chat_name": f"n{i % 4}"} for i in range(1000)]
        df = pd.DataFrame(rows)
        auditor = ProbabilisticPIIAuditor(self.parquet, seed=42)
        first = auditor._stratified_sample(df, 200).index.tolist()
        second = auditor._stratified_sample(df, 200).index.tolist()
        self.assertEqual(first, second)


class TestPreviouslyUnauditedCategories(unittest.TestCase):
    """Categories the scrubber cleans but the audit used to ignore."""

    def test_detects_secret_assignment(self):
        found = find_leaks_in_text("api_key=abcdef123456")
        self.assertIn("secret_assignments", found)

    def test_detects_private_invite(self):
        found = find_leaks_in_text("вступай https://t.me/+AbCdEf123")
        self.assertIn("private_invites", found)

    def test_detects_db_url_with_credentials(self):
        found = find_leaks_in_text("postgres://user:secretpass@db.internal:5432/app")
        self.assertIn("db_urls", found)

    def test_detects_forward_header(self):
        found = find_leaks_in_text("Forwarded from Ivan Ivanov")
        self.assertIn("forward_headers", found)

    def test_detects_raw_user_mention(self):
        found = find_leaks_in_text("пиши @ivan_petrov")
        self.assertIn("user_mentions", found)

    def test_anonymised_handles_are_not_leaks(self):
        self.assertNotIn("user_mentions", find_leaks_in_text("пиши @user_12"))
        self.assertNotIn("user_mentions", find_leaks_in_text("пиши @user_anon"))


if __name__ == "__main__":
    unittest.main()
