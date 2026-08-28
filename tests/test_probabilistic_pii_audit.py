"""Tests for the stratified probabilistic PII audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.validation.probabilistic_audit import (
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
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        found = find_leaks_in_text(f"токен {jwt}")
        self.assertIn("jwt_tokens", found)

    def test_detects_ssh_key(self):
        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA1234\n-----END RSA PRIVATE KEY-----"
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
        self.assertGreaterEqual(report["results"]["categories"]["emails"]["leaks_found"], 1)
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


if __name__ == "__main__":
    unittest.main()
