"""Tests for dataset drift monitoring (PSI / JS divergence / vocabulary)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring.drift import DatasetDriftMonitor, _js_divergence, _psi


def _make_df(n: int, seed: int, domain: str = "backend_databases", length_scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(20, 200, size=n)
    texts = [("слово " * max(1, int(length * length_scale / 10)))[: int(length * length_scale)] for length in lengths]
    return pd.DataFrame({"text_clean": texts, "domain": [domain] * n})


class TestPsiFunction:
    def test_identical_distributions_zero_psi(self):
        dist = np.array([10.0, 10.0, 10.0, 10.0])
        assert _psi(dist, dist.copy()) < 1e-6

    def test_shifted_distributions_positive_psi(self):
        expected = np.array([100.0, 0.0, 0.0, 0.0])
        actual = np.array([0.0, 0.0, 0.0, 100.0])
        assert _psi(expected, actual) > 1.0


class TestJsDivergence:
    def test_identical_distributions_zero(self):
        dist = np.array([5.0, 5.0, 5.0])
        assert _js_divergence(dist, dist.copy()) < 1e-6

    def test_disjoint_distributions_one_bit(self):
        p = np.array([10.0, 0.0])
        q = np.array([0.0, 10.0])
        assert abs(_js_divergence(p, q) - 1.0) < 1e-6


class TestDatasetDriftMonitor(unittest.TestCase):
    def test_identical_snapshots_are_stable(self):
        df = _make_df(500, seed=1)
        report = DatasetDriftMonitor(df, df.copy()).run()
        self.assertEqual(report["overall_verdict"], "stable")
        self.assertLess(report["metrics"]["length_psi"]["value"], 0.10)
        self.assertLess(report["metrics"]["domain_distribution"]["js_divergence_bits"], 0.05)
        self.assertGreaterEqual(report["metrics"]["vocabulary"]["jaccard_overlap"], 0.90)

    def test_length_shift_detected(self):
        ref = _make_df(500, seed=1, length_scale=1.0)
        cur = _make_df(500, seed=2, length_scale=4.0)  # much longer messages
        report = DatasetDriftMonitor(ref, cur).run()
        self.assertGreater(report["metrics"]["length_psi"]["value"], 0.10)
        self.assertIn(report["metrics"]["length_psi"]["verdict"], ("moderate_drift", "significant_drift"))

    def test_domain_shift_detected(self):
        ref = _make_df(500, seed=1, domain="backend_databases")
        cur = _make_df(500, seed=2, domain="frontend_ui")
        report = DatasetDriftMonitor(ref, cur).run()
        self.assertGreater(report["metrics"]["domain_distribution"]["js_divergence_bits"], 0.05)
        self.assertEqual(report["metrics"]["domain_distribution"]["biggest_share_shift_domain"], "frontend_ui")

    def test_vocabulary_shift_detected(self):
        ref = pd.DataFrame({"text_clean": ["postgres индекс репликация wal"] * 100, "domain": ["db"] * 100})
        cur = pd.DataFrame({"text_clean": ["react компонент хук состояние рендер"] * 100, "domain": ["fe"] * 100})
        report = DatasetDriftMonitor(ref, cur).run()
        self.assertLess(report["metrics"]["vocabulary"]["jaccard_overlap"], 0.75)

    def test_report_structure_and_serialization(self):
        df = _make_df(100, seed=3)
        report = DatasetDriftMonitor(df, df.copy()).run()
        for key in ("report_title", "generated_at_utc", "metrics", "overall_verdict", "thresholds"):
            self.assertIn(key, report)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "drift.json"
            written = DatasetDriftMonitor(df, df.copy()).generate_report(out)
            self.assertTrue(written.exists())
            data = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(data["overall_verdict"], "stable")


if __name__ == "__main__":
    unittest.main()
