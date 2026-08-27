"""Arithmetic consistency guards for reports/metrics_index.json.

Scope note: these checks verify INTERNAL consistency only (that aggregates
are the means of their per-scenario scores and that gains equal score
differences). They cannot prove that the underlying measurements were
actually produced by a model run; provenance requires re-running the
evaluation pipeline on real hardware.
"""

import json
import unittest
from pathlib import Path

METRICS_INDEX = Path("reports/metrics_index.json")


@unittest.skipUnless(METRICS_INDEX.exists(), "metrics_index.json not generated yet")
class TestMetricsIndexConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(METRICS_INDEX.read_text(encoding="utf-8"))
        cls.matrix = cls.data["enterprise_eval_matrix"]
        cls.scenarios = cls.matrix["scenarios"]

    def test_scenario_list_matches_declared_total(self):
        declared = self.matrix["metadata"]["total_scenarios"]
        self.assertEqual(len(self.scenarios), declared)
        domain_sum = sum(v["scenarios_count"] for v in self.matrix["domain_breakdown"].values())
        self.assertEqual(domain_sum, declared)

    def test_aggregates_are_means_of_scenario_scores(self):
        agg = self.matrix["aggregate_summary"]
        for variant in ("base", "rag", "lora", "hybrid"):
            scores = [s[variant]["score"] for s in self.scenarios]
            mean = sum(scores) / len(scores)
            # Aggregates are stored rounded to one decimal place.
            self.assertLessEqual(
                abs(mean - agg[f"{variant}_total_score"]),
                0.051,
                f"{variant}: aggregate contradicts per-scenario scores",
            )

    def test_gains_equal_score_differences(self):
        agg = self.matrix["aggregate_summary"]
        self.assertAlmostEqual(agg["rag_gain_over_base"], agg["rag_total_score"] - agg["base_total_score"], places=6)
        self.assertAlmostEqual(
            agg["hybrid_gain_over_base"], agg["hybrid_total_score"] - agg["base_total_score"], places=6
        )

    def test_domain_breakdown_covers_all_scenarios(self):
        scenario_domains = {s["domain"] for s in self.scenarios}
        breakdown_domains = set(self.matrix["domain_breakdown"].keys())
        self.assertEqual(scenario_domains, breakdown_domains)

    def test_academic_benchmark_matrix_shape(self):
        academic = self.data["academic_scientific_benchmarks_matrix"]
        variants = {"base", "rag", "lora", "hybrid"}
        for name in ("humaneval_pass_at_1", "rummlu_accuracy"):
            self.assertEqual(set(academic[name].keys()), variants)


if __name__ == "__main__":
    unittest.main()
