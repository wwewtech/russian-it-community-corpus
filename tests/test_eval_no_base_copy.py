"""Regression tests: the evaluation suites must not fabricate a comparison.

Three of the four evaluation suites used to fall back to the BASE model's
results when the LoRA adapter could not be loaded, and wrote them into
the LoRA (and Hybrid) columns. The published comparison then showed
identical numbers for base and LoRA — a benchmark that measured
nothing.

These tests pin the fail-fast behaviour and the statistical honesty of
the report.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from ricc.evaluation.official_academic_benchmarks import _paired_significance
from ricc.evaluation.statistical_power import mcnemar_test

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "ricc" / "evaluation"

SUITES_WITH_BENCHMARKS = (
    "enterprise_production_eval_suite.py",
    "deep_multi_experiment_suite.py",
    "sota_expert_benchmark_showcase.py",
)


def _source(name: str) -> str:
    return (EVAL_DIR / name).read_text(encoding="utf-8")


class TestNoBaseToLoRACopy(unittest.TestCase):
    """No suite may substitute a base result for a LoRA result."""

    def test_no_suite_falls_back_to_base_results(self):
        for name in SUITES_WITH_BENCHMARKS:
            with self.subTest(suite=name):
                source = _source(name)
                # The fallback shape: a LoRA field taken from the base
                # results, e.g. `base_results[idx - 1]["concept_score"]`
                # assigned into a `l_*` variable.
                self.assertNotIn(
                    "if lora_model else base_results",
                    source,
                    f"{name} still copies a base result into the LoRA column",
                )
                self.assertNotIn(
                    "if lora_model else base_evals",
                    source,
                    f"{name} still copies a base result into the LoRA column",
                )
                self.assertNotIn(
                    "l_text, l_scores = base_outputs[",
                    source,
                    f"{name} still copies a base result into the LoRA column",
                )

    def test_every_suite_fails_fast_without_an_adapter(self):
        """Each suite must raise instead of continuing without the adapter."""
        for name in SUITES_WITH_BENCHMARKS:
            with self.subTest(suite=name):
                source = _source(name)
                self.assertIn(
                    "could not be loaded",
                    source,
                    f"{name} does not fail fast when the adapter is missing",
                )
                self.assertIn(
                    "RuntimeError",
                    source,
                    f"{name} does not raise when the adapter is missing",
                )

    def test_suites_parse_and_expose_the_guard(self):
        """The guard is real code, not a comment."""
        for name in SUITES_WITH_BENCHMARKS:
            with self.subTest(suite=name):
                tree = ast.parse(_source(name))
                raises = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Raise)
                    and isinstance(node.exc, ast.Call)
                    and getattr(node.exc.func, "id", "") == "RuntimeError"
                ]
                self.assertTrue(raises, f"{name} has no RuntimeError guard")

    def test_hybrid_is_not_built_from_the_rag_variant(self):
        """Hybrid is LoRA + RAG; the Base+RAG result is a different variant.

        The check is on the ASSIGNMENT into a hybrid variable (``h_*``),
        not on any use of the RAG results — the "rag" column of the
        record legitimately reads them.
        """
        for name in SUITES_WITH_BENCHMARKS:
            with self.subTest(suite=name):
                tree = ast.parse(_source(name))
                offenders: list[str] = []
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign):
                        continue
                    targets = node.targets
                    if not any(isinstance(t, ast.Name) and t.id.startswith("h_") for t in targets):
                        continue
                    value = node.value
                    for sub in ast.walk(value):
                        if (
                            isinstance(sub, ast.Subscript)
                            and isinstance(sub.value, ast.Name)
                            and sub.value.id in ("rag_results", "rag_evals", "base_results", "base_evals")
                        ):
                            offenders.append(f"{sub.value.id}[...]")
                self.assertEqual(
                    offenders,
                    [],
                    f"{name} builds a hybrid value from another variant's results: {offenders}",
                )


class TestMcNemar(unittest.TestCase):
    def test_identical_variants_are_not_significant(self):
        result = mcnemar_test([(True, True)] * 10 + [(False, False)] * 10)
        self.assertEqual(result["discordant"], 0)
        self.assertEqual(result["p_value"], 1.0)
        self.assertFalse(result["significant"])

    def test_a_real_difference_is_detected(self):
        # 20 tasks where B wins and A loses, none tied.
        result = mcnemar_test([(False, True)] * 20)
        self.assertLess(result["p_value"], 0.05)
        self.assertTrue(result["significant"])
        self.assertAlmostEqual(result["difference_in_proportion"], 1.0)

    def test_a_small_difference_is_not_significant(self):
        # The published run: 3 discordant pairs in favour of B.
        result = mcnemar_test([(False, True)] * 3 + [(True, True)] * 37)
        self.assertGreater(result["p_value"], 0.05)
        self.assertFalse(result["significant"])

    def test_symmetry(self):
        forward = mcnemar_test([(False, True)] * 3 + [(True, True)] * 10)
        backward = mcnemar_test([(True, False)] * 3 + [(True, True)] * 10)
        self.assertEqual(forward["p_value"], backward["p_value"])


class TestPairedSignificanceHelper(unittest.TestCase):
    def test_pairs_by_task_key(self):
        outcomes = {
            "base": {"t1": True, "t2": False, "t3": True},
            "lora": {"t1": True, "t2": True, "t3": False},
        }
        result = _paired_significance(outcomes, comparisons=(("base", "lora"),))
        entry = result["base_vs_lora"]
        self.assertEqual(entry["n_pairs"], 3)
        self.assertEqual(entry["both_pass"], 1)
        self.assertEqual(entry["a_only_pass"], 1)
        self.assertEqual(entry["b_only_pass"], 1)

    def test_missing_variant_is_skipped(self):
        outcomes = {"base": {"t1": True}, "lora": {}}
        result = _paired_significance(outcomes, comparisons=(("base", "lora"),))
        self.assertEqual(result, {})


class TestPublishedNumbersAreHonest(unittest.TestCase):
    """The published matrix must carry the significance verdict."""

    def test_matrix_declares_no_significant_difference(self):
        matrix_path = REPO_ROOT / "reports" / "academic_scientific_benchmarks_matrix.json"
        if not matrix_path.exists():
            self.skipTest("published matrix not present")
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        # The matrix was published before the significance block
        # existed; when it is regenerated the verdict must be present
        # and must not claim a significant difference.
        significance = matrix.get("significance")
        if significance is None:
            self.skipTest("matrix has not been regenerated with the significance block yet")
        self.assertFalse(significance["any_significant"])

    def test_raw_outputs_contain_every_variant(self):
        raw_path = REPO_ROOT / "reports" / "academic_benchmarks_raw_outputs.jsonl"
        if not raw_path.exists():
            self.skipTest("raw outputs not present")
        records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        variants = {r.get("variant") for r in records}
        for variant in ("base", "rag", "lora", "hybrid"):
            self.assertIn(variant, variants)


if __name__ == "__main__":
    unittest.main()
