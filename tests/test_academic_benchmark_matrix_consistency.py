"""Published academic benchmark matrix must be reproducible from raw outputs.

The matrix (reports/academic_scientific_benchmarks_matrix.json) is derived
data: every published number must be recomputable from the raw generation
logs (reports/academic_benchmarks_raw_outputs.jsonl) without a GPU.

This guards against the class of defects that forced the original
retraction: numbers in markdown/JSON that no longer match what the harness
actually measured.
"""

from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "reports" / "academic_benchmarks_raw_outputs.jsonl"
MATRIX_PATH = REPO_ROOT / "reports" / "academic_scientific_benchmarks_matrix.json"

VARIANTS = ("base", "rag", "lora", "hybrid")


def _load() -> tuple[list[dict], dict]:
    records = [json.loads(line) for line in RAW_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    return records, matrix


class TestAcademicMatrixConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RAW_PATH.exists() or not MATRIX_PATH.exists():
            raise unittest.SkipTest("raw benchmark outputs or matrix not present")
        cls.records, cls.matrix = _load()

    def test_raw_output_coverage_is_complete(self) -> None:
        """40 HumanEval x 4 variants + 50 RuMMLU x 4 variants + PPL + ROUGE."""
        counts = Counter((r.get("benchmark"), r.get("variant")) for r in self.records)
        for variant in VARIANTS:
            self.assertEqual(counts[("HumanEval", variant)], 40, f"HumanEval/{variant}")
            self.assertEqual(counts[("RuMMLU_CS", variant)], 50, f"RuMMLU_CS/{variant}")
        self.assertEqual(counts[("Perplexity", "base")], 1)
        self.assertEqual(counts[("Perplexity", "lora")], 1)
        self.assertEqual(counts[("ROUGE", "base")], 1)
        self.assertEqual(counts[("ROUGE", "lora")], 1)

    def test_humaneval_pass_at_1_matches_raw(self) -> None:
        he = [r for r in self.records if r.get("benchmark") == "HumanEval"]
        n_tasks = len({r["task_id"] for r in he})
        self.assertEqual(n_tasks, 40)
        for variant in VARIANTS:
            passed = sum(1 for r in he if r.get("variant") == variant and r.get("passed") is True)
            recomputed = round(passed / n_tasks * 100.0, 1)
            self.assertEqual(
                recomputed,
                self.matrix["humaneval_pass_at_1"][variant],
                f"HumanEval pass@1/{variant}: matrix says {self.matrix['humaneval_pass_at_1'][variant]}, "
                f"raw logs recompute to {recomputed}",
            )

    def test_rummlu_accuracy_matches_raw(self) -> None:
        rm = [r for r in self.records if r.get("benchmark") == "RuMMLU_CS"]
        n_q = len({r.get("question") for r in rm})
        self.assertEqual(n_q, 50)
        for variant in VARIANTS:
            parsed_ok = sum(
                1 for r in rm if r.get("variant") == variant and r.get("parsed_answer") == r.get("expected_answer")
            )
            passed_flag = sum(1 for r in rm if r.get("variant") == variant and r.get("passed") is True)
            # The stored `passed` flag must agree with parsed vs expected answers
            # (this is exactly where the old substring-scoring bug diverged).
            self.assertEqual(parsed_ok, passed_flag, f"RuMMLU passed flag != parsed==expected for {variant}")
            recomputed = round(parsed_ok / n_q * 100.0, 1)
            self.assertEqual(
                recomputed,
                self.matrix["rummlu_accuracy"][variant],
                f"RuMMLU accuracy/{variant}: matrix {self.matrix['rummlu_accuracy'][variant]} != {recomputed}",
            )

    def test_perplexity_matches_raw(self) -> None:
        ppl = {r["variant"]: r["ppl"] for r in self.records if r.get("benchmark") == "Perplexity"}
        self.assertEqual(ppl, self.matrix["perplexity"])

    def test_rouge_matches_raw(self) -> None:
        rouge = {r["variant"]: r["scores"] for r in self.records if r.get("benchmark") == "ROUGE"}
        for variant in ("base", "lora"):
            self.assertIn(variant, rouge)
            self.assertEqual(rouge[variant], self.matrix["rouge"][variant])

    def test_matrix_records_provenance(self) -> None:
        """Matrix must state provenance: raw file + generation timestamp."""
        for key in ("raw_outputs_sha256", "generated_at_utc"):
            self.assertIn(key, self.matrix, f"matrix missing provenance field {key!r}")

    # --- Retraction gate ----------------------------------------------------- #

    def test_matrix_status_gate(self) -> None:
        """Published matrix must be explicitly RETRACTED, or VALID with evidence."""
        status = self.matrix.get("status")
        self.assertIn(
            status,
            ("RETRACTED", "VALID"),
            "matrix must carry an explicit status ('RETRACTED' until a reviewed GPU re-run)",
        )
        if status == "RETRACTED":
            self.assertTrue(
                self.matrix.get("retraction_reason"),
                "RETRACTED matrix must explain why (retraction_reason)",
            )
        else:
            self.assertTrue(
                self.matrix.get("rerun_evidence"),
                "matrix claiming VALID must include rerun_evidence (reviewed GPU re-run)",
            )

    def test_report_banner_matches_status(self) -> None:
        """If the markdown report is published, it must carry the RETRACTED banner."""
        if self.matrix.get("status") != "RETRACTED":
            self.skipTest("matrix is not retracted; banner not required")
        report_path = REPO_ROOT / "reports" / "OFFICIAL_ACADEMIC_SCIENTIFIC_BENCHMARKS.md"
        if not report_path.exists():
            self.skipTest("markdown report is not published in the repository")
        md = report_path.read_text(encoding="utf-8")
        head = md[:800]
        self.assertIn("RETRACTED", head, "report must carry the RETRACTED banner near the top")
        self.assertIn("do not cite", head, "report banner must say not to cite the numbers")

    # --- Heuristic (rubric-based) reports ------------------------------------- #

    def test_heuristic_report_carries_disclaimer(self) -> None:
        """Heuristic scores must be labelled as rubric-based, never as capabilities."""
        heuristic = json.loads((REPO_ROOT / "reports" / "heuristic_benchmark_eval.json").read_text(encoding="utf-8"))
        meta = heuristic.get("metadata", {})
        self.assertIn("rubric-based", str(meta.get("evaluation_type", "")).lower())
        self.assertIn("_disclaimer", heuristic, "heuristic report must carry a top-level _disclaimer")
        self.assertIn("RETRACTED", heuristic["_disclaimer"])
        self.assertIn("not capability measurements", heuristic["_disclaimer"])

    def test_readme_annotates_heuristic_link(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "rubric-based heuristics",
            readme,
            "README heuristic link must state that scores are rubric-based heuristics",
        )


if __name__ == "__main__":
    unittest.main()
