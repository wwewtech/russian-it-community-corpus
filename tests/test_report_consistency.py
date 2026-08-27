"""Consistency guards for the scientific evaluation artifacts.

These tests exist because a senior review found that two report generations
carried contradictory numbers for the same models (base PPL differing by up to
8x, effect signs flipped) and referenced a non-existent commit hash. The
canonical pipeline (scripts/run_scientific_benchmark.py) now emits all
artifacts from a single run; these tests fail if the artifacts ever drift
apart again.
"""

import json
import re
import unittest
from pathlib import Path

REPORTS = Path("reports")
METRICS = REPORTS / "scientific_evaluation_metrics.json"
MATRIX = REPORTS / "raw_model_evaluation_matrix.json"
REPRO = REPORTS / "reproducibility_audit.json"
REPORT_MD = REPORTS / "SCIENTIFIC_EVALUATION_REPORT.md"


def _load_models(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "models" in data:
        return data["models"], data.get("run_metadata", {})
    return data, {}


@unittest.skipUnless(
    METRICS.exists() and MATRIX.exists() and REPRO.exists() and REPORT_MD.exists(),
    "evaluation artifacts not generated yet",
)
class TestReportConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        metrics_raw = json.loads(METRICS.read_text(encoding="utf-8"))
        cls.metrics = (
            metrics_raw["metrics"] if isinstance(metrics_raw, dict) and "metrics" in metrics_raw else metrics_raw
        )
        cls.metrics_meta = metrics_raw.get("run_metadata", {}) if isinstance(metrics_raw, dict) else {}
        cls.matrix, cls.matrix_meta = _load_models(MATRIX)
        cls.repro, cls.repro_meta = _load_models(REPRO)
        cls.md = REPORT_MD.read_text(encoding="utf-8")

    def test_run_metadata_present_and_versioned(self):
        for meta, name in [(self.metrics_meta, "metrics"), (self.matrix_meta, "matrix"), (self.repro_meta, "repro")]:
            self.assertTrue(meta, f"{name}: missing run_metadata block")
            self.assertIn("protocol_version", meta, f"{name}: missing protocol_version")
            commit = meta.get("git", {}).get("commit", "")
            self.assertTrue(
                re.fullmatch(r"[0-9a-f]{40}", commit) or commit == "unknown",
                f"{name}: git commit is not a real 40-hex revision: {commit!r}",
            )

    def test_same_model_set_across_artifacts(self):
        metric_ids = {m["model_id"] for m in self.metrics}
        matrix_ids = {m["id"] for m in self.matrix}
        repro_ids = set(self.repro.keys())
        self.assertEqual(metric_ids, matrix_ids, "metrics vs matrix model sets differ")
        self.assertEqual(metric_ids, repro_ids, "metrics vs reproducibility model sets differ")

    def test_metrics_match_matrix_numbers(self):
        matrix_by_id = {m["id"]: m for m in self.matrix}
        for m in self.metrics:
            entry = matrix_by_id[m["model_id"]]
            self.assertAlmostEqual(
                m["base_ppl"],
                entry["base_ppl_distribution"]["mean"],
                places=2,
                msg=f"{m['model_id']}: base PPL mismatch between metrics and matrix",
            )
            self.assertAlmostEqual(
                m["lora_ppl"],
                entry["lora_ppl_distribution"]["mean"],
                places=2,
                msg=f"{m['model_id']}: lora PPL mismatch between metrics and matrix",
            )
            self.assertAlmostEqual(
                m["ppl_delta_pct"],
                entry["mean_ppl_improvement_pct"],
                places=2,
                msg=f"{m['model_id']}: delta mismatch between metrics and matrix",
            )

    def test_repro_overall_matches_matrix(self):
        matrix_by_id = {m["id"]: m for m in self.matrix}
        for m_id, rec in self.repro.items():
            summary = rec["overall_summary"]
            entry = matrix_by_id[m_id]
            self.assertAlmostEqual(
                summary["base_ppl_mean"],
                entry["base_ppl_distribution"]["mean"],
                places=2,
                msg=f"{m_id}: reproducibility vs matrix base PPL mismatch",
            )
            if summary.get("lora_ppl_mean") is not None:
                self.assertAlmostEqual(
                    summary["lora_ppl_mean"],
                    entry["lora_ppl_distribution"]["mean"],
                    places=2,
                    msg=f"{m_id}: reproducibility vs matrix lora PPL mismatch",
                )
                self.assertAlmostEqual(
                    summary["delta_pct"],
                    entry["mean_ppl_improvement_pct"],
                    places=2,
                    msg=f"{m_id}: reproducibility vs matrix delta mismatch",
                )

    def test_markdown_table_matches_metrics(self):
        """Every base-PPL cell in the MD table must equal the metrics JSON value."""
        matrix_by_id = {m["id"]: m for m in self.matrix}
        rows = re.findall(r"^\| \*\*.+?\*\* \(\w+\) \| `([^`]+)` \|[^\n]*$", self.md, flags=re.MULTILINE)
        self.assertTrue(rows, "no model rows parsed from the Markdown table")
        for model_id in rows:
            line = next(ln for ln in self.md.splitlines() if f"`{model_id}`" in ln and ln.startswith("| **"))
            cells = [c.strip() for c in line.split("|")]
            base_cell = cells[4]  # "Base PPL (Mean ± σ)" -> "34.01 ± 17.8"
            base_mean = float(base_cell.split("±")[0].strip())
            self.assertAlmostEqual(
                base_mean,
                matrix_by_id[model_id]["base_ppl_distribution"]["mean"],
                places=2,
                msg=f"{model_id}: Markdown table contradicts raw matrix",
            )

    def test_findings_are_derived_from_data(self):
        """The 'Top improvements' narrative must name the actual argmax model."""
        measured = [m for m in self.matrix if m.get("mean_ppl_improvement_pct") is not None]
        if not measured:
            self.skipTest("no LoRA measurements present")
        top = max(measured, key=lambda m: m["mean_ppl_improvement_pct"])
        findings_idx = self.md.find("Empirical Findings")
        self.assertGreater(findings_idx, -1, "findings section missing")
        findings = self.md[findings_idx : findings_idx + 1200]
        self.assertIn(
            f"`{top['id']}`", findings, f"findings do not name the actual top model {top['id']} — narrative is stale"
        )

    def test_no_hardcoded_commit_links(self):
        """Reports must never embed hand-written commit URLs (404-fabrication guard)."""
        self.assertNotRegex(
            self.md, r"github\.com/[\w.-]+/[\w.-]+/commit/[0-9a-f]{7,40}", "report contains a hand-written commit link"
        )


if __name__ == "__main__":
    unittest.main()
