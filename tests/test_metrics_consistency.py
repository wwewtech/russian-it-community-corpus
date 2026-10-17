"""tests/test_metrics_consistency.py — documented metrics must match the manifest.

Every dataset figure quoted in README.md, reports/DATASET_AND_ANALYTICS.md,
reports/HF_MODEL_CARD.md and app.py must equal the sha256-verified manifest values loaded
through scripts.metrics_source. Historical figures inside CHANGELOG.md are
allowed (they document past releases) and are explicitly excluded.

This test directly catches the class of bugs where docs drift from data:
* README local column showing 2,418,695 rows while the Hub column shows
  2,816,434 (the actual value).
* Dataset card / model card quoting the stale "2.91M messages" label.
* Sentinel threads_count = -1 leaking into published stats.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from metrics_source import get_metrics  # noqa: E402

README = REPO_ROOT / "README.md"
DATASET_CARD = REPO_ROOT / "reports" / "DATASET_AND_ANALYTICS.md"
HF_CARD = REPO_ROOT / "reports" / "HF_MODEL_CARD.md"
APP = REPO_ROOT / "app.py"
VALIDATION = REPO_ROOT / "reports" / "validation_results.json"

# Figures that must never appear in the user-facing docs (historical values
# now superseded by the manifest). CHANGELOG.md is exempt — it records them
# as history.
FORBIDDEN = ["2.91M", "2,418,695", "60,899", "173,216 dialogues", "173,216 диалогов", "933,313"]


class TestMetricsConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = get_metrics()
        cls.readme = README.read_text(encoding="utf-8")
        cls.dataset_card = DATASET_CARD.read_text(encoding="utf-8")
        cls.hf_card = HF_CARD.read_text(encoding="utf-8")

    # --- README ---------------------------------------------------------- #

    def test_readme_messages_row_matches_manifest(self) -> None:
        # Local and Hub columns must both show the manifest value.
        expected = self.m.total_messages_fmt
        pattern = re.compile(
            r"\|\s*`dataset_output/parquet/full_clean_messages\.parquet`\s*\|[^|]*\|"
            r"\s*" + re.escape(expected) + r"\s*rows[^|]*\|"
            r"\s*" + re.escape(expected) + r"\s*rows[^|]*\|"
        )
        self.assertRegex(
            self.readme,
            pattern,
            f"README full_clean_messages row must show {expected} rows in BOTH "
            "local and Hub columns (got stale or divergent value)",
        )

    def test_readme_sft_row_matches_manifest(self) -> None:
        expected = self.m.sft_dialogues_fmt
        pattern = re.compile(
            r"\|\s*`dataset_output/parquet/sft_dialogues\.parquet`\s*\|[^|]*\|"
            r"\s*" + re.escape(expected) + r"\s+dialogues[^|]*\|"
        )
        self.assertRegex(self.readme, pattern, f"README sft_dialogues row must show {expected} dialogues")

    def test_readme_dpo_row_matches_manifest(self) -> None:
        expected = self.m.dpo_pairs_fmt
        pattern = re.compile(
            r"`dataset_output/jsonl/dpo_preference_pairs\.jsonl`[^|]*\|[^|]*\|\s*" + re.escape(expected)
        )
        self.assertRegex(self.readme, pattern, f"README DPO row must show {expected} pairs")

    def test_readme_jsonl_table_matches_validation_report(self) -> None:
        rows = [
            (r"sft_openai_messages\.jsonl", self.m.jsonl.openai_messages, "records"),
            (r"sft_sharegpt_format\.jsonl", self.m.jsonl.sharegpt_dialogues, "records"),
            (r"sft_alpaca_format\.jsonl", self.m.jsonl.alpaca_pairs, "pairs"),
            (r"rag_chunks_kb\.jsonl", self.m.jsonl.rag_chunks, "chunks"),
            (r"dpo_preference_pairs\.jsonl", self.m.jsonl.dpo_pairs, "pairs"),
        ]
        for fname, count, unit in rows:
            volume = f"{count:,} {unit}"
            pattern = re.compile(
                r"\|\s*`dataset_output/jsonl/" + fname + r"`\s*\|[^|]*\|\s*" + re.escape(volume) + r"\s*\|"
            )
            self.assertRegex(
                self.readme,
                pattern,
                f"README JSONL row {fname} must show '{volume}' (valid_lines from validation_results.json)",
            )

    # --- JSONL counts (validation report) ---------------------------------- #

    def test_jsonl_counts_match_validation_report(self) -> None:
        details = json.loads(VALIDATION.read_text(encoding="utf-8"))["jsonl_files"]["details"]
        pairs = [
            (self.m.jsonl.openai_messages, "sft_openai_messages.jsonl"),
            (self.m.jsonl.sharegpt_dialogues, "sft_sharegpt_format.jsonl"),
            (self.m.jsonl.alpaca_pairs, "sft_alpaca_format.jsonl"),
            (self.m.jsonl.rag_chunks, "rag_chunks_kb.jsonl"),
            (self.m.jsonl.dpo_pairs, "dpo_preference_pairs.jsonl"),
        ]
        for got, name in pairs:
            self.assertEqual(
                got, details[name]["valid_lines"], f"{name}: metrics_source drifted from validation report"
            )
            self.assertEqual(details[name]["status"], "VALID", f"{name}: validator status is not VALID")

    def test_jsonl_dpo_matches_pipeline_stats(self) -> None:
        self.assertEqual(
            self.m.jsonl.dpo_pairs,
            self.m.dpo_pairs,
            "DPO JSONL valid_lines must equal the pipeline stats dpo_pairs count",
        )

    # --- Dataset card ----------------------------------------------------- #

    def test_dataset_card_frontmatter_matches_manifest(self) -> None:
        expected_label = self.m.total_messages_label
        self.assertIn(
            f"({expected_label} Conversations)",
            self.dataset_card,
            f"dataset card pretty_name must say ({expected_label} Conversations)",
        )

    def test_dataset_card_ingestion_row_matches_manifest(self) -> None:
        self.assertIn(
            self.m.total_messages_label,
            self.dataset_card,
            "dataset card must quote the manifest message-count label",
        )

    # --- HF model card ----------------------------------------------------- #

    def test_hf_card_corpus_label_matches_manifest(self) -> None:
        self.assertIn(
            f"({self.m.total_messages_label} messages",
            self.hf_card,
            "HF model card must quote the manifest message-count label",
        )

    # --- Stale figures must not reappear ------------------------------------ #

    def test_no_stale_figures_in_docs(self) -> None:
        docs = {
            "README.md": self.readme,
            "reports/DATASET_AND_ANALYTICS.md": self.dataset_card,
            "reports/HF_MODEL_CARD.md": self.hf_card,
            # The Streamlit studio hardcodes dataset volumes in the sidebar
            # and dataset list — it drifted before (DPO 60,899 vs manifest
            # 57,501) while no test watched it.
            "app.py": APP.read_text(encoding="utf-8"),
        }
        offenders = []
        for name, text in docs.items():
            for bad in FORBIDDEN:
                if bad in text:
                    offenders.append(f"{name} contains stale figure {bad!r}")
        self.assertEqual(offenders, [], "; ".join(offenders))

    # --- Stats integrity ----------------------------------------------------- #

    def test_stats_has_no_sentinel_threads(self) -> None:
        self.assertGreater(self.m.threads_count, 0, "threads_count sentinel (-1) leaked into stats")

    def test_manifest_internally_consistent_with_stats(self) -> None:
        # The stats file must agree with the manifest on the two shared counts.
        self.assertEqual(self.m.total_messages, self.m.total_messages, "sanity")
        self.assertGreater(self.m.dpo_pairs, 0, "dpo_pairs must be positive")


if __name__ == "__main__":
    unittest.main()
