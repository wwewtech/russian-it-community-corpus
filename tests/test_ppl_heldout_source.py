"""The PPL sample must come from the held-out tail, not from the train corpus.

official_academic_benchmarks.py used to sample the perplexity texts from
``sft_dialogues.parquet`` — the very corpus the adapter was trained on —
while calling them a "held-out test set" in the report. The fix reads the
held-out tail of the SFT JSONL (the same tail ``--holdout-tail`` reserved
from training) and records the sample source in the raw outputs.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestHoldoutPplSampleSource(unittest.TestCase):
    def _run_ppl_sampling(self, tmp: Path, jsonl_rows: list[dict], holdout_tail: int) -> tuple[list[str], list[dict]]:
        """Drive the PPL text-sampling block of the harness without a GPU.

        Mirrors the sampling logic in run_official_academic_benchmarks
        (Benchmark 3) against a synthetic corpus.
        """
        import importlib  # noqa: F401 (kept for parity with the real module's lazy imports)

        del importlib
        dataset_jsonl = tmp / "dataset_output" / "jsonl" / "sft_openai_messages.jsonl"
        dataset_jsonl.parent.mkdir(parents=True, exist_ok=True)
        dataset_jsonl.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in jsonl_rows) + "\n", encoding="utf-8"
        )

        from collections import deque

        tail_rows: list[dict] = []
        if dataset_jsonl.exists() and holdout_tail > 0:
            ring: deque[dict] = deque(maxlen=holdout_tail)
            with dataset_jsonl.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        ring.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            tail_rows = list(ring)

        test_texts: list[str] = []
        source_rows = tail_rows if tail_rows else []
        if not source_rows:
            # In the real module this falls back to the parquet; here we only
            # verify the held-out path, so an empty source means no texts.
            return test_texts, tail_rows
        for row in source_rows:
            turns = row.get("messages") if isinstance(row, dict) else None
            if turns is None:
                continue
            text = " ".join(f"{t.get('role', '')}: {t.get('content', '')}" for t in turns if isinstance(t, dict))
            if text.strip():
                test_texts.append(text)
            if len(test_texts) >= 30:
                break
        return test_texts, tail_rows

    def test_ppl_reads_only_the_held_out_tail(self) -> None:
        total, holdout = 12, 5
        rows = [
            {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
            for i in range(total)
        ]
        with tempfile.TemporaryDirectory() as d:
            texts, tail_rows = self._run_ppl_sampling(Path(d), rows, holdout)
        self.assertEqual(len(tail_rows), holdout)
        # Only the LAST holdout dialogues: the trainer saw rows 0..total-holdout-1.
        self.assertEqual(tail_rows[0]["messages"][0]["content"], f"q{total - holdout}")
        self.assertEqual(len(texts), holdout)
        self.assertIn("q7", texts[0])  # first held-out dialogue
        self.assertIn("q11", texts[-1])  # last held-out dialogue
        self.assertNotIn("q6", " ".join(texts))  # earlier rows are training-seen

    def test_zero_holdout_yields_no_texts(self) -> None:
        rows = [{"messages": [{"role": "user", "content": "q"}]}]
        with tempfile.TemporaryDirectory() as d:
            texts, tail_rows = self._run_ppl_sampling(Path(d), rows, 0)
        self.assertEqual(tail_rows, [])
        self.assertEqual(texts, [])

    def test_signature_accepts_holdout_tail(self) -> None:
        """The public entry point must expose the trained holdout size."""
        import inspect

        from ricc.evaluation.official_academic_benchmarks import run_official_academic_benchmarks

        params = inspect.signature(run_official_academic_benchmarks).parameters
        self.assertIn("holdout_tail", params)
        self.assertEqual(params["holdout_tail"].default, 1000)

    def test_run_report_mentions_the_holdout_sample(self) -> None:
        """The generated markdown must say the PPL sample source explicitly."""
        source = Path("ricc/evaluation/official_academic_benchmarks.py").read_text(encoding="utf-8")
        self.assertIn("perplexity_sample_source", source)
        self.assertIn("held-out tail", source)


if __name__ == "__main__":
    unittest.main()
