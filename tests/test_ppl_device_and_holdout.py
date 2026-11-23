"""Perplexity must use actual model placement and a genuine held-out tail."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from ricc.evaluation.official_academic_benchmarks import _compute_perplexity, _load_heldout_ppl_texts


class TestPerplexityEvaluation(unittest.TestCase):
    def test_missing_tail_is_an_error_not_training_evaluation(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(RuntimeError, "refusing to sample training-seen text"),
        ):
            _load_heldout_ppl_texts(Path(directory) / "absent.jsonl", 1000)

    def test_zero_tail_is_an_error(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(RuntimeError, "positive holdout_tail"),
        ):
            _load_heldout_ppl_texts(Path(directory) / "absent.jsonl", 0)

    def test_incomplete_tail_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sft.jsonl"
            source.write_text(json.dumps({"messages": []}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "requested 1000"):
                _load_heldout_ppl_texts(source, 1000)

    def test_complete_tail_is_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sft.jsonl"
            source.write_text(
                "\n".join(json.dumps({"messages": [{"role": "user", "content": str(index)}]}) for index in range(4)),
                encoding="utf-8",
            )
            texts, provenance = _load_heldout_ppl_texts(source, 2, max_texts=2)
            self.assertEqual(len(texts), 2)
            self.assertIn("held-out tail", provenance)

    def test_benchmark_loads_models_and_logs_provenance(self):
        from ricc.evaluation.official_academic_benchmarks import run_official_academic_benchmarks

        with (
            patch("ricc.evaluation.official_academic_benchmarks.set_deterministic_seed"),
            patch("ricc.evaluation.official_academic_benchmarks._docker_available", return_value=True),
            patch("ricc.evaluation.official_academic_benchmarks.evaluate.load"),
            patch("ricc.evaluation.official_academic_benchmarks.LocalRAGPipeline"),
            patch(
                "ricc.evaluation.official_academic_benchmarks.load_evaluation_models",
                side_effect=RuntimeError("missing adapter"),
            ),
            self.assertRaisesRegex(RuntimeError, "missing adapter"),
        ):
            run_official_academic_benchmarks()

    def test_perplexity_uses_sharded_model_device(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}
        model = MagicMock()
        model.hf_device_map = {"layers": "cuda:1"}
        model.return_value = SimpleLoss()
        with patch(
            "ricc.evaluation.official_academic_benchmarks.move_inputs_to_model", wraps=lambda inputs, target: inputs
        ) as move:
            _compute_perplexity(model, tokenizer, ["text"])
        move.assert_called_once()
        self.assertIs(move.call_args.args[1], model)


class SimpleLoss:
    loss = torch.tensor(0.0)


if __name__ == "__main__":
    unittest.main()
