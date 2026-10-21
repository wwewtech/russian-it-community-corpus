"""Tests for the training policy gate (src/lora/policy_gate.py).

The gate is the enforcement point for the binding training policy in
AGENTS.md: a run that covers less than one full epoch of the corpus, or
that was stopped by a wall-clock budget, must never be published as a
final adapter.

Before the gate existed, seven published adapters carried
``corpus_coverage_pct`` between 16.53 % and 44.62 % and were uploaded to
the Hugging Face Hub without any check.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.lora.policy_gate import (
    CANONICAL_HOLDOUT_TAIL,
    EFFECTIVE_BATCH_SIZE,
    PolicyViolation,
    assert_publishable,
    canonical_full_run,
    compute_coverage,
    evaluate_compliance,
    is_publishable,
    required_steps_for_epoch,
    trainable_dialogues,
)


def _write_meta(adapter_dir: Path, **overrides: object) -> None:
    meta: dict[str, object] = {
        "model_id": adapter_dir.name,
        "actual_steps": 21420,
        "train_dialogues": 171354,
        "required_steps_for_full_epoch": 21420,
        "corpus_coverage_pct": 100.0,
        "dataset_total_dialogues": 173216,
        "holdout_tail_dialogues": 1000,
        "eval_frac": 0.005,
        "time_budget_hours": None,
    }
    meta.update(overrides)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "training_meta.json").write_text(json.dumps(meta), encoding="utf-8")


class TestPolicyMath(unittest.TestCase):
    def test_effective_batch_size_is_eight(self):
        self.assertEqual(EFFECTIVE_BATCH_SIZE, 8)

    def test_required_steps_for_epoch(self):
        self.assertEqual(required_steps_for_epoch(171354), math.ceil(171354 / 8))
        self.assertEqual(required_steps_for_epoch(0), 0)

    def test_trainable_dialogues_subtracts_holdout_then_eval(self):
        # 173216 - 1000 holdout = 172216; minus 0.5% eval = 171354.
        self.assertEqual(trainable_dialogues(173216, holdout_tail=1000, eval_frac=0.005), 171354)
        self.assertEqual(trainable_dialogues(173216, holdout_tail=0, eval_frac=0.0), 173216)

    def test_coverage_is_measured_against_the_training_set(self):
        """The old denominator included the held-out tail and was wrong.

        A run that covered exactly one real epoch over 171354 trainable
        dialogues reported 98.93 % under the old formula and 100 % under
        the corrected one.
        """
        steps = required_steps_for_epoch(171354)
        self.assertEqual(compute_coverage(steps, 171354), 100.0)
        # The same run under the old formula (denominator = corpus size).
        old = round(min(100.0, steps * 8 / 173216 * 100.0), 2)
        self.assertEqual(old, 98.93)
        self.assertNotEqual(old, 100.0)

    def test_coverage_of_partial_run(self):
        self.assertEqual(compute_coverage(9662, 171354), 45.11)

    def test_canonical_run_is_derived_not_hardcoded(self):
        info = canonical_full_run(173216)
        self.assertEqual(info["trainable_dialogues"], 171354)
        self.assertEqual(info["required_steps"], 21420)
        self.assertIn("--steps 21420", info["command"])
        # A different corpus size must yield a different step count.
        other = canonical_full_run(200000)
        self.assertNotEqual(other["required_steps"], info["required_steps"])
        self.assertEqual(other["required_steps"], required_steps_for_epoch(other["trainable_dialogues"]))


class TestComplianceEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_adapter_without_metadata_is_not_compliant(self):
        d = self.root / "legacy"
        d.mkdir()
        report = evaluate_compliance(d)
        self.assertFalse(report.compliant)
        self.assertFalse(report.has_meta)
        self.assertIn("training_meta.json", " ".join(report.reasons))

    def test_full_epoch_adapter_is_compliant(self):
        d = self.root / "good"
        _write_meta(d)
        report = evaluate_compliance(d)
        self.assertTrue(report.compliant, report.reasons)
        self.assertTrue(is_publishable(d))

    def test_partial_epoch_adapter_is_not_compliant(self):
        d = self.root / "partial"
        _write_meta(d, actual_steps=9662, corpus_coverage_pct=45.11)
        report = evaluate_compliance(d)
        self.assertFalse(report.compliant)
        self.assertTrue(any("partial epoch" in r for r in report.reasons))
        self.assertTrue(any("corpus coverage" in r for r in report.reasons))

    def test_time_boxed_adapter_is_not_compliant_even_at_full_coverage(self):
        """A wall-clock budget means the run was stopped early by definition."""
        d = self.root / "timeboxed"
        _write_meta(d, time_budget_hours=5.8)
        report = evaluate_compliance(d)
        self.assertFalse(report.compliant)
        self.assertTrue(any("time-boxed" in r for r in report.reasons))

    def test_unreadable_metadata_is_not_compliant(self):
        d = self.root / "broken"
        d.mkdir()
        (d / "training_meta.json").write_text("{not json", encoding="utf-8")
        report = evaluate_compliance(d)
        self.assertFalse(report.compliant)

    def test_coverage_derived_when_train_dialogues_absent(self):
        """Older metadata has no train_dialogues; derive it from the corpus size."""
        d = self.root / "legacy_meta"
        _write_meta(d, train_dialogues=None, actual_steps=21420)
        report = evaluate_compliance(d)
        # 173216 - 1000 holdout - 0.5% eval = 171354 trainable -> 21420 steps needed.
        self.assertEqual(report.train_dialogues, 171354)
        self.assertEqual(report.required_steps, 21420)
        self.assertTrue(report.compliant, report.reasons)


class TestAssertPublishable(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_raises_for_non_compliant_adapter(self):
        d = self.root / "partial"
        _write_meta(d, actual_steps=3579, corpus_coverage_pct=16.71)
        with self.assertRaises(PolicyViolation) as ctx:
            assert_publishable(d)
        self.assertIn("partial", str(ctx.exception).lower())

    def test_does_not_raise_for_compliant_adapter(self):
        d = self.root / "good"
        _write_meta(d)
        assert_publishable(d)


class TestCanonicalHoldout(unittest.TestCase):
    def test_canonical_holdout_is_one_thousand(self):
        self.assertEqual(CANONICAL_HOLDOUT_TAIL, 1000)


if __name__ == "__main__":
    unittest.main()
