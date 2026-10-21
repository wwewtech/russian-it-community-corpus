"""tests/test_batch_train_zoo.py — budget tiers and run-report helpers.

Pins the honest-budget plumbing of the LoRA zoo: the ``param_tier`` mapping for
every size label used by ``ZOO_44_MODELS``, the per-tier full-profile budgets,
the wall-clock resume/holdout knobs, and the crash-safe run-report writer that
must never touch the canonical catalog.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ricc.lora.batch_train_zoo import (  # noqa: E402
    FULL_PROFILE_EVAL_FRAC,
    FULL_TIER_BUDGETS,
    PILOT_BUDGET,
    ZOO_44_MODELS,
    _fmt_metric,
    detect_target_modules,
    param_tier,
    update_run_report,
)


class TestParamTier(unittest.TestCase):
    def test_label_mapping(self):
        cases = {
            "125M": "tiny",
            "135M": "tiny",
            "0.5B": "tiny",
            "1.0B": "small",
            "1.2B": "small",
            "1.3B": "mid",
            "1.5B": "mid",
            "2.3B": "mid",
            "2.7B": "large",
            "3.8B": "large",
            "7.6B": "large",
        }
        for label, expected in cases.items():
            self.assertEqual(param_tier(label), expected, f"{label} should map to {expected}")

    def test_unparseable_label_falls_back_to_small(self):
        self.assertEqual(param_tier("unknown"), "small")

    def test_every_zoo_label_maps_to_a_budgeted_tier(self):
        for model in ZOO_44_MODELS:
            with self.subTest(adapter=model["id"]):
                self.assertIn(param_tier(model["params"]), FULL_TIER_BUDGETS)


class TestBudgets(unittest.TestCase):
    def test_pilot_budget_is_a_smoke_budget(self):
        self.assertEqual(PILOT_BUDGET["steps"], 15)
        self.assertEqual(PILOT_BUDGET["max_samples"], 400)

    def test_full_budgets_exceed_the_pilot_budget(self):
        """The full profile must not be smaller than the smoke profile.

        The step count is no longer a hardcoded tier table (it is derived
        from the dataset size at runtime), so this asserts on the sample
        budget and on the derived full-epoch step count.
        """
        for tier, budget in FULL_TIER_BUDGETS.items():
            with self.subTest(tier=tier):
                self.assertGreater(budget["max_samples"], PILOT_BUDGET["max_samples"])

    def test_full_profile_step_count_is_derived_from_the_dataset(self):
        """One full epoch over the real corpus must be far beyond the pilot budget.

        The old tier budgets (600-2000 steps) covered 2.77-9.24 % of the
        corpus and could never satisfy the binding training policy.
        """
        from ricc.lora.policy_gate import required_steps_for_epoch, trainable_dialogues

        trainable = trainable_dialogues(173216, holdout_tail=1000, eval_frac=0.005)
        steps = required_steps_for_epoch(trainable)
        self.assertEqual(steps, 21420)
        self.assertGreater(steps, PILOT_BUDGET["steps"] * 100)

    def test_eval_frac_is_a_small_holdout(self):
        self.assertGreater(FULL_PROFILE_EVAL_FRAC, 0.0)
        self.assertLess(FULL_PROFILE_EVAL_FRAC, 0.1)


class _FakeSiLUActivation(torch.nn.Module):
    """Mirrors transformers' SiLUActivation wired as ``...activation_fn``."""

    def __init__(self) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)


class _FakePhiBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = torch.nn.Linear(8, 24)
        self.o_proj = torch.nn.Linear(8, 8)
        self.activation_fn = _FakeSiLUActivation()


class Conv1D(torch.nn.Module):
    """Stand-in for transformers.pytorch_utils.Conv1D (GPT-2 style attention)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


class _FakeGpt2Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_attn = Conv1D()


class TestDetectTargetModules(unittest.TestCase):
    def test_skips_non_linear_activation_modules(self):
        # Regression: the bare "v" target used to substring-match
        # "...activation_fn" (a SiLUActivation) and crash PEFT with
        # "Target module SiLUActivation() is not supported" (Phi-3.5).
        targets = detect_target_modules(_FakePhiBlock())
        self.assertNotIn("activation_fn", targets)
        self.assertIn("qkv_proj", targets)
        self.assertIn("o_proj", targets)

    def test_conv1d_style_attention_is_still_detected(self):
        targets = detect_target_modules(_FakeGpt2Block())
        self.assertEqual(targets, ["c_attn"])

    def test_falls_back_to_default_projections(self):
        targets = detect_target_modules(torch.nn.Sequential(torch.nn.Linear(4, 4)))
        self.assertEqual(targets, ["q_proj", "v_proj"])


class TestFmtMetric(unittest.TestCase):
    def test_formats_numeric_values(self):
        self.assertEqual(_fmt_metric({"loss": 1.23456}, "loss", "{:.4f}"), "1.2346")

    def test_missing_or_non_numeric_values_render_as_dash(self):
        self.assertEqual(_fmt_metric({}, "loss", "{:.4f}"), "—")
        self.assertEqual(_fmt_metric({"loss": "n/a"}, "loss", "{:.4f}"), "—")
        self.assertEqual(_fmt_metric({"loss": None}, "loss", "{:.4f}"), "—")


class TestUpdateRunReport(unittest.TestCase):
    def _result(self) -> dict:
        return {
            "id": "demo_model",
            "model_name": "demo/model",
            "status": "SUCCESS",
            "train_loss": 1.5,
            "eval_loss": None,
            "steps": 7,
            "samples_used": 400,
            "training_time_sec": 12.0,
            "tokens_per_sec": 100.0,
        }

    def test_writes_run_report_without_touching_the_canonical_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            catalog = report_dir / "lora_zoo_index.json"
            catalog.write_text('{"canonical": true}\n', encoding="utf-8")

            update_run_report([self._result()], report_dir=report_dir)

            self.assertEqual(catalog.read_text(encoding="utf-8"), '{"canonical": true}\n')
            run_json = report_dir / "lora_zoo_training_run.json"
            self.assertTrue(run_json.exists())
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            self.assertIn("demo_model", payload["models"])
            self.assertEqual(payload["models"]["demo_model"]["steps"], 7)
            md = (report_dir / "LORA_ZOO_TRAINING_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("demo_model", md)
            self.assertIn("1.5000", md)

    def test_report_merges_previous_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            first = self._result()
            second = dict(first, id="other_model")
            update_run_report([first], report_dir=report_dir)
            update_run_report([second], report_dir=report_dir)
            payload = json.loads((report_dir / "lora_zoo_training_run.json").read_text(encoding="utf-8"))
            self.assertEqual(set(payload["models"]), {"demo_model", "other_model"})

    def test_corrupt_existing_report_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            (report_dir / "lora_zoo_training_run.json").write_text("{not json", encoding="utf-8")
            update_run_report([self._result()], report_dir=report_dir)
            payload = json.loads((report_dir / "lora_zoo_training_run.json").read_text(encoding="utf-8"))
            self.assertIn("demo_model", payload["models"])


if __name__ == "__main__":
    unittest.main()
