"""Training policy gate — the single source of truth for "may this adapter be published?".

The binding training policy (AGENTS.md) requires that a *real* training run
covers the full corpus for at least one complete epoch. Until this module
existed, nothing enforced it: adapters were uploaded to the Hugging Face Hub,
written into the registry, and shipped in model cards regardless of how much
of the corpus they had actually seen. Seven published adapters carried
``corpus_coverage_pct`` between 16.53 % and 44.62 %.

This module provides:

* :func:`effective_batch_size` — the trainer's real effective batch size;
* :func:`required_steps_for_epoch` — steps needed for one full epoch;
* :func:`compute_coverage` — coverage measured against the ACTUAL training
  set (not against a denominator that includes the held-out tail);
* :func:`evaluate_compliance` — the verdict for one adapter directory;
* :func:`assert_publishable` — raises :class:`PolicyViolation` when an
  adapter does not satisfy the policy.

Two defects this module also fixes:

1. **The coverage denominator was wrong.** The old formula was
   ``actual_steps * 8 / dataset_total`` where ``dataset_total`` included the
   1 000-dialogue held-out tail that was never trained on, so a run that
   covered 101 % of the real training set reported "100 %" while a run that
   covered exactly one real epoch reported "98.93 %". Coverage is now
   computed against the number of dialogues actually fed to the trainer.

2. **The policy's own step count was unsatisfiable.** Requiring
   ``actual_steps == ceil(total_dialogues / 8)`` while simultaneously
   subtracting the holdout and the eval split is contradictory: one real
   epoch over the trainable set is 21 420 steps, not 21 652. The canonical
   number is now DERIVED from the dataset instead of hardcoded, and
   :func:`canonical_full_run` returns the exact command to run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The trainer's effective batch size: per_device_train_batch_size (2) x
# gradient_accumulation_steps (4). Kept here so the policy and the code
# cannot drift apart silently.
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SIZE = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

# Coverage threshold required by the binding policy.
REQUIRED_COVERAGE_PCT = 100.0

# The canonical held-out tail reserved for honest benchmarks.
CANONICAL_HOLDOUT_TAIL = 1000


class PolicyViolation(RuntimeError):
    """Raised when an adapter may not be published under the binding policy."""


@dataclass
class ComplianceReport:
    """Verdict for one adapter directory against the training policy."""

    adapter_id: str
    compliant: bool
    reasons: list[str] = field(default_factory=list)
    actual_steps: int | None = None
    required_steps: int | None = None
    coverage_pct: float | None = None
    train_dialogues: int | None = None
    has_meta: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "compliant": self.compliant,
            "reasons": list(self.reasons),
            "actual_steps": self.actual_steps,
            "required_steps": self.required_steps,
            "corpus_coverage_pct": self.coverage_pct,
            "train_dialogues": self.train_dialogues,
            "has_training_meta": self.has_meta,
        }


def effective_batch_size() -> int:
    """The trainer's effective batch size (dialogues consumed per step)."""
    return EFFECTIVE_BATCH_SIZE


def required_steps_for_epoch(train_dialogues: int) -> int:
    """Optimizer steps needed to cover ``train_dialogues`` once."""
    if train_dialogues <= 0:
        return 0
    return math.ceil(train_dialogues / EFFECTIVE_BATCH_SIZE)


def trainable_dialogues(
    total_dialogues: int,
    holdout_tail: int = 0,
    eval_frac: float = 0.0,
) -> int:
    """Number of dialogues actually fed to the trainer.

    Mirrors the order of operations in ``batch_train_zoo.py``:
    the held-out tail is removed first, then the eval split is carved out
    of what remains, then the ``max_samples`` cap is applied.
    """
    remaining = total_dialogues
    if holdout_tail > 0 and remaining > holdout_tail:
        remaining -= holdout_tail
    if eval_frac > 0:
        remaining = int(remaining * (1.0 - eval_frac))
    return max(0, remaining)


def compute_coverage(actual_steps: int, train_dialogues: int) -> float:
    """Percentage of the ACTUAL training set covered by ``actual_steps``.

    The denominator is the number of dialogues the trainer saw, not the
    corpus size including the held-out tail.
    """
    if train_dialogues <= 0:
        return 0.0
    seen = actual_steps * EFFECTIVE_BATCH_SIZE
    return round(min(100.0, seen / train_dialogues * 100.0), 2)


def evaluate_compliance(
    adapter_dir: str | Path,
    *,
    required_coverage_pct: float = REQUIRED_COVERAGE_PCT,
) -> ComplianceReport:
    """Assess one adapter directory against the binding training policy."""
    path = Path(adapter_dir)
    adapter_id = path.name
    report = ComplianceReport(adapter_id=adapter_id, compliant=False)

    meta_path = path / "training_meta.json"
    if not meta_path.exists():
        report.reasons.append(
            "no training_meta.json — the adapter has no recorded training provenance "
            "(legacy/pilot checkpoint, never a full-corpus run)"
        )
        return report

    report.has_meta = True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.reasons.append(f"training_meta.json is unreadable: {exc}")
        return report

    actual_steps = meta.get("actual_steps")
    if not isinstance(actual_steps, int):
        report.reasons.append("training_meta.json has no integer 'actual_steps'")
        return report
    report.actual_steps = actual_steps

    # Prefer the explicitly recorded trainable size; fall back to the
    # corpus size minus the holdout.
    train_dialogues = meta.get("train_dialogues")
    if not isinstance(train_dialogues, int) or train_dialogues <= 0:
        total = meta.get("dataset_total_dialogues")
        holdout = meta.get("holdout_tail_dialogues")
        eval_frac = meta.get("eval_frac")
        if isinstance(total, int):
            train_dialogues = trainable_dialogues(
                total,
                holdout_tail=holdout if isinstance(holdout, int) else 0,
                eval_frac=eval_frac if isinstance(eval_frac, (int, float)) else 0.0,
            )
    if not isinstance(train_dialogues, int) or train_dialogues <= 0:
        report.reasons.append("cannot determine the number of trainable dialogues")
        return report
    report.train_dialogues = train_dialogues

    required = required_steps_for_epoch(train_dialogues)
    report.required_steps = required

    coverage = compute_coverage(actual_steps, train_dialogues)
    report.coverage_pct = coverage

    if actual_steps < required:
        report.reasons.append(
            f"partial epoch: {actual_steps} steps < {required} required for one full epoch "
            f"over {train_dialogues} trainable dialogues"
        )
    if coverage < required_coverage_pct:
        report.reasons.append(f"corpus coverage {coverage:.2f}% < {required_coverage_pct:.2f}% required")

    # A wall-clock budget means the run was stopped early by definition.
    time_budget = meta.get("time_budget_hours")
    if time_budget:
        report.reasons.append(
            f"run was time-boxed (time_budget_hours={time_budget}); a time-boxed run is "
            "never a final adapter under the binding policy"
        )

    report.compliant = not report.reasons
    return report


def assert_publishable(adapter_dir: str | Path, *, required_coverage_pct: float = REQUIRED_COVERAGE_PCT) -> None:
    """Raise :class:`PolicyViolation` unless the adapter satisfies the policy.

    Call this immediately before uploading an adapter, adding it to the
    registry, or writing its model card.
    """
    report = evaluate_compliance(adapter_dir, required_coverage_pct=required_coverage_pct)
    if not report.compliant:
        detail = "; ".join(report.reasons)
        raise PolicyViolation(
            f"adapter '{report.adapter_id}' does not satisfy the training policy: {detail}. "
            "Partial or time-boxed runs must not be published as final adapters."
        )


def is_publishable(adapter_dir: str | Path, *, required_coverage_pct: float = REQUIRED_COVERAGE_PCT) -> bool:
    """Non-raising variant of :func:`assert_publishable`."""
    return evaluate_compliance(adapter_dir, required_coverage_pct=required_coverage_pct).compliant


def canonical_full_run(
    total_dialogues: int,
    *,
    holdout_tail: int = CANONICAL_HOLDOUT_TAIL,
    eval_frac: float = 0.005,
    adapter_id: str = "<id>",
) -> dict[str, Any]:
    """The canonical full-corpus run for the current corpus size.

    The step count is DERIVED from the dataset (one full epoch over the
    trainable set) instead of being hardcoded, so it stays correct when the
    corpus grows.
    """
    trainable = trainable_dialogues(total_dialogues, holdout_tail=holdout_tail, eval_frac=eval_frac)
    steps = required_steps_for_epoch(trainable)
    return {
        "adapter_id": adapter_id,
        "total_dialogues": total_dialogues,
        "holdout_tail": holdout_tail,
        "eval_frac": eval_frac,
        "trainable_dialogues": trainable,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "required_steps": steps,
        "samples": 200_000,
        "command": (
            f"python -m ricc.lora.batch_train_zoo --only {adapter_id} --profile full "
            f"--samples 200000 --steps {steps} --eval-frac {eval_frac} "
            f"--holdout-tail {holdout_tail} --seed 42 --force --keep-checkpoints --no-upload"
        ),
    }
