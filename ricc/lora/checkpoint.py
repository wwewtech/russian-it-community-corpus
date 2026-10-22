"""Checkpoint and wall-clock training helpers."""

import shutil
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


class TimeLimitCallback(TrainerCallback):
    """Stop training gracefully once a wall-clock budget is exhausted."""

    def __init__(self, budget_sec: float) -> None:
        self.budget_sec = budget_sec
        self.started_at: float | None = None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.started_at = time.monotonic()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.started_at is not None and time.monotonic() - self.started_at >= self.budget_sec:
            control.should_training_stop = True


def latest_checkpoint(out_dir: Path) -> Path | None:
    """Return the newest numeric checkpoint directory, if one exists."""
    checkpoints = sorted(
        (p for p in out_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else -1,
    )
    return checkpoints[-1] if checkpoints else None


def clear_non_checkpoint_files(out_dir: Path) -> None:
    """Remove adapter artifacts while preserving checkpoint directories for resume."""
    checkpoints = {p for p in out_dir.glob("checkpoint-*") if p.is_dir()}
    for stale in out_dir.iterdir():
        if stale in checkpoints:
            continue
        if stale.is_dir():
            shutil.rmtree(stale)
        else:
            stale.unlink()


def remove_checkpoints(out_dir: Path) -> None:
    """Delete checkpoint directories after a successful run when not retained."""
    for checkpoint in out_dir.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
