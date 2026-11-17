# AGENTS.md

## Training policy (binding)

- When training any model (LoRA zoo), always run the **full training over the
  entire corpus to completion**. Never use wall-clock budgets (`--hours`) or
  mid-epoch stops for real training runs: adapters must be final after the run,
  so this work never has to be repeated.
- Required baseline for every real training run:
  - full SFT corpus (`--samples` covers every dialogue; do not reduce the
    sample pool), keeping only the canonical 1,000-dialogue held-out tail for
    honest benchmarks;
  - at least one **complete epoch**: `actual_steps` must reach
    `ceil(train_dialogues / effective_batch)` (effective batch = 2 x 4 = 8 per
    step) and `corpus_coverage_pct` in `training_meta.json` must read 100;
  - `--keep-checkpoints` and held-out benchmarks after training.
- Canonical full run (no `--hours`, exactly one epoch; effective batch 8):
  `python -m ricc.lora.batch_train_zoo --only <id> --profile full
  --samples 200000 --steps 21420 --eval-frac 0.005 --holdout-tail 1000
  --seed 42 --force --keep-checkpoints --no-upload` where 21420 =
  `ceil((173216 - 1000 holdout) * (1 - 0.005 eval) / 8)` — one full epoch
  over the trainable set (the 1,000-dialogue held-out tail and the eval
  split are never trained on) and 173216 is the current corpus size;
  recompute when the corpus changes, or take the number from
  `ricc.lora.policy_gate.canonical_full_run(total_dialogues)`, which derives
  it from the dataset. Verify `actual_steps == required_steps_for_full_epoch`
  and `corpus_coverage_pct == 100.0` in `training_meta.json`; also check
  that `max_samples` did not cap the pool (`max_samples >= dataset_total -
  holdout_tail`).
- Partial / time-boxed runs are allowed only for smoke tests or when the user
  explicitly asks for them; never present or publish such adapters as final.
  `scripts/night_run_full_corpus.py` is time-boxed by design and therefore
  smoke-only under this policy.
- Expect the real cost: one full epoch is roughly 25 h on a 3B model, ~13 h on
  a 1.5B model and ~5-7 h on sub-1B models on the RTX 3060. Plan around that;
  do not silently shorten it.
