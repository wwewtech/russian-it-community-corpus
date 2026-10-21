# ADR 0005: A training policy gate decides what may be published

- **Status**: Accepted
- **Date**: 2026-10-21
- **Deciders**: maintainer
- **Relates to**: `AGENTS.md` (binding training policy), `docs/user-guide/lora-training.md`

## Context

`AGENTS.md` states a binding policy: a real training run must cover the
**whole corpus for at least one complete epoch**, the acceptance criterion
being `actual_steps == ceil(dialogues / 8)` with `corpus_coverage_pct == 100`.

Nothing enforced it. The consequences were concrete:

- Seven adapters were published to the Hugging Face Hub with
  `corpus_coverage_pct` between **16.53 %** and **44.62 %**. They were
  time-boxed runs (`--hours 5.8`, `max_steps=1000000`) that stopped early.
- 49 of the 56 published adapters had **no `training_meta.json` at all** —
  legacy 50-step pilot checkpoints, indistinguishable in the registry from a
  properly trained one.
- The `full` profile itself could not satisfy the policy: its step budget
  was a hardcoded tier table (600–2000 steps) covering **2.77–9.24 %** of
  the corpus, and the sample cap (12 000–20 000) truncated the dataset
  before the epoch could complete.
- The coverage figure itself was wrong: it was computed as
  `actual_steps * 8 / dataset_total`, where `dataset_total` **included** the
  1 000-dialogue held-out tail that is never trained on. A run covering
  exactly one real epoch reported 98.93 %, and the policy's own step count
  (21 652) was unsatisfiable alongside the holdout and eval split (one real
  epoch is 21 420 steps).

So the numbers in the model cards were honest, but nothing connected them
to the decision "may this be published?".

## Decision

Introduce `ricc/lora/policy_gate.py` as the single source of truth, and
call it at every point where an adapter can leave the local machine.

1. **The gate** — `evaluate_compliance(adapter_dir)` returns a verdict with
   the reasons. An adapter is compliant when it has a `training_meta.json`,
   covered at least one full epoch of the **trainable** set, was not
   time-boxed, and has no missing provenance. `assert_publishable()` raises
   `PolicyViolation` otherwise.

2. **The step budget is derived, not hardcoded.** The `full` profile computes
   the steps for one epoch from the actual dataset size
   (`required_steps_for_epoch`), so it stays correct when the corpus grows.
   The canonical command is produced by `canonical_full_run()` rather than
   written down in a comment.

3. **Coverage is measured against the trainable set**, and the held-out tail
   is excluded from the denominator.

4. **The gate runs before publication** — in the trainer (before the Hub
   upload), in `finalize_sync_all.upload_missing_adapters` (non-compliant
   adapters are skipped and logged, not uploaded) and in the registry
   generator (which records `policy_compliant` and the coverage).

5. **Partial runs are labelled, not hidden.** A run stopped by a wall-clock
   budget is reported as `PARTIAL` with the reason, so an operator can see
   what happened instead of finding a "SUCCESS" with 16 % coverage.

6. **CI reports the compliance of the whole zoo** as
   `reports/lora_policy_compliance.json`. It does not fail the build on the
   legacy adapters — that would require retraining them, which is a GPU
   job, not a code change — but the state is visible and the release
   pipeline enforces it.

## Consequences

- A non-compliant adapter can no longer reach the Hub through any of the
  three publication paths, and the registry says which adapters are
  publishable.
- The currently published zoo is **entirely non-compliant** (0 of 56). The
  adapters that are already on the Hub stay there until they are retrained;
  this ADR does not retroactively unpublish them, and the model cards state
  the real coverage.
- The canonical full run is expensive by design: ~25 h on a 3B model,
  ~13 h on 1.5B, ~5–7 h below 1B on the RTX 3060. A "fast" run is
  possible, but it is a smoke test and says so.

## Alternatives rejected

- **Keep the policy in `AGENTS.md` only.** It was already there and was
  already ignored. A policy without an enforcement point is a comment.
- **Delete the non-compliant adapters.** They are useful as smoke
  artefacts and as evidence that the pipeline works; hiding them would
  make the problem invisible rather than solved.
- **Publish with a warning.** A published adapter is downloaded and used by
  people; the warning is read by nobody. Refusing to publish is the only
  effective line.
