# Release QC notes — LoRA smollm2_135m (full epoch, fresh GPU holdout)

**Prepared from actual artifacts on 2026-11-19.** No independent external expert
review is claimed. Files to include in a maintained release:

- full final repo path: `lora_adapters/smollm2_135m_instruct/`;
- provenance: this directory's `training_meta.json`;
- benchmarks: [matrix](../reports/academic_scientific_benchmarks_matrix.json)
  and [raw generations](../reports/academic_benchmarks_raw_outputs.jsonl);
- held-out helper report: [report](../reports/benchmark_smollm2_135m_instruct.json);
- offline package: [`dist/ricc-adapter-smollm2_135m_instruct-provenance.zip`](../dist/ricc-adapter-smollm2_135m_instruct-provenance.zip)
  (includes the final .safetensors weights, tokenizer, model card,
  benchmark matrix, raw generation logs and handwritten QC notes);
- artifact hashes: [adapter digest](adapter_release_digest.json) and
  [archive manifest](../dist/RELEASE_MANIFEST.json).

## Verified technical contract

- `actual_steps == required_steps_for_full_epoch == 21209`;
- `corpus_coverage_pct == 100.0`, `train_dialogues = 169667`, `dataset_total_dialogues = 171520`;
- `holdout_tail_dialogues = 1000`, `eval_frac = 0.005`, `seed = 42`, `time_budget_hours = null`;
- no training-epoch cap and no stale checkpoint resume;
- corpus cover and gate verified by `policy_gate.assert_publishable`;
- `10 checkpoint-* dirs preserved locally` (checkpoints do not enter release archive),
  model.safetensors hash pinned in adapter card and release digest list.

## Fresh GPU evaluation

`ricc.evaluation.official_academic_benchmarks` fixed harness ran all four variants of:
40 HumanEval questions and 50 RuMMLU questions. Raw outputs record 364 records
(160 HumanEval, 200 RuMMLU, 2 PPL, 2 ROUGE).

| metric | Base | LoRA | hybrid |
|---|---:|---:|---:|
| HumanEval pass@1 | 10.0% | 2.5% | 0.0% |
| RuMMLU | 28.0% | 0.0% | 0.0% |
| Domain perplexity (subset) | 16.27 | 12.89 | not measured |

Caution: HumanEval (N=40) and RuMMLU (N=50) do **not** support production
capability claims. The RuMMLU gap is adverse to the adapter (base advantage of
28 p.p. with McNemar p≈0.0001). LoRA and hybrid failed all 50 questions;
their short free response answers on the held-out helper report should **not**
be presented as a final chat assistant.

- Held-out full-sequence helper metric: 100 dialogues/22,647 tokens, PPL **13.9678→7.2630**.
- This is a DIFFERENT measurement from academic subset: same reserved 1000-dialogue
  tail at run_lora_benchmark first-100 prefix vs the academic 30-text parser.
- Rigged harness comparisons tests are not enough for statistically valid
  claims beyond this sample; the inherent no-treatment baseline/context choice
  is left in the report and raw outputs rather than being sanitized.

## Sandbox and review boundaries

- at fresh run, explicitly opted-in WSL Ubuntu Docker daemon provided container
  mode, readonly fs, no network and no capabilities and a lower uid.
- manual controls exercised: correct program true, assert fail and escape false.
- no host Docker fallback used for published fresh scores.
- no external expert review is represented.
- reproducibility checks inspect all equality with raw tasks per variant and
  recompute significance from measured outcomes, not from a hardcoded flag.

## Dependency confidence

- setuptools 84.2.1, rich==15.0.0, watchdog==7.1.0 are verified and lock-file refreshed;
- streamlit 1.67 UI AppTest passed under Windows; CI now runs it every lane too;
- candidate Python 3.15 image built successfully with app healthy check,
  1009 tests passed (the one other failure is a host `.git`-only ignore gate
  when mounted without repository state; 93 focused tests also pass);
- `python -m build` creates the wheel and sdist; source extracted wheel check,
  native ruff, strict mypy, registry/cards all pass;
- `trl >=1.0` STILL must not be merged automatically from dependabot: this is an API
  decision, not a working training upgrade. The production trainer uses
  Transformers Trainer, not TRL. PEFT and torch current releases are not a
  reason to re-run a completed canonical epoch arbitrarily;
- #36 stays unmerged because pins the Ruff in pyproject would disagree with precommit/CI
  (additional Transformers/Datasets/PEFT combination not checked as training guarantee);
- #35 no longer automatic refuse: a real image/test exists, but CI 3.14 matrix
  and required release version bump are needed before production adoption/page.

## What still needs a user-supplied secret

HF upload must wait for a validated `HF_TOKEN`. This package strictly does not
claim current Hub weights to be fresh. Public hub upload may start when read/write
access is proven under that credential via `HfApi.whoami()`.
