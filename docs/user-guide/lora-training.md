# LoRA Training

`ricc/lora/` fine-tunes domain adapters on the RICC corpus, sized for a
single RTX 3060 (12GB) and scaled up to a multi-model zoo.

## Entry Points

| Script | Purpose |
|---|---|
| `train_lora.py` | single adapter, full-corpus SFT, resumable checkpoints (save every ~20% steps, keep 3) |
| `batch_train_zoo.py` | 44-model zoo: train + auto-upload to `wwewtech/russian-it-community-lora` |
| `train_20_popular_models.py` | 20 popular families incl. 4-bit QLoRA for 7B-class models |
| `train_heavyweight_5_sota.py` | five 7–8B flagships, 4-bit NF4 + gradient checkpointing |
| `generate_demo.py` | inference demo against a trained adapter |

```bash
python -m ricc.lora.train_lora --model Qwen/Qwen2.5-0.5B-Instruct --steps 30
```

> **The training policy is binding.** `AGENTS.md` requires a real run to
> cover the **whole corpus for at least one complete epoch**. The
> `policy_gate` decides what may be published; see
> [ADR 0005](../adr/0005-training-policy-gate.md). A partial or
> time-boxed run is labelled `PARTIAL` and is never uploaded.

## Training Details

- **Prompt-loss masking**: `-100` over prompt tokens so CrossEntropyLoss
  covers only assistant responses (all loaders).
- **VRAM**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set by
  `setup_runtime_env(pytorch_alloc_conf=True)`; fp16 on CUDA, fp32
  fallback on CPU.
- **Full corpus**: `train_lora` trains on the whole JSONL (a silent
  1000-row truncation was removed — see module docstring); use
  `--max-samples` for smoke runs.
- **LoRA config**: r=16/α=32 for single runs, r=8/α=16 for the 20-model
  batch; target modules auto-detected (`detect_target_modules` handles
  GPT2/OPT/Falcon naming variants).

## What may be published

`ricc/lora/policy_gate.py` is the single source of truth:

| Requirement | Why |
|---|---|
| a `training_meta.json` exists | without it there is no record of what was trained |
| `actual_steps` ≥ one full epoch of the **trainable** set | the corpus is covered, not a sample of it |
| the run was not time-boxed | a `--hours` budget stops the run early by definition |
| the coverage is measured against the trainable set | the held-out tail is never trained on |

The gate runs in the trainer (before the Hub upload), in the release
pipeline (`finalize_sync_all` skips non-compliant adapters) and in the
registry generator (which records `policy_compliant` and the coverage).
CI publishes the state of the whole zoo as
`reports/lora_policy_compliance.json`.

**The currently published zoo is entirely non-compliant** (0 of 56): the
seven adapters with metadata cover 16.5–44.6 % of the corpus and the
other 49 are pilot checkpoints without provenance. Retraining them is a
GPU job (~25 h for a 3B model, ~13 h for 1.5B, ~5–7 h below 1B on the
RTX 3060), not a code change.

The canonical command for one adapter is derived from the dataset, not
written down by hand:

```bash
python -m ricc.lora.batch_train_zoo --only <adapter_id> --profile full \
    --samples 200000 --steps 21420 --eval-frac 0.005 --holdout-tail 1000 \
    --seed 42 --force --keep-checkpoints --no-upload
```

`21420` is `ceil(171354 / 8)` — one epoch over the trainable set of the
current corpus. `python -c "from ricc.lora.policy_gate import
canonical_full_run; print(canonical_full_run(173216))"` prints the current
value.

## Observability (12.2.0+)

Every run records its outcome:

- success → `record_lora_result()` with duration, final train loss, steps
  and peak VRAM (`ricc_lora_train_loss`, `ricc_lora_vram_usage_bytes`,
  `ricc_lora_train_duration_seconds`, `ricc_lora_train_total{status="success"}`).
- failure → the exception is classified via the
  [GPU/ML taxonomy](../api/errors.md)
  (`record_classified_ml_error`) and recorded with
  `ricc_lora_train_total{status="failure"}`; OOM vs adapter corruption is
  visible in `ricc_errors_total{component}`.

Grafana panels "Training loss by adapter" and "VRAM usage by adapter"
plot the gauges; the `RiccLoraTrainFailing` alert fires on failures.

## Model Zoo

`reports/LORA_MODEL_ZOO.md` and `reports/lora_zoo_index.json` catalog the
trained adapters; `make reports` regenerates the catalog from local +
Hub state. Release-side sync (finalize + upload + catalog) runs in the
release workflow via `ricc.exporter.finalize_sync_all`. The registry
carries the training provenance, so a policy-compliant adapter is
distinguishable from a legacy one.

## Provenance & Retraining (prompt-masking rollout)

An adapter carries `training_meta.json` (with `prompt_masking: true`,
final loss, steps, seed, LoRA config) only if it was trained with the
fixed prompt-masked objective. Adapters **without** that file predate the
2026-10-17 masking fix — their loss included user-prompt tokens — and
should be retrained:

```bash
# validate the fixed objective on one adapter, without touching the repo
python -m ricc.lora.batch_train_zoo --only qwen2.5_1.5b_instruct \
    --steps 15 --output-dir /tmp/lora_smoke --no-upload --no-report

# full zoo rollout: retrain every existing adapter in place
python -m ricc.lora.batch_train_zoo --force --steps 15
```

Useful `batch_train_zoo.py` flags: `--only <ids>` (comma-separated filter),
`--output-dir`, `--no-upload`, `--no-report` (legacy alias `--no-catalogs`) to
skip the run report, `--force` (retrain adapters that already exist).

## Full-corpus & night runs

`batch_train_zoo.py --profile full` runs one **complete epoch** over the
whole corpus: the step count is derived from the dataset size, the sample
cap is the corpus, there is a 2% eval holdout on top of the 1 000-dialogue
held-out tail, step-level `training_log.jsonl`, automatic checkpoints with
resume from `checkpoint-*`, and a v2 `training_meta.json`
(`actual_steps`, `required_steps_for_full_epoch`, `corpus_coverage_pct`,
`train_dialogues`, `dataset_sha256`, tok/s, peak VRAM). The aggregate run
report goes to `reports/lora_zoo_training_run.json` +
`reports/LORA_ZOO_TRAINING_REPORT.md` and never overwrites the canonical
catalog.

For long unattended runs use wall-clock budgets. `--hours N` stops training
gracefully after N hours (saving the adapter, and with `--keep-checkpoints`
also resumable checkpoints); `--holdout-tail N` excludes the last N
dialogues from training so the same dialogues stay clean for honest
post-training benchmarks. **A time-boxed run is a partial run**: it is
recorded as `PARTIAL` and is not published.

```bash
python -m ricc.lora.batch_train_zoo --only phi_3.5_mini_instruct \
    --profile full --hours 5 --holdout-tail 1000 --keep-checkpoints --no-upload
```

`scripts/run_lora_benchmark.py` then measures base vs adapter on that tail
(full-sequence perplexity + greedy generation snapshots; add
`--lm-eval-tasks humaneval` after `pip install lm-eval`):

```bash
python scripts/run_lora_benchmark.py --adapter lora_adapters/phi_3.5_mini_instruct
```

The one-command overnight flow chains time-boxed training for several
models with those benchmarks and writes `reports/LORA_NIGHT_RUN_SUMMARY.md`:

```bash
python scripts/night_run_full_corpus.py --total-hours 10 \
    --models phi_3.5_mini_instruct qwen2.5_1.5b_instruct --split-weights "60,40"
```

A 10-hour night is only a partial epoch on a 3.8B model — each adapter
records the honest fraction of the corpus it actually saw as
`corpus_coverage_pct` in `training_meta.json`, and the policy gate
refuses to publish it.
