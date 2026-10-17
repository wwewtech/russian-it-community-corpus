# LoRA Training

`src/lora/` fine-tunes domain adapters on the RICC corpus, sized for a
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
python -m src.lora.train_lora --model Qwen/Qwen2.5-0.5B-Instruct --steps 30
```

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
release workflow via `src.exporter.finalize_sync_all`.

## Provenance & Retraining (prompt-masking rollout)

An adapter carries `training_meta.json` (with `prompt_masking: true`,
final loss, steps, seed, LoRA config) only if it was trained with the
fixed prompt-masked objective. Adapters **without** that file predate the
2026-10-17 masking fix — their loss included user-prompt tokens — and
should be retrained:

```bash
# validate the fixed objective on one adapter, without touching the repo
python -m src.lora.batch_train_zoo --only qwen2.5_1.5b_instruct \
    --steps 15 --output-dir /tmp/lora_smoke --no-upload --no-catalogs

# full zoo rollout: retrain every existing adapter in place
python -m src.lora.batch_train_zoo --force --steps 15
```

Useful `batch_train_zoo.py` flags: `--only <ids>` (comma-separated filter),
`--output-dir`, `--no-upload`, `--no-catalogs`, `--force` (retrain adapters
that already exist).
