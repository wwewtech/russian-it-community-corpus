# Verification and remaining limitations

## Verification updates (2026-09-17)

- **Process-isolated HumanEval & Academic Benchmark Execution on GPU**: `src/evaluation/official_academic_benchmarks.py` was executed on local GPU (`NVIDIA GeForce RTX 3060 12GB`). Evaluated Qwen 2.5 1.5B with `qwen2.5_1.5b_instruct` LoRA adapter and local RAG. HumanEval (40 tasks), RuMMLU CS (50 questions), PPL (held-out test set), and ROUGE-1/L were physically calculated. All 364 raw model generations and test outputs were saved to `reports/academic_benchmarks_raw_outputs.jsonl`. Summary matrix saved to `reports/academic_scientific_benchmarks_matrix.json`, and markdown report rendered to `reports/OFFICIAL_ACADEMIC_SCIENTIFIC_BENCHMARKS.md`.
- **Hub dataset exact synchronization**: Pinned Hub revision `81a3495de997b0690f67d175c4ebd9cfdef35b76` downloaded and verified. Local Parquet files (`full_clean_messages.parquet`: 2,816,434 rows; `sft_dialogues.parquet`: 171,520 rows; `rag_knowledge_base.parquet`: 325,690 rows) are identical byte-for-byte and hash-for-hash (SHA-256) to published Hub artifacts. `reports/dataset_manifest.json` updated and verified; `reports/dataset_reconciliation_report.json` achieved `EXACT_MATCH` (delta: 0 rows, 0 bytes). SLO release gate evaluates to `SHIP`.
- **mypy strict typing scope expanded**: `src/inference.py`, `src/pipeline.py`, and `src/exporter/finalize_sync_all.py` refactored and added to `make typecheck-strict`. The strict gate now checks 30 modules with 0 errors under `--strict`. `pyproject.toml` overrides narrowed strictly to `src.evaluation.*` (evaluate/transformers stubs) and `src.lora.*` (external training callbacks).
- **Windows temp directory lock resolved**: Added `--basetemp=.pytest_temp` to pytest options in `pyproject.toml`, resolving OS `PermissionError` on `C:\Users\pasha\AppData\Local\Temp\pytest-of-pasha`. Full test suite passes: 474 passed, 4 skipped, 92.11% coverage.

## Local verification summary (2026-09-17)

- Full unit test suite: 474 passed, 4 skipped; coverage 92.11% (threshold >= 85%), exit 0.
- Ruff passed on all Python files (`ruff check .`, `ruff format --check .` — 188 files formatted).
- Strict type checking passed on 30 modules (`make typecheck-strict`).
- SLO release gate: verdict `SHIP` with all 6 checks passing (`validation`, `pii-audit`, `drift`, `pipeline-volume`, `artifact-manifest`, `hub-reconciliation`).
- Reconciliation status: `EXACT_MATCH` against pinned Hub revision `81a3495d`.

## Limitations / release follow-up

1. **Model Zoo Training Scope**: Published LoRA adapters in `lora_adapters/` are pilot adapters (50–100 steps on dialogue samples) designed to verify architecture and PEFT pipeline compatibility on consumer GPUs. Full multi-epoch training to convergence on the entire 171,520 dialogues dataset (~85M tokens) requires dedicated distributed GPU infrastructure (~4,100 GPU-hours on single RTX 3060 or ~60–80 hours on 8x A100 cluster) and is documented in `reports/LORA_MODEL_ZOO.md`.
2. **Typing Coverage Boundary**: 30 modules are strictly typed with `--strict` in CI. Remaining exclusions are restricted to third-party library boundaries without official complete stubs (`src/lora/*` for TRL SFTTrainer/peft callbacks, and `src/evaluation/*` for Hugging Face `evaluate` / transformers).
3. **Docker build execution was not verified locally**. The audit found root .env was not excluded from the build context; .env and .env.* exclusions and regression tests were added. Rotate credentials if an affected image was historically distributed.

These checks verify maintainability, execution safety, and artifact integrity; they are not a certification of production readiness or a guarantee of zero PII.
