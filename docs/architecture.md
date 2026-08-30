# Architecture

This document is the engineering map of the Russian IT Community Corpus
(RICC) repository. It complements the [README](../README.md) and the
[operations runbook](operations.md); the goal here is **how the code is
organised**, not how to run it.

The platform is a single-process Python data-engineering + curation +
fine-tuning stack with three surfaces:

1. A CLI orchestrator (`cli.py` / `main.py`) for batch processing.
2. A Streamlit Data Studio (`app.py` + `app_helpers.py`) for human-in-the-loop
   auditing.
3. A local inference + RAG runtime (`src/inference.py`) for evaluating the
   resulting models on a single GPU.

All three surfaces share the same Python packages in `src/`. There is no
microservice split, no message bus, and no separate API service. The
intentional design property is **a single coherent Python package that is
small enough to be audited end-to-end by a single engineer**.

---

## 1. Repository layout

```
.
├── app.py                  # Streamlit Data Studio entrypoint
├── app_helpers.py          # Pure helpers extracted from app.py (testable)
├── cli.py                  # `python cli.py ...` orchestrator
├── main.py                 # Convenience wrapper for `python cli.py`
├── Modelfile               # Ollama import recipe for the flagship model
├── Dockerfile              # Container image for batch + UI runs
├── docker-compose.yml      # Compose profile: api + ui + qdrant
├── pyproject.toml          # Ruff + pytest + coverage configuration
├── dvc.yaml / params.yaml  # DVC pipeline definition (stages 1–7)
│
├── src/                    # Library code (all business logic)
│   ├── pipeline.py         # MasterDataPipeline — the orchestrator
│   ├── bootstrap.py        # Runtime env setup (logging, HF cache, env vars)
│   ├── config.py           # Env-driven config dataclasses
│   ├── inference.py        # Interactive chat + CLI (pure helpers + I/O loop)
│   │
│   ├── ingestion/          # Stage 1: raw export → normalized records
│   ├── pii/                # Stage 2: regex + NER + declension-aware scrub
│   ├── deduplication/      # Stage 3: MinHash LSH + exact-hash dedup
│   ├── taxonomy/           # Stage 4: 8-domain classifier + tagger
│   ├── graph/              # Stage 5: thread DAG reconstruction
│   ├── exporter/           # Stage 6: Parquet + JSONL (ChatML, ShareGPT, Alpaca)
│   ├── rag/                # Stage 7: vector KB construction + LocalRAGPipeline
│   ├── monitoring/         # Quality + drift metrics (post-pipeline)
│   ├── validation/         # PII red-team + benchmark parsing + SFT regression
│   ├── analytics/          # Network / report generation (Streamlit charts)
│   └── orchestration/      # Prefect flow wrapper (optional)
│
├── scripts/                # Operational scripts (sync, regenerate, etc.)
│   ├── sync_to_hub.py      # The ONLY sanctioned HF Hub upload path
│   ├── generate_lora_registry.py
│   └── ...
│
├── lora_adapters/          # 56 LoRA adapters + registry.json (gitignored weights)
├── dataset_output/         # Parquet/JSONL artifacts (gitignored; on HF Hub)
├── reports/                # Markdown/JSON audit reports (committed)
├── tests/                  # 270+ unit tests, one module per src/ submodule
└── docs/                   # This folder
```

---

## 2. The seven-stage data pipeline

The whole batch flow is implemented as a single class —
[`src.pipeline.MasterDataPipeline`](../src/pipeline.py) — that owns seven
collaborator objects, one per stage. The stage order is fixed and linear:

```
   ┌───────────────────────────────────────────────────────────────────┐
   │                       MasterDataPipeline                          │
   │                                                                   │
   │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐       │
   │   │ 1.ingest │──▶│ 2.pii    │──▶│ 3.dedup  │──▶│ 4.taxonomy│      │
   │   └──────────┘   └──────────┘   └──────────┘   └──────────┘       │
   │                                                       │            │
   │                                                       ▼            │
   │   ┌──────────┐   ┌──────────┐   ┌──────────┐                       │
   │   │ 7.monitor│◀──│ 6.export │◀──│ 5.graph  │                       │
   │   └──────────┘   └──────────┘   └──────────┘                       │
   └───────────────────────────────────────────────────────────────────┘
```

| Stage | Module | Responsibility | Key output |
|------:|--------|----------------|------------|
| 1 | `src.ingestion` | Load raw JSON/XML exports from the 11 community nodes, normalise to a single `Message` schema, attach `community_node_id` (anonymised) | `intermediate/normalized.parquet` |
| 2 | `src.pii` | Multi-pass PII scrubbing: regex → Natasha NER → declension-aware substitutions (6 Russian cases). Tech terminology is whitelisted. | `intermediate/scrubbed.parquet` + `reports/pii_validation_report.json` |
| 3 | `src.deduplication` | Exact-hash dedup + MinHash LSH (128 permutations, Jaccard 0.80). 95k+ duplicates / spam removed. | `intermediate/clean_messages.parquet` |
| 4 | `src.taxonomy` | 8-domain classifier (Backend, Frontend, DevOps, …) + tagger. | adds `domain` and `tags` columns |
| 5 | `src.graph` | Reply-tree traversal to reconstruct conversation DAGs. Resolves temporal clusters. | `intermediate/threads.parquet` |
| 6 | `src.exporter` | Emits SFT, DPO, and RAG shards in three formats (Parquet, JSONL-ChatML, JSONL-ShareGPT/Alpaca). | `dataset_output/{parquet,jsonl}/…` |
| 7 | `src.monitoring` | Drift + SFT-quality regression scoring. Writes `reports/sft_quality_report.json`. | quality report |

The pipeline is driven by `dvc.yaml` (for DVC users) and by `cli.py run`
(for everyone else). Both call `MasterDataPipeline.run_all()`.

---

## 3. The orchestrator surface

`src.pipeline.MasterDataPipeline` is intentionally small and **fully unit
tested** (`tests/test_pipeline.py`). It does three things:

1. **Construction** — instantiates one collaborator per stage from a
   shared `PipelineConfig`. No global state, no `os.environ` reads.
2. **Sequencing** — runs the seven stages in the order above. Each
   stage is a single method on the pipeline; an individual stage can be
   re-run in isolation (e.g. `pipeline.run_export()`).
3. **Reporting** — collects per-stage metrics and writes them to
   `reports/pipeline_run.json` for downstream monitoring.

```python
from src.pipeline import MasterDataPipeline, PipelineConfig

config = PipelineConfig.from_env()        # or build one explicitly
pipeline = MasterDataPipeline(config)
pipeline.run_all()                         # all 7 stages
pipeline.run_export()                      # just stage 6
```

Because each stage collaborator is a small object with explicit
dependencies, the whole orchestrator fits on a single screen.

---

## 4. The Streamlit Data Studio

`app.py` is a 367-line Streamlit application that consumes the
artifacts produced by the pipeline. Its panels are:

1. **Overview** — `dataset_output/parquet/full_clean_messages.parquet`
   sample + per-domain distribution chart.
2. **SFT dialogues** — sample multi-turn dialogues and per-message quality
   filter controls.
3. **RAG knowledge base** — interactive top-k retrieval against
   `rag_knowledge_base.parquet` (powered by `src.rag.rag_pipeline.LocalRAGPipeline`).
4. **LoRA Zoo** — browsable list of all 56 adapters from
   `lora_adapters/registry.json`, with per-adapter metrics.
5. **Security & Compliance** — PII red-team report, leak breakdown, and
   adversarial-prompt summary. Renders the verdict produced by
   `RedTeamPIIAuditor` (`src/validation/pii_redteam.py`).
6. **Analytics** — `src.analytics.network` and
   `src.analytics.report_generator` provide the network graph and
   downloadable Markdown reports.

To keep the UI testable, every panel's pure logic is extracted into
`app_helpers.py` (loaders, filters, derived verdicts, summary builders).
`app.py` itself contains only Streamlit-specific glue plus thin
`@st.cache_data` wrappers around those helpers. The helpers are unit
tested in `tests/test_app.py`; an opt-in Streamlit AppTest smoke test
lives in `tests/test_app_streamlit_smoke.py` and is gated on
`RUN_STREAMLIT_SMOKE=1`.

---

## 5. The inference runtime

`src/inference.py` exposes two surfaces:

- **Pure helpers** (`is_exit_command`, `validate_adapter_path`,
  `build_chat_messages`, `build_prompt`, `format_rag_context`) — fully
  unit tested in `tests/test_inference.py`. They are import-safe and
  never touch `transformers` / `peft`, so the test suite can run on a
  CPU-only box.
- **The interactive chat loop** (`interactive_chat_session`) — a thin
  I/O wrapper over `transformers.AutoModelForCausalLM` +
  `peft.PeftModel`. Heavy imports are deferred inside the function so
  that the test suite does not need a GPU. CLI entry point is
  `python -m src.inference`.

The chat session deliberately **does not silently swallow LoRA loading
errors**. If `adapter_config.json` is corrupt or the weights are
missing, `validate_adapter_path` raises `RuntimeError` and the loop
exits. The previous behaviour (downgrade to base model with a misleading
"✅ Attached" log line) was a real source of false confidence and is
regression-tested by `tests/test_inference.py::TestValidateAdapterPath`.

---

## 6. The Hugging Face Hub contract

`wwewtech/russian-it-community-corpus` (dataset) and
`wwewtech/russian-it-community-lora` (LoRA Zoo) are the only two
upstream artifacts produced by this repo. Both are rebuilt from the
local `dataset_output/` and `lora_adapters/` directories by
`scripts/sync_to_hub.py`.

That script is the **only sanctioned upload path**. The contract it
enforces is documented in
[ADR 0001: Hugging Face Token Handling](adr/0001-hf-token-handling.md).
In short:

- The HF token MUST come from `HF_TOKEN` env or
  `~/.huggingface/token` — never from code, `.env`, or CLI flags.
- The script defaults to dry-run; `--apply` is required for a real
  upload.
- It never prints the token value.

`scripts/finalize_sync_all.py` is a higher-level wrapper that pushes
both the dataset and any missing LoRA adapters in one go, and rebuilds
the `LoRA_ZOO.md` index card. It is unit-tested with a mocked
`HfApi` in `tests/test_finalize_sync_all.py`.

---

## 7. Testing strategy

The test suite is the primary contract that keeps the project
maintainable. As of v12.0.3:

- **271 tests, 76.22% line coverage** (project-wide), enforced by
  `pyproject.toml`'s `--cov-fail-under=60` floor.
- **Per-module coverage floors** are not enforced, but the four
  "user-facing entry points" (the modules that the 12.0.2 audit flagged
  as 0% covered) now sit at:
  - `src/inference.py` — 44% (pure helpers 100%; the I/O loop is
    excluded because it requires a GPU and a model checkpoint, neither
    of which is sensible in CI).
  - `src/pipeline.py` — 100%.
  - `src/exporter/finalize_sync_all.py` — 77%.
  - `app_helpers.py` — 100% (every UI panel's logic is tested).
- **One test file per source module** under `tests/`, using
  `unittest.TestCase` + `unittest.mock`. No fixture-database, no shared
  mutable state.
- **Opt-in Streamlit AppTest smoke test** behind
  `RUN_STREAMLIT_SMOKE=1` — never runs in CI by default, available
  on-demand for release validation.

The full rationale and gap analysis for each entry point is in
[CHANGELOG.md § 12.0.3](../CHANGELOG.md).

---

## 8. Where to look when you change something

| You are changing… | Re-run / re-test |
|-------------------|------------------|
| Stage 1–7 logic | `python cli.py run --stages <n>` then `pytest tests/test_pipeline.py` |
| A PII rule | `pytest tests/test_pii_*.py` and re-run `scripts/generate_audit_certificate.py` against a fixture parquet |
| `app.py` | `pytest tests/test_app.py`; for full UI smoke: `RUN_STREAMLIT_SMOKE=1 pytest tests/test_app_streamlit_smoke.py` |
| `inference.py` | `pytest tests/test_inference.py`; manual: `python -m src.inference --adapter none --no-rag` |
| `finalize_sync_all.py` | `pytest tests/test_finalize_sync_all.py`; manual dry-run: `python scripts/finalize_sync_all.py --dry-run` |
| HF token handling | `pytest tests/test_finalize_sync_all.py -k token` and re-read [ADR 0001](adr/0001-hf-token-handling.md) |
| CI matrix | push to a branch; observe `lint` and `test-and-validate` jobs on the PR |

---

## 9. Design invariants

These are properties the codebase guarantees today and that future
changes must preserve. The tests in `tests/` exist to make accidental
violations fail loudly.

1. **Pure helpers, then I/O.** Every module exposes a small set of pure
   functions (no `print`, no `input`, no I/O) and an I/O wrapper.
   Tests target the pure layer.
2. **Single upload path.** HF Hub writes go through
   `scripts/sync_to_hub.py` or `scripts/finalize_sync_all.py`; nothing
   else calls `HfApi().upload_*`.
3. **No silent downgrades.** LoRA loading failures raise
   `RuntimeError`; PII leaks fail the audit; the PII red-team
   certificate must be present in `reports/`.
4. **Coverage floor = 60%.** A PR that drops overall coverage below the
   floor fails CI, even if all existing tests pass.
5. **Three data formats, one source of truth.** Parquet is canonical;
   JSONL ChatML and ShareGPT/Alpaca are derived by `src/exporter`.

See [operations.md](operations.md) for the day-2 procedures (dataset
rebuild, HF sync, CI re-runs, incident response).
