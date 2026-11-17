# Changelog

All notable changes to RICC are documented here.

> Full history: [`CHANGELOG.md`](https://github.com/wwewtech/russian-it-community-corpus/blob/main/CHANGELOG.md) in the repository root. This page mirrors the most recent releases; the root changelog is the source of truth.

## [12.2.0] - 2026-10-14

### Added
- **GPU/ML error taxonomy (P3)**: `ricc/errors.py` extended with `MLError`
  (`TrainingError`, `InferenceError`, `EvaluationError`, `TokenizerError`,
  `CheckpointError`, `AdapterLoadError`, `QuantizationError`) and `GPUError`.
- **Monitoring stack (P4)**: ServiceMonitor, Prometheus rules (12 alerts),
  Grafana dashboard (18 panels); metrics server in the Streamlit app.
- **LoRA run metrics**: `ricc_lora_train_total{model,adapter,status}` counter.
- **Drift gauges wired in**: PSI / JS-divergence / vocab-Jaccard gauges.
- **Security gate hardening (P5)**: pip-audit promoted to a hard gate.
- **mkdocs docs gate**: `mkdocs build --strict` in CI.

### Changed
- **mypy strict rollout complete**: every `ricc/` module passes `mypy --strict`.
- **pre-commit hygiene**: Black fallback hook removed; mypy hook is local.

### Fixed
- Hybrid RAG retrieval upgraded (lexical pre-filter + per-query TF-IDF re-rank).
- SLO release gate (`ricc/monitoring/slo_gate.py` + `make slo`).
- Loader crash on list-shaped exports; Docker hardening; SLO verdict mapping
  (`PASSED` accepted).

## [12.1.0] - 2026-10-10

### Added
- Real-corpus SLO gate job (`slo-real`): downloads the canonical parquet trio at the pinned HF revision `81a3495`, regenerates audit/drift/manifest/reconciliation and runs the strict gate (artifact `slo-verdict-real`).
- SLO verdict published as artifact and to the GitHub step summary.
- `requirements.lock.txt` (104 exact pins) + `scripts/lock_requirements.py`.
- `scripts/generate_adapter_cards.py`: all 56 adapter cards regenerated from `registry.json`.

### Fixed
- Committed reports regenerated from the real parquet (validation, drift, stats, manifest, reconciliation).
- PII validator community-names false positives removed (481 → 0 on 10k sample).
- SLO gate fail-closed only on the explicit synthetic marker.
- `test_pipeline.py` no longer poisons committed reports mid-session.
- CI: coverage.xml artifacts, pinned `ubuntu-24.04`, actions bumped (Node 20 warnings cleared).

## [12.0.4] - 2026-09-08

### Added
- **Unified Configuration**: `ricc/settings.py` using `pydantic-settings.BaseSettings` — single source of truth merging `config.py` + `params.yaml` with env var overrides (`RICC__`)
- **Observability Module**: `ricc/observability/__init__.py` with structured logging (structlog) + Prometheus metrics
- **Error Taxonomy**: `ricc/errors.py` with complete typed exception hierarchy
- **CI/CD Pipeline**: `.github/workflows/ci.yml` with lint (ruff), typecheck (mypy strict), test (pytest cov≥85%), Docker build/push, SLO gate, HF Hub release
- **Pre-commit Hooks**: `.pre-commit-config.yaml` mirroring CI
- **DVC Pipeline**: `dvc.yaml` with 12 reproducible stages matching `MasterDataPipeline`
- **Secrets Management**: `.env.example` template + `.secrets.baseline` for detect-secrets
- **Hardened Docker**: `Dockerfile` + `docker-compose.yml` with `read_only`, `no-new-privileges`, tmpfs mounts
- **MkDocs Documentation**: `mkdocs.yml` with Material theme

### Changed
- Migrated 11 source files from `ricc.config` imports to `ricc.settings`
- Removed `ahocorasick` from dependencies (optional, Windows compatibility)

## [12.0.3] - 2026-08-30

### Added
- Probabilistic PII audit with stratified sampling + Wilson CI + power analysis
- Drift monitoring: PSI + Jensen-Shannon + Vocabulary Jaccard
- SLO gate with 6 checks (validation, pii-audit, drift, pipeline-volume, artifact-manifest, hub-reconciliation)
- LoRA batch training for 20+ models with model zoo registry
- Hybrid RAG pipeline (lexical retrieval; no embeddings)
- Dataset reconciliation with pinned Hub revision
- Artifact manifest with SHA256 verification

### Changed
- Refactored `MasterDataPipeline` to 7 explicit stages
- Centralized runtime env setup in `ricc/bootstrap.py`
- MinHash LSH optimized with precomputed permutations

## [12.0.0] - 2026-08-28

### Added
- Initial production release
- End-to-end pipeline: ingestion → PII → dedup → taxonomy → graph → export → analytics
- Dual-layer PII scrubbing (Regex + Natasha NER) + pseudonymization
- 8 IT domain taxonomy with 268 keywords
- Thread DAG reconstruction with reply-time heuristics
- SFT/DPO/RAG extraction from conversations
- Multi-format export (Parquet, ShareGPT, Alpaca, ChatML, RAG JSONL, DPO)
- Deep analytics engine with statistical + semantic metrics
- Streamlit Data Studio for interactive exploration

---

**Versioning**: Semantic Versioning (MAJOR.MINOR.PATCH)
- MAJOR: Breaking API changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible
