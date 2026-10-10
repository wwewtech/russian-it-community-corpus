# Changelog

All notable changes to RICC are documented here.

> Full history: [`CHANGELOG.md`](https://github.com/wwewtech/russian-it-community-corpus/blob/main/CHANGELOG.md) in the repository root.

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

## [12.0.4] - 2026-10-04

### Added
- **Unified Configuration**: `src/settings.py` using `pydantic-settings.BaseSettings` — single source of truth merging `config.py` + `params.yaml` with env var overrides (`RICC__`)
- **Observability Module**: `src/observability/__init__.py` with structured logging (structlog) + Prometheus metrics for pipeline stages, PII audit, drift, validation, LoRA, SLO, HTTP, errors
- **Error Taxonomy**: `src/errors.py` with complete typed exception hierarchy (`RICCError`, `PipelineError*`, `ValidationError*`, `DataQualityError*`, `ConfigurationError*`, `ExternalServiceError*`, `ResourceError*`) + functional `Result[T, E]` type with `Ok`/`Err`
- **Public API Package**: `src/ricc/__init__.py` with `__version__`, `Settings`, `MasterDataPipeline`, all exceptions, observability, and `Result` type
- **CI/CD Pipeline**: `.github/workflows/ci.yml` with lint (ruff), typecheck (mypy strict), test (pytest cov≥85%), Docker build/push, SLO gate, HF Hub release
- **Pre-commit Hooks**: `.pre-commit-config.yaml` mirroring CI (ruff, mypy strict, YAML/JSON/TOML checks)
- **DVC Pipeline**: `dvc.yaml` with 12 reproducible stages matching `MasterDataPipeline`
- **Secrets Management**: `.env.example` template + `.secrets.baseline` for detect-secrets
- **Hardened Docker**: `Dockerfile` + `docker-compose.yml` with `read_only`, `no-new-privileges`, tmpfs mounts
- **MkDocs Documentation**: `mkdocs.yml` with Material theme, mkdocstrings for auto API reference

### Changed
- Migrated 11 source files from `src.config` imports to `src.settings`
- Updated `cli.py` with backward-compat constants for tests
- Refactored `MasterDataPipeline` to use observability metrics + structured logging
- Updated `bootstrap.py` to configure structured logging
- Removed `ahocorasick` from dependencies (optional, Windows compatibility)

### Fixed
- Coverage gate enforcement in CI (85% → 87%)
- MyPy strict mode passes on 32 modules (including new `errors.py`, `observability/`)
- SLO gate consistently returns SHIP

## [12.0.3] - 2026-09-17

### Added
- Probabilistic PII audit with stratified sampling + Wilson CI + power analysis
- Drift monitoring: PSI + Jensen-Shannon + Vocabulary Jaccard
- SLO gate with 6 checks (validation, pii-audit, drift, pipeline-volume, artifact-manifest, hub-reconciliation)
- LoRA batch training for 20+ models with model zoo registry
- Hybrid RAG pipeline (dense + sparse retrieval)
- Dataset reconciliation with pinned Hub revision
- Artifact manifest with SHA256 verification

### Changed
- Refactored `MasterDataPipeline` to 7 explicit stages
- Centralized runtime env setup in `src/bootstrap.py`
- MinHash LSH optimized with precomputed permutations

## [12.0.0] - 2026-08-15

### Added
- Initial production release
- End-to-end pipeline: ingestion → PII → dedup → taxonomy → graph → export → analytics
- Dual-layer PII scrubbing (Regex + Natasha NER) + pseudonymization
- 9 IT domain taxonomy with 500+ keywords
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