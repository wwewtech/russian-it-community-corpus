# Changelog

All notable changes to the **Russian IT Community Corpus & Data Platform** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [12.0.1] - 2026-08-29

### Fixed
- **Broken dependency pin**: `pymorphy3-dicts-ru>=2.4.417127.20260408` referenced a version that does not exist on PyPI (only `2.4.417150.4580142` is published), which made `pip install -r requirements.txt` fail. Pin corrected in `requirements.txt` and `pyproject.toml` (source: [PyPI](https://pypi.org/pypi/pymorphy3-dicts-ru/json)).
- **README benchmark table**: figures for HumanEval / RuMMLU / PPL were officially withdrawn in the dataset card but were still published as valid in the README. The table is now marked WITHDRAWN pending a GPU re-run, consistent with the dataset card and `metrics_index.json`.
- **README count mismatches**: Alpaca pairs corrected 933,331 → 933,313 and DPO pairs 60,412 → 60,899, synchronized with `reports/validation_results.json`.
- **Probabilistic PII audit report**: the committed `reports/probabilistic_pii_audit.json` was generated against a 2-message temp-folder fixture (verdict `LEAKS_DETECTED`, upper bound 0.73). Re-generated against the production corpus (2,816,434 messages, 100,000-message stratified sample, 0 leaks, verdict PASSED).
- **License detection**: `LICENSE` now contains only the MIT license text so GitHub recognizes it (`spdx_id: MIT`); the dataset usage terms moved to [`DATASET_TERMS.md`](DATASET_TERMS.md).

### Changed
- **CHANGELOG versioning note**: git tags `v7.0.0`–`v12.0.0` exist for released versions whose changelog entries were not preserved in this file; entries for 6.0.0–10.0.0 are therefore missing from this log. Historical tags are kept as-is (retagging published versions would violate SemVer).

---

## [12.0.0] - 2026-08-28

### Added
- **Probabilistic PII Audit (`src/validation/probabilistic_audit.py`)**: upgrades the point-check red-team audit to a statistically bounded one — stratified proportional sampling across community nodes, per-category leak-rate estimation with Wilson intervals and one-sided 99% upper bounds, power analysis (`required_sample_size`), and a verdict with an explicit statistical guarantee. Script `scripts/run_probabilistic_pii_audit.py` + `make audit-prob`.
- **Enlarged Benchmark Subsets for Statistical Power**: HumanEval subset 8 → **40 tasks**, RuMMLU CS subset 8 → **50 questions** (new categories: Distributed Systems, Security, Programming Languages, Systems Architecture). At N=8 the 95% Wilson CI spanned ~±20 p.p.; at N=40/50 it is ~±10/±11 p.p.
- **Wilson Confidence Intervals in Benchmark Reports (`src/evaluation/statistical_power.py`)**: dependency-free z-score/Wilson/upper-bound math; every published accuracy figure now carries a 95% CI (Markdown table rows + `*_ci95` keys in the JSON matrix).
- **Dataset Drift Monitoring (`src/monitoring/drift.py`)**: PSI over message-length distribution, Jensen–Shannon divergence over domain shares, top-k vocabulary Jaccard overlap — with stable/moderate/significant verdicts. Script `scripts/run_drift_monitoring.py` + `make drift`.
- **Prefect Orchestration (`src/orchestration/prefect_flow.py`)**: curation → validation → probabilistic audit → drift monitoring as Prefect tasks with retries; graceful sequential fallback when Prefect is not installed. `make orchestrate`.
- **DVC Pipeline (`dvc.yaml`, `params.yaml`)**: versioned corpus artifacts and reproducible stages (curate → validate → probabilistic_audit → drift_monitoring) with DVC metrics tracking; `platform` optional dependency group (`prefect`, `dvc`).

### Changed
- **Makefile**: new targets `audit-prob`, `drift`, `orchestrate`, `dvc-repro`.

---

## [11.0.0] - 2026-08-28

### Added
- **Installable Package & Console Entry Point**: `pip install .` now works — setuptools packaging config (`py-modules`, `packages.find`) plus the `it-pipeline` console command (`ricc` CLI). The Streamlit Data Studio intentionally stays on `streamlit run app.py` / `make ui`.
- **Extended PII Test Suite (`tests/test_pii_deep_coverage.py`)**: 28 new tests covering regex scrubber edge cases (TON/JWT/AWS/SSH/secret-assignment/invite links/mentions/IP filtering), deep morphological anonymizer control flow, NER scrubber logic via a fake Natasha `Doc` (no model downloads), and the `UnifiedPIIAnonymizer` facade.

### Changed
- **Toolchain Pinning**: `ruff` pinned to `==0.16.2` across pyproject dev extras, pre-commit, and CI — local lint now reproduces CI exactly; added missing dev deps (`pytest-cov`, `mypy`).
- **Coverage Gate Raised**: `--cov-fail-under` 50 → 60; `src/pii/` coverage lifted from 63–77% to 92–98% per module (total 59% → 62%).
- **Makefile**: new `install`, `lint`, and `format` targets.

---

## [5.0.0] - 2026-08-22

### Added
- **Full 2.91M Multi-Chat Ingestion & Curation**: Ingestion, normalization, and chronological merging of 11 distinct developer and engineering community nodes (`ChatExport_2026-08-15` through `ChatExport_2026-08-25`) totaling **2,911,754 raw records**.
- **Massive SFT & RAG Scale**: Extracted **164k+ curated SFT dialogues**, **307k+ RAG knowledge base chunks**, **60k+ DPO preference pairs**, and **955k+ Alpaca instruction pairs**.
- **Accelerated Ingestion & Tagging Engine**: Optimized taxonomy categorization to fast token set intersections, reducing processing time for millions of messages to seconds.
- **Integrated Progress Tracking (`tqdm`)**: Added visual progress bars across all batch processing pipelines.
- **Trained LoRA Weights on Full Corpus**: Successfully executed PEFT LoRA adaptation on RTX 3060 with loss decreasing to 2.51.

---

## [4.0.0] - 2026-08-22

### Added
- **Multi-Export Ingestion (1.27M+ Messages)**: Ingestion and chronological merge of 3 complete community chat dumps.
- **Deep Case-Aware Morphological Anonymizer (`src/pii/deep_anonymizer.py`)**: Dynamic participant name extraction with full Russian grammatical declension across all 6 cases (Им., Род., Дат., Вин., Твор., Предл.) + Natasha NER with Smart Tech Whitelist.
- **Local LoRA & PEFT Training Pipeline (`src/lora/train_lora.py`)**: QLoRA 4-bit domain adaptation optimized for consumer NVIDIA GeForce RTX 3060 (12GB VRAM).
- **Local Vector RAG Engine (`src/rag/rag_pipeline.py`)**: Sub-second semantic search across 111,659 curated IT knowledge chunks.
- **Interactive Streamlit Data Studio (`app.py`)**: Web UI with Dataset Explorer, SFT/DPO Dialogue Studio, RAG Assistant, Live Zero-PII Sandbox, and Tech Radar.
- **Automated Red-Team PII Penetration Suite (`src/validation/pii_redteam.py`)**: Generates official `zero_pii_audit_certificate.json`.
- **100-Question Domain Benchmark (`reports/domain_benchmark_100.json`)**: Quantitative evaluation suite for Backend, AI/ML, DevOps, Fintech, and Frontend.
- **Production Artifacts**: Dockerfile, docker-compose.yml, Makefile, and GitHub Actions CI workflow.

### Changed
- **Branding & Visuals**: Complete overhaul to anti-AI-slop strict minimalist vector geometry (`assets/logo_light.svg`, `assets/logo_dark.svg`, `assets/banner_light.svg`, `assets/banner_dark.svg`).
- **Export Standards**: Added Apache Parquet (zstd), ShareGPT JSONL, Alpaca JSONL, OpenAI ChatML JSONL, and DPO preference pairs.

---

## [3.0.0] - 2026-08-21

### Added
- Multi-turn conversation DAG thread reconstruction based on `reply_to_message_id` and temporal clustering.
- 8-domain taxonomy classification and multi-label technical keyword extractor.
- Deep Analytics Engine (`DeepChatAnalyzer`) with Shannon Entropy, social graph, and 8-year longitudinal trends.

---

## [1.0.0] - 2026-08-20

### Added
- Initial Telegram JSON export ingestion schema and regex-based PII scrubber.
