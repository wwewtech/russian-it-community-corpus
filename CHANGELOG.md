# Changelog

All notable changes to the **Russian IT Community Corpus & Data Platform** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [4.0.0] - 2026-08-22

### Added
- **Multi-Export Ingestion (1.27M+ Messages)**: Ingestion and chronological merge of 3 complete Telegram chat dumps (`ChatExport_2026-08-21`, `ChatExport_2026-08-22`, and `ChatExport_2026-08-23` - Wylsacom Media).
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
