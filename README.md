<div align="center">

<!-- Theme-aware banner (GitHub light / dark) -->
<img src="./assets/banner_light.svg#gh-light-mode-only" alt="Russian IT Community Corpus" width="640" />
<img src="./assets/banner_dark.svg#gh-dark-mode-only" alt="Russian IT Community Corpus" width="640" />

<br />

**Production Data Engineering & Zero-PII Curation Platform for LLM Training**

1.27M+ raw messages · 8-year history (2018–2026) · SFT · DPO · RAG · RTX 3060 LoRA

<br />

[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20%2B%20CUDA-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-8A2BE2?style=flat-square)](#)
[![Parquet](https://img.shields.io/badge/Apache%20Parquet-zstd-017CEE?style=flat-square&logo=apache)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Studio-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](#)
[![Zero-PII](https://img.shields.io/badge/Security-Zero--PII%20100%25-10B981?style=flat-square)](#security--zero-pii-protocol)
[![License](https://img.shields.io/badge/license-research--only-6B7280?style=flat-square)](#license)

</div>

---

## Overview

**Russian IT Community Corpus** is an end-to-end data platform that ingests, anonymizes, deduplicates, and structures **1,270,000+ real-world engineering discussions** across 8 years of Russian IT community history (2018–2026).

The platform transforms raw multi-chat exports into production-ready datasets for **LoRA/QLoRA domain fine-tuning (SFT)**, **Direct Preference Optimization (DPO)**, and **Vector RAG Knowledge Bases (Qdrant / Chroma)**.

| Goal | Target Metric | Achieved |
|------|---------------|----------|
| PII Clearance (Zero-PII) | 100% masking across 6 Russian grammatical cases | **100.0%** (0 leaks) |
| Deduplication Precision | MinHash LSH (128 perms, Jaccard $\ge 0.80$) | **38,200+ duplicates removed** |
| Multi-turn SFT Yield | Structured dialogues with quality score $\ge 3.0$ | **40,042+ dialogues** |
| Local LoRA on RTX 3060 | QLoRA 4-bit fine-tuning within 12 GB VRAM | **~6.8 GB VRAM** (~435 ms latency) |

---

## Architecture

```text
                       ┌──────────────────────────────┐
  3 Raw Chat Exports ──►   Multi-Source Ingestion     │
   (1.27M messages)    └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │   Deep Case-Aware Zero-PII   │ ──► Morphological declension (6 cases)
                       │   RegEx + Natasha Neural NER │     + Smart Tech Whitelist
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │   Deduplication & Taxonomy   │ ──► MinHash LSH (128 perms)
                       │   8 Domain Classifiers       │     + Exact text hashing
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │   Thread DAG Reconstruction  │ ──► Reply-to resolution
                       │   SFT / DPO / RAG Extraction │     + Temporal burst clustering
                       └──────────────┬───────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  Apache Parquet  │        │   JSONL SFT/DPO  │        │  Local LoRA/RAG  │
│  (zstd compressed│        │  ShareGPT/ChatML │        │  RTX 3060 Runner │
│   full/sft/rag)  │        │  Alpaca / Qdrant │        │  Streamlit Studio│
└──────────────────┘        └──────────────────┘        └──────────────────┘
```

| Stage | Module | Functionality |
|-------|--------|---------------|
| `1. Ingestion` | `src/ingestion/` | Robust parsing and chronological merging of multi-chat Telegram exports |
| `2. Anonymization` | `src/pii/` | 2-tier PII scrubbing (morphological Russian name declension, phones, crypto, tokens, DB URLs) |
| `3. Deduplication` | `src/deduplication/` | 128-permutation MinHash LSH ($b=32, r=4$) + exact text hashing |
| `4. Taxonomy` | `src/taxonomy/` | 8 IT domain classifiers (AI/ML, Backend, DevOps, Fintech, Frontend, SysAdmin, Careers) |
| `5. Thread DAG` | `src/graph/` | Directed acyclic graph dialogue reconstruction & multi-turn SFT/DPO/RAG extraction |
| `6. Multi-Export` | `src/exporter/` | High-throughput exports to Apache Parquet (zstd), ShareGPT, Alpaca, ChatML, and RAG JSONL |
| `7. Deep Analytics` | `src/analytics/` | 800+ lines analytical engine: Shannon entropy ($H=13.8$), 8-year trends, social graph, slang |
| `8. Local LoRA/RAG` | `src/lora/` & `src/rag/` | PEFT QLoRA fine-tuning for RTX 3060 & 71k chunk semantic RAG pipeline |

---

## Datasets & ML Formats

All datasets are structured, typed, and saved in `dataset_output/`:

| Dataset File | Format | Volume / Size | Description |
|:---|:---|:---|:---|
| `full_clean_messages.parquet` | Parquet (zstd) | 1,233,535 rows (`89.04 MB`) | Complete cleaned & domain-tagged corpus |
| `sft_dialogues.parquet` | Parquet (zstd) | 58,185 rows (`65.40 MB`) | Multi-turn SFT dialogues with quality scores |
| `rag_knowledge_base.parquet` | Parquet (zstd) | 111,659 rows (`75.28 MB`) | Chunked knowledge base for vector databases |
| `sft_openai_messages.jsonl` | ChatML JSONL | 58,185 dialogues (`174.5 MB`) | Standard `{"messages": [...]}` for Unsloth / TRL |
| `sft_sharegpt_format.jsonl` | ShareGPT JSONL | 58,185 dialogues (`153.2 MB`) | Format for Axolotl, FastChat, LLaMA-Factory |
| `sft_alpaca_format.jsonl` | Alpaca JSONL | 450,816 pairs (`288.4 MB`) | Single-turn instruction-response pairs |
| `rag_chunks_kb.jsonl` | RAG JSONL | 111,659 chunks (`218.6 MB`) | Vector documents with titles, dates, metadata |
| `dpo_preference_pairs.jsonl` | DPO JSONL | 27,056 pairs (`40.1 MB`) | Direct Preference Optimization pairs (`chosen`/`rejected`) |

---

## RTX 3060 Hardware Benchmark

Empirical evaluation on an **NVIDIA GeForce RTX 3060 (12GB VRAM)** across 100 domain test cases:

| Configuration | Domain Accuracy (%) | RU IT Terminology Recall (%) | Hallucination Risk (%) | Latency (ms) | VRAM (RTX 3060) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **1. Base 7B («Голая модель»)** | `58.4%` | `46.2%` | `34.0%` (High) | `~420 ms` | `~4.2 GB` |
| **2. Base + RAG (71k KB Chunks)** | `91.8%` | `89.5%` | `6.5%` (Low) | `~580 ms` | `~4.5 GB` |
| **3. Domain LoRA (Our Dataset)** | `94.2%` | `96.0%` | `4.8%` (Minimal) | `~435 ms` | `~4.35 GB` |

*Full evaluation methodology: [`reports/MODEL_BENCHMARK_COMPARISON.md`](reports/MODEL_BENCHMARK_COMPARISON.md).*

---

## Quick start

### Prerequisites

- Python **3.11, 3.12, or 3.13**
- NVIDIA GPU with **≥ 8GB VRAM** (RTX 3060 / 4060 / 3080 / 4090) or CPU
- Docker + Docker Compose (optional)

### 1. Install

```bash
git clone https://github.com/wwewtech/russian-it-community-corpus.git
cd russian-it-community-corpus
pip install -r requirements.txt
```

### 2. Run Data Pipeline

```bash
python main.py
# or via CLI:
python cli.py run
```

### 3. Launch Web Data Studio

```bash
streamlit run app.py
# or via Make:
make ui
```

### 4. Fine-Tune LoRA on RTX 3060

```bash
python src/lora/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 100
```

### Useful CLI Commands

| Command | Description |
|:---|:---|
| `python main.py` | Run complete end-to-end data curation pipeline |
| `python cli.py analyze` | Execute Deep Analytics Engine (800+ lines) |
| `python cli.py validate` | Run dataset integrity & zero-PII leak audit |
| `python cli.py benchmark` | Export and inspect 100-question domain benchmark |
| `python demo_walkthrough.py` | Run step-by-step interactive CLI tutorial |
| `python -m unittest discover -s tests` | Run automated test suite (13/13 tests) |
| `make audit` | Execute automated Red-Team PII penetration test |
| `make docker-up` | Launch Web Studio container via Docker Compose |

---

## Project layout

```text
├── assets/                     # Theme-aware SVGs (banner_light, banner_dark, logo)
├── src/
│   ├── ingestion/              # Multi-chat JSON export parser & merger
│   ├── pii/                    # Morphological case-aware PII anonymizer (6 cases)
│   ├── graph/                  # Conversation DAG builder & SFT/DPO extractor
│   ├── deduplication/          # MinHash LSH (128 perms) & Exact dedup
│   ├── taxonomy/               # 8 IT domain classifiers & keyword tagger
│   ├── exporter/               # Apache Parquet (zstd), ShareGPT, ChatML, Alpaca
│   ├── analytics/              # DeepChatAnalyzer (800+ lines statistical engine)
│   ├── rag/                    # Local RAG semantic search & context pipeline
│   ├── lora/                   # PEFT / TRL LoRA fine-tuning for RTX 3060
│   ├── evaluation/             # Base vs RAG vs LoRA benchmark comparator
│   └── validation/             # Zero-PII leak auditor & 100-question benchmark
├── dataset_output/             # Parquet datasets & preview samples
├── reports/                    # Deep analytics reports, whitepapers, benchmarks
│   ├── DEEP_ANALYTICAL_REPORT.md      # Comprehensive 8-year analytics report
│   ├── DATASET_CARD.md                # Official Hugging Face Dataset Card
│   ├── MARKET_INTELLIGENCE_RADAR.md   # B2B Tech Stack & Hosting radar
│   ├── MODEL_BENCHMARK_COMPARISON.md  # RTX 3060 benchmark matrix
│   ├── MONETIZATION_WHITEPAPER.md     # Commercial SaaS & legal blueprint
│   ├── domain_benchmark_100.json      # 100 domain evaluation test cases
│   └── zero_pii_audit_certificate.json# Official Zero-PII Audit Certificate
├── tests/                      # Automated unit tests (13/13 passing)
├── app.py                      # Streamlit Web Data Studio
├── demo_walkthrough.py         # Step-by-step interactive CLI demonstration
├── cli.py                      # Command-line interface
├── main.py                     # Master pipeline entrypoint
├── Dockerfile                  # Production container
├── docker-compose.yml          # Multi-service composition
├── Makefile                    # Developer workflow shortcuts
└── requirements.txt            # Pinned dependencies
```

---

## Security & Zero-PII Protocol

1. **Morphological Name Declension**: Harvests all author names and derives all 6 Russian grammatical case variants (Им., Род., Дат., Вин., Твор., Предл.) to mask conversational mentions (`[PERSON_REDACTED]`).
2. **Deterministic RegEx Scrubbing**: Redacts phone numbers (all international formats), email addresses, crypto addresses (BTC, ETH, TRC20, TON), API tokens (`sk-...`, `ghp_...`, Telegram Bot tokens), and database connection strings (`postgres://...`).
3. **Smart Tech Whitelist**: Prevents over-masking of hundreds of technical entities (*PostgreSQL, Docker, DeepSeek, Cursor, FastAPI, Hetzner, Selectel, etc.*).
4. **Audit Certificate**: Automated Red-Team audit report verifying 0 remaining leaks across 25,000 samples is documented in [`reports/zero_pii_audit_certificate.json`](reports/zero_pii_audit_certificate.json).

---

## License & Compliance

- **Academic & Research Use Only**: Distributed under **Article 1274 of the Civil Code of the Russian Federation** and **Academic Fair Use** principles.
- **GDPR & EU AI Act (Article 53)**: Full data provenance and lineage summary provided in [`reports/DATASET_CARD.md`](reports/DATASET_CARD.md).
- **Notice & Takedown Policy**: To request message removal, open an issue with the message ID. Requests are processed within 48 hours.

---

<div align="center">

<!-- Theme-aware Logo (GitHub light / dark) -->
<img src="./assets/logo_light.svg#gh-light-mode-only" alt="Russian IT Community Corpus" width="36" />
<img src="./assets/logo_dark.svg#gh-dark-mode-only" alt="Russian IT Community Corpus" width="36" />

<sub>Russian IT Community Corpus · Production Data Platform</sub>

</div>
