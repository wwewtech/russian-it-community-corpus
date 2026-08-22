<div align="center">

<!-- Theme-aware wordmark (GitHub light / dark) -->
<img src="./assets/banner_light.svg#gh-light-mode-only" alt="RICC" width="480" />
<img src="./assets/banner_dark.svg#gh-dark-mode-only" alt="RICC" width="480" />

<br />

**High-throughput data engineering and Zero-PII curation platform for language models**

1.27M+ discussions · 2018–2026 history · SFT dialogues · DPO pairs · RAG knowledge base · LoRA on RTX 3060

<br />

[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20CUDA-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20and%20QLoRA-8A2BE2?style=flat-square)](#)
[![Parquet](https://img.shields.io/badge/Apache%20Parquet-zstd-017CEE?style=flat-square&logo=apache)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-Data%20Studio-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](#)
[![Zero-PII](https://img.shields.io/badge/Security-Zero--PII%20Verified-10B981?style=flat-square)](#security-and-zero-pii-protocol)
[![License](https://img.shields.io/badge/license-research--only-6B7280?style=flat-square)](#license-and-compliance)

</div>

---

## Overview

**RICC** (**R**ussian **I**T **C**ommunity **C**orpus) is a data engineering and curation stack that ingests, cleans, deduplicates, and structures over 1,270,000 engineering and business messages from Russian developer communities over an 8-year span.

The platform produces datasets for instruction fine-tuning, direct preference optimization, and vector knowledge retrieval without manual intervention.

| Metric | Target | Verified Value |
|---|---|---|
| Zero-PII privacy guarantee | Complete redaction across Russian grammatical cases | Zero leaks across test sets |
| Deduplication accuracy | MinHash LSH with 128 permutations at 0.80 Jaccard threshold | 38,200+ duplicates removed |
| SFT dialogue quality | Multi-turn dialogues scored at 3.0 or above | 58,185 dialogues |
| Local LoRA execution | 4-bit QLoRA adaptation on consumer hardware | 6.8 GB VRAM on RTX 3060 |

---

## Architecture

```text
                       ┌──────────────────────────────┐
  3 Raw Chat Exports ──►   Multi-Source Ingestion     │
   1.27M raw records   └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │   Deep Case-Aware Zero-PII   │ ──► Declension across 6 Russian cases
                       │   RegEx + Natasha Neural NER │     Tech terminology protection whitelist
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │   Deduplication & Taxonomy   │ ──► MinHash LSH with 128 permutations
                       │   8 Domain Classifiers       │     Exact hash deduplication
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │   Thread DAG Reconstruction  │ ──► Reply tree traversal
                       │   SFT, DPO, RAG Extraction   │     Temporal cluster resolution
                       └──────────────┬───────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  Apache Parquet  │        │   JSONL Datasets │        │  Local LoRA, RAG │
│  zstd compressed │        │  ShareGPT, ChatML│        │  RTX 3060 Runner │
│   full, sft, rag │        │  Alpaca, Qdrant  │        │  Streamlit Studio│
└──────────────────┘        └──────────────────┘        └──────────────────┘
```

| Pipeline Stage | Implementation | Purpose |
|---|---|---|
| Ingestion | `src/ingestion/` | Normalizes and chronologically merges multi-chat raw exports |
| Anonymization | `src/pii/` | Redacts names across 6 cases, phone numbers, crypto wallets, API tokens, database URLs |
| Deduplication | `src/deduplication/` | Filters near-duplicate and exact spam messages via MinHash LSH |
| Taxonomy | `src/taxonomy/` | Categorizes content into 8 technical domains and extracts keyword tags |
| Thread DAG | `src/graph/` | Reconstructs conversational trees and extracts multi-turn dialogues |
| Multi-Export | `src/exporter/` | Serializes outputs into Apache Parquet with zstd compression and JSONL formats |
| Analytics | `src/analytics/` | Computes Shannon entropy, temporal patterns, social graphs, and vocabulary stats |
| Local LoRA and RAG | `src/lora/`, `src/rag/` | Provides PEFT training for RTX 3060 and semantic retrieval for 111k chunks |

---

## Datasets and Formats

All datasets are saved in `dataset_output/`:

| File | Format | Volume | Description |
|---|---|---|---|
| `full_clean_messages.parquet` | Parquet with zstd | 1,233,535 rows | Full cleaned and categorized corpus |
| `sft_dialogues.parquet` | Parquet with zstd | 58,185 dialogues | Multi-turn dialogues with quality scores |
| `rag_knowledge_base.parquet` | Parquet with zstd | 111,659 chunks | Segmented knowledge base for vector search |
| `sft_openai_messages.jsonl` | ChatML JSONL | 58,185 dialogues | OpenAI format for Unsloth and TRL |
| `sft_sharegpt_format.jsonl` | ShareGPT JSONL | 58,185 dialogues | Format for Axolotl and LLaMA-Factory |
| `sft_alpaca_format.jsonl` | Alpaca JSONL | 450,816 pairs | Single-turn instruction and response pairs |
| `rag_chunks_kb.jsonl` | RAG JSONL | 111,659 chunks | Knowledge documents with titles and metadata |
| `dpo_preference_pairs.jsonl` | DPO JSONL | 27,056 pairs | Preference pairs with chosen and rejected responses |

---

## Hardware Benchmark on RTX 3060

Empirical evaluation on an NVIDIA GeForce RTX 3060 with 12 GB VRAM:

| Setup | Domain Accuracy | Technical Terminology Recall | Hallucination Risk | Latency | VRAM Usage |
|---|:---:|:---:|:---:|:---:|:---:|
| Base 7B Model | 58.4% | 46.2% | High | ~420 ms | ~4.2 GB |
| Base Model with RAG | 91.8% | 89.5% | Low | ~580 ms | ~4.5 GB |
| Domain LoRA Fine-Tuned | 94.2% | 96.0% | Minimal | ~435 ms | ~4.35 GB |

Detailed benchmark report: [`reports/MODEL_BENCHMARK_COMPARISON.md`](reports/MODEL_BENCHMARK_COMPARISON.md).

---

## Quick start

### Requirements

- Python 3.11, 3.12, or 3.13
- NVIDIA GPU with 8 GB or more VRAM for training, or CPU for data processing
- Docker with Docker Compose for containerized runs

### 1. Install

```bash
git clone https://github.com/wwewtech/russian-it-community-corpus.git
cd russian-it-community-corpus
pip install -r requirements.txt
```

### 2. Run Data Pipeline

```bash
python main.py
```

### 3. Launch Web Data Studio

```bash
streamlit run app.py
# or:
make ui
```

### 4. Fine-Tune LoRA

```bash
python src/lora/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 100
```

### 5. Run LoRA Inference

```bash
python src/lora/generate_demo.py --prompt "Как настроить прием платежей для SaaS из РФ?"
```

### Command Reference

| Command | Action |
|---|---|
| `python main.py` | Run complete pipeline on all available chat exports |
| `python cli.py analyze` | Generate analytical reports and metrics |
| `python cli.py validate` | Validate dataset schema and check for PII leaks |
| `python cli.py benchmark` | Export and inspect 100 domain test cases |
| `python demo_walkthrough.py` | Run interactive terminal walkthrough |
| `python -m unittest discover -s tests` | Run automated test suite |
| `make audit` | Run red-team adversarial penetration test |
| `make docker-up` | Start Web Data Studio in Docker container |

---

## Project Structure

```text
├── assets/                     # Theme-aware vector graphics
├── src/
│   ├── ingestion/              # Multi-chat JSON export parser
│   ├── pii/                    # Case-aware PII anonymizer and neural NER
│   ├── graph/                  # Thread DAG builder and dialogue extractor
│   ├── deduplication/          # MinHash LSH and exact text hashing
│   ├── taxonomy/               # Domain classifiers and keyword taggers
│   ├── exporter/               # Apache Parquet and JSONL serializers
│   ├── analytics/              # DeepChatAnalyzer statistical engine
│   ├── rag/                    # Vector search and prompt augmentation pipeline
│   ├── lora/                   # PEFT training and inference scripts
│   ├── evaluation/             # Benchmark comparator
│   └── validation/             # PII auditor and domain benchmarks
├── dataset_output/             # Parquet datasets and preview samples
├── reports/                    # Analytical reports, whitepapers, benchmarks
│   ├── DEEP_ANALYTICAL_REPORT.md      # Statistical and longitudinal report
│   ├── DATASET_CARD.md                # Hugging Face Dataset Card
│   ├── MARKET_INTELLIGENCE_RADAR.md   # Industry technology radar
│   ├── MODEL_BENCHMARK_COMPARISON.md  # Model evaluation results
│   ├── MONETIZATION_WHITEPAPER.md     # Commercial and legal architecture
│   ├── domain_benchmark_100.json      # Evaluation benchmark dataset
│   └── zero_pii_audit_certificate.json# Security audit verification
├── tests/                      # Automated unit tests
├── app.py                      # Streamlit Web Data Studio
├── demo_walkthrough.py         # Terminal demonstration script
├── cli.py                      # CLI entrypoint
├── main.py                     # Pipeline master runner
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Container composition
├── Makefile                    # Task shortcuts
└── requirements.txt            # Python dependencies
```

---

## Security and Zero-PII Protocol

1. **Morphological Name Redaction**: Detects author names and inflects them across 6 Russian grammatical cases to eliminate conversational mentions.
2. **Deterministic Pattern Scrubbing**: Removes phone numbers in international formats, email addresses, cryptocurrency wallet addresses, API keys, tokens, and database connection strings.
3. **Terminology Protection**: Whitelists common technical terms, programming languages, libraries, and hosting providers to prevent false positives.
4. **Independent Audit**: Automated testing against adversarial samples verifies zero remaining personal identifiers in [`reports/zero_pii_audit_certificate.json`](reports/zero_pii_audit_certificate.json).

---

## License and Compliance

- **Academic and Research Use**: Provided in accordance with Article 1274 of the Civil Code of the Russian Federation and international Fair Use doctrines.
- **Privacy Compliance**: All personal data has been irreversibly de-identified pursuant to GDPR Recital 26 and Russian Federal Law 152-FZ.
- **Notice and Takedown**: To request message removal, open an issue using the provided takedown template. Requests are addressed within 48 hours.

---

<div align="center">

<!-- Theme-aware Logo -->
<img src="./assets/logo_light.svg#gh-light-mode-only" alt="RICC" width="36" />
<img src="./assets/logo_dark.svg#gh-dark-mode-only" alt="RICC" width="36" />

<sub>RICC · Russian IT Community Corpus</sub>

</div>
