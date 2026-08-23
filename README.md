<div align="center">

<!-- Theme-aware wordmark (GitHub light / dark) -->
<img src="./assets/banner_light.svg#gh-light-mode-only" alt="RICC" width="480" />
<img src="./assets/banner_dark.svg#gh-dark-mode-only" alt="RICC" width="480" />

<br />

**High-throughput data engineering and Zero-PII curation platform for language models**

2.91M+ discussions · 2017–2026 history · SFT dialogues · DPO pairs · RAG knowledge base · LoRA on RTX 3060

<br />

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E?style=flat-square)](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LoRA%20Zoo-orange?style=flat-square)](https://huggingface.co/wwewtech/russian-it-community-lora)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20CUDA-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20and%20QLoRA-8A2BE2?style=flat-square)](#)
[![Parquet](https://img.shields.io/badge/Apache%20Parquet-zstd-017CEE?style=flat-square&logo=apache)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-Data%20Studio-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](#)
[![Zero-PII](https://img.shields.io/badge/Security-Zero--PII%20Verified-10B981?style=flat-square)](#security-and-zero-pii-protocol)
</div>

> [!TIP]
> **🤗 Official Hugging Face Hub Integration**:
> - 📦 **Dataset**: [`wwewtech/russian-it-community-corpus`](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus) — 2.91M clean messages, 171.5k multi-turn SFT dialogues, and 325.7k RAG knowledge base chunks in Apache Parquet.
> - 🦁 **LoRA Model Zoo**: [`wwewtech/russian-it-community-lora`](https://huggingface.co/wwewtech/russian-it-community-lora) — 44+ pre-trained open adapters + Flagship 7B-8B QLoRA models (Qwen 2.5 Coder 7B, DeepSeek R1 7B, LLaMA 3.1 8B).
> 
> ```python
> from datasets import load_dataset
> dataset = load_dataset("wwewtech/russian-it-community-corpus", "sft_dialogues", split="train")
> ```

---

## Overview

**RICC** (**R**ussian **I**T **C**ommunity **C**orpus) is a data engineering and curation stack that ingests, cleans, deduplicates, and structures over 2,910,000 engineering, infrastructure, business, and software development messages from 11 community nodes over a 9-year span (2017–2026).

The platform produces datasets for instruction fine-tuning, direct preference optimization, and vector knowledge retrieval without manual intervention.

| Metric | Target | Verified Value |
|---|---|---|
| Zero-PII privacy guarantee | Complete redaction across Russian grammatical cases | 100% Zero-PII verified (25,000 samples audited) |
| Deduplication accuracy | MinHash LSH with 128 permutations at 0.80 Jaccard threshold | 95,300+ duplicates removed |
| SFT dialogue quality | Multi-turn dialogues scored at 3.0 or above | 171,533 dialogues |
| Local LoRA execution | PEFT LoRA adaptation on consumer hardware | 4.35 GB VRAM on RTX 3060 |

---

## Architecture

```text
                       ┌──────────────────────────────┐
 11 Community Nodes ──►   Multi-Source Ingestion      │
   2.91M raw records   └──────────────┬───────────────┘
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
| Ingestion | `src/ingestion/` | Normalizes and chronologically merges multi-chat raw exports (11 sources) |
| Anonymization | `src/pii/` | Redacts names across 6 cases, phone numbers, crypto wallets, API tokens, database URLs |
| Deduplication | `src/deduplication/` | Filters near-duplicate and exact spam messages via MinHash LSH |
| Taxonomy | `src/taxonomy/` | Categorizes content into 8 technical domains and extracts keyword tags |
| Thread DAG | `src/graph/` | Reconstructs conversational trees and extracts multi-turn dialogues |
| Multi-Export | `src/exporter/` | Serializes outputs into Apache Parquet with zstd compression and JSONL formats |
| Analytics | `src/analytics/` | Computes Shannon entropy, temporal patterns, social graphs, and vocabulary stats |
| Local LoRA and RAG | `src/lora/`, `src/rag/` | Provides PEFT training for RTX 3060 and semantic retrieval for 325k chunks |

---

## 🔗 Official Datasets & Direct Access Links

The corpus and trained adapters are available both remotely on **Hugging Face Hub** and locally in `dataset_output/`:

### 🤗 Hugging Face Hub Repositories:
- 📦 **Dataset Hub (Full Corpus, SFT, RAG)**:  
  👉 [**`https://huggingface.co/datasets/wwewtech/russian-it-community-corpus`**](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus)
  - 📄 [Full Clean Corpus (Parquet)](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus/blob/main/data/full_clean_messages.parquet) — 2.81M rows (189 MB)
  - 💬 [SFT Dialogues (Parquet)](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus/blob/main/data/sft_dialogues.parquet) — 171.5k multi-turn dialogues (132 MB)
  - 🔍 [RAG Knowledge Base (Parquet)](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus/blob/main/data/rag_knowledge_base.parquet) — 325.7k knowledge chunks (159 MB)
  - ⚙️ [Unified Metrics & Audit (JSON)](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus/blob/main/metrics_index.json) — 322 KB metrics index

- 🦁 **Model Hub (44+ LoRA Adapters & 7B-8B QLoRA)**:  
  👉 [**`https://huggingface.co/wwewtech/russian-it-community-lora`**](https://huggingface.co/wwewtech/russian-it-community-lora)
  - 🥇 [Flagship Qwen 2.5 Coder 7B Adapter](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/models/heavyweight_qwen2.5_coder_7b)
  - 🥈 [Flagship DeepSeek R1 Distill 7B Adapter](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/models/heavyweight_deepseek_r1_7b)
  - 🥉 [Flagship Meta LLaMA 3.1 8B Adapter](https://huggingface.co/wwewtech/russian-it-community-lora/tree/main/models/heavyweight_llama3.1_8b)

---

## 📂 Local Datasets and Formats

All datasets are automatically generated and saved in `dataset_output/`:

| File Path | Format | Volume | Description | Direct Link |
|---|---|---|---|:---:|
| `dataset_output/parquet/full_clean_messages.parquet` | Parquet (zstd) | 2,816,454 rows (189 MB) | Full cleaned corpus with metadata | [HF Mirror](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus/blob/main/data/full_clean_messages.parquet) |
| `dataset_output/parquet/sft_dialogues.parquet` | Parquet (zstd) | 171,533 dialogues (132 MB) | Multi-turn dialogues for SFT | [HF Mirror](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus/blob/main/data/sft_dialogues.parquet) |
| `dataset_output/parquet/rag_knowledge_base.parquet` | Parquet (zstd) | 325,747 chunks (159 MB) | Vector knowledge base | [HF Mirror](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus/blob/main/data/rag_knowledge_base.parquet) |
| `dataset_output/jsonl/sft_openai_messages.jsonl` | ChatML JSONL | 171,533 dialogues | OpenAI format for Unsloth / TRL | Local / HF |
| `dataset_output/jsonl/sft_sharegpt_format.jsonl` | ShareGPT JSONL | 171,533 dialogues | Axolotl & LLaMA-Factory format | Local / HF |
| `dataset_output/jsonl/sft_alpaca_format.jsonl` | Alpaca JSONL | 933,331 pairs | Single-turn instruction-response pairs | Local / HF |
| `dataset_output/jsonl/rag_chunks_kb.jsonl` | RAG JSONL | 325,747 chunks | Segmented technical documents | Local / HF |
| `dataset_output/jsonl/dpo_preference_pairs.jsonl` | DPO JSONL | 60,900 pairs | Chosen / Rejected alignment pairs | Local / HF |

---

## Hardware Benchmark on RTX 3060

Empirical evaluation on an NVIDIA GeForce RTX 3060 with 12 GB VRAM:

| Setup | Domain Accuracy | Technical Terminology Recall | Hallucination Risk | Latency | VRAM Usage |
|---|:---:|:---:|:---:|:---:|:---:|
| Base 7B Model | 58.4% | 46.2% | High | ~420 ms | ~4.2 GB |
| Base Model with RAG (325k chunks) | 94.1% | 93.4% | Low | ~590 ms | ~4.5 GB |
| Domain LoRA Fine-Tuned (171k dialogues) | 96.4% | 97.8% | Minimal | ~430 ms | ~4.35 GB |

Detailed benchmark report: [`reports/MODEL_BENCHMARK_COMPARISON.md`](reports/MODEL_BENCHMARK_COMPARISON.md).

---

## Quick start

### Requirements

- Python 3.11, 3.12, or 3.13
- NVIDIA GPU with 8 GB or more VRAM for training, or CPU for data processing
- Docker with Docker Compose for containerized runs

### Direct Loading from Hugging Face

Load pre-built dataset splits in Python with a single line:

```python
from datasets import load_dataset

# 1. Load SFT Multi-Turn Dialogues (171.5k dialogues)
sft_ds = load_dataset(
    "wwewtech/russian-it-community-corpus", "sft_dialogues", split="train"
)

# 2. Load RAG Knowledge Base Chunks (325.7k chunks)
rag_ds = load_dataset(
    "wwewtech/russian-it-community-corpus", "rag_knowledge_base", split="train"
)

# 3. Load Full Clean Messages (2.81M records)
full_ds = load_dataset(
    "wwewtech/russian-it-community-corpus", "full_corpus", split="train"
)
```

### LoRA Model Zoo (18+ Pre-Trained Adapters)

Pre-trained adapters fine-tuned on RICC dataset are available in [`lora_adapters/`](lora_adapters/) and on Hugging Face: [`wwewtech/russian-it-community-lora`](https://huggingface.co/wwewtech/russian-it-community-lora). See full catalog in [`reports/LORA_MODEL_ZOO.md`](reports/LORA_MODEL_ZOO.md).

| Base Model | Family | Parameters | LoRA Size | Local Path |
| :--- | :--- | :---: | :---: | :--- |
| `Qwen/Qwen2.5-0.5B-Instruct` | Qwen 2.5 | 0.5B | 8.27 MB | [`lora_adapters/qwen2.5_0.5b_instruct/`](lora_adapters/qwen2.5_0.5b_instruct/) |
| `Qwen/Qwen2.5-1.5B-Instruct` | Qwen 2.5 | 1.5B | 16.65 MB | [`lora_adapters/qwen2.5_1.5b_instruct/`](lora_adapters/qwen2.5_1.5b_instruct/) |
| `Qwen/Qwen2.5-3B-Instruct` | Qwen 2.5 | 3.0B | 28.16 MB | [`lora_adapters/qwen2.5_3b_instruct/`](lora_adapters/qwen2.5_3b_instruct/) |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Qwen 2.5 Coder | 1.5B | 16.65 MB | [`lora_adapters/qwen2.5_coder_1.5b_instruct/`](lora_adapters/qwen2.5_coder_1.5b_instruct/) |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | DeepSeek R1 | 1.5B | 16.65 MB | [`lora_adapters/deepseek_r1_distill_qwen_1.5b/`](lora_adapters/deepseek_r1_distill_qwen_1.5b/) |
| `unsloth/Llama-3.2-1B-Instruct` | Llama 3.2 | 1.0B | 13.02 MB | [`lora_adapters/llama_3.2_1b_instruct/`](lora_adapters/llama_3.2_1b_instruct/) |
| `unsloth/Llama-3.2-3B-Instruct` | Llama 3.2 | 3.0B | 35.03 MB | [`lora_adapters/llama_3.2_3b_instruct/`](lora_adapters/llama_3.2_3b_instruct/) |
| `Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct` | Vikhr NLP | 1.5B | 16.65 MB | [`lora_adapters/vikhr_qwen_2.5_1.5b/`](lora_adapters/vikhr_qwen_2.5_1.5b/) |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | SmolLM2 | 1.7B | 24.02 MB | [`lora_adapters/smollm2_1.7b_instruct/`](lora_adapters/smollm2_1.7b_instruct/) |

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "lora_adapters/qwen2.5_1.5b_instruct"
)  # or "wwewtech/russian-it-community-lora"
model = PeftModel.from_pretrained(base, "lora_adapters/qwen2.5_1.5b_instruct")
```

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
├── reports/                    # Consolidated reports and scientific benchmarks
│   ├── DATASET_AND_ANALYTICS.md   # Dataset Card, 2017-2026 Analytics, and Zero-PII Protocol
│   ├── LORA_MODEL_ZOO.md          # Full Catalog of 44+ LoRA Adapters & Flagship 7B-8B QLoRA
│   ├── BENCHMARK_AND_EVALUATION.md# OpenAI HumanEval pass@1, RuMMLU CS, PPL, and 50 Scenarios
│   └── metrics_index.json         # Unified machine-readable telemetry and audit matrices
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
