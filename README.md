<div align="center">

<!-- Theme-aware wordmark (GitHub light / dark) -->
<img src="./assets/banner_light.svg#gh-light-mode-only" alt="RICC" width="480" />
<img src="./assets/banner_dark.svg#gh-dark-mode-only" alt="RICC" width="480" />

<br />

**High-throughput data engineering and Probabilistic PII Audit platform for language models**

2,816,434 messages (Hub claim) · 2,816,434 (local claim) · 2017–2026 history · SFT dialogues · DPO pairs · RAG knowledge base · LoRA on RTX 3060

<br />

[![CI](https://github.com/wwewtech/russian-it-community-corpus/actions/workflows/ci.yml/badge.svg)](https://github.com/wwewtech/russian-it-community-corpus/actions/workflows/ci.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E?style=flat-square)](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LoRA%20Zoo-orange?style=flat-square)](https://huggingface.co/wwewtech/russian-it-community-lora)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20CUDA-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20and%20QLoRA-8A2BE2?style=flat-square)](#)
[![Parquet](https://img.shields.io/badge/Apache%20Parquet-zstd-017CEE?style=flat-square&logo=apache)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-Data%20Studio-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](#)
[![Probabilistic PII Audit](https://img.shields.io/badge/Security-Probabilistic%20PII%20Audit%20(99%25%20CI%20%E2%89%A4%200.01%25)-10B981?style=flat-square)](#security-and-probabilistic-pii-audit)
</div>

> [!TIP]
> **Official Hugging Face Hub Integration**:
> - **Dataset**: [`wwewtech/russian-it-community-corpus`](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus) — dataset snapshot claims 2,816,434 clean messages, 171.5k multi-turn SFT dialogues, and 325.7k RAG knowledge base chunks in Apache Parquet. *Row counts are self-reported by the snapshot; independent re-verification is pending.*
> - **LoRA Model Zoo**: [`wwewtech/russian-it-community-lora`](https://huggingface.co/wwewtech/russian-it-community-lora) — 58 adapter directories on Hugging Face Hub (55 domain adapters + 3 flagship QLoRA: Qwen2.5 Coder 7B, DeepSeek R1 Distill 7B, LLaMA 3.1 8B). Local registry [`lora_adapters/registry.json`](lora_adapters/registry.json) tracks 56 adapters; **only `smollm2_135m_instruct` is policy-compliant** (full epoch, 100% corpus coverage). The remaining 55 are pilot/smoke-test adapters and must not be treated as final.
>
> ```python
> from datasets import load_dataset
> dataset = load_dataset("wwewtech/russian-it-community-corpus", "sft_dialogues", split="train")
> ```

---

## Overview

**RICC** (**R**ussian **I**T **C**ommunity **C**orpus) is an open data engineering and curation stack that ingests, cleans, deduplicates, and structures engineering, infrastructure, business, and software development messages from 11 community nodes spanning 2017–2026.

> **Row count note**: The Hugging Face Hub snapshot claims 2,816,434 messages; the local working copy claims the same. These are **self-reported, not independently re-verified**. Earlier pipeline runs diverged from the published snapshot by up to ~14%; the authoritative check is [`reports/dataset_reconciliation_report.json`](reports/dataset_reconciliation_report.json) against the pinned Hub revision `81a3495`.

The platform produces datasets for instruction fine-tuning, direct preference optimization, and a retrieval-augmented knowledge base.

| Metric | Property | Factual Value |
|---|---|---|
| Privacy & Anonymization | Multi-pass Regex + Natasha NER + case declension | 11 community nodes anonymized (`community_node_01`..`11`) |
| Deduplication | MinHash LSH (128 permutations, 0.80 Jaccard threshold) | 95,300+ duplicate / spam messages removed *(self-reported; pending re-verification)* |
| SFT dialogue quality | Multi-turn dialogues extracted from reconstructed thread DAGs | 171,520 curated dialogues *(Hub + local claim; pending re-verification)* |
| Local LoRA execution | PEFT QLoRA adaptation on consumer hardware | ~4.35 GB VRAM on RTX 3060 (12GB) |

---

## Architecture

```text
                       ┌──────────────────────────────┐
   11 Community Nodes ──►   Multi-Source Ingestion      │
    2,816,434 raw records   └──────────────┬───────────────┘
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
| Ingestion | `ricc/ingestion/` | Normalizes and chronologically merges multi-chat raw exports (11 sources) |
| Anonymization | `ricc/pii/` | Redacts names across 6 cases, phone numbers, crypto wallets, API tokens, database URLs |
| Deduplication | `ricc/deduplication/` | Filters near-duplicate and exact spam messages via MinHash LSH |
| Taxonomy | `ricc/taxonomy/` | Categorizes content into 8 technical domains and extracts keyword tags |
| Thread DAG | `ricc/graph/` | Reconstructs conversational trees and extracts multi-turn dialogues |
| Multi-Export | `ricc/exporter/` | Serializes outputs into Apache Parquet with zstd compression and JSONL formats |
| Analytics | `ricc/analytics/` | Computes Shannon entropy, temporal patterns, social graphs, and vocabulary stats |
| Local LoRA and RAG | `ricc/lora/`, `ricc/rag/` | Provides PEFT training for RTX 3060 and hybrid lexical + TF-IDF retrieval over 325k chunks (no embedding model, no vector database) |

---

## Official Datasets & Direct Access Links

The corpus and trained adapters are available both remotely on **Hugging Face Hub** and locally in `dataset_output/`:

### Hugging Face Hub Repositories:
- **Dataset Hub (Full Corpus, SFT, RAG)**:
  [**`https://huggingface.co/datasets/wwewtech/russian-it-community-corpus`**](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus)
  - [Full Clean Corpus (Parquet)](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus/blob/main/data/full_clean_messages.parquet) — 2.82M rows *(Hub claim: 2,816,434 records, 189 MB; self-reported)*
  - [SFT Dialogues (Parquet)](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus/blob/main/data/sft_dialogues.parquet) — 171.5k multi-turn dialogues *(Hub claim; 132 MB; self-reported)*
  - [RAG Knowledge Base (Parquet)](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus/blob/main/data/rag_knowledge_base.parquet) — 325.7k knowledge chunks *(Hub claim; 159 MB; self-reported)*
  - [Heuristic Benchmark Evaluation (JSON)](reports/heuristic_benchmark_eval.json) — 50-scenario matrix of **rubric-based heuristics** (concept-overlap + AST-parseability), not capability measurements

- **Model Hub (LoRA Adapters)**:
  [**`https://huggingface.co/wwewtech/russian-it-community-lora`**](https://huggingface.co/wwewtech/russian-it-community-lora)
  - 58 adapter directories on Hub; local registry [`lora_adapters/registry.json`](lora_adapters/registry.json) tracks 56 adapters.
  - Only `smollm2_135m_instruct` is policy-compliant (full epoch, 100% corpus coverage). The remaining 55 are pilot/smoke-test adapters.
  - Flagship Qwen 2.5 Coder 7B Adapter, DeepSeek R1 Distill 7B Adapter, Meta LLaMA 3.1 8B Adapter exist on Hub but are **not policy-compliant** (partial epochs, time-boxed runs).

---

## Local Datasets and Formats

All datasets are automatically generated and saved in `dataset_output/`. Exact artifact identity — sha256, byte size, row count and schema — is snapshotted in [`reports/dataset_manifest.json`](reports/dataset_manifest.json) (`python -m ricc.validation.artifact_manifest create|verify`); the SLO gate re-verifies it on every release check. The Hugging Face Hub hosts a curated snapshot that may differ from the local working copy, so both are listed:

| File Path | Format | Local (verified) | Hub snapshot | Description | Direct Link |
|---|---|---|---|---|:---:|
| `dataset_output/parquet/full_clean_messages.parquet` | Parquet (zstd) | 2,816,434 rows *(local claim; self-reported)* | 2,816,434 rows *(Hub claim; self-reported)* | Full cleaned corpus with metadata | [HF Mirror](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus/blob/main/data/full_clean_messages.parquet) |
| `dataset_output/parquet/sft_dialogues.parquet` | Parquet (zstd) | 171,520 dialogues *(local claim)* | 171,520 dialogues *(Hub claim)* | Multi-turn dialogues for SFT | [HF Mirror](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus/blob/main/data/sft_dialogues.parquet) |
| `dataset_output/parquet/rag_knowledge_base.parquet` | Parquet (zstd) | 325,690 chunks *(local claim)* | 325,690 chunks *(Hub claim)* | Vector knowledge base | [HF Mirror](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus/blob/main/data/rag_knowledge_base.parquet) |

JSONL counts below are line-validator `valid_lines` from [`reports/validation_results.json`](reports/validation_results.json), surfaced through [`scripts/metrics_source.py`](scripts/metrics_source.py):

| File Path | Format | Verified volume | Description | Availability |
|---|---|---|---|---|
| `dataset_output/jsonl/sft_openai_messages.jsonl` | ChatML JSONL | 173,216 records *(self-reported)* | OpenAI format for Unsloth / TRL | Local / HF |
| `dataset_output/jsonl/sft_sharegpt_format.jsonl` | ShareGPT JSONL | 173,216 records *(self-reported)* | Axolotl & LLaMA-Factory format | Local / HF |
| `dataset_output/jsonl/sft_alpaca_format.jsonl` | Alpaca JSONL | 660,900 pairs *(self-reported)* | Single-turn instruction-response pairs | Local / HF |
| `dataset_output/jsonl/rag_chunks_kb.jsonl` | RAG JSONL | 332,690 chunks *(self-reported)* | Segmented technical documents | Local / HF |
| `dataset_output/jsonl/dpo_preference_pairs.jsonl` | DPO JSONL | 57,501 pairs *(self-reported)* | Chosen / Rejected alignment pairs | Local / HF |

---

## Comparative Architectural Evaluation (Base vs RAG vs LoRA vs Hybrid)

> [!WARNING]
> **Benchmark section withdrawn from README. Benchmark results RETRACTED — do not cite.**
>
> The comparison was withdrawn after a code audit of
> [`ricc/evaluation/official_academic_benchmarks.py`](ricc/evaluation/official_academic_benchmarks.py)
> found three critical defects:
>
> 1. **Substring MCQ scoring** — "A" matched "Answer: A", inflating accuracy
> 2. **PPL on empty placeholders** — perplexity computed on blank generations
> 3. **Silent Base→LoRA copy** — when the adapter failed to load, the base
>    results were copied into the LoRA/Hybrid columns without warning
>
> **All three are now fixed** (the suites fail fast without the adapter, the
> MCQ parser needs an explicit answer marker, the perplexity is computed on
> real text), and a fourth defect was found in the process: the hybrid column
> fell back to the "Base + RAG" result, and the hybrid PPL/ROUGE cells were
> copies of the LoRA cells although they were never measured.
>
> **But the published numbers are still invalid** — they were produced by the
> broken code, and re-running them needs a GPU. A fresh re-run is required
> before any accuracy claim can be made, so the dataset and the adapters ship
> **without benchmark scores**.
>
> What the raw outputs *do* show, once the comparison is done correctly: the
> variants run on the same tasks, so the comparison is paired and McNemar's
> exact test applies — and **no comparison reaches significance**
> (p ≥ 0.125 on both benchmarks). The point estimates differ; the data does
> not support calling one of them better. The report now says exactly that
> instead of claiming an improvement.
>
> The 100-question **non-executing** domain benchmark suite remains available in
> [`reports/domain_benchmark_100.json`](reports/domain_benchmark_100.json) as a corpus only.

---

## Quick start

### Requirements

- Python 3.11, 3.12, 3.13, or 3.14
- NVIDIA GPU with 8 GB or more VRAM for training, or CPU for data processing
- Docker with Docker Compose for containerized runs

### Direct Loading from Hugging Face

Load pre-built dataset splits in Python with a single line:

```python
from datasets import load_dataset

# 1. Load SFT Multi-Turn Dialogues (Hub claim: 171.5k dialogues)
sft_ds = load_dataset("wwewtech/russian-it-community-corpus", "sft_dialogues", split="train")

# 2. Load RAG Knowledge Base Chunks (Hub claim: 325.7k chunks)
rag_ds = load_dataset("wwewtech/russian-it-community-corpus", "rag_knowledge_base", split="train")

# 3. Load Full Clean Messages (Hub claim: 2.81M records)
full_ds = load_dataset("wwewtech/russian-it-community-corpus", "full_corpus", split="train")
```

### LoRA Model Zoo (56 Local Adapters / 58 Adapter Directories on Hugging Face Hub)

Pre-trained adapters fine-tuned in pilot domain-adaptation runs on RICC dataset are available locally in [`lora_adapters/`](lora_adapters/) (56 adapters tracked in `registry.json`) and on Hugging Face: [`wwewtech/russian-it-community-lora`](https://huggingface.co/wwewtech/russian-it-community-lora) (58 adapter directories). The local [`lora_adapters/registry.json`](lora_adapters/registry.json) is the single source of truth for on-disk adapters. **Only `smollm2_135m_instruct` is policy-compliant**; the remaining 55 local adapters (and all Hub directories) are pilot/smoke-test adapters and must not be treated as final. Full hand-maintained catalog lives in [`reports/LORA_MODEL_ZOO.md`](reports/LORA_MODEL_ZOO.md).

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

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto", torch_dtype="auto")
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
# a smoke test (15 steps, 400 dialogues) — proves the pipeline fits on your GPU
python -m ricc.lora.batch_train_zoo --only qwen2.5_0.5b_instruct --profile pilot

# a real run: one full epoch over the corpus (~13 h on a 1.5B model on RTX 3060; see AGENTS.md for the canonical command)
python -m ricc.lora.batch_train_zoo --only qwen2.5_1.5b_instruct --profile full
```

### 5. Run LoRA Inference

```bash
python ricc/lora/generate_demo.py --prompt "Как настроить прием платежей для SaaS из РФ?"
```

### Command Reference

| Command | Action |
|---|---|
| `python main.py` | Run complete pipeline on all available chat exports |
| `python cli.py analyze` | Generate analytical reports and metrics |
| `python cli.py validate` | Validate dataset schema and check for PII leaks |
| `python cli.py audit-prob` | Run the stratified probabilistic PII audit (Wilson 99% bounds) |
| `python cli.py drift` | Run dataset drift monitoring against a frozen reference snapshot |
| `python cli.py benchmark` | Export and inspect 100 domain test cases |
| `python cli.py orchestrate` | Run the Prefect curation orchestration flow |
| `python demo_walkthrough.py` | Run interactive terminal walkthrough |
| `python -m pytest -q` | Run the automated test suite *(the full 905 tests — pending re-verification in an environment with pytest installed)* |
| `make test` | Same, through the Makefile |
| `make coverage` | Test run with the coverage report |
| `make audit` | Run red-team adversarial penetration test |
| `make audit-prob` | Run the probabilistic PII audit |
| `make slo` | Run the SLO release gate (SHIP/HOLD) |
| `make dvc-repro` | Reproduce the DVC data pipeline |
| `make docker-up` | Start Web Data Studio in Docker container |

> **Note on the test command.** Use `pytest`, not `python -m unittest discover`:
> the unittest runner only collects 519 of the 905 tests *(claim pending re-verification)*,
> because six test modules use pytest-specific features and are silently skipped by it.

---

## Project Structure

```text
├── assets/                     # Theme-aware vector graphics
├── ricc/
│   ├── ingestion/              # Multi-chat JSON export parser
│   ├── pii/                    # Case-aware PII anonymizer and neural NER
│   ├── graph/                  # Thread DAG builder and dialogue extractor
│   ├── deduplication/          # MinHash LSH and exact text hashing
│   ├── taxonomy/               # Domain classifiers and keyword taggers
│   ├── exporter/               # Apache Parquet and JSONL serializers
│   ├── analytics/              # DeepChatAnalyzer statistical engine
│   ├── rag/                    # Hybrid lexical + TF-IDF retrieval and prompt augmentation
│   ├── lora/                   # PEFT training and inference scripts
│   ├── evaluation/             # Benchmark comparator
│   └── validation/             # PII auditor and domain benchmarks
├── dataset_output/             # Parquet datasets and preview samples
├── reports/                    # Canonical dataset & model cards and benchmark suites
│   ├── DATASET_AND_ANALYTICS.md   # Dataset Card and Zero-PII Protocol
│   ├── HF_MODEL_CARD.md           # Hugging Face Model Card
│   ├── LORA_MODEL_ZOO.md          # Catalog of LoRA Adapters on Hub (58 directories) / local registry (56)
│   ├── heuristic_benchmark_eval.json # 50-Scenario Heuristic Benchmark Matrix
│   └── domain_benchmark_100.json  # 100-Scenario Domain Benchmark Suite
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

## Privacy and Probabilistic PII Audit

1. **Morphological Name Redaction**: Detects author display names and inflects them across 6 Russian grammatical cases to eliminate conversational mentions.
2. **Deterministic Pattern Scrubbing**: Removes phone numbers in international formats, email addresses, cryptocurrency wallet addresses, API keys, tokens, and database connection strings.
3. **Source Community Anonymization**: Replaces all source channel titles with surrogate identifiers (`community_node_01`..`11`) and re-indexes technical IDs.
4. **Terminology Protection**: Whitelists common technical terms, programming languages, libraries, and hosting providers to prevent false positives.
5. **Probabilistic PII Audit**: Stratified sampling (100,000 messages, 99% confidence) with Wilson score intervals over **12 leak categories** (phones, emails, crypto wallets, API keys, JWT, SSH keys, public IPs, secret assignments, private invite links, credentialed DB URLs, Telegram forward headers, raw user mentions). **Verdict: PASSED** — the 99% upper bound on message-level leak rate is ≤ 0.01% (configurable tolerance). See [`reports/probabilistic_pii_audit.json`](reports/probabilistic_pii_audit.json) and [`ricc/validation/probabilistic_audit.py`](ricc/validation/probabilistic_audit.py).
6. **Automated Red-Team Suite**: 25,000-message sample + adversarial vectors (declensions, obfuscated phones, DB URLs, API keys, crypto wallets, Telegram forwards). Pass threshold: ≤1 vector failure AND zero production leaks. Report: [`reports/pii_validation_report.json`](reports/pii_validation_report.json).

> **Note**: "Zero-PII" is an aspirational target. The technical guarantee is a *statistical upper bound* on leak rate, not an absolute full-corpus scan.

---

## Quality Gates

The platform ships the gates that make its claims checkable. Every one of
them fails closed: a gate that cannot verify a thing reports a failure
rather than a pass.

| Gate | What it proves | Where |
|---|---|---|
| **Probabilistic PII audit** | The 99% upper bound on the message-level leak rate stays below the tolerance, over 12 categories | `ricc/validation/probabilistic_audit.py` |
| **Training policy gate** | An adapter published to the Hub covered a full epoch of the corpus, and was not a time-boxed run | `ricc/lora/policy_gate.py` |
| **SLO gate** | Validation, PII audit, drift, SFT quality and artifact provenance all pass before a release | `ricc/monitoring/slo_gate.py` |
| **Artifact manifest** | The published parquet files are byte-identical to the snapshot that was audited | `ricc/validation/artifact_manifest.py` |
| **SFT quality monitor** | The real corpus (not a fixture) meets the dialogue-quality floors | `ricc/monitoring/sft_quality.py` |
| **DVC lock** | The pipeline can be reproduced from the pinned raw exports | `dvc.lock`, `dvc.yaml` |

The gates are wired into CI, so a regression breaks the build rather than
the next release. See [`AGENTS.md`](AGENTS.md) for the binding training
policy and [`docs/development/testing.md`](docs/development/testing.md) for
the test layout.

### Reproducing the pipeline

```bash
python -m pip install ".[platform]"   # adds DVC and Prefect
python scripts/refresh_dvc_deps.py     # list the raw exports as DVC deps
dvc repro                              # rebuild every stage from the raw data
dvc push                               # publish the artefacts to the remote
```

Each stage is a real script (`scripts/dvc_stages.py <stage>`), so a single
stage can be re-run and inspected on its own.

---

## License and Ethical Use

- **Academic and Research Use**: Provided for educational, non-commercial, and machine learning research purposes.
- **Privacy Policy**: All data has undergone heuristic multi-pass PII scrubbing and anonymization.
- **Notice and Takedown**: If you identify any inadvertent personal identifier or content that should be excluded, please open a takedown issue or submit a removal request. Requests are processed within 48 hours.

---

<div align="center">

<!-- Theme-aware Logo -->
<img src="./assets/logo_light.svg#gh-light-mode-only" alt="RICC" width="36" />
<img src="./assets/logo_dark.svg#gh-dark-mode-only" alt="RICC" width="36" />

<sub>RICC · Russian IT Community Corpus</sub>

</div>
