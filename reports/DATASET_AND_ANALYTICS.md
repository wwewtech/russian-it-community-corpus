---
license: mit
pretty_name: "Russian IT Community Corpus (2.91M Conversations)"
language:
  - ru
  - en
tags:
  - russian
  - nlp
  - sft
  - dpo
  - rag
  - lora
  - zero-pii
  - software-engineering
  - system-architecture
  - developer-conversations
task_categories:
  - text-generation
  - question-answering
size_categories:
  - 1M<n<10M
configs:
  - config_name: full_corpus
    data_files:
      - split: train
        path: data/full_clean_messages.parquet
    default: true
  - config_name: sft_dialogues
    data_files:
      - split: train
        path: data/sft_dialogues.parquet
  - config_name: rag_knowledge_base
    data_files:
      - split: train
        path: data/rag_knowledge_base.parquet
dataset_info:
  features:
    - name: msg_id
      dtype: int64
    - name: chat_name
      dtype: string
    - name: timestamp
      dtype: string
    - name: unixtime
      dtype: int64
    - name: author_anon
      dtype: string
    - name: text_clean
      dtype: string
    - name: domain
      dtype: string
    - name: tags
      sequence: string
    - name: sentiment_score
      dtype: int32
    - name: token_count_approx
      dtype: int32
    - name: is_question
      dtype: bool
    - name: thread_id
      dtype: int64
---

# 📦 Russian IT Community Corpus (RICC)

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Total Messages](https://img.shields.io/badge/Clean%20Messages-2.81M-blue.svg)](#key-metrics)
[![SFT Dialogues](https://img.shields.io/badge/SFT%20Dialogues-171.5k-purple.svg)](#sft-dialogues)
[![RAG Knowledge Chunks](https://img.shields.io/badge/RAG%20Chunks-325.7k-orange.svg)](#rag-knowledge-base)
[![Chronology](https://img.shields.io/badge/Timeline-2017--2026%20(9%20years)-brightgreen.svg)](#chronology)
[![Privacy](https://img.shields.io/badge/Privacy-Anonymized%20Nodes-success.svg)](#privacy-protocol)

</div>

**Russian IT Community Corpus (RICC)** is an open, de-identified conversational dataset collected from **11 engineering community nodes** spanning a 9-year timeline (**2017–2026**). It captures authentic discussions on backend systems, cloud infrastructure, AI/ML deployment, database internals, and software architecture.

The corpus is structured into ready-to-use splits for **Instruction Fine-Tuning (SFT)**, **Direct Preference Optimization (DPO)**, and **Vector Retrieval-Augmented Generation (RAG)**.

---

## ⚡ Quick Start

```python
from datasets import load_dataset

# 1. Multi-turn SFT Dialogues (171.5k curated conversations)
sft_ds = load_dataset("wwewtech/russian-it-community-corpus", "sft_dialogues", split="train")
print(f"Loaded SFT dataset: {len(sft_ds):,} dialogues")
print("Sample dialogue:", sft_ds[0]["messages"][:2])

# 2. RAG Technical Knowledge Base (325.7k segmented documents)
rag_ds = load_dataset("wwewtech/russian-it-community-corpus", "rag_knowledge_base", split="train")
print(f"Loaded RAG knowledge base: {len(rag_ds):,} chunks")

# 3. Full Chronological Corpus (2.81M deduplicated messages)
full_ds = load_dataset("wwewtech/russian-it-community-corpus", "full_corpus", split="train")
print(f"Loaded Full corpus: {len(full_ds):,} records")
```

---

## 🏗️ Data Curation Pipeline

<div align="center">

<img src="https://huggingface.co/datasets/wwewtech/russian-it-community-corpus/resolve/main/assets/pipeline_architecture.svg" alt="RICC Pipeline Architecture" width="100%" />

</div>

| Stage | Processing Module | Description & Output |
| :---: | :--- | :--- |
| **1. Ingestion** | Multi-Source Export Ingestion | Merging 11 community nodes (`community_node_01`..`11`) across 2017–2026 into 2.91M unified records. |
| **2. Anonymization** | Natasha NER & Case-Aware Scrubber | Inflection across 6 Russian cases, Telegram handle mapping, and regex scrubbing for phones, crypto, keys. |
| **3. Deduplication** | MinHash LSH (128 Permutations) | Duplicate and spam removal at 0.80 Jaccard threshold + 8-domain taxonomy classification. |
| **4. DAG Reconstruction**| Thread Builder & Extractor | Reconstructing reply trees, extracting 171.5k SFT dialogues ($\ge 3.0$ score) and 325.7k RAG chunks. |
| **5. Multi-Split Export**| Parquet & JSONL Exporters | Generating zstd Parquet (`full`, `sft`, `rag`) and JSONL splits (`ShareGPT`, `Alpaca`, `ChatML`, `DPO`). |

---

## 📊 Key Corpus Metrics

| Metric | Verified Value | Description |
| :--- | :--- | :--- |
| **Clean Messages** | `2,816,434` | Deduplicated and anonymized messages |
| **Unique Participants** | `210,890` | Pseudonymized author identifiers (`Developer_XXXXX`) |
| **Date Range** | `Aug 06, 2017 — Aug 22, 2026` | 3,303 continuous days of community history |
| **Total Words** | `37,260,192` | Technical Russian and mixed English terminology |
| **Estimated BPE Tokens** | `49,085,532` | BPE token count approximation (~49.09M tokens) |
| **SFT Dialogues** | `171,520` | Multi-turn threads scored for technical depth ($\ge 3.0$) |
| **RAG Knowledge Chunks** | `325,690` | Cohesive problem-solving context blocks |
| **DPO Preference Pairs** | `60,899` | Pairs with chosen answers and heuristic negative baselines |

---

## 🧠 Domain & Topic Distribution

| Domain Category | Message Count | Share (%) | Core Topics |
| :--- | :---: | :---: | :--- |
| **General Tech & Architecture** | 2,683,686 | 95.3% | System design, design patterns, tooling debates, engineering culture |
| **Business, FinTech & Compliance** | 44,017 | 1.6% | Payment gateways, 152-FZ compliance, billing logic, enterprise SaaS |
| **AI, ML & LLM Engineering** | 29,411 | 1.0% | Transformers, fine-tuning, quantization, embeddings, inference infra |
| **Frontend & UI Architecture** | 18,775 | 0.7% | React, Vue, SSR, bundle optimization, state management, WebGL |
| **Engineering Management & Career** | 11,970 | 0.4% | Hiring, grading, architectural review processes, incident culture |
| **Backend & Distributed DBs** | 11,707 | 0.4% | PostgreSQL tuning, Redis caching, ClickHouse analytics, Kafka streams |
| **Sysadmin & DevSecOps** | 9,970 | 0.3% | Linux kernel, TLS certificates, vulnerability auditing, network debugging |
| **DevOps & Cloud Infrastructure** | 6,918 | 0.2% | Kubernetes, Docker, CI/CD pipelines, Prometheus monitoring |

---

## 📁 Dataset Splits & Configurations

### 1. `full_corpus` (`data/full_clean_messages.parquet`)
Full chronological sequence of clean messages with domain labels, sentiment, and structural metadata.

| Column | Type | Description |
| :--- | :--- | :--- |
| `msg_id` | `int64` | Surrogate message identifier |
| `chat_name` | `string` | Surrogate community node (`community_node_01`..`11`) |
| `timestamp` | `string` | ISO 8601 formatted timestamp |
| `unixtime` | `int64` | UNIX epoch timestamp |
| `author_anon` | `string` | Pseudonymized author label (`Developer_XXXXX`) |
| `text_clean` | `string` | Anonymized message text |
| `domain` | `string` | Primary classified engineering domain |
| `tags` | `list[str]` | Detected technical keyword tags |
| `is_question` | `bool` | True if the message contains an engineering inquiry |
| `thread_id` | `int64` | Identified conversation DAG thread ID |

### 2. `sft_dialogues` (`data/sft_dialogues.parquet`)
Reconstructed multi-turn conversation threads formatted for supervised instruction fine-tuning.

```json
{
  "thread_id": 42056,
  "chat_name": "community_node_07",
  "topic_domain": "frontend_ui",
  "topic_tags": ["vue", "js", "di_container", "architecture"],
  "quality_score": 12.25,
  "messages": [
    {
      "role": "user",
      "author": "Developer_65546",
      "content": "Как изолировать ядро CMS при использовании Vue на фронтенде?"
    },
    {
      "role": "assistant",
      "author": "Developer_38544",
      "content": "Для изоляции выносите API в независимый сервисный слой..."
    }
  ]
}
```

### 3. `rag_knowledge_base` (`data/rag_knowledge_base.parquet`)
Chunked technical discussions formatted for dense embedding indexing (Qdrant, ChromaDB, Milvus, FAISS).

---

## 📊 SFT Subset — Real Composition & Known Limitations

Computed directly from `data/sft_dialogues.parquet` (171,520 dialogues):

| Property | Value |
| :--- | :--- |
| Dialogues whose first turn contains an actual question | 20.6% (35,402) |
| Dialogues classified as `general_tech_chat` (no specific domain) | **97.1%** (166,624) |
| Dialogues in a concrete technical domain | 2.9% (4,896) |
| …of those, passing a strict QA filter (question-first opener + every turn ≥100 chars) | **283** |
| Median heuristic `quality_score` (≈1 trivial → 4+ substantive) | 2.36 (p25 1.97 / p75 2.96) |

**Known limitations — read before training on this subset:**
- This is reconstructed **community chat**, not curated instruction data. Many "assistant" turns are opinionated chat replies rather than expert answers.
- The `quality_score` heuristic rewards length and code markers; it does **not** measure factual correctness.
- For serious SFT runs, filter aggressively: exclude `general_tech_chat`, require question-first openers and multi-turn substantive answers. That leaves ~283–4,900 dialogues depending on strictness — small but much cleaner than the full set.

A representative technical exchange that does pass the filter (`thread_id` 30449, `business_legal_fintech`):

```json
{
  "role": "user",
  "content": "От map(filter(...)) уже тошнит? Серьезно, когда вы видите вот это, то хочется плакать:
              result = list(map(lambda x: x * 2, filter(lambda x: x % 2 == 0, arr)))
              А ведь можно проще..."
},
{
  "role": "assistant",
  "content": "В этом месте JS выглядит читаемее чем Python
              users.filter(user => user.gender === gender)
                .map(user => user.age)
                .reduce((acc, age, index, arr) => acc + age / arr.length, 0);"
}
```

---

## 🔬 Empirical Evaluation — Honest Status

> 🚫 **Academic metrics (HumanEval / RuMMLU / PPL / ROUGE) published earlier have been WITHDRAWN.**
> A code audit of the benchmark harness (`src/evaluation/official_academic_benchmarks.py`) found three defects that produced invalid numbers: substring-based MCQ scoring (inflated RuMMLU to an implausible 100%), PPL computed on empty placeholder strings due to wrong column names (identical base/LoRA values), and silent copying of Base results into LoRA/Hybrid columns when the adapter failed to load. All three are fixed in code with unit tests; the numbers will be republished only after a fresh GPU re-run.

What remains interpretable today — **rubric-based heuristic scores** on 50 engineering scenarios
(concept-overlap + AST-parseability judged programmatically; *not* execution-verified capability):

| Architecture Setup | Heuristic Score | AST Parse Rate |
| :--- | :---: | :---: |
| Base Model (Qwen 2.5 1.5B) | 32.9 | 69.0% |
| Base Model + RAG (325k chunks) | 44.0 | 71.0% |
| Domain LoRA (171.5k dialogues) | 34.5 | 72.2% |
| Hybrid (LoRA + RAG) | **48.6** | **73.0%** |

*(AST rates recomputed as means over all 50 per-scenario `ast_score` values in [`reports/heuristic_benchmark_eval.json`](heuristic_benchmark_eval.json); an earlier version of this card quoted different numbers that did not match the machine-readable data.)*

Pre-trained adapters for **58** base models are available in the [LoRA Model Zoo](https://huggingface.co/wwewtech/russian-it-community-lora).

---

## 🛡️ Privacy, Anonymization & Ethical Use

1. **Morphological Name Scrubbing**: Author display names are extracted and declined across all **6 Russian grammatical cases** (Им., Род., Дат., Вин., Твор., Предл.) to remove conversational references in text.
2. **Community Node Anonymization**: All 11 channel titles and supergroup IDs are strictly anonymized as surrogate nodes (`community_node_01`..`11`).
3. **Deterministic Pattern Scrubbing**: Removes phone numbers, personal emails, crypto wallet addresses (BTC, ETH, TRON, TON, SOL), API keys (`sk-proj-...`, `ghp_...`), JWT tokens, and database credentials.
4. **Terminology Whitelist**: 4,500+ standard programming keywords, frameworks, and tools are protected against accidental redaction.
5. **Notice and Takedown Policy**: Intended strictly for educational, academic, and non-commercial research. If you identify any inadvertent personal identifier, please open a takedown issue or submit a removal request. Requests are processed within **48 hours**.

---

## 📖 Citation

```bibtex
@misc{ricc2026,
  author = {Russian IT Community Open Research Group},
  title = {Russian IT Community Corpus (RICC): A Curated Multi-Domain Conversational Dataset for LLM SFT, DPO, and RAG},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/wwewtech/russian-it-community-corpus}}
}
```
