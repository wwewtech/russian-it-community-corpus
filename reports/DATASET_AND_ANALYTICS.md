---
license: mit
pretty_name: "Russian IT Community Corpus (2.91M Conversations)"
task_categories:
  - text-generation
  - question-answering
language:
  - ru
  - en
tags:
  - russian-nlp
  - sft
  - dpo
  - rag
  - lora
  - zero-pii
  - it-community
  - software-engineering
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

# 📦 Russian IT Community Corpus: Dataset Card & Statistical Analytics

> **Official Hugging Face Hub Dataset:** [`wwewtech/russian-it-community-corpus`](https://huggingface.co/datasets/wwewtech/russian-it-community-corpus)  
> **Volume:** 2.91M raw messages (2.81M clean) · 171.5k SFT dialogues · 325.7k RAG knowledge chunks  
> **Timeline:** 2017–2026 (3,303 days continuous history) · 11 community nodes (`community_node_01`..`11`)  
> **License:** MIT · Research & Educational Use

---

## 💎 1. Обзор датасета и структура данных

**Russian IT Community Corpus** — масштабный деидентифицированный корпус технических дискуссий, архитектурных разборов, решения инцидентов и кода из 11 русскоязычных сообществ разработчиков, DevOps/SRE инженеров, архитекторов и исследователей за период с 2017 по 2026 год.

```text
                       ┌──────────────────────────────┐
 11 Community Nodes ──►   Multi-Source Ingestion      │ (2.91M records)
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │   Deep Case-Aware Zero-PII   │ (Declension across 6 cases,
                       │   RegEx + Neural Scrubber    │  PII Redaction Certificate)
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │   Deduplication & Taxonomy   │ (MinHash LSH 128 permutations,
                       │   8 Domain Classifiers       │  Exact Hash deduplication)
                       └──────────────┬───────────────┘
                                      │
                       ┌──────────────▼───────────────┐
                       │   Thread DAG Reconstruction  │ (Reply-tree traversal,
                       │   SFT, DPO, RAG Extraction   │  Temporal clustering)
                       └──────────────┬───────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  Apache Parquet  │        │   JSONL Formats  │        │  Vector KB / RAG │
│  zstd compressed │        │ ShareGPT, ChatML │        │  325.7k Chunks   │
│  full, sft, rag  │        │ Alpaca, DPO      │        │  BM25 / Embeddings│
└──────────────────┘        └──────────────────┘        └──────────────────┘
```

---

## 📊 2. Ключевые метрики датасета

| Метрика | Значение | Описание |
| :--- | :--- | :--- |
| **Очищенных записей** | `2 816 454` | Сообщения после дедупликации и деидентификации |
| **Уникальных авторов** | `210 890` | Псевдонимизированные идентификаторы (`user_xxxx`) |
| **Временной диапазон** | `06.08.2017 — 22.08.2026` | 3 303 дня непрерывной хронологии |
| **Суммарный объем слов** | `37 260 192` | Слова технического корпуса |
| **Оценка объема токенов** | `49 085 532` | BPE-токены (~49.09M) |
| **SFT многоходовые диалоги** | `171 533` | Диалоги с оценкой качества $\ge 3.0$ |
| **RAG чанки базы знаний** | `325 747` | Чанки с техническим контекстом |
| **DPO пары предпочтений** | `60 412` | Пары `chosen` / `rejected` для DPO/RLHF |

---

## 📂 3. Форматы экспорта и структура таблиц

### 3.1 Apache Parquet (zstd compression) в `dataset_output/parquet/`
1. `full_clean_messages.parquet` — полный дедуплицированный корпус с метаданными.
2. `sft_dialogues.parquet` — многоходовые диалоги (ShareGPT / ChatML структуры).
3. `rag_knowledge_base.parquet` — чанки базы знаний с таксономией и тегами.
4. `dpo_pairs.parquet` — пары предпочтений для выравнивания моделей.

### 3.2 Быстрый старт в Python
```python
from datasets import load_dataset

# Загрузка многоходовых диалогов SFT
sft_ds = load_dataset("wwewtech/russian-it-community-corpus", "sft_dialogues", split="train")
print(f"Loaded {len(sft_ds)} dialogues")

# Загрузка базы знаний RAG
rag_ds = load_dataset("wwewtech/russian-it-community-corpus", "rag_knowledge_base", split="train")
print(f"Loaded {len(rag_ds)} knowledge chunks")
```

---

## 🧠 4. Тематическая структура и доменное распределение

| Домен / Направление | Сообщений | Доля | Описание |
| :--- | :--- | :--- | :--- |
| **General Tech & Architecture** | 2,683,686 | 95.3% | Системный дизайн, паттерны, обсуждение технологий |
| **Business, Legal & FinTech** | 44,017 | 1.6% | Платежные шлюзы, комплаенс 152-ФЗ, PCI-DSS, b2b |
| **AI / ML / NLP & LLMs** | 29,411 | 1.0% | Обучение сетей, эмбеддинги, LoRA, Transformers |
| **Frontend & UI Architecture** | 18,775 | 0.7% | React, Vue, SSR, оптимизация бандлов, WebGL |
| **Management & Career** | 11,970 | 0.4% | Найм, грейды, онбординг, процессы в командах |
| **Backend & Distributed DBs** | 11,707 | 0.4% | PostgreSQL, Redis, Kafka, ClickHouse, шардинг |
| **Sysadmin & DevSecOps** | 9,970 | 0.3% | Безопасность, TLS, аудит уязвимостей, Linux kernel |
| **DevOps, K8s & Infrastructure**| 6,918 | 0.2% | Kubernetes, Terraform, CI/CD, мониторинг Prometheus |

---

## ⏰ 5. Временные паттерны и статистика активности

- **Пиковый час активности:** `21:00` (вечерний инженерный трафик)
- **Пиковый день недели:** `Вторник`
- **Средняя интенсивность:** `852.7` сообщений / день

```text
00:00 | ████████████████████                110,601
03:00 | █████                               29,200
06:00 | ██                                  13,738
09:00 | ████████████                        65,299
12:00 | ███████████████████████████████     171,781
15:00 | ████████████████████████████████    174,037
18:00 | ████████████████████████████████    173,991
21:00 | ███████████████████████████████████ 183,165
```

---

## 🛡️ 6. Протокол деидентификации и валидации
 
1. **Морфологическая деидентификация имен**: Детекция отображаемых имен участников и автоматическое склонение по **6 падежам русского языка** для вычищения упоминаний в тексте.
2. **Анонимизация источников**: Все названия чатов заменены на суррогатные идентификаторы узлов (`community_node_01`..`11`), а технические ID переиндексированы.
3. **Детерминированная очистка паттернов**:
   - Телефонные номера (RU/KZ/BY/International).
   - E-mail адреса и домены.
   - Криптовалютные кошельки (BTC, ETH, TON, TRON, Solana).
   - API-ключи, JWT токены, пароли и Database Connection Strings (`postgres://...`).
4. **Белый список терминов (Terminology Whitelist)**: 4,500+ технических терминов защищены от ложных срабатываний (`nginx`, `redis`, `docker`, `clickhouse`, `postgres`, `kubernetes`, `golang`).
5. **Процедура Notice & Takedown**: Обработка запросов на исключение сообщений в течение 48 часов.
