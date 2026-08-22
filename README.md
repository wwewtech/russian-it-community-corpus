# 🚀 Russian IT Community Data Engineering & Curation Pipeline

Production-grade конвейер обработки, деидентификации (Zero-PII), дедупликации (MinHash LSH), реконструкции диалоговых графов (Thread DAG), глубокой семантической аналитики и экспорта датасетов для обучения языковых моделей (LLM SFT, DPO, RAG).

---

## 🌟 Возможности проекта

1. **Масштабный Ingestion & Merging**:
   - Автоматическая загрузка, нормализация и хронологическое слияние экспортов чатов Telegram (`ChatExport_2026-08-21` и `ChatExport_2026-08-22`) объемом свыше **530,000 сообщений за 8 лет (2018–2026)**.

2. **Двухуровневая деидентификация (Zero-PII Protocol)**:
   - **RegEx-контур**: Детерминированное маскирование телефонов РФ/мира, email-адресов, криптокошельков (BTC, ETH, TRON TRC20, TON), API-ключей (OpenAI `sk-...`, GitHub `ghp_...`, Telegram Bot tokens, AWS, JWT) и приватных ссылок.
   - **NER-контур (Natasha)**: Распознавание личных имен (`PER`) и локаций (`LOC`) с защитным списком (*Smart Tech Whitelist*) сотен IT-брендов, языков и библиотек.
   - **Сквозная псевдонимизация**: Преобразование пользователей в стабильные идентификаторы `Developer_XXXXX`.

3. **Нечеткая дедупликация (MinHash LSH + Exact Hashing)**:
   - 128 перестановок MinHash с Locality-Sensitive Hashing для отсева спама, ботов и кросс-постов (Jaccard similarity $\ge 0.80$).

4. **Реконструкция диалоговых деревьев (DAG Resolution)**:
   - Восстановление иерархических графов бесед по связям `reply_to_message_id` и временным окнам.
   - Извлечение многоходовых диалогов (Multi-turn SFT) и пар предпочтений (DPO).

5. **Мультиформатный экспортер для ML**:
   - **Apache Parquet (`.parquet`)**: сжатый формат с zstd для быстрой загрузки.
   - **ShareGPT JSONL**: для `Axolotl`, `FastChat`, `LLaMA-Factory`.
   - **Alpaca JSONL**: формат `instruction` / `output`.
   - **OpenAI ChatML JSONL**: формат `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}` для `Unsloth` и `TRL`.
   - **RAG Knowledge Base**: сегментированные чанки (500–1000 токенов) с метаданными для векторных баз (`Qdrant`, `Chroma`, `pgvector`).
   - **DPO Preference Pairs**: пары для Direct Preference Optimization.

6. **Глубокий аналитический модуль (Deep Analytics v4.0 Enterprise)**:
   - 800+ строк аналитического кода: расчет статистик длин, лексического разнообразия (Shannon Entropy), динамики активности по часам, дням и годам (2018–2026), выявление IT-сленга, социальный граф и рейтинг инфлюенсеров, LDA тематическое моделирование и оценка коммерческой готовности датасета.

7. **Пакет валидации и доменный бенчмарк**:
   - Автоматический аудит на отсутствие утечек PII, валидация схем данных и 100 тестовых вопросов по IT-бизнесу, бэкенду, DevOps и AI.

---

## 📁 Структура проекта

```
D:\project_x\
├── src\
│   ├── ingestion\              # Загрузка и нормализация Telegram экспортов
│   │   ├── loader.py
│   │   └── schema.py           # Pydantic модели данных
│   ├── pii\                    # Двухконтурная очистка PII и псевдонимизация
│   │   ├── regex_scrubber.py   # Телефоны, email, крипта, токены, ключи
│   │   ├── ner_scrubber.py     # Natasha NER для персон и локаций + Whitelist
│   │   └── anonymizer.py       # Менеджер консистентных авторов
│   ├── graph\                  # Реконструкция диалоговых деревьев (DAG)
│   │   ├── thread_builder.py   # Граф ответов и временная кластеризация
│   │   └── conversation_extractor.py # Извлечение SFT, DPO и RAG данных
│   ├── deduplication\          # Дедупликация сообщений
│   │   ├── minhash_lsh.py      # Нечеткая LSH дедупликация (Jaccard >= 0.80)
│   │   └── exact_dedup.py      # Хэш-дедупликация дубликатов
│   ├── taxonomy\               # Доменная классификация и тегирование
│   │   ├── classifier.py       # 8 IT-доменов
│   │   └── tagger.py           # Извлечение тегов и тональности
│   ├── exporter\               # Экспорт во все стандарты ML
│   │   ├── parquet_exporter.py # Apache Parquet (zstd)
│   │   ├── jsonl_exporter.py   # ShareGPT, Alpaca, OpenAI ChatML
│   │   ├── rag_exporter.py     # Векторная база знаний
│   │   └── dpo_exporter.py     # DPO пары предпочтений
│   ├── analytics\              # Глубокий аналитический движок (800+ строк)
│   │   ├── engine.py           # Расчет статистик, энтропии, LDA, трендов
│   │   ├── metrics.py          # Шеннон, перцентили, тональность, токенизация
│   │   ├── network.py          # Социальный граф и инфлюенсеры
│   │   └── report_generator.py # Генератор Markdown, JSON и Rich консоли
│   └── validation\             # Валидация качества и бенчмарк
│       ├── validator.py        # Проверка целостности и Zero-PII аудит
│       └── benchmark.py        # 100 контрольных вопросов домена
├── dataset_output\             # Сгенерированные готовые датасеты
│   ├── parquet\
│   │   ├── full_clean_messages.parquet
│   │   ├── sft_dialogues.parquet
│   │   └── rag_knowledge_base.parquet
│   ├── jsonl\
│   │   ├── sft_sharegpt_format.jsonl
│   │   ├── sft_alpaca_format.jsonl
│   │   ├── sft_openai_messages.jsonl
│   │   ├── rag_chunks_kb.jsonl
│   │   └── dpo_preference_pairs.jsonl
│   └── samples\                # Превью семплов датасета
│       ├── sft_sample_preview.json
│       ├── rag_sample_preview.json
│       └── dpo_sample_preview.json
├── reports\                    # Аналитика, белые книги и Dataset Card
│   ├── DEEP_ANALYTICAL_REPORT.md      # Полный аналитический отчет v4.0
│   ├── DATASET_CARD.md                # Hugging Face Dataset Card с кодом
│   ├── MARKET_INTELLIGENCE_RADAR.md   # Отраслевой отчет: стек, облака, финтех
│   ├── MONETIZATION_WHITEPAPER.md     # Юридическая и коммерческая архитектура
│   ├── domain_benchmark_100.json      # Набор 100 тестовых вопросов
│   ├── analytics_summary.json         # JSON дамп аналитики
│   └── validation_results.json        # Результаты тестов валидации
├── tests\                      # Модульные тесты (unittest / pytest)
│   ├── test_pii_scrubbing.py
│   ├── test_graph_reconstruction.py
│   ├── test_deduplication.py
│   └── test_export_formats.py
├── cli.py                      # Консольный интерфейс управления
├── main.py                     # Главный мастер-скрипт запуска
└── README.md                   # Документация проекта
```

---

## ⚡ Быстрый запуск

### 1. Запуск полного конвейера (End-to-End Pipeline)

```bash
python main.py
```
*или через CLI:*
```bash
python cli.py run
```

### 2. Запуск только глубокой аналитики

```bash
python cli.py analyze
```

### 3. Запуск валидации датасетов и аудита безопасности

```bash
python cli.py validate
```

### 4. Экспорт и просмотр доменного бенчмарка

```bash
python cli.py benchmark
```

### 5. Запуск интерактивной веб-студии (Streamlit Data Studio)

```bash
streamlit run app.py
```
*или через Make:*
```bash
make ui
```

### 6. Запуск интерактивной пошаговой демонстрации

```bash
python demo_walkthrough.py
```

### 7. Запуск Red-Team аудита безопасности (Zero-PII)

```bash
make audit
```

### 8. Запуск набора модульных тестов

```bash
python -m unittest discover -s tests
```

### 9. Запуск через Docker / Docker Compose

```bash
docker-compose up data-studio
```

---

## 📑 Правовой статус и комплаенс (GDPR / EU AI Act / 152-ФЗ)

Датасет подготовлен с соблюдением требований регламентов **EU AI Act (Article 53 Data Lineage)** и **GDPR (Articles 6, 14, 17)**:
- Все персональные идентификаторы замаскированы.
- Назначение корпуса — **Strictly for Academic Research and Educational Purposes** (в соответствии со **ст. 1274 ГК РФ** и **Educational Fair Use**).
- Предусмотрен прозрачный регламент удаления данных по запросу (**Notice and Takedown Policy** в `DATASET_CARD.md`).
- Официальный сертификат аудита безопасности сохранен в [`reports/zero_pii_audit_certificate.json`](reports/zero_pii_audit_certificate.json).

