---
license: other
license_name: research-and-education-only
task_categories:
  - conversational
  - text-generation
  - question-answering
language:
  - ru
  - en
tags:
  - it-discussions
  - russian-it-community
  - software-engineering
  - devops
  - backend
  - ai-ml
  - instruction-tuning
  - sft
  - dpo
  - rag
  - anonymized
pretty_name: Russian IT Community Conversational Corpus (Anonymized 2018-2026)
size_categories:
  - 100K<n<1M
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

# RICC: Russian IT Community Corpus

## Описание датасета

**RICC** — масштабный деидентифицированный корпус технических сообщений из русскоязычных сообществ разработчиков, архитекторов, основателей стартапов и ML-инженеров за период с ноября 2018 по август 2026 года.

Датасет очищен от спама и бот-сообщений, структурирован по ориентированным графам бесед и экспортирован в форматы машинного обучения:
- **Apache Parquet**: сжатый бинарный формат с компрессией zstd
- **ShareGPT JSONL**: многоходовые диалоги для Axolotl, FastChat и LLaMA-Factory
- **Alpaca JSONL**: пары инструкций и ответов
- **OpenAI ChatML JSONL**: диалоги сообщений для Unsloth и TRL
- **RAG Knowledge Base**: база знаний для векторных систем поиска Qdrant, Chroma и pgvector
- **DPO Preference Pairs**: пары предпочтений для оптимизации ответов модели

---

## Ключевые метрики датасета

| Метрика | Значение | Описание |
| :--- | :--- | :--- |
| **Очищенных сообщений** | `1 233 535` | Сообщения после дедупликации и деидентификации |
| **Участников** | `163 049` | Псевдонимизированные авторы |
| **Период сбора** | `17.11.2018 — 22.08.2026` | 8 лет непрерывной хронологии |
| **Суммарный объём слов** | `17 969 211` | Технический словарный корпус |
| **Оценка объёма токенов** | `~23.72M токенов` | Tiktoken cl100k и LLaMA-3 BPE |
| **SFT диалоговых веток** | `58 185` | Диалоги с подтвержденным качеством |
| **DPO пар предпочтений** | `27 056` | Пары для обучения предпочтениям |
| **RAG чанков базы знаний** | `111 659` | Сегментированные документы с метаданными |
| **Индекс Шеннона** | `13.8` | Показатель лексического разнообразия |

---

## 🧠 Доменная таксономия и темы

Корпус размечен по 8 специализированным технологическим доменам:

1. **`ai_ml_nlp` (AI, Machine Learning, RAG, LLM)**: DeepSeek, PyTorch, LoRA, QLoRA, vLLM, Ollama, HuggingFace, Cursor, RAG, эмбеддинги, Qdrant.
2. **`backend_databases` (Бэкенд и Базы данных)**: Python, FastAPI, Django, Asyncio, Go, Rust, Java, PostgreSQL, Redis, ClickHouse, Kafka, SQLAlchemy.
3. **`devops_infra` (Инфраструктура, Облака, DevOps)**: Docker, Kubernetes, Nginx, Linux, CI/CD, Hetzner, Selectel, Timeweb, Yandex Cloud, Prometheus, Loki.
4. **`business_legal_fintech` (IT-бизнес, Налоги, Финтех)**: Стартапы, ИП/ООО, УСН, Stripe, PayPal, эквайринг, международные платежи, релокация (ОАЭ, Кипр, Грузия, Армения), крипта (USDT, TON).
5. **`frontend_ui` (Фронтенд и Мобильная разработка)**: React, Next.js, TypeScript, Vue, Tailwind CSS, Shadcn UI, Flutter.
6. **`sysadmin_security` (Информационная безопасность и сети)**: VPN, WireGuard, VLESS, Shadowsocks, SSL/TLS, DDoS, Auth, JWT.
7. **`career_team_management` (Карьера и Управление командами)**: Зарплаты, найм, собеседования, грейды (Junior/Middle/Senior/Lead), удаленка, Upwork, фриланс.
8. **`general_tech_chat` (Общие технические обсуждения)**: Железо (MacBook M-серии), IDE, инструменты, телеграм-боты.

---

## 🛡️ Политика деидентификации и безопасность данных (Zero-PII Protocol)

Датасет прошел двухконтурную автоматическую очистку:

1. **Детерминированный RegEx-контур**:
   - Номера телефонов РФ и международные (`+7...`, `89...`, `+1...`) заменены на `[PHONE_REDACTED]`.
   - Email-адреса заменены на `[EMAIL_REDACTED]`.
   - Криптовалютные кошельки (Bitcoin, Ethereum, TRON TRC20, TON) заменены на токены вида `[CRYPTO_WALLET_XXX]`.
   - Секретные ключи API и токены (`sk-...`, `ghp_...`, Telegram Bot tokens, AWS keys, JWT) заменены на `[API_KEY_REDACTED]`.
   - IP-адреса серверов заменены на `[IP_REDACTED]`.
   - Приватные инвайт-ссылки `t.me/+...` заменены на `t.me/[INVITE_LINK_REDACTED]`.
   - Юзернеймы `@username` заменены на консистентные псевдонимы `@user_XXXXX`.

2. **Нейросетевой NER-контур (Natasha / Slovnet)**:
   - Личные имена авторов и упоминания персон (`PER`) заменены на `[PERSON_REDACTED]`.
   - Приватные локации и адреса (`LOC`) заменены на `[LOCATION_REDACTED]`.
   - **Smart Tech Whitelist**: Все технические термины, языки программирования, библиотеки и технологические бренды (PostgreSQL, Docker, DeepSeek, Google, Hetzner, FastAPI, Python и сотни других) защищены от маскирования.

3. **Сквозная псевдонимизация авторов**:
   - Идентификаторы авторов детерминированно отображены в псевдонимы `Developer_00001`, `Developer_00002`..., сохраняя структуру диалогов и авторские ветки.

---

## 💻 Быстрый старт: Загрузка и использование

### Загрузка через библиотеку `datasets` (Hugging Face)

```python
from datasets import load_dataset

# Загрузка SFT диалогов из Parquet
dataset = load_dataset("parquet", data_files="dataset_output/parquet/sft_dialogues.parquet")
print(f"Загружено диалогов: {len(dataset['train'])}")
print("Пример диалога:", dataset['train'][0])
```

### Дообучение модели (LoRA / SFT) с помощью `unsloth`

```python
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Загрузка базовой модели
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# 2. Добавление LoRA адаптеров
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

# 3. Загрузка датасета ChatML
dataset = load_dataset("json", data_files="dataset_output/jsonl/sft_openai_messages.jsonl", split="train")

# 4. Обучение
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="messages",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
    ),
)
trainer.train()
```

---

## ⚖️ Правовой статус и Политика удаления (Disclaimer & Takedown Policy)

- **Назначение:** Датасет распространяется **исключительно в некоммерческих исследовательских, образовательных и научных целях** (Strictly for Academic Research & Educational Purposes) в соответствии со **ст. 1274 ГК РФ** и принципами **Educational Fair Use**.
- **Авторские права:** Создатели датасета не претендуют на авторские права пользовательских текстов. Все персональные данные деидентифицированы.
- **Политика Opt-Out (Notice and Takedown):** Если вы являетесь автором сообщений и желаете удалить свои реплики из выборки, создайте **Issue** в репозитории проекта с указанием `msg_id` или контекста. Запрос будет удовлетворен в течение 48 часов.
