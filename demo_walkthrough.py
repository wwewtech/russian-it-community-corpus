"""
Interactive Demonstration & Tutorial Walkthrough for Russian IT Community Data Platform.
Run: python demo_walkthrough.py
"""

import sys
from pathlib import Path

import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.pii.deep_anonymizer import DeepPIIAnonymizer
from src.validation.benchmark import BenchmarkRunner

BASE_DIR = Path(__file__).resolve().parent
PARQUET_DIR = BASE_DIR / "dataset_output" / "parquet"
JSONL_DIR = BASE_DIR / "dataset_output" / "jsonl"
REPORTS_DIR = BASE_DIR / "reports"


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"📌 {title}")
    print("=" * 80)


def demo_sft_loading():
    print_section("1. ДЕМОНСТРАЦИЯ ЗАГРУЗКИ SFT ДИАЛОГОВ (171.5k ДИАЛОГОВ)")
    sft_path = PARQUET_DIR / "sft_dialogues.parquet"
    if not sft_path.exists():
        print(f"❌ Файл {sft_path} не найден.")
        return
    df = pd.read_parquet(sft_path)
    print(f"✅ Успешно загружено {len(df):,} SFT диалогов из Apache Parquet.")
    sample = df.iloc[0]
    print(f"\nПример диалога #{sample['thread_id']}:")
    print(f"Домен: {sample['topic_domain']} | Теги: {sample['topic_tags']} | Скор качества: {sample['quality_score']}★")
    print("-" * 50)
    for msg in sample["messages"][:4]:
        role = msg.get("role", "user")
        author = msg.get("author", "Dev")
        text = msg.get("content", "")
        print(f"[{role.upper()} - {author}]:\n{text}\n")


def demo_rag_search():
    print_section("2. ДЕМОНСТРАЦИЯ ПОИСКА ПО БАЗЕ ЗНАНИЙ RAG (325.7k ЧАНКОВ)")
    rag_path = PARQUET_DIR / "rag_knowledge_base.parquet"
    if not rag_path.exists():
        print(f"❌ Файл {rag_path} не найден.")
        return
    df = pd.read_parquet(rag_path)
    print(f"✅ База знаний RAG содержит {len(df):,} документов.")
    query = "FastAPI"
    matches = df[df["content"].str.contains(query, case=False, na=False)].head(2)
    print(f"\nПоисковый запрос: '{query}' -> Найдено {len(matches)} релевантных кейсов:")
    for _idx, row in matches.iterrows():
        print(f"\n[Чанк {row['chunk_id']}] {row['title']} ({row['date_range']})")
        print(row["content"][:300] + "...\n")


def demo_deep_pii():
    print_section("3. ДЕМОНСТРАЦИЯ ГЛУБОКОЙ ZERO-PII ДЕИДЕНТИФИКАЦИИ")
    anonymizer = DeepPIIAnonymizer(enable_ner=True)
    anonymizer.name_forms_to_mask.update(
        ["максим", "максиму", "максима", "денис", "денису", "алексей", "алексею", "смирнов", "смирнову"]
    )
    anonymizer._recompile_name_patterns()

    raw_sample = (
        "Привет, Максим! Меня зовут Денис. Мой телефон +7 (999) 123-45-67, "
        "почта denis@mail.ru, кошелек USDT TRC20: TLsV52sRDL79HXGGm9yzwKibb6BeruhUzy. "
        "Ключ OpenAI: sk-proj-1234567890abcdef1234567890abcdef. Спроси у Алексея Смирнова про Docker и FastAPI."
    )
    print(f"📥 ИСХОДНЫЙ ТЕКСТ:\n{raw_sample}\n")
    cleaned = anonymizer.scrub_text(raw_sample)
    print(f"🛡️ ОЧИЩЕННЫЙ ZERO-PII ТЕКСТ:\n{cleaned}\n")


def demo_benchmark():
    print_section("4. ДЕМОНСТРАЦИЯ ДОМЕННОГО БЕНЧМАРКА (100 ВОПРОСОВ)")
    bench = BenchmarkRunner()
    print(f"✅ Бенчмарк содержит {len(bench.questions)} вопросов по 5 направлениям:")
    for q in bench.questions[:3]:
        print(f"[{q['id']}] ({q['domain']}): {q['query']}")
        print(f"   Фокус оценки: {q['eval_focus']}\n")


if __name__ == "__main__":
    print("\n🚀 ЗАПУСК ИНТЕРАКТИВНОЙ ДЕМОНСТРАЦИИ DATA PLATFORM")
    demo_sft_loading()
    demo_rag_search()
    demo_deep_pii()
    demo_benchmark()
    print("\n" + "=" * 80)
    print("🎉 ДЕМОНСТРАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
    print("Запустите Web Data Studio: streamlit run app.py")
    print("=" * 80 + "\n")
