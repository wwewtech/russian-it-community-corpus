"""
Interactive Demonstration & Tutorial Walkthrough for Russian IT Community Data Platform.
Run: python demo_walkthrough.py [--query TEXT] [--limit N] [--full]
"""

import argparse
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

# Must stay in sync with the sandbox list in app.py (PII tab) — both feed
# the same DeepPIIAnonymizer demo names through all Russian cases.
DEMO_NAME_FORMS = [
    "максим",
    "максима",
    "максиму",
    "денис",
    "дениса",
    "денису",
    "алексей",
    "алексея",
    "смирнов",
    "смирнова",
]


def read_parquet_head(path: Path, n: int, columns: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """Return ``(head_df, total_rows)`` without loading the whole file.

    Total row count comes from the Parquet footer metadata; only the
    first ``n`` rows are materialised via an iterator batch.
    """
    import pyarrow.parquet as pq

    total = pq.read_metadata(path).num_rows
    if total == 0:
        return pd.DataFrame(), 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=n, columns=columns):
        return batch.to_pandas(), total
    return pd.DataFrame(), total


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def demo_sft_loading(limit: int):
    print_section("1. ДЕМОНСТРАЦИЯ ЗАГРУЗКИ SFT ДИАЛОГОВ (171.5k ДИАЛОГОВ)")
    sft_path = PARQUET_DIR / "sft_dialogues.parquet"
    if not sft_path.exists():
        print(f"Файл {sft_path} не найден.")
        return
    df, total = read_parquet_head(sft_path, max(limit, 5))
    if df.empty:
        print("Parquet пуст — нечего показать.")
        return
    print(f"SFT Parquet содержит {total:,} диалогов (показан 1 пример из первых {len(df):,}).")
    sample = df.iloc[0]
    print(f"\nПример диалога #{sample['thread_id']}:")
    print(f"Домен: {sample['topic_domain']} | Теги: {sample['topic_tags']} | Скор качества: {sample['quality_score']}")
    print("-" * 50)
    for msg in sample["messages"][:4]:
        role = msg.get("role", "user")
        author = msg.get("author", "Dev")
        text = msg.get("content", "")
        print(f"[{role.upper()} - {author}]:\n{text}\n")


def demo_rag_search(query: str, limit: int):
    print_section("2. ДЕМОНСТРАЦИЯ ПОИСКА ПО БАЗЕ ЗНАНИЙ RAG (325.7k ЧАНКОВ)")
    rag_path = PARQUET_DIR / "rag_knowledge_base.parquet"
    if not rag_path.exists():
        print(f"Файл {rag_path} не найден.")
        return
    df, total = read_parquet_head(
        rag_path, limit, columns=["chunk_id", "title", "content", "date_range", "topic_domain"]
    )
    print(f"База знаний RAG содержит {total:,} документов (поиск по первым {len(df):,}).")
    matches = df[df["content"].str.contains(query, case=False, na=False, regex=False)].head(2)
    print(f"\nПоисковый запрос: '{query}' -> Найдено {len(matches)} релевантных кейсов:")
    for _idx, row in matches.iterrows():
        print(f"\n[Чанк {row['chunk_id']}] {row['title']} ({row['date_range']})")
        print(row["content"][:300] + "...\n")


def demo_deep_pii():
    print_section("3. ДЕМОНСТРАЦИЯ ГЛУБОКОЙ ZERO-PII ДЕИДЕНТИФИКАЦИИ")
    anonymizer = DeepPIIAnonymizer(enable_ner=True)
    anonymizer.name_forms_to_mask.update(DEMO_NAME_FORMS)
    anonymizer._recompile_name_patterns()

    raw_sample = (
        "Привет, Максим! Меня зовут Денис. Мой телефон +7 (999) 123-45-67, "
        "почта denis@mail.ru, кошелек USDT TRC20: TLsV52sRDL79HXGGm9yzwKibb6BeruhUzy. "
        "Ключ OpenAI: sk-proj-1234567890abcdef1234567890abcdef. Спроси у Алексея Смирнова про Docker и FastAPI."
    )
    print(f"ИСХОДНЫЙ ТЕКСТ:\n{raw_sample}\n")
    cleaned = anonymizer.scrub_text(raw_sample)
    print(f"ОЧИЩЕННЫЙ ZERO-PII ТЕКСТ:\n{cleaned}\n")


def demo_benchmark():
    print_section("4. ДЕМОНСТРАЦИЯ ДОМЕННОГО БЕНЧМАРКА (100 ВОПРОСОВ)")
    bench = BenchmarkRunner()
    print(f"Бенчмарк содержит {len(bench.questions)} вопросов по 5 направлениям:")
    for q in bench.questions[:3]:
        print(f"[{q['id']}] ({q['domain']}): {q['query']}")
        print(f"   Фокус оценки: {q['eval_focus']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RICC interactive data platform walkthrough")
    parser.add_argument("--query", default="FastAPI", help="RAG demo search query (literal substring)")
    parser.add_argument(
        "--limit",
        type=int,
        default=20000,
        help="How many leading parquet rows to materialise for the SFT/RAG demos",
    )
    parser.add_argument("--full", action="store_true", help="Load full parquet files (needs ~GBs of RAM)")
    args = parser.parse_args()
    limit = 2_000_000_000 if args.full else max(args.limit, 5)

    print("\nЗАПУСК ИНТЕРАКТИВНОЙ ДЕМОНСТРАЦИИ DATA PLATFORM")
    demo_sft_loading(limit)
    demo_rag_search(args.query, limit)
    demo_deep_pii()
    demo_benchmark()
    print("\n" + "=" * 80)
    print("ДЕМОНСТРАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
    print("Запустите Web Data Studio: streamlit run app.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
