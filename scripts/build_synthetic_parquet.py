"""
Build a tiny synthetic Parquet fixture for the data-dependent CI steps.

The three PII / validate / SFT-quality jobs in ``.github/workflows/ci.yml``
all run against Parquet artifacts that are **gitignored** (they live on the
Hugging Face Hub, not in the repository). On a vanilla GitHub-hosted runner
those files are absent, so the jobs short-circuit to "skipped" and never
actually exercise the code path they exist to test.

This script generates a 100-row synthetic Parquet for each of the three
expected files, with the same schema as the production exporter. The
fixtures are intentionally:

* small (~5 KB each) so they fit comfortably inside the checkout
* self-contained — no external resources, no LFS
* deterministic — the same `random_state=42` is used for every column,
  so byte-for-byte diffs across PRs are easy to reason about
* safe — every PII-style string is the **already-scrubbed** form
  (``community_node_NN``, ``[PHONE_REDACTED]``, ``[EMAIL_REDACTED]``),
  so the PII red-team suite passes against them by construction

Usage
-----

::

    # Write the three fixtures into the same directory the production
    # exporter uses. The CI workflow invokes this script with --out
    # pointing at dataset_output/parquet/.
    python scripts/build_synthetic_parquet.py --out dataset_output/parquet

    # Custom row count for stress tests:
    python scripts/build_synthetic_parquet.py --out /tmp/fixtures --rows 5000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

# --- Schemas --------------------------------------------------------------- #

# Required columns for src/validation/pii_redteam.py (audit_production_parquet
# reads `text_clean`, `chat_name`, `chat_id`).
_FULL_MESSAGES_COLUMNS = [
    "msg_id",
    "chat_id",
    "chat_name",
    "timestamp",
    "unixtime",
    "author_anon",
    "author_id_anon",
    "text_clean",
    "reply_to_id",
    "domain",
    "tags",
    "sentiment_score",
    "token_count_approx",
    "is_question",
    "thread_id",
]

# Required column for src/monitoring/sft_quality.py is `messages` (a JSON
# string of [{role, content}, ...]) plus a handful of trace columns so
# the regression report has something to summarise.
_SFT_DIALOGUES_COLUMNS = [
    "dialogue_id",
    "thread_id",
    "messages",
    "source_domain",
    "quality_score",
    "n_turns",
]

# Required column for the RAG knowledge base index. The exporter emits
# {id, content, domain, embedding, source}. We skip the embedding column
# (the LocalRAGPipeline only uses it at query time and is happy with None
# for the fixture); the validator only checks that the file exists and
# has rows.
_RAG_COLUMNS = ["id", "content", "domain", "source"]


# --- Sample data ----------------------------------------------------------- #

_DOMAINS = [
    "backend",
    "frontend",
    "devops",
    "data_engineering",
    "ml_engineering",
    "mobile",
    "security",
    "general_tech_chat",
]

# Each turn is a {"role": ..., "content": ...} dict — the shape the
# SFT quality monitor's _iter_turns() expects. Using plain tuples
# would silently drop every turn because the monitor's filter is
# ``isinstance(t, dict)``.
_RUSSIAN_SAMPLE_TURNS = [
    {"role": "user", "content": "Привет! Подскажи, как лучше настроить nginx для отдачи статики?"},
    {
        "role": "assistant",
        "content": (
            "Рекомендую вынести статику в отдельный location и включить gzip. "
            "Пример конфигурации: location /static/ { alias /var/www/static/; "
            "expires 30d; }"
        ),
    },
    {"role": "user", "content": "А что по поводу HTTP/2 — есть ли смысл включать поверх nginx?"},
    {
        "role": "assistant",
        "content": (
            "Да, для HTTPS включение HTTP/2 заметно ускоряет загрузку за счёт "
            "мультиплексирования. Достаточно добавить http2 в директиву listen."
        ),
    },
    {"role": "user", "content": "Спасибо, помогло! Какие best practices по кешированию на клиенте?"},
    {
        "role": "assistant",
        "content": (
            "Используйте ETag и Cache-Control: immutable для хешированных ассетов. "
            "Для HTML ставьте no-cache и валидируйте через If-None-Match."
        ),
    },
]


def _make_full_messages(n: int, *, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic ``full_clean_messages`` DataFrame.

    All values are already-anonymised (community_node_NN, [REDACTED] tokens),
    so the PII red-team audit against this fixture is expected to pass.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        chat_id = rng.randint(1, 11)
        rows.append(
            {
                "msg_id": i,
                "chat_id": chat_id,
                "chat_name": f"community_node_{chat_id:02d}",
                "timestamp": f"2026-08-{1 + (i % 28):02d}T10:00:00Z",
                "unixtime": 1_700_000_000 + i * 60,
                "author_anon": f"user_{i:05d}",
                "author_id_anon": f"uid_{i:08d}",
                "text_clean": (
                    f"Пример сообщения #{i} на тему {rng.choice(_DOMAINS)}. "
                    "Контакт: [PHONE_REDACTED], email: [EMAIL_REDACTED]. "
                    "Обсуждаем best practices и архитектурные решения."
                ),
                "reply_to_id": i - 1 if i > 0 and rng.random() < 0.3 else None,
                "domain": rng.choice(_DOMAINS),
                "tags": ["synthetic_fixture", rng.choice(["question", "answer", "discussion"])],
                "sentiment_score": rng.randint(-1, 1),
                "token_count_approx": rng.randint(20, 200),
                "is_question": rng.random() < 0.25,
                "thread_id": rng.randint(0, max(1, n // 4)),
            }
        )
    return pd.DataFrame(rows, columns=_FULL_MESSAGES_COLUMNS)


def _make_sft_dialogues(n: int, *, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic ``sft_dialogues`` DataFrame with a `messages` column.

    The dialogues are 6-turn exchanges in Russian so the SFT quality
    regression check finds:
      * role_balance near 0 (3 user, 3 assistant)
      * russian_ratio = 1.0
      * median assistant turn length > 80 chars
      * zero duplicate adjacent turns
      * zero empty assistant turns
    All five sub-scores must pass their floors.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        rows.append(
            {
                "dialogue_id": i,
                "thread_id": rng.randint(0, max(1, n // 4)),
                "messages": json.dumps(_RUSSIAN_SAMPLE_TURNS, ensure_ascii=False),
                "source_domain": rng.choice(_DOMAINS),
                "quality_score": round(rng.uniform(3.5, 5.0), 2),
                "n_turns": len(_RUSSIAN_SAMPLE_TURNS),
            }
        )
    return pd.DataFrame(rows, columns=_SFT_DIALOGUES_COLUMNS)


def _make_rag_kb(n: int, *, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic ``rag_knowledge_base`` DataFrame.

    The LocalRAGPipeline tokenizer-free index only requires ``content`` and
    optionally ``domain``. We add a stable ``id`` so the exporter's
    byte-for-byte determinism is preserved.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": i,
                "content": (
                    f"Фрагмент #{i}: описание архитектурного решения в домене "
                    f"{rng.choice(_DOMAINS)}. Содержит практические рекомендации, "
                    "примеры конфигураций и ссылки на документацию."
                ),
                "domain": rng.choice(_DOMAINS),
                "source": f"synthetic_fixture/{i}",
            }
        )
    return pd.DataFrame(rows, columns=_RAG_COLUMNS)


def _make_sft_openai_jsonl(n: int, path: Path) -> Path:
    """Build ``sft_openai_messages.jsonl`` consumed by the validator.

    The validator's ``audit_pii_leakage`` and ``validate_sft_turn_structures``
    both read this single JSONL file (one dialogue per line, each line a
    ``{"messages": [{"role": ..., "content": ...}, ...]}`` object). The
    fixture reuses the Russian sample turns so the dialogue structure
    passes the alternation check, and the bodies contain no live PII
    (just synthetic Russian text and ``[PHONE_REDACTED]`` / ``[EMAIL_REDACTED]``
    tokens that the PII redactor recognises as already-scrubbed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for _ in range(n):
            fh.write(
                json.dumps(
                    {"messages": _RUSSIAN_SAMPLE_TURNS},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def _make_sharegpt_jsonl(n: int, path: Path) -> Path:
    """``sft_sharegpt_format.jsonl``: ``{conversations: [{from, value}]}`` per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    role_to_from = {"user": "human", "assistant": "gpt"}
    with path.open("w", encoding="utf-8") as fh:
        for _ in range(n):
            convs = [{"from": role_to_from[t["role"]], "value": t["content"]} for t in _RUSSIAN_SAMPLE_TURNS]
            fh.write(json.dumps({"conversations": convs}, ensure_ascii=False) + "\n")
    return path


def _make_alpaca_jsonl(n: int, path: Path) -> Path:
    """``sft_alpaca_format.jsonl``: ``{instruction, input, output}`` per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    user_msgs = [t for t in _RUSSIAN_SAMPLE_TURNS if t["role"] == "user"]
    asst_msgs = [t for t in _RUSSIAN_SAMPLE_TURNS if t["role"] == "assistant"]
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            q = user_msgs[i % len(user_msgs)]["content"]
            a = asst_msgs[i % len(asst_msgs)]["content"]
            fh.write(
                json.dumps(
                    {"instruction": q, "input": "", "output": a},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def _make_rag_chunks_jsonl(n: int, path: Path) -> Path:
    """``rag_chunks_kb.jsonl``: ``{id, content, domain}`` per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(
                json.dumps(
                    {
                        "id": i,
                        "content": f"Фрагмент #{i}: " + "Обсуждение nginx, HTTP/2, кеширования. " * 5,
                        "domain": ["backend", "devops", "frontend"][i % 3],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def _make_dpo_jsonl(n: int, path: Path) -> Path:
    """``dpo_preference_pairs.jsonl``: ``{prompt, chosen, rejected}`` per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    user_msgs = [t for t in _RUSSIAN_SAMPLE_TURNS if t["role"] == "user"]
    asst_msgs = [t for t in _RUSSIAN_SAMPLE_TURNS if t["role"] == "assistant"]
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            q = user_msgs[i % len(user_msgs)]["content"]
            a = asst_msgs[i % len(asst_msgs)]["content"]
            fh.write(
                json.dumps(
                    {
                        "prompt": q,
                        "chosen": a,
                        "rejected": "Не знаю.",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


# --- Entry point ----------------------------------------------------------- #


def build(out_dir: Path, rows: int) -> list[Path]:
    """Write the fixtures into ``out_dir`` and return their paths.

    Output layout mirrors the production exporter::

        <out_dir>/
            parquet/
                full_clean_messages.parquet
                sft_dialogues.parquet
                rag_knowledge_base.parquet
            jsonl/
                sft_openai_messages.jsonl
    """
    parquet_dir = out_dir / "parquet"
    jsonl_dir = out_dir / "jsonl"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for name, builder in (
        ("full_clean_messages.parquet", _make_full_messages),
        ("sft_dialogues.parquet", _make_sft_dialogues),
        ("rag_knowledge_base.parquet", _make_rag_kb),
    ):
        path = parquet_dir / name
        builder(rows).to_parquet(path, index=False, compression="zstd")
        paths.append(path)

    paths.append(_make_sft_openai_jsonl(rows, jsonl_dir / "sft_openai_messages.jsonl"))
    paths.append(_make_sharegpt_jsonl(rows, jsonl_dir / "sft_sharegpt_format.jsonl"))
    paths.append(_make_alpaca_jsonl(rows, jsonl_dir / "sft_alpaca_format.jsonl"))
    paths.append(_make_rag_chunks_jsonl(rows, jsonl_dir / "rag_chunks_kb.jsonl"))
    paths.append(_make_dpo_jsonl(rows, jsonl_dir / "dpo_preference_pairs.jsonl"))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset_output"),
        help=(
            "Root of the dataset export. Must match the directory the validator "
            "receives: ``<out>/parquet/*.parquet`` and ``<out>/jsonl/*.jsonl``."
        ),
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Number of rows per fixture. Default 100; ~5 KB on disk.",
    )
    args = parser.parse_args()

    written = build(args.out, args.rows)
    for p in written:
        size_kb = round(p.stat().st_size / 1024, 1)
        print(f"  wrote {p} ({size_kb} KB, {args.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
