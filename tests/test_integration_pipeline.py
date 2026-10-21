"""
End-to-end integration test for :class:`ricc.pipeline.MasterDataPipeline`.

This is the FIRST real end-to-end exercise of ``run_all()``: every stage
(ingestion -> PII anonymization -> dedup -> taxonomy -> thread DAG ->
export -> analytics/validation) runs for real, with NO mocking of the
pipeline's collaborators.

The fixture is a small but realistic Telegram export: three chats, a
reply chain, an exact duplicate, a near-duplicate (MinHash), obvious PII
(a phone number and an e-mail), a multi-turn thread, out-of-order
timestamps and a Russian first name that must be declined by the
morphological anonymizer.

The test is hermetic: all inputs and outputs live in a temporary
directory, the repository's ``dataset_output/`` and ``reports/`` are
never touched, and no network / GPU / optional dependency is required.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from ricc.pipeline import MasterDataPipeline
from ricc.settings import settings

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Fixture constants
# ---------------------------------------------------------------------------

BASE_UNIXTIME = 1767225600  # 2026-01-01T00:00:00Z

PHONE = "+7 999 123-45-67"
EMAIL = "ivan.petrov@example.com"
AUTHOR_NAME = "Алексей"  # Russian first name -> morphological declension path
AUTHOR_INFLECTED = "Алексея"  # genitive form, produced only by pymorphy
OTHER_AUTHOR = "Мария"

# Near-duplicate pair: long enough for the 3-word shingle window, and
# different in exactly one word so that the exact-hash dedup keeps both
# while the MinHash LSH (threshold 0.80) collapses them into one.
NEAR_DUP_A = (
    "Для мониторинга production PostgreSQL используйте Prometheus и Grafana вместе с exporters, "
    "алертами, метриками и логами, а также настройте регулярные проверки доступности базы данных, "
    "чтобы дежурный инженер получал уведомление заранее до появления деградации сервиса."
)
NEAR_DUP_B = (
    "Для мониторинга production PostgreSQL используйте Prometheus и Grafana вместе с exporter, "
    "алертами, метриками и логами, а также настройте регулярные проверки доступности базы данных, "
    "чтобы дежурный инженер получал уведомление заранее до появления деградации сервиса."
)

# A short answer reused verbatim in a second chat -> exact-hash duplicate.
# It is deliberately SHORTER than 30 characters so that the MinHash stage
# skips it (``deduplicate_messages`` ignores texts below 30 chars): only
# the exact-hash deduplicator can remove it, which makes the assertion
# in the test specifically about the exact-dedup stage.
EXACT_DUP_ANSWER = "Ставьте proxy_pass в блок."


def _message(
    msg_id: int,
    offset: int,
    author: str,
    author_id: str,
    text: str,
    reply_to: int | None = None,
) -> dict[str, Any]:
    """Build one raw Telegram-export message record."""
    record: dict[str, Any] = {
        "id": msg_id,
        "type": "message",
        "date_unixtime": str(BASE_UNIXTIME + offset),
        "from": author,
        "from_id": author_id,
        "text": text,
    }
    if reply_to is not None:
        record["reply_to_message_id"] = reply_to
    return record


# Chat A: a multi-turn DevOps thread with an explicit reply chain.
CHAT_A_MESSAGES: list[dict[str, Any]] = [
    _message(
        1,
        0,
        "Алексей",
        "user_alexey",
        "Коллеги, подскажите, как правильно настроить nginx для проксирования запросов к backend-сервису?",
    ),
    _message(
        2,
        60,
        "Мария",
        "user_mariya",
        EXACT_DUP_ANSWER,
        reply_to=1,
    ),
    _message(
        3,
        120,
        "Алексей",
        "user_alexey",
        "Спасибо! А как настроить таймауты, если backend отвечает долго, и как балансировать нагрузку?",
        reply_to=2,
    ),
    _message(
        4,
        180,
        "Мария",
        "user_mariya",
        "Алексея, посмотрите настройку proxy_next_upstream timeout: при таймаутах nginx возвращает 502, "
        "и помогает только явное перечисление error timeout в списке proxy_next_upstream.",
        reply_to=3,
    ),
    _message(
        5,
        240,
        "Ольга",
        "user_olga",
        "Алексей, посмотрите ещё настройку keepalive в upstream, это заметно снижает нагрузку на backend. "
        "Мария, спасибо за развёрнутый ответ по proxy_next_upstream.",
    ),
    # A short verbatim repeat of message #2 -> removed by exact dedup only.
    _message(
        6,
        300,
        "Ольга",
        "user_olga",
        EXACT_DUP_ANSWER,
    ),
    # A message whose domain scores TIE (one "docker" keyword for
    # devops_infra, one "python" keyword for backend_databases). The
    # domain classifier has to break that tie deterministically, otherwise
    # the "domain" column - and therefore the parquet sha256 - depends on
    # the interpreter's per-process hash seed.
    _message(
        7,
        360,
        "Ольга",
        "user_olga",
        "Мы используем docker и python в одном проекте, конфигурация у них общая.",
    ),
]

# Chat B: an AI/LLM thread plus the near-duplicate pair.
CHAT_B_MESSAGES: list[dict[str, Any]] = [
    _message(
        10,
        30,
        "Иван",
        "user_ivan",
        "Подскажите, как внедрить LoRA fine-tuning для русскоязычной LLM на одной видеокарте с 12 GB VRAM?",
    ),
    _message(
        11,
        90,
        "Ольга",
        "user_olga",
        "Используйте QLoRA с 4-bit квантованием весов и gradient checkpointing, этого хватает для 12 GB VRAM.",
        reply_to=10,
    ),
    _message(
        12,
        150,
        "Иван",
        "user_ivan",
        "А какие библиотеки посоветуете для обучения и какой transformers поставить?",
        reply_to=11,
    ),
    _message(
        13,
        210,
        "Ольга",
        "user_olga",
        "Unsloth или HuggingFace Transformers с peft, актуальные инструкции есть в документации проекта.",
        reply_to=12,
    ),
    _message(14, 270, "Иван", "user_ivan", NEAR_DUP_A),
    _message(15, 300, "Ольга", "user_olga", NEAR_DUP_B),
]

# Chat C: a support thread carrying obvious PII plus the exact duplicate.
CHAT_C_MESSAGES: list[dict[str, Any]] = [
    _message(
        20,
        45,
        "Пётр",
        "user_petr",
        "Как настроить postgresql для продакшена, если нужен failover и репликация базы?",
    ),
    _message(
        21,
        105,
        "Алексей",
        "user_alexey",
        f"Для отказоустойчивости postgresql включите streaming replication и Patroni. "
        f"Мой номер {PHONE}, а почта {EMAIL} — напишите туда, если что-то не заработает.",
        reply_to=20,
    ),
    _message(
        22,
        165,
        "Ольга",
        "user_olga",
        "Не знаю",
        reply_to=20,
    ),
    _message(
        23,
        225,
        "Мария",
        "user_mariya",
        "Ещё посмотрите в сторону logical replication для отдельных таблиц, это проще в настройке.",
        reply_to=20,
    ),
    # Verbatim repeat of chat A's answer -> must be removed by exact dedup.
    _message(24, 285, "Пётр", "user_petr", EXACT_DUP_ANSWER),
]

CHATS: list[tuple[str, list[dict[str, Any]]]] = [
    ("ChatExport_2026-01-01", CHAT_A_MESSAGES),
    ("ChatExport_2026-01-02", CHAT_B_MESSAGES),
    ("ChatExport_2026-01-03", CHAT_C_MESSAGES),
]

TOTAL_INPUT_MESSAGES = sum(len(msgs) for _name, msgs in CHATS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_exports(root: Path) -> list[Path]:
    """Materialize the synthetic Telegram exports under ``root``."""
    export_dirs: list[Path] = []
    for name, messages in CHATS:
        export_dir = root / name
        export_dir.mkdir(parents=True, exist_ok=True)
        # Deliberately store the messages in a NON-chronological order:
        # the loader is responsible for sorting them by unixtime.
        shuffled = list(reversed(messages))
        payload = {
            "name": name,
            "type": "personal_chat",
            "id": 42,
            "messages": shuffled,
        }
        (export_dir / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        export_dirs.append(export_dir)
    return export_dirs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextlib.contextmanager
def _redirected_output_dirs(data_dir: Path, reports_dir: Path) -> Iterator[None]:
    """Point every settings-driven output directory at the temp tree."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(settings, "output_dir", data_dir))
        stack.enter_context(patch.object(settings, "parquet_output_dir", data_dir / "parquet"))
        stack.enter_context(patch.object(settings, "jsonl_output_dir", data_dir / "jsonl"))
        stack.enter_context(patch.object(settings, "samples_output_dir", data_dir / "samples"))
        stack.enter_context(patch.object(settings, "reports_dir", reports_dir))
        yield


def _run_pipeline(export_dirs: list[Path], data_dir: Path, reports_dir: Path) -> dict[str, Any]:
    """Run the whole pipeline and return the execution summary."""
    with _redirected_output_dirs(data_dir, reports_dir):
        pipeline = MasterDataPipeline(raw_export_dirs=export_dirs)
        # The terminal summary is pure console output; capture it so the
        # test does not depend on the Windows console code page.
        with contextlib.redirect_stdout(io.StringIO()):
            summary = pipeline.run_all()
    return summary


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# Runner executed in a SUBPROCESS with a forced PYTHONHASHSEED. It re-uses
# the very same fixture helpers, runs the pipeline and prints the sha256 of
# every exported parquet file. Used to prove that the corpus artifacts do
# not depend on the interpreter's per-process hash seed (the regression
# that made the ``domain`` column - and therefore the parquet sha256 -
# flip between runs).
_SUBPROCESS_RUNNER = """
import contextlib, hashlib, io, json, sys
from pathlib import Path

sys.path.insert(0, %(repo_root)r)
from tests.test_integration_pipeline import _redirected_output_dirs, _write_exports

from ricc.pipeline import MasterDataPipeline

out_root = Path(sys.argv[1])
exports = _write_exports(out_root / "exports")
data = out_root / "dataset_output"
reports = out_root / "reports"
with _redirected_output_dirs(data, reports):
    pipeline = MasterDataPipeline(raw_export_dirs=exports)
    with contextlib.redirect_stdout(io.StringIO()):
        summary = pipeline.run_all()

digests = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((data / "parquet").glob("*.parquet"))
}
print(json.dumps({
    "cleaned_messages_count": summary["cleaned_messages_count"],
    "sft_dialogues_count": summary["sft_dialogues_count"],
    "rag_chunks_count": summary["rag_chunks_count"],
    "digests": digests,
}))
"""


def _run_pipeline_in_subprocess(out_root: Path, hash_seed: str) -> dict[str, Any]:
    """Run the pipeline in a fresh interpreter with a fixed PYTHONHASHSEED."""
    script = _SUBPROCESS_RUNNER % {"repo_root": str(REPO_ROOT)}
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(out_root)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, f"subprocess run failed:\n{completed.stdout}\n{completed.stderr}"
    payload: dict[str, Any] = json.loads(completed.stdout.strip().splitlines()[-1])
    return payload


def _all_exported_text(messages: pd.DataFrame, sft: pd.DataFrame) -> str:
    """Concatenate every exported message / SFT turn body into one blob."""
    parts: list[str] = [str(t) for t in messages["text_clean"].tolist()]
    for turn_list in sft["messages"].tolist():
        parts.extend(str(turn["content"]) for turn in turn_list)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_pipeline_end_to_end_is_deterministic_and_pii_free(tmp_path: Path) -> None:
    """Full pipeline run on a synthetic export: real artifacts, zero PII, deterministic output."""
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    export_dirs = _write_exports(exports_root)

    run_a_data = tmp_path / "run_a" / "dataset_output"
    run_a_reports = tmp_path / "run_a" / "reports"
    run_b_data = tmp_path / "run_b" / "dataset_output"
    run_b_reports = tmp_path / "run_b" / "reports"

    summary_a = _run_pipeline(export_dirs, run_a_data, run_a_reports)

    # --- sanity: the pipeline really consumed the fixture -----------------
    assert summary_a["raw_messages_count"] == TOTAL_INPUT_MESSAGES
    assert summary_a["cleaned_messages_count"] < TOTAL_INPUT_MESSAGES

    parquet_dir = run_a_data / "parquet"
    jsonl_dir = run_a_data / "jsonl"

    messages_path = parquet_dir / "full_clean_messages.parquet"
    sft_path = parquet_dir / "sft_dialogues.parquet"
    rag_path = parquet_dir / "rag_knowledge_base.parquet"

    # --- parquet artifacts exist and are non-empty ------------------------
    for path in (messages_path, sft_path, rag_path):
        assert path.exists(), f"missing parquet artifact: {path}"
        assert path.stat().st_size > 0, f"empty parquet artifact: {path}"

    messages = _read_parquet(messages_path)
    sft = _read_parquet(sft_path)
    rag = _read_parquet(rag_path)

    assert len(messages) > 0
    assert len(sft) > 0
    assert len(rag) > 0

    # --- deduplication actually removed something -------------------------
    assert len(messages) < TOTAL_INPUT_MESSAGES, "dedup removed nothing from the fixture"
    texts = [str(t) for t in messages["text_clean"].tolist()]
    # The exact duplicate is collapsed to a single occurrence.
    assert texts.count(EXACT_DUP_ANSWER) == 1, "exact duplicate was not removed by deduplication"
    # The near-duplicate pair is collapsed to a single occurrence.
    surviving_near_dupes = [t for t in texts if t in (NEAR_DUP_A, NEAR_DUP_B)]
    assert len(surviving_near_dupes) == 1, "MinHash LSH did not collapse the near-duplicate pair"

    # --- SFT dialogues: at least one dialogue with an assistant turn ------
    assistant_turns = 0
    for turn_list in sft["messages"].tolist():
        assistant_turns += sum(1 for turn in turn_list if turn["role"] == "assistant")
    assert assistant_turns >= 1, "no SFT dialogue contains an assistant turn"

    # --- zero-PII guarantee (the most important assertion) ---------------
    exported_text = _all_exported_text(messages, sft)
    assert PHONE not in exported_text, "phone number leaked into the exported dataset"
    assert EMAIL not in exported_text, "e-mail leaked into the exported dataset"
    # ... and the same must hold for the JSONL exports.
    jsonl_blob = "\n".join(path.read_text(encoding="utf-8") for path in sorted(jsonl_dir.glob("*.jsonl")))
    assert PHONE not in jsonl_blob, "phone number leaked into the JSONL exports"
    assert EMAIL not in jsonl_blob, "e-mail leaked into the JSONL exports"
    for real_name in (AUTHOR_NAME, AUTHOR_INFLECTED, OTHER_AUTHOR):
        assert real_name not in jsonl_blob, f"author name {real_name!r} leaked into the JSONL exports"

    # --- author pseudonymisation -----------------------------------------
    # The fixture puts the real first names into message BODIES (not only
    # into the author column), so this asserts the morphological scrubber
    # really removed the names from the exported text.
    assert AUTHOR_NAME not in exported_text, "author name was not pseudonymised"
    assert AUTHOR_INFLECTED not in exported_text, "inflected author name was not pseudonymised"
    assert OTHER_AUTHOR not in exported_text, "author name was not pseudonymised"
    assert "[PERSON_REDACTED]" in exported_text, "the name scrubber did not run at all"
    author_anon_values = set(str(v) for v in messages["author_anon"].tolist())
    assert author_anon_values, "no pseudonymised authors in the export"
    assert all(name.startswith("Developer_") for name in author_anon_values), (
        f"unexpected author pseudonym format: {author_anon_values}"
    )

    # --- the morphological (case-aware declension) path really ran ------
    # The fixture contains the genitive form "Алексея"; only pymorphy's
    # case inflection can harvest and mask that form, so a non-zero
    # ``morph_names`` counter proves the morphological scrubber was used
    # and not just a plain regex on the nominative.
    pii_stats = summary_a["pii_stats"]
    assert pii_stats.get("morph_names", 0) >= 1, "the morphological name scrubber did not run"
    assert pii_stats.get("phones", 0) >= 1, "the phone scrubber did not run"
    assert pii_stats.get("emails", 0) >= 1, "the e-mail scrubber did not run"
    assert pii_stats.get("authors_anonymized", 0) >= 1, "no author was pseudonymised"

    # --- JSONL exports exist and every line is valid JSON -----------------
    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    assert jsonl_files, "no JSONL exports were produced"
    for path in jsonl_files:
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            if not line.strip():
                continue
            json.loads(line)  # raises on malformed JSON

    # --- reports were written into the temp tree, not the repository -----
    assert (run_a_reports / "analytics_summary.json").exists()
    assert (run_a_reports / "validation_results.json").exists()
    assert (run_a_reports / "pipeline_execution_stats.json").exists()

    # --- determinism: a second, independent run must be byte-identical ---
    summary_b = _run_pipeline(export_dirs, run_b_data, run_b_reports)

    assert summary_b["raw_messages_count"] == summary_a["raw_messages_count"]
    assert summary_b["cleaned_messages_count"] == summary_a["cleaned_messages_count"]
    assert summary_b["threads_count"] == summary_a["threads_count"]
    assert summary_b["sft_dialogues_count"] == summary_a["sft_dialogues_count"]
    assert summary_b["rag_chunks_count"] == summary_a["rag_chunks_count"]
    assert summary_b["dpo_pairs_count"] == summary_a["dpo_pairs_count"]

    messages_b = _read_parquet(run_b_data / "parquet" / "full_clean_messages.parquet")
    assert len(messages_b) == len(messages)
    assert messages_b["text_clean"].tolist() == messages["text_clean"].tolist()
    assert messages_b["domain"].tolist() == messages["domain"].tolist()

    for name in (
        "full_clean_messages.parquet",
        "sft_dialogues.parquet",
        "rag_knowledge_base.parquet",
    ):
        digest_a = _sha256(parquet_dir / name)
        digest_b = _sha256(run_b_data / "parquet" / name)
        assert digest_a == digest_b, f"non-deterministic parquet output for {name}"


@pytest.mark.integration
def test_pipeline_end_to_end_is_hash_seed_independent(tmp_path: Path) -> None:
    """The exported parquet bytes must not depend on PYTHONHASHSEED.

    Regression guard: the domain classifier used to break score ties by
    dict iteration order, so the ``domain`` column - and therefore the
    sha256 of ``full_clean_messages.parquet`` - flipped between runs
    depending on the interpreter's per-process hash seed.

    The fixture contains a message whose domain scores tie (one keyword
    for ``devops_infra``, one for ``backend_databases``), so a
    non-deterministic tie-break really does change the exported bytes.
    Several seeds are probed in parallel, because any *pair* of seeds may
    happen to agree even with a broken tie-break.
    """
    seeds = [str(i) for i in range(8)]
    with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        futures = {seed: pool.submit(_run_pipeline_in_subprocess, tmp_path / f"seed_{seed}", seed) for seed in seeds}
        results = {seed: future.result() for seed, future in futures.items()}

    reference = results[seeds[0]]
    for seed in seeds[1:]:
        result = results[seed]
        assert result["cleaned_messages_count"] == reference["cleaned_messages_count"], (
            f"row count differs for PYTHONHASHSEED={seed}"
        )
        assert result["sft_dialogues_count"] == reference["sft_dialogues_count"], (
            f"SFT dialogue count differs for PYTHONHASHSEED={seed}"
        )
        assert result["rag_chunks_count"] == reference["rag_chunks_count"], (
            f"RAG chunk count differs for PYTHONHASHSEED={seed}"
        )
        assert result["digests"] == reference["digests"], (
            f"parquet sha256 differs for PYTHONHASHSEED={seed}: {result['digests']} != {reference['digests']}"
        )


@pytest.mark.integration
def test_pipeline_end_to_end_leaves_repository_artifacts_untouched(tmp_path: Path) -> None:
    """The end-to-end run must not write into the repository's dataset_output/ or reports/.

    Only the directories the pipeline itself writes are snapshotted:
    ``dataset_output/parquet``, ``dataset_output/jsonl`` and
    ``dataset_output/samples``. ``dataset_output/intermediate/`` is
    written by the separate DVC stage script (``scripts/dvc_stages.py``)
    and is therefore not touched by ``MasterDataPipeline.run_all()``.
    """
    repo_root = Path(__file__).resolve().parents[1]
    watched_dirs = [
        repo_root / "dataset_output" / "parquet",
        repo_root / "dataset_output" / "jsonl",
        repo_root / "dataset_output" / "samples",
        repo_root / "reports",
    ]

    def _snapshot() -> dict[str, tuple[int, float]]:
        snapshot: dict[str, tuple[int, float]] = {}
        for directory in watched_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    snapshot[str(path)] = (path.stat().st_size, path.stat().st_mtime)
        return snapshot

    before = _snapshot()

    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    export_dirs = _write_exports(exports_root)

    _run_pipeline(export_dirs, tmp_path / "dataset_output", tmp_path / "reports")

    assert _snapshot() == before, "the pipeline modified the repository's dataset_output/ or reports/"


@pytest.fixture(autouse=True)
def _cleanup_tmp_path(tmp_path: Path) -> Iterator[None]:
    """Guarantee the temp tree is removed even when the test fails."""
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)
