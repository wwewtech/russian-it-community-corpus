"""Round-trip tests for Parquet and JSONL exporters (data contract with the outside world)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.exporter.jsonl_exporter import JSONLExporter
from src.exporter.parquet_exporter import ParquetExporter
from src.ingestion.schema import CleanedMessage, RAGChunk, SFTDialogue, SFTTurn


def _msg(msg_id: int = 1, text: str = "Как настроить репликацию?") -> CleanedMessage:
    return CleanedMessage(
        msg_id=msg_id,
        chat_id=100,
        chat_name="community_node_01",
        timestamp="2026-01-01T12:00:00",
        unixtime=1767264000,
        author_raw="user",
        author_id_raw="user",
        author_anon="Developer_00001",
        author_id_anon="Developer_00001",
        text_clean=text,
        reply_to_id=None,
        domain="backend_databases",
        tags=["postgres"],
        sentiment_score=1,
        token_count_approx=7,
        is_question=True,
        thread_id=42,
    )


def _dialogue(thread_id: int = 1) -> SFTDialogue:
    return SFTDialogue(
        thread_id=thread_id,
        chat_name="community_node_01",
        topic_domain="backend_databases",
        topic_tags=["postgres", "sql"],
        messages=[
            SFTTurn(role="user", author="Developer_00001", content="Как настроить репликацию?"),
            SFTTurn(role="assistant", author="Developer_00002", content="Используйте streaming replication."),
        ],
        quality_score=3.5,
        turn_count=2,
        total_tokens=20,
    )


def _chunk(chunk_id: str = "c1") -> RAGChunk:
    return RAGChunk(
        chunk_id=chunk_id,
        thread_id=42,
        chat_name="community_node_01",
        title="Репликация PostgreSQL",
        topic_domain="backend_databases",
        topic_tags=["postgres"],
        content="Streaming replication настраивается через wal_level=replica.",
        date_range="2026-01-01 — 2026-01-02",
        participants_count=2,
        message_count=2,
        token_count=15,
    )


class TestParquetExporter:
    def test_creates_output_dir(self, tmp_path: Path):
        out_dir = tmp_path / "nested" / "parquet"
        ParquetExporter(out_dir)
        assert out_dir.exists()

    def test_messages_roundtrip(self, tmp_path: Path):
        exp = ParquetExporter(tmp_path)
        out = exp.export_messages([_msg(1), _msg(2, "Второе сообщение")])
        assert out.exists() and out.suffix == ".parquet"
        df = pd.read_parquet(out)
        assert len(df) == 2
        assert df.loc[0, "msg_id"] == 1
        assert df.loc[0, "text_clean"] == "Как настроить репликацию?"
        assert df.loc[0, "domain"] == "backend_databases"
        assert list(df.loc[0, "tags"]) == ["postgres"]
        assert bool(df.loc[0, "is_question"]) is True
        assert df.loc[1, "thread_id"] == 42

    def test_sft_dialogues_roundtrip(self, tmp_path: Path):
        exp = ParquetExporter(tmp_path)
        out = exp.export_sft_dialogues([_dialogue(1), _dialogue(2)])
        df = pd.read_parquet(out)
        assert len(df) == 2
        assert df.loc[0, "thread_id"] == 1
        assert df.loc[0, "topic_domain"] == "backend_databases"
        assert df.loc[0, "messages"][0]["role"] == "user"

    def test_rag_chunks_roundtrip(self, tmp_path: Path):
        exp = ParquetExporter(tmp_path)
        out = exp.export_rag_chunks([_chunk("c1"), _chunk("c2")])
        df = pd.read_parquet(out)
        assert len(df) == 2
        assert df.loc[1, "chunk_id"] == "c2"
        assert df.loc[0, "token_count"] == 15


class TestJSONLExporterShareGPT:
    def test_format_shape(self, tmp_path: Path):
        exp = JSONLExporter(tmp_path)
        out = exp.export_sharegpt([_dialogue(7)])
        parsed = json.loads(out.read_text(encoding="utf-8").strip())
        assert parsed["id"] == "dialogue_7"
        assert parsed["domain"] == "backend_databases"
        conv = parsed["conversations"]
        assert conv[0] == {"from": "human", "value": "Как настроить репликацию?"}
        assert conv[1]["from"] == "gpt"


class TestJSONLExporterAlpaca:
    def test_pairs_user_assistant(self, tmp_path: Path):
        exp = JSONLExporter(tmp_path)
        out = exp.export_alpaca([_dialogue(3)])
        parsed = json.loads(out.read_text(encoding="utf-8").strip())
        assert parsed["instruction"] == "Как настроить репликацию?"
        assert parsed["output"] == "Используйте streaming replication."
        assert parsed["input"] == ""
        assert parsed["domain"] == "backend_databases"


class TestJSONLExporterChatML:
    def test_system_prompt_included(self, tmp_path: Path):
        exp = JSONLExporter(tmp_path)
        out = exp.export_openai_chatml([_dialogue(9)], system_prompt="Ты инженер.")
        parsed = json.loads(out.read_text(encoding="utf-8").strip())
        assert parsed["id"] == "chat_9"
        roles = [m["role"] for m in parsed["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert parsed["messages"][0]["content"] == "Ты инженер."

    def test_custom_system_prompt(self, tmp_path: Path):
        exp = JSONLExporter(tmp_path)
        out = exp.export_openai_chatml([_dialogue(9)], system_prompt="Custom")
        parsed = json.loads(out.read_text(encoding="utf-8").strip())
        assert parsed["messages"][0]["content"] == "Custom"