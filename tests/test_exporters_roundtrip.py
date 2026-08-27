"""Round-trip tests for Parquet and JSONL exporters (data contract with the outside world)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.exporter.jsonl_exporter import export_sharegpt_format, export_sft_dialogues_to_jsonl
from src.exporter.parquet_exporter import export_messages_to_parquet
from src.ingestion.schema import CleanedMessage, SFTDialogue, SFTTurn


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


class TestParquetExporter:
    def test_roundtrip_preserves_fields(self, tmp_path: Path):
        msgs = [_msg(1), _msg(2, "Второе сообщение")]
        out = export_messages_to_parquet(msgs, tmp_path / "out.parquet")
        assert out.exists()
        table = pq.read_table(out)
        assert table.num_rows == 2
        df = table.to_pandas()
        assert df.loc[0, "msg_id"] == 1
        assert df.loc[0, "text_clean"] == "Как настроить репликацию?"
        assert df.loc[0, "domain"] == "backend_databases"
        assert list(df.loc[0, "tags"]) == ["postgres"]
        assert bool(df.loc[0, "is_question"]) is True
        assert df.loc[1, "thread_id"] == 42

    def test_creates_parent_dirs(self, tmp_path: Path):
        out = export_messages_to_parquet([_msg()], tmp_path / "a" / "b" / "out.parquet")
        assert out.exists()

    def test_empty_list_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no messages"):
            export_messages_to_parquet([], tmp_path / "out.parquet")


class TestSFTJsonlExporter:
    def test_roundtrip(self, tmp_path: Path):
        dialogues = [_dialogue(1), _dialogue(2)]
        out = export_sft_dialogues_to_jsonl(dialogues, tmp_path / "sft.jsonl")
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["thread_id"] == 1
        assert parsed["messages"][0]["role"] == "user"
        assert parsed["messages"][1]["content"] == "Используйте streaming replication."

    def test_creates_parent_dirs(self, tmp_path: Path):
        out = export_sft_dialogues_to_jsonl([_dialogue()], tmp_path / "x" / "y" / "sft.jsonl")
        assert out.exists()

    def test_empty_list_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no dialogues"):
            export_sft_dialogues_to_jsonl([], tmp_path / "sft.jsonl")


class TestShareGptExporter:
    def test_format_shape(self, tmp_path: Path):
        out = export_sharegpt_format([_dialogue(7)], tmp_path / "sharegpt.jsonl")
        parsed = json.loads(out.read_text(encoding="utf-8").strip())
        conv = parsed["conversations"]
        assert conv[0] == {"from": "user", "value": "Как настроить репликацию?"}
        assert conv[1]["from"] == "assistant"

    def test_empty_list_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no dialogues"):
            export_sharegpt_format([], tmp_path / "sharegpt.jsonl")


