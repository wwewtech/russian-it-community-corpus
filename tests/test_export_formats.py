"""
Unit tests for Parquet and JSONL dataset exporters.
"""

import json
import tempfile
import unittest
from pathlib import Path
from src.exporter.jsonl_exporter import JSONLExporter
from src.exporter.parquet_exporter import ParquetExporter
from src.ingestion.schema import CleanedMessage, SFTDialogue, SFTTurn


class TestExportFormats(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_path = Path(self.temp_dir.name)
        self.parquet_exporter = ParquetExporter(self.out_path / "parquet")
        self.jsonl_exporter = JSONLExporter(self.out_path / "jsonl")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parquet_export(self):
        msgs = [
            CleanedMessage(
                msg_id=1, chat_id=10, chat_name="DevChat", timestamp="2026-08-20T10:00:00",
                unixtime=1000, author_anon="Dev_1", author_id_anon="u_1", text_clean="FastAPI vs Django",
                domain="backend_databases", tags=["fastapi", "django"],
            )
        ]
        p_path = self.parquet_exporter.export_messages(msgs, "test_msgs.parquet")
        self.assertTrue(p_path.exists())
        self.assertGreater(p_path.stat().st_size, 0)

    def test_jsonl_sharegpt_and_chatml(self):
        turns = [
            SFTTurn(role="user", author="Dev_1", content="Как оптимизировать ClickHouse?"),
            SFTTurn(role="assistant", author="Dev_2", content="Используйте MergeTree и партиционирование по датам."),
        ]
        dialogues = [
            SFTDialogue(
                thread_id=42, chat_name="DevChat", topic_domain="backend_databases",
                topic_tags=["clickhouse", "sql"], messages=turns, quality_score=4.5,
                turn_count=2, total_tokens=25
            )
        ]

        sg_path = self.jsonl_exporter.export_sharegpt(dialogues, "test_sharegpt.jsonl")
        self.assertTrue(sg_path.exists())
        with open(sg_path, "r", encoding="utf-8") as f:
            line = f.readline()
            data = json.loads(line)
            self.assertIn("conversations", data)
            self.assertEqual(len(data["conversations"]), 2)

        cml_path = self.jsonl_exporter.export_openai_chatml(dialogues, "test_chatml.jsonl")
        self.assertTrue(cml_path.exists())
        with open(cml_path, "r", encoding="utf-8") as f:
            line = f.readline()
            data = json.loads(line)
            self.assertIn("messages", data)
            self.assertEqual(data["messages"][0]["role"], "system")
            self.assertEqual(data["messages"][1]["role"], "user")
            self.assertEqual(data["messages"][2]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
