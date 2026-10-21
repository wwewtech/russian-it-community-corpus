"""Tests for the streaming Parquet exporter.

The exporter used to materialise the whole corpus as a list of dicts
plus a DataFrame and write it in a single ``to_parquet`` call, which
produced a file with only THREE row groups for 2.8M rows (~1M rows per
group) and a peak memory footprint that grew with the corpus.

These tests pin the streaming behaviour: many row groups, an unchanged
output contract, and a documented handling of a batch whose keys differ
from the first one.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from ricc.exporter.parquet_exporter import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ROW_GROUP_SIZE,
    ParquetExporter,
    write_records_streaming,
)
from ricc.ingestion.schema import CleanedMessage


def _message(i: int, text: str = "Сообщение с текстом про архитектуру сервисов.") -> CleanedMessage:
    return CleanedMessage(
        msg_id=i,
        chat_id=1,
        chat_name="community_node_01",
        timestamp="2026-01-01T00:00:00",
        unixtime=1767225600 + i,
        author_anon="user_00001",
        author_id_anon="user_00001",
        text_clean=text,
        reply_to_id=None,
        domain="backend_databases",
        tags=["a", "b"],
        sentiment_score=0,
        token_count_approx=7,
        is_question=False,
        thread_id=1,
    )


def _records(n: int) -> list[dict[str, Any]]:
    return [
        {
            "msg_id": i,
            "chat_id": 1,
            "chat_name": "community_node_01",
            "timestamp": "2026-01-01T00:00:00",
            "unixtime": 1767225600 + i,
            "author_anon": "user_00001",
            "author_id_anon": "user_00001",
            "text_clean": f"Сообщение номер {i} про архитектуру распределённых сервисов.",
            "reply_to_id": None,
            "domain": "backend_databases",
            "tags": ["tag"],
            "sentiment_score": 0,
            "token_count_approx": 7,
            "is_question": False,
            "thread_id": 1,
        }
        for i in range(n)
    ]


class TestStreamingRowGroups(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_output_has_many_row_groups(self) -> None:
        """The regression: a single-shot writer produced ONE row group."""
        out = self.root / "streamed.parquet"
        rows, row_groups = write_records_streaming(
            iter(_records(5000)),
            out,
            batch_size=500,
            row_group_size=500,
        )
        self.assertEqual(rows, 5000)
        metadata = pq.ParquetFile(out).metadata
        self.assertEqual(metadata.num_rows, 5000)
        self.assertGreater(
            metadata.num_row_groups,
            1,
            "the streaming writer must emit more than one row group",
        )
        self.assertEqual(metadata.num_row_groups, 10)

    def test_exporter_uses_multiple_row_groups(self) -> None:
        out = self.root / "messages.parquet"
        exporter = ParquetExporter(self.root, batch_size=500, row_group_size=500)
        written = exporter.export_messages([_message(i) for i in range(5000)], file_name="messages.parquet")
        self.assertEqual(written, out)
        self.assertGreater(pq.ParquetFile(out).metadata.num_row_groups, 1)


class TestOutputContractUnchanged(unittest.TestCase):
    """The published dataset contract must not change."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_streamed_output_matches_the_dataframe_reference(self) -> None:
        records = _records(1000)
        streamed = self.root / "streamed.parquet"
        write_records_streaming(iter(records), streamed, batch_size=100, row_group_size=100)

        # The reference shape: one DataFrame, one to_parquet call.
        reference = self.root / "reference.parquet"
        pd.DataFrame.from_records(records).to_parquet(reference, engine="pyarrow", compression="zstd", index=False)

        left = pq.ParquetFile(streamed)
        right = pq.ParquetFile(reference)
        self.assertEqual(left.schema_arrow.names, right.schema_arrow.names)
        self.assertEqual(left.metadata.num_rows, right.metadata.num_rows)

        left_df = pd.read_parquet(streamed)
        right_df = pd.read_parquet(reference)
        self.assertEqual(left_df.columns.tolist(), right_df.columns.tolist())
        pd.testing.assert_frame_equal(left_df, right_df)

    def test_column_order_and_names_are_preserved(self) -> None:
        out = self.root / "ordered.parquet"
        write_records_streaming(iter(_records(10)), out, batch_size=5, row_group_size=5)
        expected = [
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
        self.assertEqual(pq.ParquetFile(out).schema_arrow.names, expected)

    def test_empty_text_filtering_still_applies(self) -> None:
        out = self.root / "filtered.parquet"
        messages = [_message(i) for i in range(10)]
        messages[3] = _message(3, text="   ")
        messages[7] = _message(7, text="")
        ParquetExporter(self.root).export_messages(messages, file_name="filtered.parquet")
        self.assertEqual(pq.ParquetFile(out).metadata.num_rows, 8)


class TestSchemaAlignment(unittest.TestCase):
    """A later batch with different keys is handled as documented."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_unknown_key_is_dropped_and_missing_key_filled(self) -> None:
        out = self.root / "mixed.parquet"
        first = _records(3)
        second = _records(3)
        second[0]["unexpected_column"] = "should not appear"
        del second[1]["domain"]

        rows, _groups = write_records_streaming(iter(first + second), out, batch_size=3, row_group_size=3)

        self.assertEqual(rows, 6)
        names = pq.ParquetFile(out).schema_arrow.names
        self.assertNotIn("unexpected_column", names)
        self.assertIn("domain", names)

        frame = pd.read_parquet(out)
        # The missing key became null, the row was NOT dropped.
        self.assertEqual(len(frame), 6)
        self.assertIsNone(frame["domain"].iloc[4])

    def test_schema_is_taken_from_the_first_batch(self) -> None:
        out = self.root / "first.parquet"
        first = _records(2)
        # A second batch that introduces a column must not change the schema.
        second = _records(2)
        second[0]["extra"] = 1
        write_records_streaming(iter(first + second), out, batch_size=2, row_group_size=2)
        self.assertNotIn("extra", pq.ParquetFile(out).schema_arrow.names)


class TestDefaults(unittest.TestCase):
    def test_default_sizes_are_sane(self) -> None:
        self.assertGreater(DEFAULT_BATCH_SIZE, 0)
        self.assertGreater(DEFAULT_ROW_GROUP_SIZE, 0)
        self.assertLessEqual(DEFAULT_BATCH_SIZE, DEFAULT_ROW_GROUP_SIZE)


if __name__ == "__main__":
    unittest.main()
