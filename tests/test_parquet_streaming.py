"""Tests for the streaming Parquet exporter.

The exporter used to materialise the whole corpus as a list of dicts plus a
DataFrame and write it in a single ``to_parquet`` call, which produced a file
with only THREE row groups for 2.8M rows (~1M rows per group) and a peak
memory footprint that grew with the corpus.

These tests pin the streaming behaviour: many row groups, an unchanged output
contract, and a documented handling of a batch whose keys differ from the
first one.
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from ricc.exporter.parquet_exporter import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ROW_GROUP_SIZE,
    MESSAGE_SCHEMA,
    ParquetExporter,
    ParquetSchemaError,
    write_records_streaming,
)
from ricc.ingestion.schema import CleanedMessage, RAGChunk, SFTDialogue, SFTTurn


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
        # The real corpus has replies; a mix of null and int is what makes
        # ``reply_to_id`` a float64 column in the legacy whole-corpus schema.
        reply_to_id=None if i % 3 == 0 else max(1, i - 1),
        domain="backend_databases",
        tags=["a", "b"],
        sentiment_score=0,
        token_count_approx=7,
        is_question=False,
        thread_id=1,
    )


def _dialogue(i: int) -> SFTDialogue:
    return SFTDialogue(
        thread_id=i,
        chat_name="community_node_01",
        topic_domain="backend_databases",
        topic_tags=["postgres", "sql"],
        messages=[
            SFTTurn(role="user", author="Dev_1", content="Как настроить репликацию?"),
            SFTTurn(role="assistant", author="Dev_2", content="Используйте streaming replication."),
        ],
        quality_score=3.5,
        turn_count=2,
        total_tokens=20,
    )


def _chunk(i: int) -> RAGChunk:
    return RAGChunk(
        chunk_id=f"c{i}",
        thread_id=i,
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


def _record_stream(n: int) -> Iterator[dict[str, Any]]:
    """The same records, produced lazily.

    The memory test needs a generator: a list of N dicts would itself
    scale with N and hide the property under test.
    """
    for i in range(n):
        yield {
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


class TestStreamingRowGroups(unittest.TestCase):
    """The regression: the single-shot writer produced ONE row group for SFT/RAG."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_output_has_many_row_groups(self) -> None:
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
        self.assertEqual(row_groups, 10)

    def test_exporter_uses_multiple_row_groups(self) -> None:
        out = self.root / "messages.parquet"
        exporter = ParquetExporter(self.root, batch_size=500, row_group_size=500)
        written = exporter.export_messages([_message(i) for i in range(5000)], file_name="messages.parquet")
        self.assertEqual(written, out)
        self.assertGreater(pq.ParquetFile(out).metadata.num_row_groups, 1)

    def test_sft_dialogues_have_more_than_one_row_group(self) -> None:
        """The SFT file used to be a single row group - a consumer could not read a part of it."""
        out = self.root / "sft.parquet"
        exporter = ParquetExporter(self.root, batch_size=100, row_group_size=100)
        exporter.export_sft_dialogues([_dialogue(i) for i in range(1000)], file_name="sft.parquet")
        metadata = pq.ParquetFile(out).metadata
        self.assertGreater(metadata.num_row_groups, 1)
        self.assertEqual(metadata.num_rows, 1000)

    def test_rag_chunks_have_more_than_one_row_group(self) -> None:
        out = self.root / "rag.parquet"
        exporter = ParquetExporter(self.root, batch_size=100, row_group_size=100)
        exporter.export_rag_chunks([_chunk(i) for i in range(1000)], file_name="rag.parquet")
        metadata = pq.ParquetFile(out).metadata
        self.assertGreater(metadata.num_row_groups, 1)
        self.assertEqual(metadata.num_rows, 1000)

    def test_row_group_size_is_honoured(self) -> None:
        out = self.root / "sized.parquet"
        write_records_streaming(iter(_records(1000)), out, batch_size=100, row_group_size=250)
        metadata = pq.ParquetFile(out).metadata
        self.assertEqual(metadata.num_row_groups, 4)
        for index in range(metadata.num_row_groups):
            self.assertEqual(metadata.row_group(index).num_rows, 250)

    def test_exporter_accepts_an_iterable_not_only_a_list(self) -> None:
        """The memory-safe path: the caller feeds the writer lazily."""
        out = self.root / "lazy.parquet"
        exporter = ParquetExporter(self.root, batch_size=100, row_group_size=100)
        written = exporter.export_records(
            (record for record in _records(1000)),
            file_name="lazy.parquet",
        )
        self.assertEqual(written, out)
        self.assertEqual(pq.ParquetFile(out).metadata.num_rows, 1000)
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

    def test_exported_messages_match_the_dataframe_reference(self) -> None:
        messages = [_message(i) for i in range(500)]
        streamed = self.root / "streamed.parquet"
        ParquetExporter(self.root, batch_size=100, row_group_size=100).export_messages(
            messages, file_name="streamed.parquet"
        )

        reference_records = [
            {
                "msg_id": m.msg_id,
                "chat_id": m.chat_id,
                "chat_name": m.chat_name,
                "timestamp": m.timestamp,
                "unixtime": m.unixtime,
                "author_anon": m.author_anon,
                "author_id_anon": m.author_id_anon,
                "text_clean": m.text_clean,
                "reply_to_id": m.reply_to_id,
                "domain": m.domain,
                "tags": m.tags,
                "sentiment_score": m.sentiment_score,
                "token_count_approx": m.token_count_approx,
                "is_question": m.is_question,
                "thread_id": m.thread_id,
            }
            for m in messages
        ]
        reference = self.root / "reference.parquet"
        pd.DataFrame.from_records(reference_records).to_parquet(
            reference, engine="pyarrow", compression="zstd", index=False
        )

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

    def test_published_message_schema_is_preserved(self) -> None:
        """The pinned schema must stay identical to the published artifact schema."""
        out = self.root / "pinned.parquet"
        ParquetExporter(self.root).export_messages([_message(i) for i in range(10)], file_name="pinned.parquet")
        self.assertEqual(pq.ParquetFile(out).schema_arrow.names, MESSAGE_SCHEMA.names)
        self.assertEqual(
            [field.type for field in pq.ParquetFile(out).schema_arrow],
            [field.type for field in MESSAGE_SCHEMA],
        )

    def test_compression_is_zstd(self) -> None:
        out = self.root / "compressed.parquet"
        write_records_streaming(iter(_records(100)), out, batch_size=10, row_group_size=10)
        metadata = pq.ParquetFile(out).metadata
        compression = metadata.row_group(0).column(0).compression
        self.assertEqual(compression, "ZSTD")

    def test_empty_text_filtering_still_applies(self) -> None:
        out = self.root / "filtered.parquet"
        messages = [_message(i) for i in range(10)]
        messages[3] = _message(3, text="   ")
        messages[7] = _message(7, text="")
        ParquetExporter(self.root).export_messages(messages, file_name="filtered.parquet")
        self.assertEqual(pq.ParquetFile(out).metadata.num_rows, 8)

    def test_empty_text_filtering_still_applies_for_a_lazy_iterable(self) -> None:
        out = self.root / "filtered_lazy.parquet"
        messages = [_message(i) for i in range(10)]
        messages[3] = _message(3, text="   ")
        ParquetExporter(self.root).export_messages(iter(messages), file_name="filtered_lazy.parquet")
        self.assertEqual(pq.ParquetFile(out).metadata.num_rows, 9)


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

    def test_strict_schema_raises_on_a_different_key_set(self) -> None:
        out = self.root / "strict.parquet"
        first = _records(2)
        second = _records(2)
        second[0]["extra"] = 1
        with self.assertRaises(ParquetSchemaError):
            write_records_streaming(iter(first + second), out, batch_size=2, row_group_size=2, strict_schema=True)
        # The failed export must not leave a half-written file behind.
        self.assertFalse(out.exists())

    def test_incompatible_value_type_raises_a_clear_error(self) -> None:
        out = self.root / "incompatible.parquet"
        first = _records(2)
        second = _records(2)
        second[0]["msg_id"] = "not-an-int"
        with self.assertRaises(ParquetSchemaError) as ctx:
            write_records_streaming(iter(first + second), out, batch_size=2, row_group_size=2)
        self.assertIn("msg_id", str(ctx.exception))
        self.assertFalse(out.exists())

    def test_pinned_schema_drops_an_unknown_key(self) -> None:
        """With a pinned schema an unknown key is dropped, not an export failure."""
        out = self.root / "pinned.parquet"
        records = _records(2)
        records[1]["unexpected"] = "x"
        rows, _groups = write_records_streaming(
            iter(records), out, batch_size=1, row_group_size=1, schema=MESSAGE_SCHEMA
        )
        self.assertEqual(rows, 2)
        self.assertNotIn("unexpected", pq.ParquetFile(out).schema_arrow.names)

    def test_strict_schema_raises_for_a_later_batch_with_an_extra_key(self) -> None:
        """``strict_schema=True`` turns the alignment into an error.

        The FIRST batch defines the schema by construction, so the check
        applies to the batches that follow it.
        """
        out = self.root / "strict.parquet"
        first = _records(2)
        second = _records(2)
        second[0]["unexpected"] = "x"
        with self.assertRaises(ParquetSchemaError):
            write_records_streaming(iter(first + second), out, batch_size=2, row_group_size=2, strict_schema=True)
        self.assertFalse(out.exists())


class TestAtomicWrite(unittest.TestCase):
    """A failed export must never leave a truncated file behind."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_failed_export_leaves_no_target_and_no_partial(self) -> None:
        out = self.root / "boom.parquet"
        partial = out.with_name(out.name + ".partial")

        def exploding():
            yield _records(1)[0]
            raise RuntimeError("mid-export")

        with self.assertRaises(RuntimeError):
            write_records_streaming(exploding(), out, batch_size=1, row_group_size=1)
        self.assertFalse(out.exists(), "a failed export left a target file behind")
        self.assertFalse(partial.exists(), "a failed export left a .partial file behind")

    def test_successful_export_leaves_no_partial(self) -> None:
        out = self.root / "ok.parquet"
        write_records_streaming(iter(_records(10)), out, batch_size=5, row_group_size=5)
        self.assertTrue(out.exists())
        self.assertFalse(out.with_name(out.name + ".partial").exists())


class TestMemoryIsBounded(unittest.TestCase):
    """The peak memory must not grow with the number of records.

    This is the property that makes the export usable at the corpus
    scale: the old implementation materialised the whole corpus as a
    list of dicts plus a DataFrame.
    """

    def test_peak_memory_does_not_scale_with_the_corpus(self) -> None:
        import tracemalloc

        with tempfile.TemporaryDirectory() as tmp:
            peaks = []
            # The row group is the unit the writer buffers, so the memory
            # bound is the ROW GROUP size, not the corpus size. The batch
            # size only controls the conversion work. Measuring with
            # equal batch and row-group sizes is therefore the meaningful
            # comparison: the peak must stay flat as the corpus grows.
            for n in (20_000, 80_000, 320_000):
                target = Path(tmp) / f"m{n}.parquet"
                tracemalloc.start()
                write_records_streaming(_record_stream(n), target, batch_size=10_000, row_group_size=10_000)
                _cur, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peaks.append(peak)
                target.unlink(missing_ok=True)

        # 16x the records must not mean 16x the memory: the peak is set
        # by the row group, so it stays flat.
        self.assertLess(
            peaks[-1],
            peaks[0] * 1.5,
            f"peak memory grew with the corpus: {peaks[0] / 1e6:.1f} MB -> {peaks[-1] / 1e6:.1f} MB",
        )


class TestDefaults(unittest.TestCase):
    def test_default_sizes_are_sane(self) -> None:
        self.assertGreater(DEFAULT_BATCH_SIZE, 0)
        self.assertGreater(DEFAULT_ROW_GROUP_SIZE, 0)
        self.assertLessEqual(DEFAULT_BATCH_SIZE, DEFAULT_ROW_GROUP_SIZE)

    def test_invalid_sizes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_records_streaming(iter(_records(1)), Path(tmp) / "x.parquet", batch_size=0)
            with self.assertRaises(ValueError):
                write_records_streaming(iter(_records(1)), Path(tmp) / "x.parquet", row_group_size=0)


if __name__ == "__main__":
    unittest.main()
