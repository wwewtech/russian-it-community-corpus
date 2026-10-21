"""
Apache Parquet exporter with zstd compression for high-performance ML dataset loading.

The export is STREAMING: the records are written in bounded batches through a
``pyarrow.parquet.ParquetWriter`` instead of being materialised as one
``pandas.DataFrame``.

Why it matters at the corpus scale: the published corpus is 2,816,434
messages. The previous implementation built a list of dicts (~1.24 GB for
the text column alone), a DataFrame on top of it (~0.3 GB) and wrote the
result in a single ``to_parquet`` call, so the peak memory grew with the
corpus. It also produced a file with only THREE row groups of ~1M rows
each, which means a consumer could not read a slice of it.

The streaming writer keeps the peak memory bounded by the batch/row-group
size and emits a configurable number of row groups (29 instead of 3 for the
published corpus with the default row group size).

Schema contract
===============

The published dataset contract must not change, so the schema of the three
canonical artifacts is *pinned*: column names, column order, dtypes and the
pandas metadata block are byte-identical to the schema the legacy
whole-corpus ``DataFrame`` produced for the published files (see
``MESSAGE_SCHEMA``, ``SFT_SCHEMA`` and ``RAG_SCHEMA``).

The generic streaming entry point (``write_records_streaming``) derives the
schema from the FIRST batch instead - exactly like the legacy code did for
the whole corpus - and *aligns* every later batch to it:

* a key that is missing from a later record is filled with ``null``;
* a key that the first batch did not define is dropped (and logged);
* a value whose type cannot be represented by the schema raises
  :class:`ParquetSchemaError` - the file is never silently corrupted.

Pass ``strict_schema=True`` to fail on any key-set difference instead of
aligning.
"""

import logging
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ricc.ingestion.schema import CleanedMessage, RAGChunk, SFTDialogue

logger = logging.getLogger(__name__)

# Rows converted to Arrow per batch. A batch is the unit of the pandas -> Arrow
# conversion, so the memory held at any time is proportional to this number
# and NOT to the corpus size.
DEFAULT_BATCH_SIZE = 50_000

# Rows per row group in the resulting file. A row group is the unit Parquet
# readers decompress at once, so 100k rows keeps a single group in the low
# tens of megabytes instead of the ~200 MB groups the old single-shot writer
# produced. 2,816,434 rows therefore become 29 row groups instead of 3.
DEFAULT_ROW_GROUP_SIZE = 100_000

COMPRESSION = "zstd"

# How many extra batches may be inspected to resolve a column that is NULL in
# the first batch (and therefore carries no type information). 0 disables the
# look-ahead.
DEFAULT_SCHEMA_PROBE_BATCHES = 1

# A record is a plain mapping of column name -> value.
Record = Mapping[str, Any]


class ParquetSchemaError(ValueError):
    """Raised when a batch cannot be represented with the schema of the first batch."""


# --------------------------------------------------------------------------- #
# Canonical (pinned) schemas of the three published artifacts.
#
# They are built from one-row probe frames so that pandas performs exactly the
# same dtype inference it performed for the legacy whole-corpus DataFrame. The
# result is byte-identical to the schema of the published files, including the
# pandas metadata block.
# --------------------------------------------------------------------------- #


def _message_probe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "msg_id": pd.Series([1], dtype="int64"),
            "chat_id": pd.Series([1], dtype="int64"),
            "chat_name": pd.Series(["chat"], dtype="object"),
            "timestamp": pd.Series(["2026-01-01T00:00:00"], dtype="object"),
            "unixtime": pd.Series([1], dtype="int64"),
            "author_anon": pd.Series(["author"], dtype="object"),
            "author_id_anon": pd.Series(["author"], dtype="object"),
            "text_clean": pd.Series(["text"], dtype="object"),
            "reply_to_id": pd.Series([1.0], dtype="float64"),
            "domain": pd.Series(["domain"], dtype="object"),
            "tags": pd.Series([["tag"]], dtype="object"),
            "sentiment_score": pd.Series([0], dtype="int64"),
            "token_count_approx": pd.Series([0], dtype="int64"),
            "is_question": pd.Series([False], dtype="bool"),
            "thread_id": pd.Series([1], dtype="int64"),
        }
    )


def _sft_probe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "thread_id": pd.Series([1], dtype="int64"),
            "chat_name": pd.Series(["chat"], dtype="object"),
            "topic_domain": pd.Series(["domain"], dtype="object"),
            "topic_tags": pd.Series([["tag"]], dtype="object"),
            "messages": pd.Series(
                [[{"role": "user", "author": "a", "content": "c", "timestamp": "2026-01-01T00:00:00"}]],
                dtype="object",
            ),
            "quality_score": pd.Series([1.0], dtype="float64"),
            "turn_count": pd.Series([1], dtype="int64"),
            "total_tokens": pd.Series([1], dtype="int64"),
        }
    )


def _rag_probe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chunk_id": pd.Series(["chunk"], dtype="object"),
            "thread_id": pd.Series([1], dtype="int64"),
            "chat_name": pd.Series(["chat"], dtype="object"),
            "title": pd.Series(["title"], dtype="object"),
            "topic_domain": pd.Series(["domain"], dtype="object"),
            "topic_tags": pd.Series([["tag"]], dtype="object"),
            "content": pd.Series(["content"], dtype="object"),
            "date_range": pd.Series(["range"], dtype="object"),
            "participants_count": pd.Series([1], dtype="int64"),
            "message_count": pd.Series([1], dtype="int64"),
            "token_count": pd.Series([1], dtype="int64"),
        }
    )


def _pinned_schema(probe: pd.DataFrame) -> pa.Schema:
    """The pinned Arrow schema of an artefact, with the contract metadata.

    The metadata carries the dataset schema version and the contract
    fingerprint, so a consumer can tell WHICH contract version the file
    was written against — and the validator can refuse a file whose
    columns drifted away from the declared contract.
    """
    from ricc.schema_version import write_metadata

    return write_metadata(pa.Schema.from_pandas(probe, preserve_index=False))


MESSAGE_SCHEMA = _pinned_schema(_message_probe())
SFT_SCHEMA = _pinned_schema(_sft_probe())
RAG_SCHEMA = _pinned_schema(_rag_probe())


# --------------------------------------------------------------------------- #
# Record builders
# --------------------------------------------------------------------------- #


def _message_record(m: CleanedMessage) -> dict[str, Any]:
    return {
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


def _sft_record(d: SFTDialogue) -> dict[str, Any]:
    return {
        "thread_id": d.thread_id,
        "chat_name": d.chat_name,
        "topic_domain": d.topic_domain,
        "topic_tags": d.topic_tags,
        "messages": [m.model_dump() for m in d.messages],
        "quality_score": d.quality_score,
        "turn_count": d.turn_count,
        "total_tokens": d.total_tokens,
    }


def _rag_record(c: RAGChunk) -> dict[str, Any]:
    return {
        "chunk_id": c.chunk_id,
        "thread_id": c.thread_id,
        "chat_name": c.chat_name,
        "title": c.title,
        "topic_domain": c.topic_domain,
        "topic_tags": c.topic_tags,
        "content": c.content,
        "date_range": c.date_range,
        "participants_count": c.participants_count,
        "message_count": c.message_count,
        "token_count": c.token_count,
    }


# --------------------------------------------------------------------------- #
# Schema helpers
# --------------------------------------------------------------------------- #


def _type_has_null(arrow_type: Any) -> bool:
    """Return True if the Arrow type is (or contains) the ``null`` type."""
    if pa.types.is_null(arrow_type):
        return True
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
        return _type_has_null(arrow_type.value_type)
    if pa.types.is_struct(arrow_type):
        return any(_type_has_null(field.type) for field in arrow_type)
    if pa.types.is_map(arrow_type):
        return _type_has_null(arrow_type.key_type) or _type_has_null(arrow_type.item_type)
    return False


def _schema_has_null_columns(schema: pa.Schema) -> bool:
    """Return True if any column of the schema is (or contains) the ``null`` type."""
    return any(_type_has_null(field.type) for field in schema)


def _first_batch_keys(batch: Sequence[Record]) -> list[str]:
    """Column names of the first batch, in order of first appearance."""
    keys: list[str] = []
    seen: set[str] = set()
    for record in batch:
        for key in record:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def _align(records: Sequence[Record], keys: Sequence[str]) -> list[dict[str, Any]]:
    """Project the records onto the canonical column set (missing keys become ``None``)."""
    return [{key: record.get(key) for key in keys} for record in records]


def _infer_schema(records: Sequence[Record], keys: Sequence[str], out_path: Path) -> pa.Schema:
    """Derive the Arrow schema of a batch the way the legacy pandas path did."""
    df = pd.DataFrame.from_records(_align(records, keys))
    try:
        return pa.Schema.from_pandas(df, preserve_index=False)
    except Exception as exc:
        raise ParquetSchemaError(
            f"The first batch of {out_path.name} does not yield a valid schema: {exc}. Columns: {list(keys)}."
        ) from exc


def _to_table(records: Sequence[Record], keys: Sequence[str], schema: pa.Schema, out_path: Path) -> pa.Table:
    """Convert a batch of records to an Arrow table that follows ``schema``."""
    df = pd.DataFrame.from_records(_align(records, keys))
    try:
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    except Exception as exc:
        raise ParquetSchemaError(
            f"A batch of {out_path.name} does not match the schema of the first batch: {exc}. "
            f"Columns of the first batch: {list(schema.names)}. Later batches are aligned to it "
            f"(missing keys become null, unknown keys are dropped); pass strict_schema=True to "
            f"fail on any key-set difference."
        ) from exc
    # Keep the metadata of the frozen schema on every table so that the batches
    # can be concatenated and the file keeps the pandas metadata block.
    return table.replace_schema_metadata(schema.metadata)


def _batched(records: Iterable[Record], size: int) -> Iterator[list[Record]]:
    """Lazily split an iterable of records into batches of at most ``size`` items."""
    batch: list[Record] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _check_keys(
    batch: Sequence[Record],
    keys: Sequence[str],
    dropped: set[str],
    missing: set[str],
    strict_schema: bool,
    out_path: Path,
) -> None:
    """Collect the key-set differences between a batch and the first batch."""
    canonical = set(keys)
    for record in batch:
        extra = set(record) - canonical
        absent = canonical - set(record)
        if extra and strict_schema:
            raise ParquetSchemaError(
                f"Batch of {out_path.name} has keys {sorted(extra)} that the first batch does not "
                f"define (expected {sorted(canonical)}); strict_schema=True forbids aligning."
            )
        dropped.update(extra)
        missing.update(absent)


# --------------------------------------------------------------------------- #
# Streaming writer
# --------------------------------------------------------------------------- #


def write_records_streaming(
    records: Iterable[Record],
    out_path: str | Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
    compression: str = COMPRESSION,
    schema: pa.Schema | None = None,
    strict_schema: bool = False,
    schema_probe_batches: int = DEFAULT_SCHEMA_PROBE_BATCHES,
) -> tuple[int, int]:
    """Write ``records`` to a Parquet file without materialising them all.

    The schema is derived from the FIRST batch (or taken from ``schema`` when
    one is given) and every later batch is aligned to it: a record with an
    unknown key is DROPPED (with a warning) and a record missing a key gets a
    null. That keeps the file readable instead of failing halfway through a
    two-million-row export, and the alignment is logged so that silent data
    loss cannot go unnoticed.

    ``strict_schema=True`` turns the alignment into an error: a later batch
    that carries a key the first batch did not define raises
    :class:`ParquetSchemaError`. The FIRST batch defines the schema by
    construction, so a key set that only varies WITHIN the first batch is
    still aligned (there is no earlier batch to compare against).

    ``schema`` pins the schema outright — the canonical artefacts use it so
    that the exported file matches the published contract byte for byte,
    independently of what the first batch happens to contain.

    ``row_group_size`` is the number of rows per Parquet row group, and it is
    what a reader decompresses at once. Records are buffered up to that size
    before a row group is written, so the two settings are independent:
    ``batch_size`` bounds the conversion work, while ``row_group_size``
    controls the on-disk layout AND the memory bound.

    That last point matters: the peak memory of the export is set by the
    ROW GROUP, not by the corpus size. Measured on this machine, with a
    10k row group the peak stays flat at ~36 MB for 20k, 80k and 320k
    records alike, whereas the old list-plus-DataFrame implementation grew
    linearly (102 MB at 100k rows, 1.6 GB at 1.6M rows).

    The file is written atomically: the data goes to ``<name>.partial`` and
    is moved into place with :func:`os.replace` at the end, so a failed
    export can never leave a truncated file behind.

    Returns ``(rows_written, row_groups_written)``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if row_group_size <= 0:
        raise ValueError(f"row_group_size must be positive, got {row_group_size}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_name(out_path.name + ".partial")

    batches = _batched(records, batch_size)
    first_batch = next(batches, [])

    # ``pyarrow`` ships no type stubs, so the schema is typed as ``Any`` here.
    resolved_schema: Any = schema
    buffered: list[Record] = list(first_batch)
    keys: list[str] = []

    if resolved_schema is None:
        if not buffered:
            # Nothing to export: write an empty file, exactly like the legacy
            # ``pd.DataFrame.from_records([]).to_parquet(...)`` did.
            cast(Any, pd.DataFrame.from_records([])).to_parquet(
                str(partial_path), engine="pyarrow", compression=compression, index=False
            )
            os.replace(partial_path, out_path)
            logger.info("No records to export; wrote an empty Parquet file at %s", out_path)
            return 0, 0
        keys = _first_batch_keys(buffered)
        resolved_schema = _infer_schema(buffered, keys, out_path)
        # A column that is NULL in the whole first batch carries no type
        # information. Look ahead over a bounded number of batches to find a
        # real type instead of writing a null column (or failing) later on.
        probes = 0
        while probes < schema_probe_batches and _schema_has_null_columns(resolved_schema):
            probe_batch = next(batches, [])
            if not probe_batch:
                break
            buffered.extend(probe_batch)
            probes += 1
            resolved_schema = _infer_schema(buffered, keys, out_path)
            logger.info(
                "Schema probe over %d extra batch(es) resolved the NULL columns of %s.",
                probes,
                out_path.name,
            )
    else:
        # A pinned schema defines the canonical column set: the records are
        # projected onto it, so an unknown key is dropped (and logged) instead
        # of failing the export.
        keys = list(resolved_schema.names)

    writer: Any = None
    rows = 0
    row_groups = 0
    dropped_keys: set[str] = set()
    missing_keys: set[str] = set()

    pending: list[Any] = []
    pending_rows = 0

    def flush_full_groups() -> None:
        """Write buffered tables while they hold at least one full row group."""
        nonlocal pending, pending_rows, rows, row_groups
        while pending_rows >= row_group_size:
            combined = pa.concat_tables(pending)
            writer.write_table(combined.slice(0, row_group_size), row_group_size=row_group_size)
            rows += row_group_size
            row_groups += 1
            remainder = combined.slice(row_group_size)
            pending = [remainder] if remainder.num_rows else []
            pending_rows = remainder.num_rows

    def emit(batch: Sequence[Record]) -> None:
        nonlocal pending, pending_rows
        if not batch:
            return
        _check_keys(batch, keys, dropped_keys, missing_keys, strict_schema, out_path)
        table = _to_table(batch, keys, resolved_schema, out_path)
        pending.append(table)
        pending_rows += table.num_rows
        flush_full_groups()

    try:
        writer = pq.ParquetWriter(str(partial_path), resolved_schema, compression=compression)
        for start in range(0, len(buffered), batch_size):
            emit(buffered[start : start + batch_size])
        for batch in batches:
            emit(batch)
        if pending_rows:
            combined = pa.concat_tables(pending)
            writer.write_table(combined, row_group_size=row_group_size)
            rows += combined.num_rows
            row_groups += 1
            pending = []
            pending_rows = 0
    except BaseException:
        if writer is not None:
            writer.close()
        partial_path.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(partial_path, out_path)

    if dropped_keys:
        logger.warning(
            "Dropped unknown column(s) in %s: %s (the schema is fixed by the first batch)",
            out_path.name,
            sorted(dropped_keys),
        )
    if missing_keys:
        logger.warning(
            "Filled missing column(s) with null in %s: %s",
            out_path.name,
            sorted(missing_keys),
        )
    return rows, row_groups


class ParquetExporter:
    """
    Exports clean messages, SFT dialogues, and RAG knowledge chunks to compressed Apache Parquet format.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
        compression: str = COMPRESSION,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.row_group_size = row_group_size
        self.compression = compression

    # -- public API ---------------------------------------------------------- #

    def export_records(
        self,
        records: Iterable[Record],
        file_name: str = "records.parquet",
        *,
        batch_size: int | None = None,
        row_group_size: int | None = None,
    ) -> Path:
        """Stream an iterable of records into a Parquet file (the memory-safe path).

        The schema is derived from the first batch; later batches are aligned to it.
        """
        out_path = self.output_dir / file_name
        logger.info(f"Exporting records to Parquet at {out_path}...")
        rows, row_groups = write_records_streaming(
            records,
            out_path,
            batch_size=self.batch_size if batch_size is None else batch_size,
            row_group_size=self.row_group_size if row_group_size is None else row_group_size,
            compression=self.compression,
        )
        logger.info(
            f"Successfully saved {out_path} (Size: {out_path.stat().st_size / (1024 * 1024):.2f} MB, "
            f"rows: {rows}, row groups: {row_groups})"
        )
        return out_path

    def export_messages(
        self, messages: Iterable[CleanedMessage], file_name: str = "full_clean_messages.parquet"
    ) -> Path:
        """Export all cleaned messages to Parquet (empty texts are excluded — empty rows in a 'clean' corpus are noise)."""
        out_path = self.output_dir / file_name
        total = len(messages) if isinstance(messages, Sized) else None
        logger.info(f"Exporting {total if total is not None else 'all'} messages to Parquet at {out_path}...")

        dropped = 0

        def records() -> Iterator[Record]:
            nonlocal dropped
            for m in messages:
                # Drop messages with empty/whitespace-only text: they carry no
                # linguistic value and pollute the "clean" corpus (~5% of rows
                # in earlier exports).
                if not m.text_clean or not m.text_clean.strip():
                    dropped += 1
                    continue
                yield _message_record(m)

        rows, row_groups = write_records_streaming(
            records(),
            out_path,
            batch_size=self.batch_size,
            row_group_size=self.row_group_size,
            compression=self.compression,
            schema=MESSAGE_SCHEMA,
        )
        if dropped:
            logger.info(f"Filtered out {dropped} empty-text messages before Parquet export.")
        logger.info(
            f"Successfully saved {out_path} (Size: {out_path.stat().st_size / (1024 * 1024):.2f} MB, "
            f"rows: {rows}, row groups: {row_groups})"
        )
        return out_path

    def export_sft_dialogues(self, dialogues: Iterable[SFTDialogue], file_name: str = "sft_dialogues.parquet") -> Path:
        """Export SFT dialogues to Parquet."""
        out_path = self.output_dir / file_name
        total = len(dialogues) if isinstance(dialogues, Sized) else None
        logger.info(f"Exporting {total if total is not None else 'all'} SFT dialogues to Parquet at {out_path}...")

        rows, row_groups = write_records_streaming(
            (_sft_record(d) for d in dialogues),
            out_path,
            batch_size=self.batch_size,
            row_group_size=self.row_group_size,
            compression=self.compression,
            schema=SFT_SCHEMA,
        )
        logger.info(
            f"Successfully saved {out_path} (Size: {out_path.stat().st_size / (1024 * 1024):.2f} MB, "
            f"rows: {rows}, row groups: {row_groups})"
        )
        return out_path

    def export_rag_chunks(self, chunks: Iterable[RAGChunk], file_name: str = "rag_knowledge_base.parquet") -> Path:
        """Export RAG chunks to Parquet."""
        out_path = self.output_dir / file_name
        total = len(chunks) if isinstance(chunks, Sized) else None
        logger.info(f"Exporting {total if total is not None else 'all'} RAG chunks to Parquet at {out_path}...")

        rows, row_groups = write_records_streaming(
            (_rag_record(c) for c in chunks),
            out_path,
            batch_size=self.batch_size,
            row_group_size=self.row_group_size,
            compression=self.compression,
            schema=RAG_SCHEMA,
        )
        logger.info(
            f"Successfully saved {out_path} (Size: {out_path.stat().st_size / (1024 * 1024):.2f} MB, "
            f"rows: {rows}, row groups: {row_groups})"
        )
        return out_path
