"""The published dataset contract: columns, types and a version.

The public contract of this project is the COLUMN SET of the exported
Parquet files and JSONL exports. A consumer who downloads the dataset
must be able to tell whether a column was renamed, removed or retyped
between releases — otherwise a silent contract break looks like a data
change.

Nothing carried that information before: ``artifact_manifest.SCHEMA_VERSION``
versions the MANIFEST FORMAT, not the data, and ``DatasetValidator``
claimed to check "schema conformance" while only counting columns
(``col_count = table.num_columns``) and comparing nothing.

This module is the single declaration of the contract:

* :data:`DATASET_SCHEMA_VERSION` versions the contract as a whole;
* :func:`schema_fingerprint` hashes the whole contract, so any change to
  any column is detectable;
* :func:`get_schema` returns the declared schema of one artefact;
* :func:`write_metadata` / :func:`read_metadata` put the version and the
  fingerprint into the Parquet key-value metadata and read them back.

The declared schemas below were taken from the PUBLISHED files
(``dataset_output/parquet/*.parquet``) and are verified against them by
``tests/test_dataset_schema.py``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa

# The version of the published data contract. Bump the MAJOR version
# when a column is removed or its type changes (a breaking change for a
# consumer), the MINOR version when a column is added (a compatible
# change), and the PATCH version for a documentation-only change.
DATASET_SCHEMA_VERSION = "1.0.0"

# Arrow key-value metadata keys.
METADATA_VERSION_KEY = b"dataset_schema_version"
METADATA_FINGERPRINT_KEY = b"dataset_schema_fingerprint"


@dataclass(frozen=True)
class ColumnSpec:
    """One column of the published contract."""

    name: str
    arrow_type: pa.DataType
    nullable: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": str(self.arrow_type),
            "nullable": self.nullable,
            "description": self.description,
        }


@dataclass(frozen=True)
class ArtifactSchema:
    """The contract of one exported artefact."""

    artifact: str
    columns: tuple[ColumnSpec, ...] = field(default_factory=tuple)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "columns": [c.to_dict() for c in self.columns],
        }


# --------------------------------------------------------------------------- #
# The canonical artefacts
# --------------------------------------------------------------------------- #

MESSAGES_SCHEMA = ArtifactSchema(
    artifact="full_clean_messages",
    columns=(
        ColumnSpec("msg_id", pa.int64(), description="Message id, unique within a chat"),
        ColumnSpec("chat_id", pa.int64(), description="Source community node id"),
        ColumnSpec("chat_name", pa.string(), description="Surrogate community node name"),
        ColumnSpec("timestamp", pa.string(), description="ISO-8601 UTC timestamp"),
        ColumnSpec("unixtime", pa.int64(), description="Unix timestamp, seconds"),
        ColumnSpec("author_anon", pa.string(), description="Pseudonymised author display name"),
        ColumnSpec("author_id_anon", pa.string(), description="Pseudonymised author id"),
        ColumnSpec("text_clean", pa.string(), description="Anonymised message text"),
        ColumnSpec("reply_to_id", pa.float64(), description="Replied-to message id (null when not a reply)"),
        ColumnSpec("domain", pa.string(), description="Taxonomy domain"),
        ColumnSpec("tags", pa.list_(pa.string()), description="Technical keyword tags"),
        ColumnSpec("sentiment_score", pa.int64(), description="Sentiment score"),
        ColumnSpec("token_count_approx", pa.int64(), description="Approximate token count"),
        ColumnSpec("is_question", pa.bool_(), description="Whether the message is a question"),
        ColumnSpec("thread_id", pa.int64(), description="Thread id (null when unthreaded)"),
    ),
)

SFT_SCHEMA = ArtifactSchema(
    artifact="sft_dialogues",
    columns=(
        ColumnSpec("thread_id", pa.int64(), description="Thread id"),
        ColumnSpec("chat_name", pa.string(), description="Surrogate community node name"),
        ColumnSpec("topic_domain", pa.string(), description="Taxonomy domain of the thread"),
        ColumnSpec("topic_tags", pa.list_(pa.string()), description="Topic keyword tags"),
        ColumnSpec(
            "messages",
            pa.list_(
                pa.struct(
                    [
                        ("author", pa.string()),
                        ("content", pa.string()),
                        ("role", pa.string()),
                        ("timestamp", pa.string()),
                    ]
                )
            ),
            description="The dialogue turns",
        ),
        ColumnSpec("quality_score", pa.float64(), description="Quality score of the dialogue"),
        ColumnSpec("turn_count", pa.int64(), description="Number of turns"),
        ColumnSpec("total_tokens", pa.int64(), description="Approximate token count"),
    ),
)

RAG_SCHEMA = ArtifactSchema(
    artifact="rag_knowledge_base",
    columns=(
        ColumnSpec("chunk_id", pa.string(), description="Chunk id"),
        ColumnSpec("thread_id", pa.int64(), description="Source thread id"),
        ColumnSpec("chat_name", pa.string(), description="Surrogate community node name"),
        ColumnSpec("title", pa.string(), description="Chunk title"),
        ColumnSpec("topic_domain", pa.string(), description="Taxonomy domain"),
        ColumnSpec("topic_tags", pa.list_(pa.string()), description="Topic keyword tags"),
        ColumnSpec("content", pa.string(), description="Chunk text"),
        ColumnSpec("date_range", pa.string(), description="Date range of the source thread"),
        ColumnSpec("participants_count", pa.int64(), description="Participants in the source thread"),
        ColumnSpec("message_count", pa.int64(), description="Messages in the source thread"),
        ColumnSpec("token_count", pa.int64(), description="Approximate token count"),
    ),
)

# The JSONL exports and the shape each record has.
JSONL_SCHEMAS: dict[str, dict[str, Any]] = {
    "sft_sharegpt_format": {
        "format": "sharegpt",
        "record": 'one conversation per line: {"conversations": [{"from", "value"}, ...]}',
    },
    "sft_openai_messages": {
        "format": "openai-chatml",
        "record": 'one conversation per line: {"messages": [{"role", "content"}, ...]}',
    },
    "sft_alpaca_format": {
        "format": "alpaca",
        "record": 'one pair per line: {"instruction", "input", "output"}',
    },
    "rag_chunks_kb": {
        "format": "rag-chunk",
        "record": "one chunk per line: the RAG chunk object",
    },
    "dpo_preference_pairs": {
        "format": "dpo",
        "record": 'one pair per line: {"prompt", "chosen", "rejected"}',
    },
}

ARTIFACT_SCHEMAS: dict[str, ArtifactSchema] = {
    "full_clean_messages": MESSAGES_SCHEMA,
    "sft_dialogues": SFT_SCHEMA,
    "rag_knowledge_base": RAG_SCHEMA,
}


def get_schema(artifact: str) -> ArtifactSchema:
    """The declared schema of ``artifact``."""
    try:
        return ARTIFACT_SCHEMAS[artifact]
    except KeyError:
        raise KeyError(f"unknown artefact {artifact!r}; known: {sorted(ARTIFACT_SCHEMAS)}") from None


def schema_fingerprint() -> str:
    """A stable hash of the WHOLE contract.

    Any change to any column of any artefact changes the fingerprint, so
    it can be compared against a snapshot to detect a contract break.
    """
    payload = {
        "version": DATASET_SCHEMA_VERSION,
        "artifacts": {name: schema.to_dict() for name, schema in sorted(ARTIFACT_SCHEMAS.items())},
        "jsonl": {name: JSONL_SCHEMAS[name] for name in sorted(JSONL_SCHEMAS)},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_metadata(schema: pa.Schema) -> pa.Schema:
    """Return ``schema`` with the contract version and fingerprint attached."""
    metadata = dict(schema.metadata or {})
    metadata[METADATA_VERSION_KEY] = DATASET_SCHEMA_VERSION.encode("utf-8")
    metadata[METADATA_FINGERPRINT_KEY] = schema_fingerprint().encode("utf-8")
    return schema.with_metadata(metadata)


def read_metadata(schema: pa.Schema) -> dict[str, str]:
    """The contract metadata carried by a Parquet schema."""
    metadata = schema.metadata or {}
    return {
        "version": metadata.get(METADATA_VERSION_KEY, b"").decode("utf-8", errors="replace"),
        "fingerprint": metadata.get(METADATA_FINGERPRINT_KEY, b"").decode("utf-8", errors="replace"),
    }


def contract_manifest() -> dict[str, Any]:
    """The full contract as a JSON-serialisable manifest."""
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "schema_fingerprint": schema_fingerprint(),
        "artifacts": {name: schema.to_dict() for name, schema in sorted(ARTIFACT_SCHEMAS.items())},
        "jsonl": {name: JSONL_SCHEMAS[name] for name in sorted(JSONL_SCHEMAS)},
    }


def write_contract_manifest(path: str | Path) -> Path:
    """Write the contract manifest next to the other reports."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract_manifest(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def verify_against_file(parquet_path: str | Path, artifact: str) -> list[str]:
    """Check a Parquet file against the declared contract.

    Returns the list of problems; an empty list means the file matches.
    """
    import pyarrow.parquet as pq

    path = Path(parquet_path)
    if not path.exists():
        return [f"{path} does not exist"]

    file_schema = pq.ParquetFile(path).schema_arrow
    declared = get_schema(artifact)
    problems: list[str] = []

    expected_names = declared.names
    actual_names = list(file_schema.names)
    if actual_names != expected_names:
        missing = [n for n in expected_names if n not in actual_names]
        extra = [n for n in actual_names if n not in expected_names]
        if missing:
            problems.append(f"missing column(s): {missing}")
        if extra:
            problems.append(f"unexpected column(s): {extra}")
        if not missing and not extra:
            problems.append(f"column order differs: expected {expected_names}, got {actual_names}")

    for spec in declared.columns:
        if spec.name not in actual_names:
            continue
        actual_type = file_schema.field(spec.name).type
        if actual_type != spec.arrow_type:
            problems.append(f"column {spec.name!r}: expected {spec.arrow_type}, got {actual_type}")

    meta = read_metadata(file_schema)
    if meta["version"] != DATASET_SCHEMA_VERSION:
        problems.append(f"schema version metadata is {meta['version']!r}, expected {DATASET_SCHEMA_VERSION!r}")
    if meta["fingerprint"] != schema_fingerprint():
        problems.append("schema fingerprint metadata does not match the current contract")

    return problems
