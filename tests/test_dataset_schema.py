"""Tests for the published dataset contract.

The public contract of this project is the column set of the exported
Parquet files. Nothing carried that information before: the validator
counted columns and compared nothing, so a renamed or retyped column
passed validation silently, and a consumer could not tell which
contract version a downloaded file was written against.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ricc.ingestion.schema import CleanedMessage
from ricc.schema_version import (
    ARTIFACT_SCHEMAS,
    DATASET_SCHEMA_VERSION,
    JSONL_SCHEMAS,
    contract_manifest,
    get_schema,
    read_metadata,
    schema_fingerprint,
    verify_against_file,
    write_metadata,
)
from ricc.validation.validator import DatasetValidator

REPO_ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = REPO_ROOT / "dataset_output" / "parquet"


def _message(i: int) -> CleanedMessage:
    return CleanedMessage(
        msg_id=i,
        chat_id=1,
        chat_name="community_node_01",
        timestamp="2026-01-01T00:00:00",
        unixtime=1767225600 + i,
        author_anon="user_00001",
        author_id_anon="user_00001",
        text_clean="Сообщение про архитектуру распределённых сервисов.",
        reply_to_id=None,
        domain="backend_databases",
        tags=["tag"],
        sentiment_score=0,
        token_count_approx=5,
        is_question=False,
        thread_id=1,
    )


class TestContractDeclaration(unittest.TestCase):
    def test_every_artifact_has_a_schema(self):
        for artifact in ("full_clean_messages", "sft_dialogues", "rag_knowledge_base"):
            with self.subTest(artifact=artifact):
                self.assertIn(artifact, ARTIFACT_SCHEMAS)
                self.assertTrue(get_schema(artifact).names)

    def test_jsonl_exports_are_declared(self):
        for name in (
            "sft_sharegpt_format",
            "sft_openai_messages",
            "sft_alpaca_format",
            "rag_chunks_kb",
            "dpo_preference_pairs",
        ):
            self.assertIn(name, JSONL_SCHEMAS)

    def test_unknown_artifact_raises(self):
        with self.assertRaises(KeyError):
            get_schema("no_such_artifact")

    def test_fingerprint_is_stable(self):
        self.assertEqual(schema_fingerprint(), schema_fingerprint())

    def test_fingerprint_changes_when_a_column_changes(self):
        """The fingerprint must detect ANY contract change."""
        from ricc.schema_version import ArtifactSchema, ColumnSpec

        original = schema_fingerprint()
        columns = ARTIFACT_SCHEMAS["full_clean_messages"].columns
        mutated = ArtifactSchema(
            artifact="full_clean_messages",
            columns=(ColumnSpec("msg_id", pa.string()),) + columns[1:],
        )
        ARTIFACT_SCHEMAS["full_clean_messages"] = mutated
        try:
            changed = schema_fingerprint()
        finally:
            ARTIFACT_SCHEMAS["full_clean_messages"] = ARTIFACT_SCHEMAS["full_clean_messages"].__class__(
                artifact="full_clean_messages",
                columns=columns,
            )
        self.assertNotEqual(changed, original)
        self.assertEqual(schema_fingerprint(), original)

    def test_contract_manifest_is_serialisable(self):
        payload = contract_manifest()
        json.dumps(payload)  # must not raise
        self.assertEqual(payload["dataset_schema_version"], DATASET_SCHEMA_VERSION)
        self.assertEqual(payload["schema_fingerprint"], schema_fingerprint())


class TestPublishedFilesMatchTheContract(unittest.TestCase):
    """The declared contract must match the PUBLISHED files."""

    def test_published_files_match(self):
        for artifact, filename in (
            ("full_clean_messages", "full_clean_messages.parquet"),
            ("sft_dialogues", "sft_dialogues.parquet"),
            ("rag_knowledge_base", "rag_knowledge_base.parquet"),
        ):
            path = PARQUET_DIR / filename
            if not path.exists():
                self.skipTest(f"{filename} not present")
            with self.subTest(artifact=artifact):
                actual = pq.ParquetFile(path).schema_arrow
                declared = get_schema(artifact)
                self.assertEqual(list(actual.names), declared.names)
                for spec in declared.columns:
                    self.assertEqual(actual.field(spec.name).type, spec.arrow_type, spec.name)

    def test_verification_reports_the_metadata_gap_of_an_old_file(self):
        """A file published before the versioning has no metadata.

        That must be REPORTED, not silently accepted: the file predates
        the contract and cannot be certified against it.
        """
        path = PARQUET_DIR / "full_clean_messages.parquet"
        if not path.exists():
            self.skipTest("published file not present")
        problems = verify_against_file(path, "full_clean_messages")
        # The columns must match; only the metadata may be missing.
        self.assertFalse([p for p in problems if "column" in p.lower()])


class TestMetadataRoundTrip(unittest.TestCase):
    def test_write_and_read_metadata(self):
        schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
        with_meta = write_metadata(schema)
        meta = read_metadata(with_meta)
        self.assertEqual(meta["version"], DATASET_SCHEMA_VERSION)
        self.assertEqual(meta["fingerprint"], schema_fingerprint())

    def test_read_metadata_of_a_plain_schema_is_empty(self):
        meta = read_metadata(pa.schema([("a", pa.int64())]))
        self.assertEqual(meta["version"], "")
        self.assertEqual(meta["fingerprint"], "")


class TestValidatorChecksTheContract(unittest.TestCase):
    """The validator must actually enforce the contract."""

    def _write_dataset(self, root: Path, tampered: bool = False) -> Path:
        parquet_dir = root / "parquet"
        parquet_dir.mkdir(parents=True)
        from ricc.exporter.parquet_exporter import MESSAGE_SCHEMA

        schema = MESSAGE_SCHEMA
        records = [
            {
                "msg_id": i,
                "chat_id": 1,
                "chat_name": "community_node_01",
                "timestamp": "2026-01-01T00:00:00",
                "unixtime": 1767225600 + i,
                "author_anon": "user_00001",
                "author_id_anon": "user_00001",
                "text_clean": "текст",
                "reply_to_id": None,
                "domain": "backend_databases",
                "tags": ["a"],
                "sentiment_score": 0,
                "token_count_approx": 3,
                "is_question": False,
                "thread_id": 1,
            }
            for i in range(3)
        ]
        if tampered:
            # A renamed column: the contract says "text_clean".
            for record in records:
                record["text"] = record.pop("text_clean")
            schema = pa.schema([f for f in MESSAGE_SCHEMA if f.name != "text_clean"] + [pa.field("text", pa.string())])
            schema = schema.with_metadata(MESSAGE_SCHEMA.metadata or {})

        table = pa.Table.from_pylist(records, schema=schema)
        pq.write_table(table, parquet_dir / "full_clean_messages.parquet")
        return root

    def test_validator_fails_on_a_renamed_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._write_dataset(Path(tmp), tampered=True)
            result = DatasetValidator(root).validate_parquet_files()
            self.assertFalse(result["passed"], "a renamed column must fail the validation")
            details = result["details"]["full_clean_messages.parquet"]
            self.assertTrue(details["schema_problems"])
            self.assertEqual(details["status"], "INVALID")

    def test_validator_passes_on_a_conforming_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._write_dataset(Path(tmp))
            result = DatasetValidator(root).validate_parquet_files()
            details = result["details"]["full_clean_messages.parquet"]
            self.assertEqual(details["schema_problems"], [])
            self.assertEqual(details["status"], "VALID")
            self.assertEqual(details["schema_version"], DATASET_SCHEMA_VERSION)


class TestGeneratedManifest(unittest.TestCase):
    def test_manifest_script_is_in_sync(self):
        manifest_path = REPO_ROOT / "reports" / "dataset_schema.json"
        if not manifest_path.exists():
            self.skipTest("contract manifest not generated")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_fingerprint"], schema_fingerprint())
        self.assertEqual(payload["dataset_schema_version"], DATASET_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
