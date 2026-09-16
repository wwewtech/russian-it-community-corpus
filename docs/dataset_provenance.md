# Dataset Artifact Manifest

What it does and does not do:

- **Does**: records the exact identity of the three canonical parquet artifacts
  (`dataset_output/parquet/{full_clean_messages,sft_dialogues,rag_knowledge_base}.parquet`)
  — streaming sha256, byte size, row count, and Arrow schema — into a
  schema-versioned snapshot at `reports/dataset_manifest.json`.
- **Does NOT**: establish original training or source provenance. The manifest
  explicitly says: "Local artifact identity snapshots; not original source or
  training provenance." Where the local artifacts came from (which pipeline run,
  which upstream exports) must be tracked separately (see DVC and
  `reports/pipeline_execution_stats.json`).

## Usage

```bash
# Create / refresh the snapshot (fails closed on missing/corrupt artifacts)
python -m src.validation.artifact_manifest create --root . --manifest reports/dataset_manifest.json

# Verify artifacts against the snapshot (used by the SLO gate)
python -m src.validation.artifact_manifest verify --root . --manifest reports/dataset_manifest.json
```

Exit codes: `0` = OK, `1` = fail (missing artifact, hash mismatch, schema
change, size/count mismatch, invalid manifest JSON, path traversal, or any
inconsistency). All failures are fail-closed: there is no "skip" path.

## Verification properties

1. **Streaming sha256** — bounded memory (1 MiB chunks), works for multi-GB files.
2. **Row counts and schema summary** from parquet metadata — catches silent
   column additions/removals and row-count drift that byte hashes alone
   would also catch, but reports them in human-readable form.
3. **Atomic writes** — the snapshot is written to a temp file in the same
   directory and `os.replace`d; a crash never leaves a half-written manifest.
4. **Tamper detection beyond content** — if a file is replaced (new inode),
   or its mtime changes mid-inventory, verification fails. Same-size,
   same-row-count tampering is caught by the hash.
5. **Fail-closed everywhere** — absent manifest, unreadable files, structurally
   invalid JSON, duplicate JSON keys, wrong schema version, or wrong UTC
   timestamp format all raise `ManifestError`; nothing degrades to a "pass".
6. **Path safety** — traversal (`..`), absolute paths outside root, symlinks
   and NTFS reparse points below root, and alternate data streams are rejected.

## Release gate integration

`src/monitoring/slo_gate.py` (`make slo`, used by release checks) verifies the
manifest against the artifacts rooted at the reports directory's parent. Any
mismatch — including a *missing* manifest — flips the verdict to **HOLD**,
because published numbers must correspond to exactly these bytes.

## Current snapshot (2026-09-16)

| Artifact | Rows | Size | sha256 (first 16) |
|---|---:|---:|---|
| full_clean_messages.parquet | 2,418,695 | 156.3 MB | `419bd0284a7d4f84…` |
| sft_dialogues.parquet | 173,216 | 88.6 MB | `183fde9a7ea72558…` |
| rag_knowledge_base.parquet | 332,690 | 128.7 MB | `cafe594c5fa20338…` |

Note: these local working-copy numbers differ from the Hugging Face Hub
snapshot (2,816,434 / 171,520 / 325,690). The README table lists both.
Reconciling the two (re-running the pipeline or re-uploading) is tracked
separately; until then the manifest makes the discrepancy explicit and
machine-checkable instead of silent.
