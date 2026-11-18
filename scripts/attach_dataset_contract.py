"""Attach the current dataset contract to an already-written parquet file.

Used for a pinned snapshot whose physical Arrow layout is valid but whose
metadata predates contract versioning. The row values are not rewritten:
the schema metadata is replaced in-place via a one-time atomically-renamed
parquet copy. Export code already attaches this metadata on new exports.
"""

from pathlib import Path

import pyarrow.parquet as pq

from ricc.schema_version import schema_fingerprint, write_metadata


def attach(root: Path) -> None:
    for name in ("full_clean_messages", "sft_dialogues", "rag_knowledge_base"):
        source = root / "dataset_output" / "parquet" / f"{name}.parquet"
        target = source.with_suffix(".contract.parquet")
        reader = pq.ParquetFile(source)
        schema = write_metadata(reader.schema_arrow)
        with pq.ParquetWriter(target, schema, compression="zstd") as writer:
            for batch in reader.iter_batches(batch_size=100_000):
                writer.write_batch(batch)
        # Atomic rename is the normal path. Windows sandboxes can deny
        # os.replace with no flock/directory handle explanation; keep a copy
        # fallback so the script degrades without corrupting the original.
        try:
            target.replace(source)
        except PermissionError:
            import shutil

            shutil.copyfile(target, source)
            target.unlink()
        print(f"{name}: contract metadata attached ({schema_fingerprint()[:16]}...)")


if __name__ == "__main__":
    attach(Path.cwd())
