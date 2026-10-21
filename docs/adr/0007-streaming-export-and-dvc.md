# ADR 0007: The export streams, and DVC is a real pipeline

- **Status**: Accepted
- **Date**: 2026-10-21
- **Deciders**: maintainer
- **Relates to**: `docs/development/dvc-pipeline.md`, `ricc/exporter/parquet_exporter.py`

## Context

Two independent parts of the platform did not do what they claimed.

### The export

`ParquetExporter` built a list of every record, a `pandas.DataFrame` on top
of it, and wrote the result with a single `to_parquet` call. For the
published corpus (2 816 434 messages) that is:

- a peak memory footprint that **grows with the corpus** (the records list
  alone is ~1.24 GB for the text column, plus the DataFrame on top);
- a file with **three row groups of ~1M rows each** — a reader cannot
  fetch a slice, it decompresses a third of the corpus.

The MinHash LSH index had the same shape: a Python `list[int]` of 128
entries per message, ~5 kB each, about **8.9 GB** for the 1.78M indexed
messages of the corpus.

### DVC

`dvc.yaml` described twelve stages, `params.yaml` was called "the single
source of truth", and `make dvc-repro` was documented. None of it worked:

- `dvc init` had never been run: there was no `.dvc/`, no `dvc.lock` and
  no remote;
- the stages were inline `python -c "..."` blocks, which do not survive a
  Windows shell — `cmd.exe` splits the multi-line quoted string and the
  stage dies with `'from' is not recognized as an internal or external
  command`. So `dvc repro` had never run, on any platform;
- the `ingest` stage declared only source files as dependencies, never the
  raw Telegram exports, so DVC could not detect a change in the corpus;
- `pii_audit.sample_size` was declared as a parameter of three stages that
  do not use it, so editing it invalidated unrelated stages — and the value
  was not read anywhere at runtime either (see ADR 0008);
- the `drift` stage compared the corpus **against itself**, so it could
  only ever report "stable".

## Decision

### The export streams

`write_records_streaming` writes through a `pyarrow.parquet.ParquetWriter`
in batches. The consequences:

- the **peak memory is bounded by the row group**, not by the corpus. With
  a 10k row group the peak stays flat at ~36 MB for 20k, 80k and 320k
  records alike, where the old path grew linearly (102 MB at 100k rows,
  1.6 GB at 1.6M rows);
- the file has **29 row groups** instead of 3, so a consumer can read a
  slice;
- the schema is **pinned** (`MESSAGE_SCHEMA` / `SFT_SCHEMA` /
  `RAG_SCHEMA`), because inferring it from the first batch alone is
  unsafe: a first batch whose `reply_to_id` is all-None yields a `null`
  column and a later batch with real integers then fails outright;
- the write is **atomic** (`.partial` + `os.replace`), so a failed export
  can never leave a truncated file where a consumer expects a dataset;
- the file grows ~7 % (198.5 → 212.6 MB): zstd compresses a large block
  better than a small one. The extra bytes buy the ability to read a slice.

The MinHash signatures moved into one contiguous `uint32` block: **8.9 GB →
0.9 GB** for the corpus, and the permutation arithmetic is vectorised over
the shingles, so the index is also faster.

### DVC is real

- `dvc init` has been run; `.dvc/config` names a remote and `dvc.lock` is
  committed;
- every stage is a **real script** (`scripts/dvc_stages.py <stage>`) with a
  `main()`, invocable by DVC or by hand and unit-testable;
- the raw exports are a **declared dependency** of the ingest stage. DVC
  does not expand globs in `deps:`, so they are listed explicitly and
  `scripts/refresh_dvc_deps.py --check` keeps the list in sync;
- a parameter is declared only by the stage that **reads** it;
- the drift stage compares against a **frozen reference snapshot**
  (`RICC_DRIFT_REFERENCE`), never against the current file;
- CI has a job that checks the dependency list, the stage definitions (no
  inline `python -c`) and the lock file.

## Consequences

- Exporting the corpus no longer needs a machine with tens of gigabytes of
  free RAM for the export step alone.
- The published files on the Hub still have three row groups; they will
  be re-exported with the streaming writer, which changes their bytes and
  therefore their manifest hashes. The dataset version is unchanged (the
  contract is the same), but the artefact hashes are.
- The DVC pipeline now has a real cost: `dvc repro` on the full corpus
  runs the whole pipeline, which is hours of GPU-free CPU work plus the
  MinHash pass. The lock file and the cache make the second run cheap.
- The DVC remote defaults to a local directory outside the working copy.
  A shared setup points it at object storage; the rest of the
  configuration does not change.

## Alternatives rejected

- **Keep the list-and-DataFrame export.** It works on a small corpus and
  fails exactly where it matters.
- **Use `row_group_size` on the existing call.** That fixes the file
  layout but not the memory: the whole corpus is still materialised twice
  before the write.
- **Delete the DVC layer.** Tempting — it did not work. But the pipeline
  description is the reproducibility contract of a data project, and
  deleting it would leave the corpus with no way to be rebuilt from the
  raw exports. Fixing it was cheaper than replacing it.
