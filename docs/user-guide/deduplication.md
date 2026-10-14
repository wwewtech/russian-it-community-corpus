# Deduplication

`src/deduplication/` removes duplicate and near-duplicate messages before
taxonomy tagging and export. Two stages, both strict-typed.

## Stage 1 — Exact Duplicates

`ExactDeduplicator` (`exact_dedup.py`):

- Normalizes text (whitespace, unicode NFC) and hashes with **xxHash
  (XXH3)** and SHA-256 for provenance.
- O(n) set lookups; the hash of every kept message is recorded in the
  artifact manifest (`reports/dataset_manifest.json`) so downstream
  consumers can verify corpus identity.

## Stage 2 — Near Duplicates (MinHash LSH)

`MinHashLSH` (`minhash_lsh.py`):

- Shingles the normalized text into character n-grams.
- MinHash signature (configurable permutations) + LSH banding for
  candidate pairs — sublinear in corpus size.
- Jaccard similarity above the threshold marks a near-duplicate cluster;
  the lexicographically-first message id is kept as the canonical one.

## Why Both

Exact hashing is cheap and catches copy-paste/fwd spam; MinHash catches
reposts with edited punctuation, trailing quotes or emoji. Both stages
record removed counts into the pipeline log (`stage_end(removed=...)`) and
the messages counter (`ricc_pipeline_messages_processed_total`).

## Tuning

Thresholds and shingle size are module constants in `minhash_lsh.py` —
they are conservative by default. If you observe false merges on short
technical messages (e.g. `+1`, `ok`), raise the min-length floor before
touching the similarity threshold, and re-run `make validate`.

## Downstream Effects

- Removed messages never reach `full_clean_messages.parquet`.
- Dedup statistics feed the analytics report and the dataset manifest.
