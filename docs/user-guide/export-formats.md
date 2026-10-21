# Export Formats

`ricc/exporter/` turns cleaned messages, SFT dialogues, RAG chunks and DPO
pairs into training-ready files. All exporters are strict-typed and
covered by round-trip tests (`tests/test_exporters_roundtrip.py`).

## Outputs

| File | Exporter | Format |
|---|---|---|
| `dataset_output/parquet/full_clean_messages.parquet` | `ParquetExporter` | columnar corpus (canonical artifact) |
| `dataset_output/parquet/sft_dialogues.parquet` | `ParquetExporter` | SFT dialogue table |
| `dataset_output/parquet/rag_knowledge_base.parquet` | `ParquetExporter` | RAG chunk table |
| `dataset_output/jsonl/sft_openai_messages.jsonl` | `JSONLExporter` | OpenAI ChatML `{"messages": [...]}` |
| `dataset_output/jsonl/sft_sharegpt.jsonl` | `JSONLExporter` | ShareGPT `conversations` |
| `dataset_output/jsonl/sft_alpaca.jsonl` | `JSONLExporter` | Alpaca instruction/input/output |
| `dataset_output/jsonl/rag_knowledge_base.jsonl` | `RAGExporter` | chunk records with metadata |
| `dataset_output/jsonl/dpo_preference_pairs.jsonl` | `DPOExporter` | chosen/rejected pairs |

The three parquet files are the **canonical artifacts**: the dataset
manifest (`reports/dataset_manifest.json`) pins their sha256 digests, and
the [SLO gate](slo-gate.md) verifies provenance against it.

## The contract is versioned

The **column set** of these files is the public contract of the project.
It is declared in `ricc/schema_version.py` — the column names, their
order and their Arrow types — and:

- every exported Parquet file carries the contract **version and
  fingerprint** in its Arrow metadata, so a downloaded file says which
  contract it was written against;
- the validator **enforces** it: a renamed, reordered or retyped column
  fails the validation instead of passing silently;
- `reports/dataset_schema.json` records the whole contract, and CI keeps
  it in sync with the code.

See [ADR 0006](../adr/0006-dataset-contract-versioning.md).

## The export streams

`ParquetExporter` writes through a `pyarrow.parquet.ParquetWriter` in
batches, so the peak memory is bounded by the row group rather than
growing with the corpus, and the file has many row groups (29 instead of
3 for the message corpus) — a consumer can read a slice instead of
decompressing a third of the file.

Three properties the streaming writer guarantees:

- the **schema is pinned** (`MESSAGE_SCHEMA` / `SFT_SCHEMA` /
  `RAG_SCHEMA`), because inferring it from the first batch alone is
  unsafe: a first batch whose `reply_to_id` is all-`None` yields a `null`
  column and a later batch with real integers then fails outright;
- the write is **atomic** (a `.partial` file plus `os.replace`), so a
  failed export never leaves a truncated file where a consumer expects a
  dataset;
- a **later batch with a different key set** is aligned to the schema
  (missing key → null, unknown key → dropped and logged) instead of
  producing a corrupt file. `strict_schema=True` turns the alignment into
  an error.

The file grows ~7 % compared to the single-shot writer: zstd compresses
a large block better than a small one. The extra bytes buy the ability to
read a slice.

## ChatML Masking

SFT exporters apply prompt-loss masking (-100 over the prompt tokens) as
a training-time transform in the LoRA loaders — the JSONL stores the raw
dialogue, masking happens in [LoRA Training](lora-training.md).

## Provenance & Sync

`ricc/exporter/finalize_sync_all.py` is the release-side finisher:
verifies the artifact manifest, reconciles local artifacts with the
pinned Hugging Face Hub revision, uploads the adapters that satisfy the
[training policy](lora-training.md), and regenerates
`reports/LORA_MODEL_ZOO.md`. It fails closed: an upload error aborts the
release instead of being swallowed with a warning.

## Regenerating Reports

```bash
make reports       # model zoo catalog
make reports-real  # JSON reports from REAL local parquet (CI-fresh stats)
```

`reports-real` refuses synthetic markers — it is the honest-numbers path.
