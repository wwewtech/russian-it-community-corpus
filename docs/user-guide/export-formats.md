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

## ChatML Masking

SFT exporters apply prompt-loss masking (-100 over the prompt tokens) as
a training-time transform in the LoRA loaders — the JSONL stores the raw
dialogue, masking happens in [LoRA Training](lora-training.md).

## Provenance & Sync

`ricc/exporter/finalize_sync_all.py` is the release-side finisher:
reconciles local artifacts with the pinned Hugging Face Hub revision,
uploads missing adapters, and regenerates `reports/LORA_MODEL_ZOO.md`.
It runs in the release workflow only.

## Regenerating Reports

```bash
make reports       # model zoo catalog
make reports-real  # JSON reports from REAL local parquet (CI-fresh stats)
```

`reports-real` refuses synthetic markers — it is the honest-numbers path.
