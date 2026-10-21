# Pipeline Stages

`MasterDataPipeline` (`ricc/pipeline.py`) runs seven stages in sequence.
Each stage is wrapped in `time_stage()`, so its duration lands in the
`ricc_pipeline_stage_duration_seconds` histogram and its outcome in
`ricc_pipeline_stage_total{stage, status}`.

## Stage Diagram

```
ChatExport_*/result.json
        │
        ▼
[1] Ingestion ─── merge_multiple_exports() ──> list[NormalizedMessage]
        ▼
[2] PII anonymization ── UnifiedPIIAnonymizer ──> list[CleanedMessage]
        ▼
[3] Deduplication ── ExactDeduplicator + MinHashLSH
        ▼
[4] Taxonomy tagging ── TechnicalTagger ──> domain + tags
        ▼
[5] Graph extraction ── ThreadDAGBuilder + ConversationExtractor
        │                 ├──> list[SFTDialogue]
        │                 └──> list[RAGChunk]
        ▼
[6] Export ── Parquet / JSONL (SFT, RAG, DPO)
        ▼
[7] Analytics & validation ── DeepChatAnalyzer + DatasetValidator
        ▼
dataset_output/ + reports/
```

## Stage Contracts

| Stage | Input → Output | Failure mode |
|---|---|---|
| 1 Ingestion | raw exports → `NormalizedMessage` | `IngestionError` on unreadable/malformed export |
| 2 PII | raw text → masked text + anon ids | `PIIAnonymizationError` |
| 3 Dedup | messages → deduplicated | `DeduplicationError` |
| 4 Taxonomy | text → domain/tags | `TaxonomyError` |
| 5 Graph | messages → threads/dialogues/chunks | `GraphExtractionError` |
| 6 Export | objects → parquet/jsonl | `ExportError` |
| 7 Analytics | artifacts → reports | `AnalyticsError` |

Typed errors carry `stage` context — see [Errors](../api/errors.md).

## Idempotence & State

The pipeline is stateless between runs: every run rebuilds from the raw
exports. Partial artifacts from a crashed run are overwritten; the
artifact manifest must be re-created after any successful run
(`python -m ricc.validation.artifact_manifest create ...`) so the
[SLO gate](../user-guide/slo-gate.md) can verify provenance.

## Runtime Environment

`ricc/bootstrap.setup_runtime_env()` centralizes UTF-8 stdio (Windows
cp1251 protection), `HF_HOME` → repo-local `.hf_cache`, tokenizers env
and structlog configuration. Entry points call it before importing heavy
third-party libraries — keep that ordering in new scripts.
