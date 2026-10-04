# RICC — Russian IT Community Corpus

**RICC** (Russian IT Community Corpus) is a production-grade data engineering platform for LLM dataset curation with Zero-PII guarantees.

## Features

- **End-to-end Pipeline**: 7-stage data curation from raw Telegram exports to SFT/DPO/RAG datasets
- **Zero-PII Guarantees**: Dual-layer scrubbing (Regex + NER) + pseudonymization + probabilistic audit
- **Deduplication**: Exact hashing + MinHash LSH for fuzzy near-duplicate detection
- **Taxonomy Classification**: 9 IT domains with 500+ technical keywords via Aho-Corasick
- **Conversation Graph**: Thread DAG reconstruction with reply-time heuristics
- **Multi-format Export**: Parquet, ShareGPT, Alpaca, OpenAI ChatML, RAG JSONL, DPO pairs
- **Analytics Engine**: Deep statistical + semantic analysis with interactive reports
- **Probabilistic PII Audit**: Stratified sampling + Wilson CI + power analysis (α=0.01)
- **Drift Monitoring**: PSI + Jensen-Shannon + Vocabulary Jaccard with thresholds
- **SLO Gate**: Fail-closed privacy, fail-open informational — SHIP/HOLD verdicts
- **LoRA Training**: Multi-model batch training with MLflow tracking & model registry
- **RAG Pipeline**: Hybrid dense+sparse retrieval with BM25 + embeddings
- **Observability**: Structured logging (structlog) + Prometheus metrics
- **Typed Exceptions**: Complete error hierarchy with context & recoverability

## Quick Links

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Architecture Overview](architecture/overview.md)
- [API Reference](api/ricc.md)

## Installation

```bash
pip install ricc
```

Or with optional platform dependencies (Prefect, DVC):

```bash
pip install ricc[platform]
```

## Quick Start

```python
from ricc import Settings, MasterDataPipeline

# Configure (or use defaults)
settings = Settings()

# Run full pipeline
pipeline = MasterDataPipeline()
summary = pipeline.run_all()

print(f"Processed {summary['cleaned_messages_count']:,} messages")
print(f"Extracted {summary['sft_dialogues_count']:,} SFT dialogues")
print(f"Validation: {'PASSED' if summary['validation_passed'] else 'FAILED'}")
```

## CLI Usage

```bash
# Run full pipeline
it-pipeline run

# Validate datasets
it-pipeline validate

# Run PII audit
it-pipeline audit-prob

# Run drift monitoring
it-pipeline drift

# Export benchmarks
it-pipeline benchmark
```

## Streamlit Data Studio

```bash
make ui
# or
streamlit run app.py
```

## Citation

```bibtex
@software{ricc2026,
  title = {RICC: Russian IT Community Corpus — Data Engineering & Zero-PII Curation Platform},
  author = {wwewtech},
  year = {2026},
  url = {https://github.com/wwewtech/russian-it-community-corpus}
}
```

## License

MIT License — see [LICENSE](https://github.com/wwewtech/russian-it-community-corpus/blob/main/LICENSE)