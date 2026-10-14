# RAG Pipeline

`src/rag/rag_pipeline.py` (`LocalRAGPipeline`) is a dependency-light
hybrid retrieval engine over the 325k-chunk knowledge base: CPU-only,
built on scikit-learn (already a core dependency).

## Design

```
query ──> Stage 1: lexical str.contains pre-filter (cap 5000)
      ──> Stage 2: per-query TF-IDF (1-2 grams, 20k features) + cosine re-rank
      ──> hybrid score = 0.65·tfidf + 0.35·lexical (scaled 0..10)
```

- **Per-query TF-IDF fit** keeps memory flat: no global 325k×30k matrix;
  the vectorizer is fitted on the candidate set plus the query.
- **Graceful degradation**: without sklearn, or with < 2 candidates, the
  pipeline falls back to lexical scores — `search()` never raises on
  retrieval errors, it returns `[]`.
- `mode="hybrid"` (default) | `"lexical"` | `"tfidf"`.

## Usage

```python
from pathlib import Path
from src.rag.rag_pipeline import LocalRAGPipeline

kb = LocalRAGPipeline(Path("dataset_output/parquet/rag_knowledge_base.parquet"))
hits = kb.search("как настроить nginx reverse proxy", top_k=3, domain_filter="devops_infra")
# hits: [{"chunk_id", "title", "domain", "tags", "content", "date_range", "score"}, ...]

prompt = kb.format_rag_prompt("вопрос", hits)  # context-augmented LLM prompt
```

## Consumers

- `src/inference.py` — terminal chat (`cli.py chat`), injects top-k
  context into the system prompt.
- `official_academic_benchmarks.py` — RAG-augmented benchmark variants
  (HumanEval / RuMMLU with retrieved context).
- Streamlit Studio — "База знаний RAG" search page.

## Quality

`tests/test_rag_pipeline.py` covers ranking order (hybrid > lexical on
domain-targeted queries), domain filtering, empty-KB and missing-sklearn
fallbacks, and the `BenchmarkComparator` keyword-recall metric.
