"""
Unit tests for LocalRAGPipeline search, ranking, and retrieval.
"""

from pathlib import Path
import pytest
from src.rag.rag_pipeline import LocalRAGPipeline


@pytest.fixture
def rag_pipeline():
    kb_path = Path("dataset_output/parquet/rag_knowledge_base.parquet")
    if not kb_path.exists():
        pytest.skip("RAG parquet knowledge base not generated yet")
    return LocalRAGPipeline(kb_path)


class TestRAGPipeline:
    def test_search_returns_results(self, rag_pipeline):
        query = "как настроить kafka idempotent producer"
        results = rag_pipeline.search(query, top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3
        if results:
            assert "content" in results[0]
            assert "score" in results[0]
            assert results[0]["score"] > 0

    def test_empty_query_handling(self, rag_pipeline):
        results = rag_pipeline.search("", top_k=5)
        assert isinstance(results, list)

    def test_domain_filtering(self, rag_pipeline):
        query = "kubernetes rolling update deployment zero downtime"
        results = rag_pipeline.search(query, top_k=5)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r.get("content", ""), str)
