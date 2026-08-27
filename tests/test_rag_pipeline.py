"""
Unit tests for LocalRAGPipeline search, ranking, and retrieval.
"""

import unittest
from pathlib import Path

from src.rag.rag_pipeline import LocalRAGPipeline


class TestRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb_path = Path("dataset_output/parquet/rag_knowledge_base.parquet")
        if not cls.kb_path.exists():
            raise unittest.SkipTest("RAG parquet knowledge base not generated yet")
        cls.rag_pipeline = LocalRAGPipeline(cls.kb_path)

    def test_search_returns_results(self):
        query = "как настроить kafka idempotent producer"
        results = self.rag_pipeline.search(query, top_k=3)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        if results:
            self.assertIn("content", results[0])
            self.assertIn("score", results[0])
            self.assertGreater(results[0]["score"], 0)

    def test_empty_query_handling(self):
        results = self.rag_pipeline.search("", top_k=5)
        self.assertIsInstance(results, list)

    def test_domain_filtering(self):
        query = "kubernetes rolling update deployment zero downtime"
        results = self.rag_pipeline.search(query, top_k=5)
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIsInstance(r.get("content", ""), str)

    def test_rag_properties_and_len(self):
        self.assertIsNotNone(self.rag_pipeline.df)
        self.assertEqual(len(self.rag_pipeline), len(self.rag_pipeline.df_kb))
        self.assertEqual(len(self.rag_pipeline.df), len(self.rag_pipeline.df_kb))

    def test_format_rag_prompt(self):
        query = "как настроить nginx"
        sample_contexts = [{"title": "Nginx Config", "date_range": "2024", "content": "worker_processes auto;"}]
        formatted = self.rag_pipeline.format_rag_prompt(query, sample_contexts)
        self.assertIn("Nginx Config", formatted)
        self.assertIn("worker_processes auto;", formatted)
        self.assertIn(query, formatted)


if __name__ == "__main__":
    unittest.main()
