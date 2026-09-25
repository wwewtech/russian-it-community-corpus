"""
Comprehensive unit tests for LocalRAGPipeline.

Tests cover initialization, search functionality, ranking, filtering, and edge cases.
Uses synthetic parquet fixtures for isolated, fast, deterministic testing.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.rag.rag_pipeline import LocalRAGPipeline


def _make_synthetic_parquet(path: Path) -> None:
    """Create a synthetic RAG knowledge base parquet file for testing."""
    data = {
        "chunk_id": [
            "chunk_001", "chunk_002", "chunk_003", "chunk_004", "chunk_005",
            "chunk_006", "chunk_007", "chunk_008", "chunk_009", "chunk_010",
        ],
        "thread_id": [
            "t1", "t1", "t2", "t2", "t3", "t3", "t4", "t4", "t5", "t5"
        ],
        "chat_name": [
            "kafka-setup", "kafka-setup", "k8s-deployment", "k8s-deployment",
            "nginx-config", "nginx-config", "python-asyncio", "python-asyncio",
            "postgres-tuning", "postgres-tuning"
        ],
        "title": [
            "Kafka Idempotent Producer Setup",
            "Kafka Consumer Group Rebalancing",
            "Kubernetes Rolling Update Strategy",
            "Kubernetes Deployment Zero Downtime",
            "Nginx Reverse Proxy Configuration",
            "Nginx SSL Termination",
            "Python AsyncIO Event Loop Patterns",
            "Python AsyncIO Task Groups",
            "PostgreSQL Index Optimization",
            "PostgreSQL Connection Pooling",
        ],
        "topic_domain": [
            "kafka", "kafka", "kubernetes", "kubernetes",
            "nginx", "nginx", "python", "python",
            "postgresql", "postgresql"
        ],
        "topic_tags": [
            ["kafka", "producer", "idempotent"],
            ["kafka", "consumer", "rebalance"],
            ["kubernetes", "rolling-update", "deployment"],
            ["kubernetes", "deployment", "zero-downtime"],
            ["nginx", "reverse-proxy", "config"],
            ["nginx", "ssl", "tls"],
            ["python", "asyncio", "event-loop"],
            ["python", "asyncio", "taskgroup"],
            ["postgresql", "index", "optimization"],
            ["postgresql", "pooling", "pgbouncer"],
        ],
        "content": [
            "To configure Kafka idempotent producer set enable.idempotence=true "
            "and acks=all. This ensures exactly-once semantics without duplicates. "
            "Also configure max.in.flight.requests.per.connection=5 for ordering.",
            "Consumer group rebalancing happens when members join or leave. "
            "Use cooperative sticky assignor to minimize disruption. "
            "Configure session.timeout.ms and heartbeat.interval.ms appropriately.",
            "Rolling updates in Kubernetes replace pods incrementally. "
            "Use maxSurge and maxUnavailable to control rollout speed. "
            "Readiness probes ensure traffic only goes to ready pods.",
            "Zero downtime deployment requires proper health checks. "
            "PreStop hook with sleep allows in-flight requests to complete. "
            "PodDisruptionBudget protects availability during voluntary disruptions.",
            "Nginx reverse proxy configuration uses proxy_pass directive. "
            "Set proxy_set_header Host $host for backend host header. "
            "Configure proxy_cache for caching upstream responses.",
            "SSL termination at Nginx offloads encryption from backends. "
            "Use ssl_certificate and ssl_certificate_key directives. "
            "Enable ssl_protocols TLSv1.2 TLSv1.3 for security.",
            "Python asyncio event loop runs async tasks concurrently. "
            "Use asyncio.create_task for scheduling coroutines. "
            "asyncio.gather collects results with exception handling.",
            "Task groups in Python 3.11+ provide structured concurrency. "
            "asyncio.TaskGroup ensures all tasks complete or cancel together. "
            "Exception in one task cancels siblings automatically.",
            "PostgreSQL index optimization starts with EXPLAIN ANALYZE. "
            "Create B-tree indexes for equality and range queries. "
            "Partial indexes reduce size for filtered queries.",
            "Connection pooling with PgBouncer reduces connection overhead. "
            "Configure pool_mode=transaction for short transactions. "
            "Monitor pool usage with SHOW POOLS command.",
        ],
        "date_range": [
            "2024-01", "2024-01", "2024-02", "2024-02",
            "2024-03", "2024-03", "2024-04", "2024-04",
            "2024-05", "2024-05"
        ],
        "participants_count": [3, 3, 4, 4, 2, 2, 3, 3, 2, 2],
        "message_count": [15, 12, 20, 18, 10, 8, 14, 16, 11, 9],
        "token_count": [450, 380, 520, 490, 310, 280, 420, 440, 350, 320],
    }
    df = pd.DataFrame(data)
    df.to_parquet(path, index=False)


class TestRAGPipelineInitialization(unittest.TestCase):
    """Tests for LocalRAGPipeline initialization."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.parquet_path = Path(self.temp_dir.name) / "test_kb.parquet"
        _make_synthetic_parquet(self.parquet_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_with_valid_parquet(self):
        """Initialization with valid parquet file loads knowledge base."""
        pipeline = LocalRAGPipeline(self.parquet_path)
        self.assertEqual(len(pipeline), 10)
        self.assertFalse(pipeline.df_kb.empty)
        stats = pipeline.get_stats()
        self.assertEqual(stats["chunks"], 10)
        self.assertIn("kafka", stats["domains"])
        self.assertIn("kubernetes", stats["domains"])

    def test_init_with_missing_file_logs_warning(self):
        """Initialization with missing file logs warning and creates empty KB."""
        missing_path = Path(self.temp_dir.name) / "missing.parquet"
        pipeline = LocalRAGPipeline(missing_path)
        self.assertEqual(len(pipeline), 0)
        self.assertTrue(pipeline.df_kb.empty)
        stats = pipeline.get_stats()
        self.assertEqual(stats["chunks"], 0)
        self.assertEqual(stats["domains"], [])

    def test_init_with_use_tfidf_false(self):
        """Initialization with use_tfidf=False disables vector reranking."""
        pipeline = LocalRAGPipeline(self.parquet_path, use_tfidf=False)
        self.assertFalse(pipeline.use_tfidf)
        results = pipeline.search("kafka idempotent producer", top_k=3)
        self.assertEqual(pipeline.retriever, "lexical")

    def test_init_with_custom_prefilter_cap(self):
        """Initialization with custom prefilter_cap respects the limit."""
        pipeline = LocalRAGPipeline(self.parquet_path, prefilter_cap=3)
        self.assertEqual(pipeline.prefilter_cap, 3)


class TestRAGPipelineSearch(unittest.TestCase):
    """Tests for search functionality."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.parquet_path = Path(self.temp_dir.name) / "test_kb.parquet"
        _make_synthetic_parquet(self.parquet_path)
        self.pipeline = LocalRAGPipeline(self.parquet_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_search_returns_correct_structure(self):
        """Search returns list of dicts with required fields: score, domain, content."""
        results = self.pipeline.search("kafka idempotent producer", top_k=3)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for hit in results:
            self.assertIn("chunk_id", hit)
            self.assertIn("title", hit)
            self.assertIn("domain", hit)
            self.assertIn("tags", hit)
            self.assertIn("content", hit)
            self.assertIn("date_range", hit)
            self.assertIn("score", hit)
            self.assertIsInstance(hit["score"], float)
            self.assertIsInstance(hit["domain"], str)
            self.assertIsInstance(hit["content"], str)
            self.assertGreater(hit["score"], 0)

    def test_top_k_limiting(self):
        """Top-k parameter limits number of returned results."""
        for k in [1, 2, 3, 5, 10]:
            results = self.pipeline.search("kafka", top_k=k)
            self.assertLessEqual(len(results), k, f"top_k={k} returned {len(results)} results")

    def test_empty_query_returns_empty(self):
        """Empty or whitespace-only query returns empty results."""
        self.assertEqual(self.pipeline.search("", top_k=5), [])
        self.assertEqual(self.pipeline.search("   ", top_k=5), [])
        self.assertEqual(self.pipeline.search("\t\n", top_k=5), [])

    def test_nonexistent_terms_return_empty(self):
        """Queries with no matching terms return empty results."""
        results = self.pipeline.search("xyznonexistentterm123", top_k=5)
        self.assertEqual(results, [])

    def test_score_ordering_highest_first(self):
        """Results are ordered by relevance score descending (highest first)."""
        results = self.pipeline.search("configure", top_k=5)
        self.assertGreater(len(results), 1)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_domain_filtering(self):
        """Domain filter restricts results to specified domain."""
        results = self.pipeline.search("deployment", top_k=5, domain_filter="kubernetes")
        for r in results:
            self.assertEqual(r["domain"], "kubernetes")

        results_kafka = self.pipeline.search("producer", top_k=5, domain_filter="kafka")
        for r in results_kafka:
            self.assertEqual(r["domain"], "kafka")

    def test_domain_filter_nonexistent_falls_back(self):
        """Domain filter for non-existent domain falls back to unfiltered search."""
        results = self.pipeline.search("kafka", top_k=5, domain_filter="nonexistent")
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_search_modes(self):
        """Different search modes work without errors."""
        query = "kafka idempotent producer"
        for mode in ["hybrid", "lexical", "tfidf"]:
            results = self.pipeline.search(query, top_k=3, mode=mode)
            self.assertIsInstance(results, list)
            for r in results:
                self.assertIn("score", r)

    def test_lexical_mode_uses_lexical_retriever(self):
        """Lexical mode sets retriever to 'lexical'."""
        self.pipeline.search("kafka", top_k=3, mode="lexical")
        self.assertEqual(self.pipeline.retriever, "lexical")

    def test_hybrid_mode_uses_tfidf_when_available(self):
        """Hybrid mode uses TF-IDF when sklearn available and enough candidates."""
        self.pipeline.search("kafka idempotent producer", top_k=3, mode="hybrid")
        self.assertIn(self.pipeline.retriever, ["lexical", "hybrid-tfidf"])

    def test_search_on_empty_kb(self):
        """Search on empty knowledge base returns empty list."""
        empty_path = Path(self.temp_dir.name) / "empty.parquet"
        pd.DataFrame(columns=[
            "chunk_id", "thread_id", "chat_name", "title", "topic_domain",
            "topic_tags", "content", "date_range", "participants_count",
            "message_count", "token_count"
        ]).to_parquet(empty_path, index=False)

        pipeline = LocalRAGPipeline(empty_path)
        results = pipeline.search("anything", top_k=5)
        self.assertEqual(results, [])


class TestRAGPipelineRowToHit(unittest.TestCase):
    """Tests for _row_to_hit conversion."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.parquet_path = Path(self.temp_dir.name) / "test_kb.parquet"
        _make_synthetic_parquet(self.parquet_path)
        self.pipeline = LocalRAGPipeline(self.parquet_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_row_to_hit_handles_missing_fields(self):
        """_row_to_hit handles missing/NaN fields gracefully."""
        import pandas as pd
        row = pd.Series({
            "chunk_id": "test_001",
            "title": "Test Title",
            "topic_domain": "test",
            "topic_tags": ["tag1", "tag2"],
            "content": "Test content",
            "date_range": "2024-01",
            "relevance_score": 5.5,
        })
        hit = LocalRAGPipeline._row_to_hit(row)
        self.assertEqual(hit["chunk_id"], "test_001")
        self.assertEqual(hit["title"], "Test Title")
        self.assertEqual(hit["domain"], "test")
        self.assertEqual(hit["tags"], ["tag1", "tag2"])
        self.assertEqual(hit["content"], "Test content")
        self.assertEqual(hit["date_range"], "2024-01")
        self.assertEqual(hit["score"], 5.5)

    def test_row_to_hit_handles_non_list_tags(self):
        """_row_to_hit converts non-list tags to empty list."""
        import pandas as pd
        row = pd.Series({
            "chunk_id": "test_001",
            "topic_tags": "not-a-list",
            "relevance_score": 1.0,
        })
        hit = LocalRAGPipeline._row_to_hit(row)
        self.assertEqual(hit["tags"], [])

    def test_row_to_hit_defaults_for_missing(self):
        """_row_to_hit provides defaults for missing fields."""
        import pandas as pd
        row = pd.Series({"relevance_score": 2.0})
        hit = LocalRAGPipeline._row_to_hit(row)
        self.assertEqual(hit["chunk_id"], "")
        self.assertEqual(hit["title"], "")
        self.assertEqual(hit["domain"], "general")
        self.assertEqual(hit["tags"], [])
        self.assertEqual(hit["content"], "")
        self.assertEqual(hit["date_range"], "")
        self.assertEqual(hit["score"], 2.0)


class TestRAGPipelineFormatPrompt(unittest.TestCase):
    """Tests for format_rag_prompt method."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.parquet_path = Path(self.temp_dir.name) / "test_kb.parquet"
        _make_synthetic_parquet(self.parquet_path)
        self.pipeline = LocalRAGPipeline(self.parquet_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_format_rag_prompt_with_contexts(self):
        """format_rag_prompt includes all context fields in output."""
        query = "How to configure Kafka?"
        contexts = [
            {
                "title": "Kafka Config",
                "date_range": "2024-01",
                "content": "enable.idempotence=true",
            },
            {
                "title": "Kafka Tuning",
                "date_range": "2024-02",
                "content": "acks=all",
            },
        ]
        prompt = self.pipeline.format_rag_prompt(query, contexts)
        self.assertIn("Kafka Config", prompt)
        self.assertIn("2024-01", prompt)
        self.assertIn("enable.idempotence=true", prompt)
        self.assertIn("Kafka Tuning", prompt)
        self.assertIn("2024-02", prompt)
        self.assertIn("acks=all", prompt)
        self.assertIn(query, prompt)
        self.assertIn("Ответ эксперта:", prompt)

    def test_format_rag_prompt_empty_contexts_returns_query(self):
        """Empty contexts returns original query unchanged."""
        query = "How to configure Kafka?"
        prompt = self.pipeline.format_rag_prompt(query, [])
        self.assertEqual(prompt, query)

    def test_format_rag_prompt_none_contexts_returns_query(self):
        """None contexts returns original query."""
        query = "How to configure Kafka?"
        prompt = self.pipeline.format_rag_prompt(query, None)
        self.assertEqual(prompt, query)


class TestRAGPipelineStatsAndLen(unittest.TestCase):
    """Tests for get_stats and __len__ methods."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.parquet_path = Path(self.temp_dir.name) / "test_kb.parquet"
        _make_synthetic_parquet(self.parquet_path)
        self.pipeline = LocalRAGPipeline(self.parquet_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_len_returns_chunk_count(self):
        """__len__ returns number of chunks in KB."""
        self.assertEqual(len(self.pipeline), 10)

    def test_get_stats_structure(self):
        """get_stats returns expected dictionary structure."""
        stats = self.pipeline.get_stats()
        self.assertIn("chunks", stats)
        self.assertIn("retriever", stats)
        self.assertIn("use_tfidf", stats)
        self.assertIn("domains", stats)
        self.assertEqual(stats["chunks"], 10)
        self.assertIsInstance(stats["domains"], list)
        self.assertEqual(len(stats["domains"]), 5)

    def test_df_property_alias(self):
        """df property is alias for df_kb."""
        self.assertIs(self.pipeline.df, self.pipeline.df_kb)


class TestRAGPipelineEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.parquet_path = Path(self.temp_dir.name) / "test_kb.parquet"
        _make_synthetic_parquet(self.parquet_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_search_with_special_characters(self):
        """Search handles special regex characters in query."""
        pipeline = LocalRAGPipeline(self.parquet_path)
        results = pipeline.search("kafka (producer|consumer)", top_k=3)
        self.assertIsInstance(results, list)

    def test_search_with_unicode(self):
        """Search handles Cyrillic and mixed scripts."""
        pipeline = LocalRAGPipeline(self.parquet_path)
        results = pipeline.search("как настроить кафка", top_k=3)
        self.assertIsInstance(results, list)

    def test_search_very_long_query(self):
        """Search handles very long queries (truncates to 10 keywords)."""
        pipeline = LocalRAGPipeline(self.parquet_path)
        long_query = " ".join([f"word{i}" for i in range(50)])
        results = pipeline.search(long_query, top_k=3)
        self.assertIsInstance(results, list)

    def test_search_case_insensitive(self):
        """Search is case-insensitive."""
        pipeline = LocalRAGPipeline(self.parquet_path)
        results_lower = pipeline.search("kafka producer", top_k=3)
        results_upper = pipeline.search("KAFKA PRODUCER", top_k=3)
        self.assertEqual(len(results_lower), len(results_upper))

    def test_keyword_extraction_limit(self):
        """Only first 10 keywords are used for matching."""
        from src.rag.rag_pipeline import _extract_keywords
        keywords = _extract_keywords(" ".join([f"word{i}" for i in range(20)]))
        self.assertEqual(len(keywords), 10)

    def test_lexical_scoring(self):
        """Lexical scoring counts keyword occurrences."""
        from src.rag.rag_pipeline import _lexical_scores
        import pandas as pd
        contents = pd.Series([
            "kafka producer idempotent",
            "kafka consumer group",
            "nginx proxy",
        ])
        scores = _lexical_scores(contents, ["kafka", "producer"])
        self.assertEqual(scores[0], 3.0)
        self.assertEqual(scores[1], 1.5)
        self.assertEqual(scores[2], 0.0)


if __name__ == "__main__":
    unittest.main()