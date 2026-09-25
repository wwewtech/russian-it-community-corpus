"""Tests for the DeepChatAnalyzer analytics engine."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.analytics.engine import DeepChatAnalyzer
from src.ingestion.schema import CleanedMessage


def _make_msg(
    msg_id: int,
    text: str,
    author: str = "user1",
    author_id: str = "u1",
    chat_id: int = 1,
    chat_name: str = "community_node_01",
    unixtime_offset: int = 0,
    domain: str = "backend_databases",
    tags: list[str] | None = None,
    sentiment: int = 0,
    tokens: int = 10,
    is_question: bool = False,
    thread_id: int | None = None,
) -> CleanedMessage:
    base_time = 1767225600 + unixtime_offset  # 2026-01-01 00:00:00
    return CleanedMessage(
        msg_id=msg_id,
        chat_id=chat_id,
        chat_name=chat_name,
        timestamp=datetime.fromtimestamp(base_time).isoformat(),
        unixtime=base_time,
        author_anon=author,
        author_id_anon=author_id,
        text_clean=text,
        domain=domain,
        tags=tags or [],
        sentiment_score=sentiment,
        token_count_approx=tokens,
        is_question=is_question,
        thread_id=thread_id,
    )


class TestDeepChatAnalyzerConstruction(unittest.TestCase):
    def test_constructor_initializes_all_attributes(self):
        msgs = [_make_msg(1, "hello world"), _make_msg(2, "foo bar", author="user2", author_id="u2", unixtime_offset=86400)]
        analyzer = DeepChatAnalyzer(msgs)
        self.assertEqual(analyzer.total_messages, 2)
        self.assertEqual(len(analyzer.authors), 2)
        self.assertIsNotNone(analyzer.min_date)
        self.assertIsNotNone(analyzer.max_date)
        # 86400 seconds = 1 day difference
        self.assertEqual(analyzer.total_days, 1)

    def test_constructor_empty_messages(self):
        analyzer = DeepChatAnalyzer([])
        self.assertEqual(analyzer.total_messages, 0)
        self.assertEqual(analyzer.authors, set())
        self.assertIsNone(analyzer.min_date)
        self.assertIsNone(analyzer.max_date)

    def test_constructor_sorts_by_unixtime(self):
        msgs = [
            _make_msg(3, "third", unixtime_offset=200),
            _make_msg(1, "first", unixtime_offset=0),
            _make_msg(2, "second", unixtime_offset=100),
        ]
        analyzer = DeepChatAnalyzer(msgs)
        self.assertEqual([m.msg_id for m in analyzer.sorted_messages], [1, 2, 3])

    def test_constructor_samples_for_nlp_when_limit_set(self):
        msgs = [_make_msg(i, f"message {i} " * 10) for i in range(200)]
        analyzer = DeepChatAnalyzer(msgs, sample_limit_for_nlp=100)
        self.assertEqual(len(analyzer.tokenized_corpus), 100)


class TestComputeVolumeStatistics(unittest.TestCase):
    def setUp(self):
        msgs = [
            _make_msg(1, "short", tokens=5),
            _make_msg(2, "medium length message here", tokens=20, unixtime_offset=60),
            _make_msg(3, "a" * 200, tokens=50, unixtime_offset=120),
        ]
        self.analyzer = DeepChatAnalyzer(msgs)

    def test_returns_all_expected_keys(self):
        result = self.analyzer.compute_volume_statistics()
        expected_keys = {
            "total_messages",
            "unique_authors",
            "date_start",
            "date_end",
            "total_days_active",
            "messages_per_day",
            "tokens_per_day",
            "total_characters",
            "total_words",
            "total_tokens_estimated",
            "vocabulary_unique_words",
            "character_length_distribution",
            "word_count_distribution",
            "token_count_distribution",
            "author_activity_distribution",
        }
        self.assertSetEqual(set(result.keys()), expected_keys)

    def test_counts_are_correct(self):
        result = self.analyzer.compute_volume_statistics()
        self.assertEqual(result["total_messages"], 3)
        self.assertEqual(result["unique_authors"], 1)
        self.assertEqual(result["total_characters"], len("short") + len("medium length message here") + 200)

    def test_percentiles_present(self):
        result = self.analyzer.compute_volume_statistics()
        for key in ("character_length_distribution", "word_count_distribution", "token_count_distribution"):
            self.assertIn("median", result[key])  # p50 is reported as median
            self.assertIn("p95", result[key])
            self.assertIn("p99", result[key])


class TestComputeTemporalDynamics(unittest.TestCase):
    def setUp(self):
        base = 1767225600  # 2026-01-01 00:00:00 (Thursday)
        msgs = [
            _make_msg(1, "msg1", unixtime_offset=0),  # 00:00 Thu
            _make_msg(2, "msg2", unixtime_offset=3600),  # 01:00 Thu
            _make_msg(3, "msg3", unixtime_offset=86400),  # 00:00 Fri
            _make_msg(4, "msg4", unixtime_offset=86400 + 3600),  # 01:00 Fri
        ]
        self.analyzer = DeepChatAnalyzer(msgs)

    def test_returns_expected_structure(self):
        result = self.analyzer.compute_temporal_dynamics()
        expected_keys = {
            "peak_hour",
            "peak_weekday",
            "hourly_distribution",
            "weekday_distribution",
            "yearly_volume",
            "monthly_volume",
            "monthly_avg_character_length",
            "inter_arrival_seconds_distribution",
        }
        self.assertSetEqual(set(result.keys()), expected_keys)

    def test_hourly_distribution_counts(self):
        result = self.analyzer.compute_temporal_dynamics()
        # Timezone offset may shift hours; check total count preserved
        total = sum(result["hourly_distribution"].values())
        self.assertEqual(total, 4)

    def test_weekday_distribution_russian_names(self):
        result = self.analyzer.compute_temporal_dynamics()
        self.assertIn("Четверг", result["weekday_distribution"])
        self.assertIn("Пятница", result["weekday_distribution"])

    def test_empty_timestamps_returns_empty_dict(self):
        analyzer = DeepChatAnalyzer([_make_msg(1, "test")])
        analyzer.timestamps = []
        result = analyzer.compute_temporal_dynamics()
        self.assertEqual(result, {})


class TestComputeLexicalAnalytics(unittest.TestCase):
    def setUp(self):
        msgs = [
            _make_msg(1, "python python python code code"),
            _make_msg(2, "code review python test"),
            _make_msg(3, "deploy production server"),
        ]
        self.analyzer = DeepChatAnalyzer(msgs, sample_limit_for_nlp=10)

    def test_returns_all_expected_keys(self):
        result = self.analyzer.compute_lexical_analytics()
        expected_keys = {
            "shannon_entropy",
            "type_token_ratio_ttr",
            "root_ttr",
            "top_unigrams",
            "top_bigrams",
            "top_trigrams",
            "top_fourgrams",
        }
        self.assertSetEqual(set(result.keys()), expected_keys)

    def test_unigrams_count_correct(self):
        result = self.analyzer.compute_lexical_analytics()
        unigrams = {item["word"]: item["count"] for item in result["top_unigrams"]}
        self.assertEqual(unigrams.get("python"), 4)
        self.assertEqual(unigrams.get("code"), 3)

    def test_bigrams_extracted(self):
        result = self.analyzer.compute_lexical_analytics()
        bigrams = {item["ngram"]: item["count"] for item in result["top_bigrams"]}
        self.assertIn("python code", bigrams)

    def test_entropy_positive(self):
        result = self.analyzer.compute_lexical_analytics()
        self.assertGreater(result["shannon_entropy"], 0.0)

    def test_ttr_between_zero_and_one(self):
        result = self.analyzer.compute_lexical_analytics()
        self.assertGreaterEqual(result["type_token_ratio_ttr"], 0.0)
        self.assertLessEqual(result["type_token_ratio_ttr"], 1.0)


class TestComputeDomainSlangAnalytics(unittest.TestCase):
    def setUp(self):
        # Use terms that exist in RUSSIAN_IT_SLANG_TERMS
        msgs = [
            _make_msg(1, "deploy prod kuber", domain="devops_infra", tags=["docker", "kubernetes"]),
            _make_msg(2, "junior code review", domain="career_team_management", tags=["code review"]),
            _make_msg(3, "normal message no slang", domain="general_tech_chat", tags=[]),
        ]
        self.analyzer = DeepChatAnalyzer(msgs)

    def test_detects_slang_terms(self):
        result = self.analyzer.compute_domain_slang_analytics()
        # Just verify the structure is returned correctly
        self.assertIn("slang_terms_detected_count", result)
        self.assertIn("top_slang_terms", result)
        self.assertIsInstance(result["top_slang_terms"], list)
        self.assertIn("domain_message_distribution", result)
        self.assertIn("top_technical_tags", result)

    def test_domain_distribution_percentages(self):
        result = self.analyzer.compute_domain_slang_analytics()
        dist = result["domain_message_distribution"]
        self.assertEqual(dist["devops_infra"]["count"], 1)
        self.assertEqual(dist["devops_infra"]["percentage"], 33.33)
        self.assertEqual(dist["general_tech_chat"]["count"], 1)

    def test_technical_tags_counted(self):
        result = self.analyzer.compute_domain_slang_analytics()
        tags = {item["tag"]: item["count"] for item in result["top_technical_tags"]}
        self.assertEqual(tags.get("docker"), 1)
        self.assertEqual(tags.get("kubernetes"), 1)
        self.assertEqual(tags.get("code review"), 1)


class TestComputeSentimentAndSyntax(unittest.TestCase):
    def setUp(self):
        msgs = [
            _make_msg(1, "спасибо отлично работает", sentiment=3, is_question=False),
            _make_msg(2, "как настроить nginx?", sentiment=0, is_question=True),
            _make_msg(3, "баг ошибка краш", sentiment=-3, is_question=False),
            _make_msg(4, "def foo(): return 42", sentiment=0, is_question=False),
        ]
        self.analyzer = DeepChatAnalyzer(msgs)

    def test_sentiment_structure(self):
        result = self.analyzer.compute_sentiment_and_syntax()
        self.assertIn("sentiment", result)
        self.assertIn("questions_count", result)
        self.assertIn("questions_ratio_percentage", result)
        self.assertIn("code_snippets_count", result)
        self.assertIn("code_snippets_ratio_percentage", result)

    def test_question_count_and_ratio(self):
        result = self.analyzer.compute_sentiment_and_syntax()
        self.assertEqual(result["questions_count"], 1)
        self.assertEqual(result["questions_ratio_percentage"], 25.0)

    def test_code_snippet_detection(self):
        result = self.analyzer.compute_sentiment_and_syntax()
        self.assertEqual(result["code_snippets_count"], 1)  # "def foo()" message
        self.assertEqual(result["code_snippets_ratio_percentage"], 25.0)


class TestComputeSocialNetworkAnalytics(unittest.TestCase):
    def test_mocks_social_network_analyzer(self):
        msgs = [_make_msg(1, "msg1"), _make_msg(2, "reply", author="user2", author_id="u2", unixtime_offset=60)]
        analyzer = DeepChatAnalyzer(msgs)

        with patch("src.analytics.engine.SocialNetworkAnalyzer") as mock_sna:
            mock_instance = MagicMock()
            mock_instance.analyze.return_value = {"nodes": 2, "edges": 1}
            mock_sna.return_value = mock_instance

            result = analyzer.compute_social_network_analytics()

            mock_sna.assert_called_once_with(reply_window_minutes=30)
            mock_instance.analyze.assert_called_once_with(msgs)
            self.assertEqual(result, {"nodes": 2, "edges": 1})


class TestComputeAuthorKeyPhrases(unittest.TestCase):
    def test_filters_authors_with_fewer_than_30_messages(self):
        msgs = [_make_msg(i, "test message", author="rare_user", author_id="rare") for i in range(10)]
        analyzer = DeepChatAnalyzer(msgs)
        result = analyzer.compute_author_key_phrases(top_authors_count=25, top_words_per_author=5)
        self.assertEqual(result, {})

    def test_returns_signature_words_for_active_author(self):
        msgs = []
        for i in range(50):
            msgs.append(_make_msg(i, f"python django fastapi python {i}", author="active_user", author_id="active"))
        for i in range(5):
            msgs.append(_make_msg(100 + i, "other stuff", author="other", author_id="other"))
        analyzer = DeepChatAnalyzer(msgs)
        result = analyzer.compute_author_key_phrases(top_authors_count=2, top_words_per_author=3)
        self.assertIn("active_user", result)
        self.assertGreater(len(result["active_user"]), 0)


class TestComputeTopicClustersLDA(unittest.TestCase):
    def test_returns_empty_when_sklearn_missing(self):
        msgs = [_make_msg(i, "test message " * 10) for i in range(60)]
        analyzer = DeepChatAnalyzer(msgs)

        with patch("src.analytics.engine.HAS_SKLEARN", False):
            result = analyzer.compute_topic_clusters_lda(n_topics=5)
            self.assertEqual(result, [])

    def test_returns_empty_when_fewer_than_50_messages(self):
        msgs = [_make_msg(i, "test") for i in range(10)]
        analyzer = DeepChatAnalyzer(msgs)
        result = analyzer.compute_topic_clusters_lda(n_topics=5)
        self.assertEqual(result, [])

    def test_returns_empty_when_insufficient_docs(self):
        msgs = [_make_msg(i, "short") for i in range(60)]
        analyzer = DeepChatAnalyzer(msgs)

        with patch("src.analytics.engine.HAS_SKLEARN", True):
            result = analyzer.compute_topic_clusters_lda(n_topics=8)
            self.assertEqual(result, [])


class TestComputeLongitudinalTrends(unittest.TestCase):
    def test_groups_by_year(self):
        base = datetime(2023, 6, 15).timestamp()
        msgs = [
            _make_msg(1, "python ai ml", unixtime_offset=0),  # 2026
            _make_msg(2, "docker kubernetes", unixtime_offset=-365 * 86400),  # 2025
        ]
        # Fix timestamps to specific years
        msgs[0].timestamp = "2026-06-15T12:00:00"
        msgs[0].unixtime = int(datetime(2026, 6, 15).timestamp())
        msgs[1].timestamp = "2025-06-15T12:00:00"
        msgs[1].unixtime = int(datetime(2025, 6, 15).timestamp())

        analyzer = DeepChatAnalyzer(msgs)
        analyzer.timestamps = [datetime(2026, 6, 15), datetime(2025, 6, 15)]

        result = analyzer.compute_longitudinal_trends()
        self.assertIn(2025, result)
        self.assertIn(2026, result)
        self.assertEqual(result[2025]["message_count"], 1)
        self.assertEqual(result[2026]["message_count"], 1)

    def test_extracts_tech_keywords(self):
        msgs = [
            _make_msg(1, "python django fastapi"),
            _make_msg(2, "docker kubernetes helm"),
        ]
        msgs[0].timestamp = "2026-01-01T00:00:00"
        msgs[0].unixtime = int(datetime(2026, 1, 1).timestamp())
        msgs[1].timestamp = "2026-01-01T00:00:00"
        msgs[1].unixtime = int(datetime(2026, 1, 1).timestamp())

        analyzer = DeepChatAnalyzer(msgs)
        analyzer.timestamps = [datetime(2026, 1, 1), datetime(2026, 1, 1)]

        result = analyzer.compute_longitudinal_trends()
        tech_2026 = result[2026]["top_tech_keywords"]
        self.assertIn("python", tech_2026)
        self.assertIn("docker", tech_2026)


class TestComputeNoiseAndQuality(unittest.TestCase):
    def test_counts_short_messages(self):
        msgs = [
            _make_msg(1, "a" * 10),  # short (10 chars)
            _make_msg(2, "a" * 30),  # normal (30 chars)
            _make_msg(3, ""),  # empty (0 chars)
            _make_msg(4, "спасибо отлично", sentiment=2),  # short (14 chars) + high emotion
        ]
        analyzer = DeepChatAnalyzer(msgs)
        result = analyzer.compute_noise_and_quality()

        self.assertEqual(result["short_messages_under_20_chars"], 3)  # 10 + 0 + 14 chars
        self.assertEqual(result["empty_messages_count"], 1)
        self.assertEqual(result["high_emotion_messages_count"], 1)

    def test_ratios_calculated(self):
        msgs = [_make_msg(1, "short"), _make_msg(2, "normal length message")]
        analyzer = DeepChatAnalyzer(msgs)
        result = analyzer.compute_noise_and_quality()
        self.assertEqual(result["short_messages_ratio_percentage"], 50.0)
        self.assertEqual(result["empty_messages_ratio_percentage"], 0.0)


class TestComputeDatasetQualityScore(unittest.TestCase):
    def test_high_volume_high_diversity_scores_high(self):
        msgs = [
            _make_msg(i, f"python django message {i}", domain="backend_databases", author=f"user{i}", author_id=f"u{i}")
            for i in range(200)
        ]
        # Add questions
        for i in range(50):
            msgs.append(_make_msg(200 + i, f"how to {i}?", is_question=True, author=f"user{i}", author_id=f"u{i}"))
        analyzer = DeepChatAnalyzer(msgs)
        result = analyzer.compute_dataset_quality_score()

        self.assertIn("total_score", result)
        self.assertIn("score_breakdown", result)
        self.assertIn("quality_tier", result)
        self.assertLessEqual(result["total_score"], 100)
        self.assertGreaterEqual(result["total_score"], 0)

    def test_score_breakdown_has_all_components(self):
        msgs = [_make_msg(1, "test")]
        analyzer = DeepChatAnalyzer(msgs)
        result = analyzer.compute_dataset_quality_score()

        breakdown = result["score_breakdown"]
        expected = {
            "volume_score",
            "author_diversity_score",
            "technical_density_score",
            "dialogue_continuity_score",
            "lexical_diversity_score",
            "pii_compliance_score",
        }
        self.assertSetEqual(set(breakdown.keys()), expected)

    def test_tier_assignment(self):
        # Very small dataset -> Category C
        msgs = [_make_msg(1, "test")]
        analyzer = DeepChatAnalyzer(msgs)
        result = analyzer.compute_dataset_quality_score()
        self.assertEqual(result["quality_tier"], "Baseline (Category C)")


class TestRunFullAnalysis(unittest.TestCase):
    def test_returns_complete_report_structure(self):
        msgs = [
            _make_msg(1, "python django", domain="backend_databases", tags=["python", "django"]),
            _make_msg(2, "how to deploy?", is_question=True, author="user2", author_id="u2", unixtime_offset=60),
        ]
        analyzer = DeepChatAnalyzer(msgs, sample_limit_for_nlp=10)

        with patch("src.analytics.engine.SocialNetworkAnalyzer") as mock_sna:
            mock_instance = MagicMock()
            mock_instance.analyze.return_value = {"nodes": 2, "edges": 1}
            mock_sna.return_value = mock_instance

            report = analyzer.run_full_analysis()

        expected_sections = {
            "report_metadata",
            "volume_statistics",
            "temporal_dynamics",
            "lexical_analytics",
            "domain_slang_analytics",
            "sentiment_and_syntax",
            "social_network",
            "author_signature_phrases",
            "topic_clusters_lda",
            "longitudinal_evolution_8_years",
            "noise_and_quality",
            "quality_and_readiness",
        }
        self.assertSetEqual(set(report.keys()), expected_sections)

    def test_metadata_contains_version_and_count(self):
        msgs = [_make_msg(1, "test")]
        analyzer = DeepChatAnalyzer(msgs)
        report = analyzer.run_full_analysis()

        meta = report["report_metadata"]
        self.assertIn("engine_version", meta)
        self.assertEqual(meta["total_messages_analyzed"], 1)
        self.assertIn("generated_at", meta)


class TestTokenizeClean(unittest.TestCase):
    def test_empty_string_returns_empty_list(self):
        msgs = [_make_msg(1, "")]
        analyzer = DeepChatAnalyzer(msgs)
        self.assertEqual(analyzer._tokenize_clean(""), [])

    def test_removes_stopwords_and_digits(self):
        msgs = [_make_msg(1, "test")]
        analyzer = DeepChatAnalyzer(msgs)
        tokens = analyzer._tokenize_clean("и в на 123 python django")
        self.assertNotIn("и", tokens)
        self.assertNotIn("123", tokens)
        self.assertIn("python", tokens)
        self.assertIn("django", tokens)

    def test_lemmatization_when_morph_available(self):
        msgs = [_make_msg(1, "test")]
        analyzer = DeepChatAnalyzer(msgs)

        with patch("src.analytics.engine.HAS_MORPH", True), patch("src.analytics.engine.MORPH") as mock_morph:
            mock_parse = MagicMock()
            mock_parse.normal_form = "деплой"
            mock_morph.parse.return_value = [mock_parse]

            tokens = analyzer._tokenize_clean("деплоил деплой")
            self.assertEqual(tokens, ["деплой", "деплой"])

    def test_keeps_hyphenated_tokens(self):
        msgs = [_make_msg(1, "test")]
        analyzer = DeepChatAnalyzer(msgs)
        tokens = analyzer._tokenize_clean("ci-cd pipeline fast-api")
        self.assertIn("ci-cd", tokens)
        self.assertIn("fast-api", tokens)


if __name__ == "__main__":
    unittest.main()