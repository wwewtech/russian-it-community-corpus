"""Coverage boost wave 3: engine scale branches, LDA failure, thread builder bursts, extractor branches, drift verdicts."""

from __future__ import annotations

import unittest

import pandas as pd

from tests.test_coverage_boost import _msg


class TestEngineScaleBranches(unittest.TestCase):
    def test_moderate_tier_large_corpus(self) -> None:
        from src.analytics.engine import DeepChatAnalyzer

        pool = [f"w{i:04d}" for i in range(2500)]
        msgs = []
        for i in range(500):
            words = " ".join(pool[(i * 5 + j) % len(pool)] for j in range(8))
            msgs.append(
                _msg(
                    i,
                    f"{words} kafka",
                    domain="backend_databases" if i % 5 < 2 else "general_tech_chat",
                    is_question=(i % 5 == 0),
                    author_anon=f"p{i:04d}",
                    author_id_anon=f"q{i}",
                )
            )
        a = DeepChatAnalyzer(msgs)
        q = a.compute_dataset_quality_score()
        self.assertEqual(
            q["score_breakdown"],
            {
                "volume_score": 5,
                "author_diversity_score": 15,
                "technical_density_score": 20,
                "dialogue_continuity_score": 15,
                "lexical_diversity_score": 10,
                "pii_compliance_score": 10,
            },
        )
        self.assertEqual(q["total_score"], 75)
        self.assertEqual(q["quality_tier"], "Moderate Quality (Category B)")

    def test_low_tech_no_questions(self) -> None:
        from src.analytics.engine import DeepChatAnalyzer

        msgs = [_msg(i, f"просто поболтать о погоде номер {i} сегодня", domain="general_tech_chat") for i in range(8)]
        q = DeepChatAnalyzer(msgs).compute_dataset_quality_score()
        self.assertEqual(q["score_breakdown"]["technical_density_score"], 10)
        self.assertEqual(q["score_breakdown"]["dialogue_continuity_score"], 8)
        self.assertEqual(q["quality_tier"], "Baseline (Category C)")

    def test_lda_failure_returns_empty(self) -> None:
        from src.analytics.engine import DeepChatAnalyzer

        msgs = []
        k = 0
        for author in range(8):
            for _m in range(8):
                msgs.append(_msg(k, "kafka кластер настройка", author_anon=f"u{author}", author_id_anon=f"h{author}"))
                k += 1
        self.assertEqual(DeepChatAnalyzer(msgs).compute_topic_clusters_lda(n_topics=8), [])

    def test_lda_early_return_few_docs(self) -> None:
        from src.analytics.engine import DeepChatAnalyzer

        pool = [f"doc{i:02d}" for i in range(20)]
        msgs = []
        k = 0
        for author in range(5):
            for _m in range(12):
                words = " ".join(pool[(author * 3 + j) % len(pool)] for j in range(10))
                msgs.append(_msg(k, f"{words} настройка", author_anon=f"e{author}", author_id_anon=f"f{author}"))
                k += 1
        self.assertEqual(DeepChatAnalyzer(msgs).compute_topic_clusters_lda(n_topics=8), [])

    def test_tokenize_long_text_skips_morph(self) -> None:
        from src.analytics.engine import DeepChatAnalyzer

        a = DeepChatAnalyzer([_msg(1, "тест")])
        tokens = a._tokenize_clean(" ".join(f"слово{i}" for i in range(60)))
        self.assertGreater(len(tokens), 50)

    def test_entropy_mid_branch(self) -> None:
        from src.analytics.engine import DeepChatAnalyzer

        pool = [f"e{i:04d}" for i in range(300)]
        msgs = [
            _msg(
                i,
                " ".join(pool[(i * 3 + j) % len(pool)] for j in range(8)) + " kafka",
                author_anon=f"z{i:03d}",
                author_id_anon=f"y{i}",
            )
            for i in range(150)
        ]
        q = DeepChatAnalyzer(msgs).compute_dataset_quality_score()
        self.assertEqual(q["score_breakdown"]["lexical_diversity_score"], 7)


class TestThreadBuilderBranches(unittest.TestCase):
    def test_duplicate_keys_hit_visited_continue(self) -> None:
        from src.graph.thread_builder import ThreadDAGBuilder

        p = _msg(1, "корневой вопрос про kafka настройку кластера")
        d1 = _msg(2, "ответ первый подробности конфигурации", reply_to_id=1)
        d2 = _msg(2, "ответ второй дубликат идентификатора", reply_to_id=1)
        _, threads = ThreadDAGBuilder().build_threads([p, d1, d2])
        self.assertGreaterEqual(len(threads), 1)

    def test_burst_flush_at_visited_tree_node(self) -> None:
        from src.graph.thread_builder import ThreadDAGBuilder

        t = 1704000000
        o1 = _msg(1, "первое сообщение в чате про деплой", unixtime=t)
        o2 = _msg(2, "второе сообщение следом про докер", unixtime=t + 60)
        r = _msg(3, "корневой вопрос про kubernetes ingress", unixtime=t + 10000)
        c = _msg(4, "ответ про ingress controller nginx", unixtime=t + 10060, reply_to_id=3)
        _, threads = ThreadDAGBuilder().build_threads([o1, o2, r, c])
        sizes = sorted(len(v) for v in threads.values())
        self.assertIn(2, sizes)

    def test_burst_flush_at_time_gap(self) -> None:
        from src.graph.thread_builder import ThreadDAGBuilder

        t = 1704000000
        o1 = _msg(1, "первое сообщение в чате про деплой", unixtime=t)
        o2 = _msg(2, "второе сообщение следом про докер", unixtime=t + 60)
        o3 = _msg(3, "третье сообщение через долгое время", unixtime=t + 100000)
        _, threads = ThreadDAGBuilder().build_threads([o1, o2, o3])
        self.assertEqual(sorted(len(v) for v in threads.values()), [1, 2])


class TestExtractorBranches(unittest.TestCase):
    def test_empty_text_not_contaminated(self) -> None:
        from src.graph.conversation_extractor import is_llm_contaminated

        self.assertFalse(is_llm_contaminated(""))
        self.assertTrue(is_llm_contaminated("как языковая модель я не могу помочь"))

    def test_single_message_thread_skipped(self) -> None:
        from src.graph.conversation_extractor import ConversationExtractor

        self.assertEqual(ConversationExtractor().extract_sft_dialogues({1: [_msg(1, "одиночка")]}), [])

    def test_single_author_thread_skipped(self) -> None:
        from src.graph.conversation_extractor import ConversationExtractor

        msgs = [
            _msg(1, "первое длинное сообщение про kafka", author_anon="solo", author_id_anon="s1"),
            _msg(2, "второе длинное сообщение про kafka", author_anon="solo", author_id_anon="s1"),
        ]
        self.assertEqual(ConversationExtractor().extract_sft_dialogues({1: msgs}), [])

    def test_short_texts_merge_below_two(self) -> None:
        from src.graph.conversation_extractor import ConversationExtractor

        msgs = [
            _msg(1, "длинный вопрос про настройку kafka кластера"),
            _msg(2, "ок", author_anon="user_02", author_id_anon="h2"),
        ]
        self.assertEqual(ConversationExtractor().extract_sft_dialogues({1: msgs}), [])

    def test_same_author_merge_and_trailing_pop(self) -> None:
        from src.graph.conversation_extractor import ConversationExtractor

        msgs = [
            _msg(1, "вопрос про настройку kafka кластера подробно", author_anon="user_01", author_id_anon="hash_1"),
            _msg(2, "дополнение к вопросу про репликацию", author_anon="user_01", author_id_anon="hash_1"),
            _msg(3, "ответ подробный про idempotent producer", author_anon="user_02", author_id_anon="h2"),
            _msg(4, "еще вопрос вдогонку про retention", author_anon="user_01", author_id_anon="hash_1"),
        ]
        dialogues = ConversationExtractor().extract_sft_dialogues({1: msgs})
        self.assertEqual(len(dialogues), 1)
        self.assertEqual(dialogues[0].messages[-1].role, "assistant")

    def test_three_authors_flip_and_mega_thread_cap(self) -> None:
        from src.graph.conversation_extractor import ConversationExtractor

        msgs = [
            _msg(1, "вопрос про настройку kafka подробно"),
            _msg(2, "ответ от второго участника развернуто", author_anon="user_02", author_id_anon="h2"),
            _msg(3, "комментарий третьего участника по делу", author_anon="user_03", author_id_anon="h3"),
        ]
        dialogues = ConversationExtractor().extract_sft_dialogues({1: msgs})
        self.assertEqual(len(dialogues), 1)
        big = []
        for i in range(40):
            author = "user_01" if i % 2 == 0 else "user_02"
            aid = "hash_1" if i % 2 == 0 else "h2"
            big.append(
                _msg(
                    i,
                    f"реплика номер {i} про kafka настройку кластера подробно",
                    author_anon=author,
                    author_id_anon=aid,
                )
            )
        capped = ConversationExtractor().extract_sft_dialogues({9: big})
        self.assertEqual(len(capped), 1)
        self.assertLessEqual(capped[0].turn_count, 32)

    def test_dpo_skips(self) -> None:
        from src.graph.conversation_extractor import ConversationExtractor

        ext = ConversationExtractor()
        self.assertEqual(
            ext.extract_dpo_pairs(
                {1: [_msg(1, "коротко"), _msg(2, "тоже коротко", author_anon="user_02", author_id_anon="h2")]}
            ),
            [],
        )
        same_author = [
            _msg(1, "длинный вопрос про настройку kafka кластера подробно", author_anon="root", author_id_anon="r1"),
            _msg(2, "ответ автора самому себе", author_anon="root", author_id_anon="r1"),
            _msg(3, "еще ответ себе же", author_anon="root", author_id_anon="r1"),
        ]
        self.assertEqual(ext.extract_dpo_pairs({1: same_author}), [])

    def test_rag_empty_thread_skipped(self) -> None:
        from src.graph.conversation_extractor import ConversationExtractor

        self.assertEqual(ConversationExtractor().extract_rag_chunks({1: []}), [])


class TestDriftVerdicts(unittest.TestCase):
    def test_zero_buckets_psi(self) -> None:
        from src.monitoring.drift import DatasetDriftMonitor

        ref = pd.DataFrame({"text_clean": ["aaa"] * 10, "domain": ["backend"] * 10})
        self.assertEqual(DatasetDriftMonitor(ref, ref, length_buckets=0).compute_length_psi(), 0.0)

    def test_js_moderate(self) -> None:
        from src.monitoring.drift import DatasetDriftMonitor

        ref = pd.DataFrame(
            {
                "text_clean": ["kafka broker topic partition replica setup config file service port"] * 60,
                "domain": ["backend"] * 60,
            }
        )
        cur = pd.DataFrame(
            {
                "text_clean": ["kafka broker topic partition replica setup config file service port extra option"] * 60,
                "domain": ["backend"] * 50 + ["devops"] * 10,
            }
        )
        rep = DatasetDriftMonitor(ref, cur).run()
        self.assertEqual(rep["metrics"]["domain_distribution"]["verdict"], "moderate_drift")

    def test_vocab_moderate(self) -> None:
        from src.monitoring.drift import DatasetDriftMonitor

        ref = pd.DataFrame({"text_clean": ["kafka broker topic config service port"] * 60, "domain": ["backend"] * 60})
        cur = pd.DataFrame(
            {
                "text_clean": ["kafka broker topic config service port"] * 58 + ["kafka broker ubuntu kernel"] * 2,
                "domain": ["backend"] * 60,
            }
        )
        rep = DatasetDriftMonitor(ref, cur).run()
        self.assertEqual(rep["metrics"]["vocabulary"]["verdict"], "moderate_drift")

    def test_psi_moderate(self) -> None:
        from src.monitoring.drift import DatasetDriftMonitor

        ref = pd.DataFrame({"text_clean": ["x" * 20] * 30 + ["x" * 29] * 30, "domain": ["backend"] * 60})
        cur = pd.DataFrame({"text_clean": ["x" * 20] * 40 + ["x" * 29] * 20, "domain": ["backend"] * 60})
        psi = DatasetDriftMonitor(ref, cur, length_buckets=2).compute_length_psi()
        self.assertGreaterEqual(psi, 0.10)
        self.assertLess(psi, 0.25)
        rep = DatasetDriftMonitor(ref, cur, length_buckets=2).run()
        self.assertEqual(rep["metrics"]["length_psi"]["verdict"], "moderate_drift")


if __name__ == "__main__":
    unittest.main()
