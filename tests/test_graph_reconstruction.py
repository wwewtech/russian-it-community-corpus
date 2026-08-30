"""
Unit tests for Thread DAG reconstruction and Conversation Extraction.
"""

import unittest
from datetime import datetime

from src.graph.conversation_extractor import ConversationExtractor
from src.graph.thread_builder import ThreadDAGBuilder
from src.ingestion.schema import CleanedMessage


class TestGraphReconstruction(unittest.TestCase):
    def setUp(self):
        self.builder = ThreadDAGBuilder()
        self.extractor = ConversationExtractor()

    def test_reply_chain_clustering(self):
        now = datetime.now()
        msgs = [
            CleanedMessage(
                msg_id=1,
                chat_id=100,
                chat_name="TestChat",
                timestamp=now.isoformat(),
                unixtime=1000,
                author_anon="Developer_001",
                author_id_anon="user_001",
                text_clean="Кто пробовал переносить проект с Flask на FastAPI? Как прирост RPS?",
                reply_to_id=None,
                is_question=True,
            ),
            CleanedMessage(
                msg_id=2,
                chat_id=100,
                chat_name="TestChat",
                timestamp=now.isoformat(),
                unixtime=1060,
                author_anon="Developer_002",
                author_id_anon="user_002",
                text_clean="Мы перенесли пару сервисов, прирост в 2-3 раза под нагрузкой за счет асинхронных эндпоинтов и uvicorn.",
                reply_to_id=1,
            ),
            CleanedMessage(
                msg_id=3,
                chat_id=100,
                chat_name="TestChat",
                timestamp=now.isoformat(),
                unixtime=1120,
                author_anon="Developer_001",
                author_id_anon="user_001",
                text_clean="А как с валидацией через Pydantic v2 по скорости?",
                reply_to_id=2,
                is_question=True,
            ),
            CleanedMessage(
                msg_id=4,
                chat_id=100,
                chat_name="TestChat",
                timestamp=now.isoformat(),
                unixtime=1180,
                author_anon="Developer_002",
                author_id_anon="user_002",
                text_clean="Pydantic core на Rust написан, поэтому валидация быстрее в 5-10 раз.",
                reply_to_id=3,
            ),
        ]

        updated_msgs, threads = self.builder.build_threads(msgs)
        self.assertEqual(len(threads), 1)
        self.assertEqual(len(list(threads.values())[0]), 4)

        # Test SFT extraction
        sft_dialogues = self.extractor.extract_sft_dialogues(threads)
        self.assertEqual(len(sft_dialogues), 1)
        dialogue = sft_dialogues[0]
        self.assertEqual(dialogue.turn_count, 4)
        self.assertEqual(dialogue.messages[0].role, "user")
        self.assertEqual(dialogue.messages[1].role, "assistant")
        self.assertEqual(dialogue.messages[2].role, "user")
        self.assertEqual(dialogue.messages[3].role, "assistant")

    def _make_msg(self, msg_id, chat_id, unixtime, author_id, text, reply_to_id=None):
        return CleanedMessage(
            msg_id=msg_id,
            chat_id=chat_id,
            chat_name="BurstChat",
            timestamp=datetime.fromtimestamp(unixtime).isoformat(),
            unixtime=unixtime,
            author_anon=f"Dev_{author_id}",
            author_id_anon=author_id,
            text_clean=text,
            reply_to_id=reply_to_id,
        )

    def test_temporal_burst_clustering(self):
        """Messages without explicit replies but close in time form a burst thread."""
        msgs = [
            self._make_msg(10, 200, 5000, "u_1", "Кто-нибудь сравнивал производительность Redis и KeyDB?"),
            self._make_msg(
                11, 200, 5060, "u_2", "KeyDB быстрее на многопоточной нагрузке, но экосистема у Redis больше."
            ),
        ]
        _, threads = self.builder.build_threads(msgs)
        self.assertEqual(len(threads), 1)
        self.assertEqual(len(list(threads.values())[0]), 2)

    def test_stale_reply_is_not_linked(self):
        """A reply pointing to a message older than max_reply_gap_hours must not merge threads."""
        stale = self._make_msg(20, 300, 1000, "u_1", "Какой менеджер очередей выбрать для новой системы?")
        late_reply = self._make_msg(
            21, 300, 1000 + 72 * 3600, "u_2", "Мы взяли RabbitMQ, отрабатывает отлично.", reply_to_id=20
        )
        updated, threads = self.builder.build_threads([stale, late_reply])
        # No link -> each message ends up alone in its own thread bucket
        self.assertEqual(len(threads), 2)
        self.assertIsNot(updated[0].thread_id, updated[1].thread_id)

    def test_build_threads_empty_input(self):
        updated, threads = self.builder.build_threads([])
        self.assertEqual(updated, [])
        self.assertEqual(threads, {})

    def test_compute_message_quality(self):
        empty = self._make_msg(30, 400, 7000, "u_1", "")
        self.assertEqual(self.extractor.compute_message_quality(empty), 0.0)

        single_word = self._make_msg(31, 400, 7010, "u_1", "ок")
        self.assertEqual(self.extractor.compute_message_quality(single_word), 0.0)

        # NOTE: single-word messages hit the word_count < 2 early return (0.0)
        # before the trivial-reaction branch, so a TWO-word trivial phrase is
        # required to exercise the 0.2 path.
        trivial = self._make_msg(32, 400, 7020, "u_1", "не знаю")
        self.assertEqual(self.extractor.compute_message_quality(trivial), 0.2)

    def test_extract_dpo_pairs_prefers_higher_quality_answer(self):
        # 12.0.2 fix: the compute_message_quality heuristic no longer awards
        # +2.0 for the mere presence of a code keyword — it now scores only
        # length and technical-keyword density. Make the chosen message
        # long enough and dense enough in technical terms to cross the
        # ``best_score >= 3.0`` threshold that gates DPO pair creation.
        good_text = (
            "Для продакшена настройте синхронную репликацию PostgreSQL, укажите "
            "primary_conninfo и standby_conninfo в postgresql.conf, включите wal_level=replica "
            "и используйте pg_basebackup для начальной инициализации standby. Затем следите за "
            "отставанием реплик через SELECT client_state, sent_lsn, write_lsn, replay_lsn FROM "
            "pg_stat_replication; и используйте repmgr или Patroni для автоматического failover "
            "при сбое primary. В кластере Kubernetes обязательно настройте readinessProbe через "
            "pg_is_in_recovery() и livenessProbe на pg_stat_replication, чтобы rolling update не "
            "приводил к split-brain при двух одновременно активных primary."
        )
        bad_text = "вроде где то читал что это не нужно вообще настраивать"
        root = self._make_msg(40, 500, 9000, "u_root", "Как правильно настроить репликацию в PostgreSQL кластере?")
        good = self._make_msg(41, 500, 9060, "u_a", good_text, reply_to_id=40)
        bad = self._make_msg(42, 500, 9120, "u_b", bad_text, reply_to_id=40)

        pairs = self.extractor.extract_dpo_pairs({7: [root, good, bad]})
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertEqual(pair["prompt"], root.text_clean)
        self.assertEqual(pair["chosen"], good_text)
        self.assertEqual(pair["rejected"], bad_text)
        self.assertGreater(pair["chosen_quality"], pair["rejected_quality"])
        self.assertGreaterEqual(pair["chosen_quality"], 3.0)

    def test_extract_dpo_pairs_requires_enough_words_in_prompt(self):
        short_root = self._make_msg(43, 500, 9200, "u_root", "а?")
        r1 = self._make_msg(44, 500, 9260, "u_a", "Первый вариант ответа с достаточной длиной текста.")
        r2 = self._make_msg(45, 500, 9320, "u_b", "Второй вариант ответа с достаточной длиной текста.")
        self.assertEqual(self.extractor.extract_dpo_pairs({8: [short_root, r1, r2]}), [])

    def test_extract_rag_chunks_metadata(self):
        m1 = self._make_msg(50, 600, 11000, "u_1", "Расскажи, как устроен Raft консенсус в распределённых системах?")
        m2 = self._make_msg(
            51,
            600,
            11060,
            "u_2",
            "Raft выбирает лидера через таймауты выборов, лог реплицируется по порядку, коммит происходит после кворума.",
        )
        chunks = self.extractor.extract_rag_chunks({7: [m1, m2]})
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.chunk_id, "rag_kb_000007")
        self.assertEqual(chunk.thread_id, 7)
        self.assertEqual(chunk.participants_count, 2)
        self.assertEqual(chunk.message_count, 2)
        self.assertIn("u_1", chunk.content)
        self.assertIn("u_2", chunk.content)

    def test_extract_rag_chunks_skips_low_content_threads(self):
        tiny = [self._make_msg(52, 600, 12000, "u_1", "+"), self._make_msg(53, 600, 12060, "u_2", "ок")]
        self.assertEqual(self.extractor.extract_rag_chunks({9: tiny}), [])


if __name__ == "__main__":
    unittest.main()
