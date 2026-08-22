"""
Unit tests for Exact and MinHash LSH fuzzy deduplication.
"""

import unittest
from src.deduplication.exact_dedup import ExactDeduplicator
from src.deduplication.minhash_lsh import MinHashLSH
from src.ingestion.schema import CleanedMessage


class TestDeduplication(unittest.TestCase):

    def setUp(self):
        self.exact_dedup = ExactDeduplicator(min_len_to_dedup=10)
        self.minhash_lsh = MinHashLSH(threshold=0.70)

    def test_exact_dedup(self):
        msgs = [
            CleanedMessage(msg_id=1, chat_id=1, chat_name="C", timestamp="2026-01-01", unixtime=1, author_anon="A", author_id_anon="u1", text_clean="Привет всем разработчикам в этом чате!"),
            CleanedMessage(msg_id=2, chat_id=1, chat_name="C", timestamp="2026-01-01", unixtime=2, author_anon="B", author_id_anon="u2", text_clean="Привет всем разработчикам в этом чате!"),
            CleanedMessage(msg_id=3, chat_id=1, chat_name="C", timestamp="2026-01-01", unixtime=3, author_anon="C", author_id_anon="u3", text_clean="Совершенно другое сообщение по делу"),
        ]
        unique, removed = self.exact_dedup.deduplicate(msgs)
        self.assertEqual(len(unique), 2)
        self.assertEqual(removed, 1)

    def test_minhash_fuzzy_dedup(self):
        t1 = "Внимание! Подписывайтесь на наш официальный канал про IT и разработку сервисов."
        t2 = "Внимание!! Подписывайтесь на наш официальный канал про IT и разработку полезных сервисов."
        t3 = "Как правильно поднять PostgreSQL в Docker контейнере с постоянным volume хранилищем?"

        is_dup1, _, _ = self.minhash_lsh.insert(1, t1)
        self.assertFalse(is_dup1)

        is_dup2, _, sim = self.minhash_lsh.insert(2, t2)
        self.assertTrue(is_dup2)
        self.assertGreaterEqual(sim, 0.70)

        is_dup3, _, _ = self.minhash_lsh.insert(3, t3)
        self.assertFalse(is_dup3)


if __name__ == "__main__":
    unittest.main()
