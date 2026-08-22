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


if __name__ == "__main__":
    unittest.main()
