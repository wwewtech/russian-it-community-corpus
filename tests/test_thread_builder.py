"""Unit tests for :mod:`src.graph.thread_builder`."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from src.graph.thread_builder import ThreadDAGBuilder
from src.ingestion.schema import CleanedMessage


def _make_msg(
    msg_id: int,
    chat_id: int = 1,
    chat_name: str = "test_chat",
    unixtime: int = 1767225600,
    author_anon: str = "user_1",
    author_id_anon: str = "id_1",
    text_clean: str = "test message",
    reply_to_id: int | None = None,
    domain: str = "general_tech_chat",
    tags: list[str] | None = None,
    sentiment_score: int = 0,
    token_count_approx: int = 10,
    is_question: bool = False,
) -> CleanedMessage:
    """Factory for CleanedMessage with sensible defaults."""
    return CleanedMessage(
        msg_id=msg_id,
        chat_id=chat_id,
        chat_name=chat_name,
        timestamp=datetime.fromtimestamp(unixtime).isoformat(),
        unixtime=unixtime,
        author_anon=author_anon,
        author_id_anon=author_id_anon,
        text_clean=text_clean,
        reply_to_id=reply_to_id,
        domain=domain,
        tags=tags or [],
        sentiment_score=sentiment_score,
        token_count_approx=token_count_approx,
        is_question=is_question,
        thread_id=None,
    )


class TestThreadDAGBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = ThreadDAGBuilder(max_reply_gap_hours=48, burst_window_minutes=5)

    # 1. Explicit reply chains: A→B→C forms single thread with thread_id=1
    def test_explicit_reply_chain(self):
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=1, unixtime=base_time),
            _make_msg(msg_id=2, unixtime=base_time + 60, reply_to_id=1),
            _make_msg(msg_id=3, unixtime=base_time + 120, reply_to_id=2),
        ]
        updated, threads = self.builder.build_threads(msgs)

        self.assertEqual(len(threads), 1)
        self.assertIn(1, threads)
        self.assertEqual(len(threads[1]), 3)
        for m in updated:
            self.assertEqual(m.thread_id, 1)
        # Chronological order in thread
        self.assertEqual([m.msg_id for m in threads[1]], [1, 2, 3])

    # 2. Time gap filtering: Reply >48h after parent should NOT link
    def test_time_gap_filters_reply(self):
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=1, unixtime=base_time),
            _make_msg(msg_id=2, unixtime=base_time + 49 * 3600, reply_to_id=1),  # >48h
        ]
        updated, threads = self.builder.build_threads(msgs)

        self.assertEqual(len(threads), 2)
        self.assertEqual(threads[1], [updated[0]])
        self.assertEqual(threads[2], [updated[1]])
        self.assertEqual(updated[0].thread_id, 1)
        self.assertEqual(updated[1].thread_id, 2)

    # 3. Multiple root threads: Two separate reply trees get different thread_ids
    def test_multiple_root_threads(self):
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=1, unixtime=base_time),
            _make_msg(msg_id=2, unixtime=base_time + 60, reply_to_id=1),
            _make_msg(msg_id=10, unixtime=base_time + 200),
            _make_msg(msg_id=11, unixtime=base_time + 260, reply_to_id=10),
        ]
        updated, threads = self.builder.build_threads(msgs)

        self.assertEqual(len(threads), 2)
        self.assertEqual(len(threads[1]), 2)
        self.assertEqual(len(threads[2]), 2)
        self.assertEqual([m.msg_id for m in threads[1]], [1, 2])
        self.assertEqual([m.msg_id for m in threads[2]], [10, 11])

    # 4. Orphan messages: Standalone messages get unique thread_ids
    def test_orphan_messages_get_unique_threads(self):
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=1, unixtime=base_time),
            _make_msg(msg_id=2, unixtime=base_time + 10000),
            _make_msg(msg_id=3, unixtime=base_time + 20000),
        ]
        updated, threads = self.builder.build_threads(msgs)

        self.assertEqual(len(threads), 3)
        for i, m in enumerate(updated, 1):
            self.assertEqual(m.thread_id, i)
            self.assertEqual(threads[i], [m])

    # 5. Temporal burst clustering: Messages within 5min in same chat grouped
    def test_temporal_burst_clustering(self):
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=1, unixtime=base_time),
            _make_msg(msg_id=2, unixtime=base_time + 60),
            _make_msg(msg_id=3, unixtime=base_time + 120),
            _make_msg(msg_id=10, unixtime=base_time + 10000),  # far apart
        ]
        updated, threads = self.builder.build_threads(msgs)

        self.assertEqual(len(threads), 2)
        self.assertEqual(len(threads[1]), 3)
        self.assertEqual([m.msg_id for m in threads[1]], [1, 2, 3])
        self.assertEqual(threads[2], [updated[3]])
        self.assertEqual(updated[3].thread_id, 2)

    # 6. Mixed scenario: Explicit replies + orphans + bursts in same chat
    def test_mixed_scenario(self):
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=1, unixtime=base_time),              # root of reply chain
            _make_msg(msg_id=2, unixtime=base_time + 60, reply_to_id=1),
            _make_msg(msg_id=3, unixtime=base_time + 120, reply_to_id=2),  # chain of 3
            _make_msg(msg_id=10, unixtime=base_time + 10000),     # orphan
            _make_msg(msg_id=11, unixtime=base_time + 10060),     # burst with 12
            _make_msg(msg_id=12, unixtime=base_time + 10120),     # burst (within 5 min)
            _make_msg(msg_id=20, unixtime=base_time + 20000),     # another orphan
        ]
        updated, threads = self.builder.build_threads(msgs)

        # Thread 1: explicit chain (3 msgs: 1,2,3)
        # Thread 2: burst (10, 11, 12) - all within 5min window
        # Thread 3: orphan 20 (alone, far from others)
        self.assertEqual(len(threads), 3)
        self.assertEqual(len(threads[1]), 3)
        self.assertEqual([m.msg_id for m in threads[1]], [1, 2, 3])
        self.assertEqual(len(threads[2]), 3)
        self.assertEqual([m.msg_id for m in threads[2]], [10, 11, 12])
        self.assertEqual(len(threads[3]), 1)
        self.assertEqual(threads[3][0].msg_id, 20)

    # 7. Cross-chat isolation: Messages from different chats never in same thread
    def test_cross_chat_isolation(self):
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=1, chat_id=1, chat_name="chat_a", unixtime=base_time),
            _make_msg(msg_id=2, chat_id=1, chat_name="chat_a", unixtime=base_time + 60, reply_to_id=1),
            _make_msg(msg_id=1, chat_id=2, chat_name="chat_b", unixtime=base_time),
            _make_msg(msg_id=2, chat_id=2, chat_name="chat_b", unixtime=base_time + 60, reply_to_id=1),
        ]
        updated, threads = self.builder.build_threads(msgs)

        self.assertEqual(len(threads), 2)
        # Each chat gets its own thread
        chat_a_msgs = [m for m in updated if m.chat_id == 1]
        chat_b_msgs = [m for m in updated if m.chat_id == 2]
        self.assertEqual(chat_a_msgs[0].thread_id, chat_a_msgs[1].thread_id)
        self.assertEqual(chat_b_msgs[0].thread_id, chat_b_msgs[1].thread_id)
        self.assertNotEqual(chat_a_msgs[0].thread_id, chat_b_msgs[0].thread_id)

    # 8. Empty input: Returns ([], {})
    def test_empty_input(self):
        updated, threads = self.builder.build_threads([])

        self.assertEqual(updated, [])
        self.assertEqual(threads, {})

    # 9. Single message: Returns ([msg], {thread_id: [msg]})
    def test_single_message(self):
        msg = _make_msg(msg_id=42, unixtime=1767225600)
        updated, threads = self.builder.build_threads([msg])

        self.assertEqual(len(updated), 1)
        self.assertEqual(len(threads), 1)
        self.assertEqual(updated[0].thread_id, 1)
        self.assertEqual(threads[1], [updated[0]])

    # Additional edge cases

    def test_reply_to_nonexistent_message(self):
        """Reply to non-existent parent falls back to temporal burst clustering."""
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=1, unixtime=base_time),
            _make_msg(msg_id=2, unixtime=base_time + 60, reply_to_id=999),  # parent doesn't exist
        ]
        updated, threads = self.builder.build_threads(msgs)

        # Both messages are within burst window (5 min) and same chat,
        # so they form a burst thread together (reply_to missing parent is ignored)
        self.assertEqual(len(threads), 1)
        self.assertEqual(len(threads[1]), 2)
        self.assertEqual(updated[0].thread_id, 1)
        self.assertEqual(updated[1].thread_id, 1)

    def test_reply_out_of_order(self):
        """Messages processed in timestamp order regardless of input order."""
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=2, unixtime=base_time + 60, reply_to_id=1),
            _make_msg(msg_id=1, unixtime=base_time),
        ]
        updated, threads = self.builder.build_threads(msgs)

        self.assertEqual(len(threads), 1)
        self.assertEqual(len(threads[1]), 2)
        self.assertEqual([m.msg_id for m in threads[1]], [1, 2])

    def test_burst_requires_min_two_messages(self):
        """Single message in burst window remains orphan (not a thread)."""
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=1, unixtime=base_time),
            _make_msg(msg_id=2, unixtime=base_time + 60),  # within 5 min but only 2 msgs
        ]
        updated, threads = self.builder.build_threads(msgs)

        # Should form a burst thread of 2 messages
        self.assertEqual(len(threads), 1)
        self.assertEqual(len(threads[1]), 2)

    def test_burst_single_message_is_orphan(self):
        """A single message with no neighbors within burst window gets unique thread."""
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=1, unixtime=base_time),
            _make_msg(msg_id=2, unixtime=base_time + 10 * 60),  # 10 min apart - outside burst window
        ]
        updated, threads = self.builder.build_threads(msgs)

        self.assertEqual(len(threads), 2)
        self.assertEqual(threads[1], [updated[0]])
        self.assertEqual(threads[2], [updated[1]])

    def test_burst_interrupted_by_explicit_thread(self):
        """Explicit reply thread flushes current burst accumulation."""
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=1, unixtime=base_time),
            _make_msg(msg_id=2, unixtime=base_time + 60),  # burst start
            _make_msg(msg_id=3, unixtime=base_time + 120), # burst continues
            # Explicit thread root with valid child (but parent far in time to avoid linking to burst)
            _make_msg(msg_id=10, unixtime=base_time + 10000),  # root of explicit thread
            _make_msg(msg_id=11, unixtime=base_time + 10060, reply_to_id=10),  # child
            # New burst after explicit thread
            _make_msg(msg_id=4, unixtime=base_time + 20000), # should start new burst
            _make_msg(msg_id=5, unixtime=base_time + 20060), # burst continues
        ]
        updated, threads = self.builder.build_threads(msgs)

        # Thread 1: explicit chain [10, 11] (processed first)
        # Thread 2: first burst [1, 2, 3] (flushed when hitting explicit thread msg 10)
        # Thread 3: second burst [4, 5]
        self.assertEqual(len(threads), 3)
        self.assertEqual(len(threads[1]), 2)
        self.assertEqual([m.msg_id for m in threads[1]], [10, 11])
        self.assertEqual(len(threads[2]), 3)
        self.assertEqual([m.msg_id for m in threads[2]], [1, 2, 3])
        self.assertEqual(len(threads[3]), 2)
        self.assertEqual([m.msg_id for m in threads[3]], [4, 5])

    def test_custom_parameters(self):
        """Custom max_reply_gap_hours and burst_window_minutes are respected."""
        # 24h reply gap, 10min burst window
        builder = ThreadDAGBuilder(max_reply_gap_hours=24, burst_window_minutes=10)
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=1, unixtime=base_time),
            _make_msg(msg_id=2, unixtime=base_time + 30 * 3600, reply_to_id=1),  # 30h - should NOT link with 24h limit
            _make_msg(msg_id=3, unixtime=base_time + 5 * 60),  # 5min - should burst with 10min window
        ]
        updated, threads = builder.build_threads(msgs)

        # 1 and 2 should be separate (30h > 24h)
        # 3 should burst with 1 (5min < 10min)
        # But wait - 3 comes after 1 in sorted order, so burst starts with 1, then 3
        # Actually: sorted by time: 1, 3, 2
        # burst: 1 and 3 within 10 min -> thread 1
        # 2 is 30h after 1 -> separate thread
        self.assertEqual(len(threads), 2)
        thread_1_ids = [m.msg_id for m in threads[1]]
        thread_2_ids = [m.msg_id for m in threads[2]]
        self.assertIn(1, thread_1_ids)
        self.assertIn(3, thread_1_ids)
        self.assertIn(2, thread_2_ids)

    def test_thread_id_assignment_increments(self):
        """Thread IDs are assigned sequentially starting from 1."""
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=1, unixtime=base_time),
            _make_msg(msg_id=2, unixtime=base_time + 60, reply_to_id=1),
            _make_msg(msg_id=3, unixtime=base_time + 10000),
            _make_msg(msg_id=4, unixtime=base_time + 10060),
        ]
        updated, threads = self.builder.build_threads(msgs)

        thread_ids = set(m.thread_id for m in updated)
        self.assertEqual(thread_ids, {1, 2})
        self.assertEqual(len(threads[1]), 2)
        self.assertEqual(len(threads[2]), 2)

    def test_returns_updated_messages_with_thread_id(self):
        """Input messages are modified in place with thread_id."""
        base_time = 1767225600
        msgs = [
            _make_msg(msg_id=1, unixtime=base_time),
            _make_msg(msg_id=2, unixtime=base_time + 60, reply_to_id=1),
        ]
        original_ids = [id(m) for m in msgs]
        updated, threads = self.builder.build_threads(msgs)

        # Same objects returned
        self.assertEqual([id(m) for m in updated], original_ids)
        # thread_id set on all
        for m in updated:
            self.assertIsNotNone(m.thread_id)


if __name__ == "__main__":
    unittest.main()