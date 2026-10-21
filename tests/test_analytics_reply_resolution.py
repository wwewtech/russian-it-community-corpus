"""Regression tests for intra-chat reply resolution in the social network.

msg_id is only unique WITHIN a chat.  The analyzer used to index messages globally by
msg_id, so replies were resolved against messages of other chats (measured: 42.77%
of 2,309,365 replies resolved cross-chat), corrupting the social graph.
"""

from __future__ import annotations

from datetime import datetime

from src.analytics.network import SocialNetworkAnalyzer
from src.ingestion.schema import CleanedMessage


def _msg(
    msg_id: int,
    chat_id: int,
    author: str,
    reply_to_id: int | None = None,
    unixtime: int = 1704067200,
) -> CleanedMessage:
    return CleanedMessage(
        msg_id=msg_id,
        chat_id=chat_id,
        chat_name=f"chat_{chat_id}",
        timestamp=datetime.fromtimestamp(unixtime).isoformat(),
        unixtime=unixtime,
        author_anon=author,
        author_id_anon=author,
        text_clean="hello world",
        reply_to_id=reply_to_id,
    )


class TestReplyResolutionIsIntraChat:
    def test_cross_chat_reply_does_not_create_edge(self):
        # Chat 100 and chat 200 deliberately reuse msg_id values.
        # Bob (chat 200) replies to msg_id 5, which exists ONLY in chat 100 (Eve).
        # With a global msg_id index this resolves to Eve and creates a cross-chat edge.
        msgs = [
            _msg(1, 100, "Alice", unixtime=1000),
            _msg(5, 100, "Eve", unixtime=1005),
            _msg(1, 200, "Mallory", unixtime=1000),
            _msg(2, 200, "Bob", reply_to_id=5, unixtime=1100),
        ]
        # reply_window_minutes=0 disables the sequential fallback so only the
        # explicit-reply path can produce an edge.
        adj, indeg, outdeg = SocialNetworkAnalyzer(reply_window_minutes=0).build_network(msgs)
        assert "Eve" not in adj.get("Bob", {})
        assert "Eve" not in indeg
        assert "Bob" not in outdeg
        assert not adj

    def test_intra_chat_reply_creates_edge(self):
        # Same layout, but now chat 200 also contains msg_id 5 (Frank) and Bob's
        # reply to msg_id 5 must resolve to Frank, not to Eve from chat 100.
        msgs = [
            _msg(1, 100, "Alice", unixtime=1000),
            _msg(5, 100, "Eve", unixtime=1005),
            _msg(1, 200, "Mallory", unixtime=1000),
            _msg(5, 200, "Frank", unixtime=1005),
            _msg(2, 200, "Bob", reply_to_id=5, unixtime=1100),
        ]
        adj, indeg, outdeg = SocialNetworkAnalyzer(reply_window_minutes=0).build_network(msgs)
        assert adj["Bob"]["Frank"] == 1
        assert "Eve" not in adj.get("Bob", {})
        assert indeg["Frank"] == 1
        assert "Eve" not in indeg
        assert outdeg["Bob"] == 1

    def test_reused_msg_id_resolves_within_own_chat(self):
        # Two chats with the very same msg_id values: each reply must point to the
        # author inside its own chat.
        msgs = [
            _msg(1, 100, "Alice", unixtime=1000),
            _msg(2, 100, "Bob", reply_to_id=1, unixtime=1100),
            _msg(1, 200, "Carol", unixtime=1000),
            _msg(2, 200, "Dave", reply_to_id=1, unixtime=1100),
        ]
        adj, indeg, _ = SocialNetworkAnalyzer(reply_window_minutes=0).build_network(msgs)
        assert adj["Bob"]["Alice"] == 1
        assert adj["Dave"]["Carol"] == 1
        assert "Carol" not in adj.get("Bob", {})
        assert "Alice" not in adj.get("Dave", {})
        assert indeg["Alice"] == 1
        assert indeg["Carol"] == 1

    def test_sequential_fallback_stays_within_chat(self):
        # Interleaved chats: the globally previous message belongs to another chat,
        # so the sequential fallback must not create a cross-chat edge.
        msgs = [
            _msg(7, 1, "Alice", unixtime=1000),
            _msg(7, 2, "Carol", unixtime=1005),
            _msg(8, 1, "Bob", reply_to_id=7, unixtime=1010),
        ]
        adj, _, _ = SocialNetworkAnalyzer(reply_window_minutes=30).build_network(msgs)
        # Bob's explicit reply resolves within chat 1 -> Alice.
        assert adj["Bob"]["Alice"] == 1
        assert "Bob" not in adj.get("Carol", {})
        assert "Carol" not in adj.get("Bob", {})

    def test_analyzer_report_has_no_cross_chat_edges(self):
        msgs = [
            _msg(1, 100, "Alice", unixtime=1000),
            _msg(5, 100, "Eve", unixtime=1005),
            _msg(1, 200, "Mallory", unixtime=1000),
            _msg(2, 200, "Bob", reply_to_id=5, unixtime=1100),
        ]
        report = SocialNetworkAnalyzer(reply_window_minutes=0).analyze(msgs)
        assert report["total_edges"] == 0
        assert report["total_interactions"] == 0

    def test_engine_message_index_uses_composite_key(self):
        from src.analytics.engine import DeepChatAnalyzer

        msgs = [
            _msg(1, 100, "Alice", unixtime=1000),
            _msg(1, 200, "Mallory", unixtime=1000),
        ]
        analyzer = DeepChatAnalyzer(msgs)
        assert (100, 1) in analyzer.msg_by_id
        assert (200, 1) in analyzer.msg_by_id
        assert analyzer.msg_by_id[(200, 1)].author_anon == "Mallory"
        assert analyzer.msg_by_id[(100, 1)].author_anon == "Alice"


if __name__ == "__main__":
    import unittest

    unittest.main()
