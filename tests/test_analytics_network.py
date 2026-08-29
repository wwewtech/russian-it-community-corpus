"""Tests for the social-network analyzer (src/analytics/network.py)."""

from __future__ import annotations

from datetime import datetime

from src.analytics.network import SocialNetworkAnalyzer
from src.ingestion.schema import CleanedMessage


def _msg(msg_id: int, author: str, reply_to_id: int | None = None, unixtime: int = 1704067200) -> CleanedMessage:
    return CleanedMessage(
        msg_id=msg_id,
        chat_id=1,
        chat_name="node_01",
        timestamp=datetime.fromtimestamp(unixtime).isoformat(),
        unixtime=unixtime,
        author_anon=author,
        author_id_anon=author,
        text_clean="hello world",
        reply_to_id=reply_to_id,
    )


class TestBuildNetwork:
    def test_reply_creates_directed_edge(self):
        msgs = [_msg(1, "Alice"), _msg(2, "Bob", reply_to_id=1)]
        adj, indeg, outdeg = SocialNetworkAnalyzer().build_network(msgs)
        assert adj["Bob"]["Alice"] == 1
        assert outdeg["Bob"] == 1
        assert indeg["Alice"] == 1

    def test_self_reply_is_ignored(self):
        msgs = [_msg(1, "Alice"), _msg(2, "Alice", reply_to_id=1)]
        adj, _, _ = SocialNetworkAnalyzer().build_network(msgs)
        assert not adj

    def test_missing_reply_target_falls_back_to_sequential(self):
        msgs = [_msg(1, "Alice", unixtime=1000), _msg(2, "Bob", reply_to_id=99, unixtime=1100)]
        adj, _, _ = SocialNetworkAnalyzer(reply_window_minutes=30).build_network(msgs)
        assert adj["Bob"]["Alice"] == 1

    def test_sequential_beyond_window_is_ignored(self):
        msgs = [_msg(1, "Alice", unixtime=1000), _msg(2, "Bob", reply_to_id=99, unixtime=1000 + 31 * 60)]
        adj, _, _ = SocialNetworkAnalyzer(reply_window_minutes=30).build_network(msgs)
        assert "Bob" not in adj

    def test_different_chat_no_sequential_edge(self):
        msgs = [
            CleanedMessage(
                msg_id=1, chat_id=1, chat_name="a",
                timestamp=datetime.fromtimestamp(1000).isoformat(), unixtime=1000,
                author_anon="Alice", author_id_anon="Alice", text_clean="hi",
            ),
            CleanedMessage(
                msg_id=2, chat_id=2, chat_name="b",
                timestamp=datetime.fromtimestamp(1100).isoformat(), unixtime=1100,
                author_anon="Bob", author_id_anon="Bob", text_clean="hi",
            ),
        ]
        adj, _, _ = SocialNetworkAnalyzer(reply_window_minutes=30).build_network(msgs)
        assert not adj

    def test_empty_messages(self):
        adj, indeg, outdeg = SocialNetworkAnalyzer().build_network([])
        assert not adj and not indeg and not outdeg


class TestAnalyze:
    def test_report_structure(self):
        msgs = [
            _msg(1, "Alice", unixtime=100),
            _msg(2, "Bob", reply_to_id=1, unixtime=110),
            _msg(3, "Alice", reply_to_id=2, unixtime=120),
        ]
        report = SocialNetworkAnalyzer().analyze(msgs)
        for key in (
            "total_nodes", "total_edges", "total_interactions", "density",
            "reciprocal_connections", "top_influencers", "top_responders",
            "strongest_collaboration_pairs",
        ):
            assert key in report
        assert report["total_nodes"] == 2
        assert report["total_edges"] == 2

    def test_density_range(self):
        msgs = [_msg(1, "Alice", unixtime=100), _msg(2, "Bob", reply_to_id=1, unixtime=110)]
        report = SocialNetworkAnalyzer().analyze(msgs)
        assert 0.0 <= report["density"] <= 1.0

    def test_reciprocal_pair_detected(self):
        msgs = [
            _msg(1, "Alice", unixtime=100),
            _msg(2, "Bob", reply_to_id=1, unixtime=110),
            _msg(3, "Alice", reply_to_id=2, unixtime=120),
        ]
        report = SocialNetworkAnalyzer().analyze(msgs)
        assert report["reciprocal_connections"] >= 1

    def test_single_node_density_zero(self):
        msgs = [_msg(1, "Alice", unixtime=100)]
        report = SocialNetworkAnalyzer().analyze(msgs)
        assert report["density"] == 0.0
        assert report["top_influencers"] == []


if __name__ == "__main__":
    import unittest

    unittest.main()
