"""
Social Network Analysis & Interaction Graph for Telegram Community Dialogues.
"""

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from src.ingestion.schema import CleanedMessage

logger = logging.getLogger(__name__)


class SocialNetworkAnalyzer:
    """
    Constructs and analyzes the directed communication and reply network of chat participants.
    """

    def __init__(self, reply_window_minutes: int = 30):
        self.reply_window_minutes = reply_window_minutes

    def build_network(
        self, messages: List[CleanedMessage]
    ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int], Dict[str, int]]:
        """
        Build directed interaction graph: sender -> recipient -> reply_count.
        Returns (adjacency_dict, in_degree_dict, out_degree_dict).
        """
        adjacency: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        in_degree: Dict[str, int] = defaultdict(int)
        out_degree: Dict[str, int] = defaultdict(int)

        msg_map = {m.msg_id: m for m in messages}
        sorted_msgs = sorted(messages, key=lambda m: m.unixtime)

        for i, msg in enumerate(sorted_msgs):
            sender = msg.author_anon

            # 1. Explicit reply_to
            if msg.reply_to_id and msg.reply_to_id in msg_map:
                target_msg = msg_map[msg.reply_to_id]
                recipient = target_msg.author_anon
                if sender != recipient:
                    adjacency[sender][recipient] += 1
                    out_degree[sender] += 1
                    in_degree[recipient] += 1
            else:
                # 2. Sequential response within time window
                if i > 0:
                    prev_msg = sorted_msgs[i - 1]
                    if (
                        prev_msg.chat_id == msg.chat_id
                        and (msg.unixtime - prev_msg.unixtime) <= (self.reply_window_minutes * 60)
                        and prev_msg.author_anon != sender
                    ):
                        recipient = prev_msg.author_anon
                        adjacency[sender][recipient] += 1
                        out_degree[sender] += 1
                        in_degree[recipient] += 1

        return adjacency, in_degree, out_degree

    def analyze(self, messages: List[CleanedMessage]) -> Dict[str, Any]:
        """
        Run comprehensive network graph analytics.
        """
        adjacency, in_degree, out_degree = self.build_network(messages)
        
        all_nodes = set(adjacency.keys()) | set(in_degree.keys()) | set(out_degree.keys())
        total_nodes = len(all_nodes)
        
        total_edges = sum(len(targets) for targets in adjacency.values())
        total_interactions = sum(sum(targets.values()) for targets in adjacency.values())

        # Top influencers (users who receive the most questions/answers)
        top_influencers = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # Top active conversationalists (users who reply the most)
        top_responders = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:15]

        # Calculate reciprocity (mutual interactions)
        reciprocal_pairs = 0
        strong_pairs: List[Tuple[str, str, int]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        adj_dict = {u: dict(targets) for u, targets in adjacency.items()}

        for u, targets in adj_dict.items():
            for v, weight in targets.items():
                if (v, u) in seen_pairs:
                    continue
                reverse_weight = adj_dict.get(v, {}).get(u, 0)
                if reverse_weight > 0:
                    reciprocal_pairs += 1
                    total_weight = weight + reverse_weight
                    if total_weight >= 10:
                        strong_pairs.append((u, v, total_weight))
                seen_pairs.add((u, v))

        strong_pairs.sort(key=lambda x: x[2], reverse=True)

        # Graph density
        max_possible_edges = total_nodes * (total_nodes - 1) if total_nodes > 1 else 1
        density = round(total_edges / max_possible_edges, 6) if max_possible_edges else 0.0

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "total_interactions": total_interactions,
            "density": density,
            "reciprocal_connections": reciprocal_pairs,
            "top_influencers": [{"author": a, "replies_received": cnt} for a, cnt in top_influencers],
            "top_responders": [{"author": a, "replies_given": cnt} for a, cnt in top_responders],
            "strongest_collaboration_pairs": [
                {"user_a": a, "user_b": b, "total_replies": w} for a, b, w in strong_pairs[:12]
            ],
        }
