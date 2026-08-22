"""
Conversation Thread Builder and DAG Reconstruction for Telegram Chat Exports.
"""

import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from src.ingestion.schema import CleanedMessage

logger = logging.getLogger(__name__)


class ThreadDAGBuilder:
    """
    Reconstructs conversational DAGs (Directed Acyclic Graphs) and linear threads from reply chains.
    """

    def __init__(self, max_reply_gap_hours: int = 48, burst_window_minutes: int = 5):
        self.max_reply_gap_hours = max_reply_gap_hours
        self.burst_window_minutes = burst_window_minutes

    def build_threads(self, messages: List[CleanedMessage]) -> Tuple[List[CleanedMessage], Dict[int, List[CleanedMessage]]]:
        """
        Group messages into threads based on reply_to_id and temporal proximity.
        Assigns thread_id to each message in place and returns both updated messages and threads dictionary.
        """
        if not messages:
            return [], {}

        # Index messages by (chat_id, msg_id)
        msg_map: Dict[Tuple[int, int], CleanedMessage] = {
            (m.chat_id, m.msg_id): m for m in messages
        }

        # Parent to children mapping
        children_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        parent_map: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # Sort messages by timestamp
        sorted_msgs = sorted(messages, key=lambda m: (m.chat_id, m.unixtime, m.msg_id))

        # First pass: Link explicit reply_to_id
        for msg in sorted_msgs:
            cur_key = (msg.chat_id, msg.msg_id)
            if msg.reply_to_id:
                parent_key = (msg.chat_id, msg.reply_to_id)
                if parent_key in msg_map:
                    parent_msg = msg_map[parent_key]
                    # Check time gap
                    time_diff = msg.unixtime - parent_msg.unixtime
                    if 0 <= time_diff <= self.max_reply_gap_hours * 3600:
                        children_map[parent_key].append(cur_key)
                        parent_map[cur_key] = parent_key

        # Second pass: Cluster reply chains into connected thread components
        visited: Set[Tuple[int, int]] = set()
        threads: Dict[int, List[CleanedMessage]] = defaultdict(list)
        current_thread_id = 1

        # Process roots with explicit replies first
        for msg in sorted_msgs:
            key = (msg.chat_id, msg.msg_id)
            if key in visited:
                continue

            # If message is a root of an explicit reply tree
            if key not in parent_map and key in children_map:
                # Traverse tree in BFS/DFS order
                queue = deque([key])
                thread_nodes: List[CleanedMessage] = []
                while queue:
                    node_key = queue.popleft()
                    if node_key in visited:
                        continue
                    visited.add(node_key)
                    node_msg = msg_map[node_key]
                    node_msg.thread_id = current_thread_id
                    thread_nodes.append(node_msg)
                    for child_key in children_map.get(node_key, []):
                        if child_key not in visited:
                            queue.append(child_key)

                if thread_nodes:
                    # Sort thread nodes chronologically
                    thread_nodes.sort(key=lambda m: m.unixtime)
                    threads[current_thread_id] = thread_nodes
                    current_thread_id += 1

        # Third pass: Group remaining orphan messages into temporal burst threads if they are part of active dialogues
        last_chat_id = None
        last_time = 0
        burst_thread: List[CleanedMessage] = []
        burst_id = current_thread_id

        for msg in sorted_msgs:
            key = (msg.chat_id, msg.msg_id)
            if key in visited:
                # If we hit an already visited tree node, flush current burst
                if len(burst_thread) >= 2:
                    threads[burst_id] = list(burst_thread)
                    for bm in burst_thread:
                        bm.thread_id = burst_id
                    burst_id += 1
                burst_thread = []
                continue

            # Standalone message
            if last_chat_id == msg.chat_id and (msg.unixtime - last_time) <= (self.burst_window_minutes * 60):
                burst_thread.append(msg)
                visited.add(key)
            else:
                if len(burst_thread) >= 2:
                    threads[burst_id] = list(burst_thread)
                    for bm in burst_thread:
                        bm.thread_id = burst_id
                    burst_id += 1
                burst_thread = [msg]
                visited.add(key)

            last_chat_id = msg.chat_id
            last_time = msg.unixtime

        if len(burst_thread) >= 2:
            threads[burst_id] = list(burst_thread)
            for bm in burst_thread:
                bm.thread_id = burst_id
            burst_id += 1

        # Assign unique thread ID for single remaining standalone messages
        for msg in messages:
            if msg.thread_id is None:
                msg.thread_id = burst_id
                threads[burst_id] = [msg]
                burst_id += 1

        logger.info(f"Reconstructed {len(threads)} conversation threads from {len(messages)} messages.")
        return messages, threads
