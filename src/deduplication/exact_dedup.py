"""
Exact and normalized hash deduplication.
"""

import hashlib
import logging
import re

from src.ingestion.schema import CleanedMessage

logger = logging.getLogger(__name__)


def normalize_text_for_hash(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation and collapsing whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ExactDeduplicator:
    """
    Deduplicator based on MD5/SHA-256 hashes of normalized strings.
    """

    def __init__(self, min_len_to_dedup: int = 15):
        self.min_len_to_dedup = min_len_to_dedup
        self.seen_hashes: set[str] = set()

    def deduplicate(self, messages: list[CleanedMessage]) -> tuple[list[CleanedMessage], int]:
        """
        Filter out exact identical messages (e.g. repeated bot warnings, spam slogans).
        """
        unique_msgs: list[CleanedMessage] = []
        dupes_count = 0

        for msg in messages:
            norm_text = normalize_text_for_hash(msg.text_clean)
            if len(norm_text) < self.min_len_to_dedup:
                # Keep short conversational acknowledgments
                unique_msgs.append(msg)
                continue

            h = hashlib.md5(norm_text.encode("utf-8")).hexdigest()
            if h in self.seen_hashes:
                dupes_count += 1
            else:
                self.seen_hashes.add(h)
                unique_msgs.append(msg)

        logger.info(f"Exact deduplication: filtered {dupes_count} duplicate messages.")
        return unique_msgs, dupes_count
