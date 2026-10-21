"""
Exact and normalized hash deduplication.
"""

import hashlib
import logging
import re
from collections.abc import Iterable, Iterator

from ricc.ingestion.schema import CleanedMessage

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

    The fingerprints are stored as raw 8-byte digests rather than 32
    character hex strings: at the corpus scale that is 8 bytes per
    message instead of ~90 (a Python str plus the hex representation), so
    the index stays small while the corpus is streamed.
    """

    def __init__(self, min_len_to_dedup: int = 15):
        self.min_len_to_dedup = min_len_to_dedup
        self.seen_hashes: set[bytes] = set()

    def deduplicate(self, messages: list[CleanedMessage]) -> tuple[list[CleanedMessage], int]:
        """
        Filter out exact identical messages (e.g. repeated bot warnings, spam slogans).
        """
        counter = [0]
        unique_msgs: list[CleanedMessage] = []
        for message in self._unique_stream(messages, counter):
            unique_msgs.append(message)
        return unique_msgs, counter[0]

    def deduplicate_stream(self, messages: Iterable[CleanedMessage]) -> Iterator[CleanedMessage]:
        """Yield the unique messages, dropping exact duplicates.

        The corpus is 2.8M messages, so the list and the result must not
        be held at the same time. This generator keeps only the
        fingerprints.
        """
        counter = [0]
        yield from self._unique_stream(messages, counter)

    def _unique_stream(self, messages: Iterable[CleanedMessage], counter: list[int]) -> Iterator[CleanedMessage]:
        """The shared implementation; ``counter[0]`` receives the duplicate count."""
        for msg in messages:
            norm_text = normalize_text_for_hash(msg.text_clean)
            if len(norm_text) < self.min_len_to_dedup:
                # Keep short conversational acknowledgments
                yield msg
                continue

            # MD5 here is a non-cryptographic content fingerprint for
            # duplicate detection only (not passwords/signatures), so
            # usedforsecurity=False documents that intent for scanners.
            digest = hashlib.md5(norm_text.encode("utf-8"), usedforsecurity=False).digest()
            if digest in self.seen_hashes:
                counter[0] += 1
                continue
            self.seen_hashes.add(digest)
            yield msg

        logger.info(f"Exact deduplication: filtered {counter[0]} duplicate messages.")
