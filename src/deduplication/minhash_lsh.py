"""
MinHash LSH (Locality-Sensitive Hashing) for fuzzy deduplication of chat messages and threads.
"""

import hashlib
import logging
import re
import struct
from typing import Dict, List, Optional, Set, Tuple

from src.config import MINHASH_NUM_PERM, MINHASH_SHINGLE_SIZE, MINHASH_THRESHOLD
from src.ingestion.schema import CleanedMessage

logger = logging.getLogger(__name__)

# Precomputed deterministic random seeds for 128 hash permutations
_PERM_A = [
    (i * 10007 + 34229) & 0xFFFFFFFF for i in range(MINHASH_NUM_PERM)
]
_PERM_B = [
    (i * 49999 + 88123) & 0xFFFFFFFF for i in range(MINHASH_NUM_PERM)
]
_PRIME = 4294967311  # 2^32 + 15 (large 32-bit prime)


class MinHashLSH:
    """
    High-performance pure Python MinHash LSH index for near-duplicate text detection (Jaccard similarity >= threshold).
    """

    def __init__(
        self,
        num_perm: int = MINHASH_NUM_PERM,
        threshold: float = MINHASH_THRESHOLD,
        shingle_size: int = MINHASH_SHINGLE_SIZE,
    ):
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        
        # Calculate optimal number of bands and rows for high recall at threshold ~0.70-0.80
        self.b = 32
        self.r = num_perm // self.b
        
        # LSH buckets: band_idx -> bucket_hash -> list of item_ids
        self.buckets: List[Dict[int, List[int]]] = [dict() for _ in range(self.b)]
        self.signatures: Dict[int, List[int]] = {}

    def _get_shingles(self, text: str) -> Set[str]:
        """Extract word shingles of length k."""
        words = re.findall(r'\w+', text.lower())
        if len(words) < self.shingle_size:
            return {" ".join(words)} if words else set()
        shingles = set()
        for i in range(len(words) - self.shingle_size + 1):
            shingles.add(" ".join(words[i : i + self.shingle_size]))
        return shingles

    def compute_minhash(self, text: str) -> List[int]:
        """Compute MinHash signature vector of length num_perm."""
        shingles = self._get_shingles(text)
        if not shingles:
            return [0] * self.num_perm

        # Hash each shingle to uint32
        shingle_hashes = []
        for s in shingles:
            h = int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)
            shingle_hashes.append(h)

        sig = []
        for i in range(self.num_perm):
            a = _PERM_A[i]
            b = _PERM_B[i]
            min_val = min(((a * h + b) % _PRIME) for h in shingle_hashes)
            sig.append(min_val)

        return sig

    def jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity between two MinHash signatures."""
        if not sig1 or not sig2:
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / float(self.num_perm)

    def insert(self, item_id: int, text: str) -> Tuple[bool, Optional[int], float]:
        """
        Check if text is a near-duplicate of an already indexed item.
        If duplicate found (Jaccard >= threshold), returns (True, duplicate_of_item_id, similarity).
        Otherwise, indexes the item and returns (False, None, 0.0).
        """
        if len(text.strip()) < 20:
            return False, None, 0.0

        sig = self.compute_minhash(text)
        candidates: Set[int] = set()

        # Query buckets for each band
        for band_idx in range(self.b):
            band_sig = tuple(sig[band_idx * self.r : (band_idx + 1) * self.r])
            bucket_key = hash(band_sig)
            bucket = self.buckets[band_idx]
            if bucket_key in bucket:
                candidates.update(bucket[bucket_key])

        # Verify exact signature similarity against candidates
        for cand_id in candidates:
            if cand_id in self.signatures:
                cand_sig = self.signatures[cand_id]
                sim = self.jaccard_similarity(sig, cand_sig)
                if sim >= self.threshold:
                    return True, cand_id, sim

        # Not a duplicate, insert into index
        self.signatures[item_id] = sig
        for band_idx in range(self.b):
            band_sig = tuple(sig[band_idx * self.r : (band_idx + 1) * self.r])
            bucket_key = hash(band_sig)
            bucket = self.buckets[band_idx]
            if bucket_key not in bucket:
                bucket[bucket_key] = []
            bucket[bucket_key].append(item_id)

        return False, None, 0.0

    def deduplicate_messages(
        self, messages: List[CleanedMessage]
    ) -> Tuple[List[CleanedMessage], int]:
        """
        Deduplicate a list of messages in place.
        Returns list of unique messages and number of removed duplicates.
        """
        unique_msgs: List[CleanedMessage] = []
        dupes_count = 0

        for m in messages:
            # Skip very short messages from fuzzy dedup to avoid over-filtering
            if len(m.text_clean.strip()) < 30:
                unique_msgs.append(m)
                continue

            is_dup, dup_of, sim = self.insert(m.msg_id, m.text_clean)
            if is_dup:
                dupes_count += 1
            else:
                unique_msgs.append(m)

        logger.info(f"MinHash LSH deduplication: removed {dupes_count} near-duplicate messages.")
        return unique_msgs, dupes_count
