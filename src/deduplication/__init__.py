"""
Deduplication module with exact and MinHash LSH fuzzy deduplication algorithms.
"""

from src.deduplication.exact_dedup import ExactDeduplicator, normalize_text_for_hash
from src.deduplication.minhash_lsh import MinHashLSH

__all__ = ["MinHashLSH", "ExactDeduplicator", "normalize_text_for_hash"]
