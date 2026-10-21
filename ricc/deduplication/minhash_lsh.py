"""
MinHash LSH (Locality-Sensitive Hashing) for fuzzy deduplication of chat messages and threads.

Memory note: the signature index is the memory bottleneck at the corpus
scale. 63 % of the 2.8M messages are long enough to be indexed, and a
Python ``list[int]`` of 128 entries costs ~5.3 kB per message — about
9.5 GB for the corpus. The signatures are therefore stored in a single
contiguous ``numpy.uint32`` block (512 B per message, ~0.9 GB) with a
dict mapping the item id to its row offset. The dict of ids is
unavoidable, but the per-item overhead is what dominates.
"""

import logging
import re
from collections.abc import Hashable, Iterable, Iterator
from typing import cast

import numpy as np
import numpy.typing as npt
import xxhash
from tqdm import tqdm

from ricc.ingestion.schema import CleanedMessage
from ricc.settings import settings

logger = logging.getLogger(__name__)

# Precomputed deterministic random seeds for 128 hash permutations
_PERM_A = [(i * 10007 + 34229) & 0xFFFFFFFF for i in range(settings.MINHASH_NUM_PERM)]
_PERM_B = [(i * 49999 + 88123) & 0xFFFFFFFF for i in range(settings.MINHASH_NUM_PERM)]
_PRIME = 4294967311  # 2^32 + 15 (large 32-bit prime)


# The permutation parameters. They are derived from a FIXED formula
# (not from a random seed) so the signature of a text is reproducible
# across processes and machines. The arrays are built for the CONFIGURED
# number of permutations; an index constructed with a different
# ``num_perm`` derives its own permutations below, because the old
# module-level constants were always 128 long — so an index built with
# ``num_perm=32`` produced a 128-dimensional signature while its bands
# were computed for 32, silently corrupting the similarity estimate.
def _permutations(num_perm: int) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """The (a, b) permutation coefficients for ``num_perm`` hashes."""
    a = np.array([(i * 10007 + 34229) & 0xFFFFFFFF for i in range(num_perm)], dtype=np.uint64)
    b = np.array([(i * 49999 + 88123) & 0xFFFFFFFF for i in range(num_perm)], dtype=np.uint64)
    return a, b


# The permutation parameters as numpy arrays, so the signature of a
# message is computed with vectorised arithmetic instead of a Python
# loop over 128 permutations x N shingles.
_PERM_A_ARR, _PERM_B_ARR = _permutations(settings.MINHASH_NUM_PERM)


def _band_key(band_sig: tuple[int, ...]) -> int:
    """Stable bucket key for one LSH band.

    The builtin ``hash()`` is SipHash with a per-process random seed
    (PYTHONHASHSEED), so bucket placement — and therefore which
    near-duplicate survives — changed between runs. xxh64 with a fixed
    seed is deterministic across processes, machines and restarts.
    Signature values are < _PRIME (< 2**33), hence 8 bytes LE each.
    """
    h = xxhash.xxh64(seed=0)
    for v in band_sig:
        h.update(int(v).to_bytes(8, "little", signed=False))
    return h.intdigest()


class MinHashLSH:
    """
    High-performance pure Python MinHash LSH index for near-duplicate text detection (Jaccard similarity >= threshold).
    """

    def __init__(
        self,
        num_perm: int = settings.MINHASH_NUM_PERM,
        threshold: float = settings.MINHASH_THRESHOLD,
        shingle_size: int = settings.MINHASH_SHINGLE_SIZE,
    ):
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size

        # The permutations belong to THIS index: they must have exactly
        # num_perm entries, or the signature length and the band layout
        # disagree.
        self._perm_a, self._perm_b = _permutations(num_perm)

        # Calculate optimal number of bands and rows for high recall at threshold ~0.70-0.80
        self.b = 32
        self.r = num_perm // self.b

        # LSH buckets: band_idx -> bucket_hash -> list of item_ids.
        # Item ids are opaque hashables: deduplicate_messages() uses
        # (chat_id, msg_id) tuples because msg_id is unique only within
        # one chat — a bare msg_id collides across community nodes and
        # would overwrite foreign signatures/candidates.
        self.buckets: list[dict[int, list[Hashable]]] = [dict() for _ in range(self.b)]

        # Signatures live in ONE contiguous uint32 block: row `i` holds
        # the signature of the i-th indexed item. A Python list of 128
        # ints per item cost ~5.3 kB (9.5 GB for the corpus); a uint32
        # row costs 512 B (0.9 GB).
        self._signatures = np.zeros((0, self.num_perm), dtype=np.uint32)
        self._signature_rows: dict[Hashable, int] = {}
        self._signature_count = 0

    def _get_shingles(self, text: str) -> set[str]:
        """Extract word shingles of length k."""
        words = re.findall(r"\w+", text.lower())
        if len(words) < self.shingle_size:
            return {" ".join(words)} if words else set()
        shingles = set()
        for i in range(len(words) - self.shingle_size + 1):
            shingles.add(" ".join(words[i : i + self.shingle_size]))
        return shingles

    def compute_minhash(self, text: str) -> npt.NDArray[np.uint32]:
        """Compute the MinHash signature vector of length num_perm.

        Returns a ``uint32`` array. The permutation arithmetic is
        vectorised over the shingles, which is both faster and the
        reason the signatures fit in a compact block.
        """
        shingles = self._get_shingles(text)
        if not shingles:
            return np.zeros(self.num_perm, dtype=np.uint32)

        # Hash each shingle to uint32 using xxh32 (faster than MD5)
        shingle_hashes = np.fromiter(
            (xxhash.xxh32(s.encode("utf-8"), seed=0).intdigest() & 0xFFFFFFFF for s in shingles),
            dtype=np.uint64,
            count=len(shingles),
        )

        # sig[i] = min over shingles of (a_i * h + b_i) % PRIME
        permuted = (self._perm_a[:, None] * shingle_hashes[None, :] + self._perm_b[:, None]) % _PRIME
        sig = permuted.min(axis=1)
        return cast("npt.NDArray[np.uint32]", sig.astype(np.uint32))

    def jaccard_similarity(
        self, sig1: npt.NDArray[np.uint32] | list[int], sig2: npt.NDArray[np.uint32] | list[int]
    ) -> float:
        """Estimate Jaccard similarity between two MinHash signatures.

        Accepts a numpy array (what :meth:`compute_minhash` returns) or
        a plain list (the historical shape), so existing callers keep
        working.
        """
        a = np.asarray(sig1)
        b = np.asarray(sig2)
        if a.size == 0 or b.size == 0:
            return 0.0
        matches = int(np.count_nonzero(a == b))
        return matches / float(a.size)

    @property
    def signatures(self) -> dict[Hashable, list[int]]:
        """The indexed signatures, as a plain dict (compatibility view).

        The index itself is a contiguous uint32 block; this property
        materialises the old dict-of-lists shape for callers that need
        it. Do NOT use it on a large index — it defeats the whole point
        of the compact storage.
        """
        return {item_id: self._signatures[row].tolist() for item_id, row in self._signature_rows.items()}

    def _append_signature(self, item_id: Hashable, sig: npt.NDArray[np.uint32]) -> None:
        """Append one signature to the contiguous block.

        The block grows geometrically so that appending stays amortised
        O(1) per item, while the memory stays one contiguous allocation
        instead of 1.8M separate Python lists.

        ``_signatures.shape[0]`` is the CAPACITY of the block, not the
        number of signatures in it, so the fill level is tracked
        separately — growing from the capacity would double the block
        on every reallocation.
        """
        if self._signature_count >= self._signatures.shape[0]:
            # Grow geometrically: double the capacity, keeping the block
            # contiguous. 64k rows is ~32 MB at 128 uint32 columns.
            new_rows = max(65_536, self._signatures.shape[0] * 2)
            grown = np.zeros((new_rows, self.num_perm), dtype=np.uint32)
            if self._signature_count:
                grown[: self._signature_count] = self._signatures[: self._signature_count]
            self._signatures = grown
        self._signatures[self._signature_count] = sig
        self._signature_rows[item_id] = self._signature_count
        self._signature_count += 1

    def insert(self, item_id: Hashable, text: str) -> tuple[bool, Hashable | None, float]:
        """
        Check if text is a near-duplicate of an already indexed item.
        If duplicate found (Jaccard >= threshold), returns (True, duplicate_of_item_id, similarity).
        Otherwise, indexes the item and returns (False, None, 0.0).
        """
        if len(text.strip()) < 20:
            return False, None, 0.0

        sig = self.compute_minhash(text)
        candidates: set[Hashable] = set()

        # Query buckets for each band
        for band_idx in range(self.b):
            band_sig = tuple(sig[band_idx * self.r : (band_idx + 1) * self.r])
            bucket_key = _band_key(band_sig)
            bucket = self.buckets[band_idx]
            if bucket_key in bucket:
                candidates.update(bucket[bucket_key])

        # Verify exact signature similarity against candidates
        for cand_id in candidates:
            row = self._signature_rows.get(cand_id)
            if row is not None:
                cand_sig = self._signatures[row]
                sim = self.jaccard_similarity(sig, cand_sig)
                if sim >= self.threshold:
                    return True, cand_id, sim

        # Not a duplicate, insert into index
        self._append_signature(item_id, sig)
        for band_idx in range(self.b):
            band_sig = tuple(sig[band_idx * self.r : (band_idx + 1) * self.r])
            bucket_key = _band_key(band_sig)
            bucket = self.buckets[band_idx]
            if bucket_key not in bucket:
                bucket[bucket_key] = []
            bucket[bucket_key].append(item_id)

        return False, None, 0.0

    def deduplicate_messages(self, messages: list[CleanedMessage]) -> tuple[list[CleanedMessage], int]:
        """
        Deduplicate a list of messages in place with a progress bar.
        Returns list of unique messages and number of removed duplicates.
        """
        counter = [0]
        unique_msgs: list[CleanedMessage] = []
        for message in self._unique_stream(messages, counter, show_progress=True):
            unique_msgs.append(message)
        return unique_msgs, counter[0]

    def deduplicate_stream(
        self,
        messages: Iterable[CleanedMessage],
        *,
        show_progress: bool = False,
    ) -> Iterator[CleanedMessage]:
        """Yield the unique messages, dropping near-duplicates.

        The corpus is 2.8M messages, so the input and the output must
        not be held at the same time. The index itself is bounded by the
        signature block (512 B per indexed message).
        """
        counter = [0]
        yield from self._unique_stream(messages, counter, show_progress=show_progress)

    def _unique_stream(
        self,
        messages: Iterable[CleanedMessage],
        counter: list[int],
        *,
        show_progress: bool = False,
    ) -> Iterator[CleanedMessage]:
        """The shared implementation; ``counter[0]`` receives the duplicate count."""
        source = tqdm(messages, desc="MinHash LSH Dedup", unit="msg") if show_progress else messages
        for m in source:
            # Skip very short messages from fuzzy dedup to avoid over-filtering
            if len(m.text_clean.strip()) < 30:
                yield m
                continue

            is_dup, _dup_of, _sim = self.insert((m.chat_id, m.msg_id), m.text_clean)
            if is_dup:
                counter[0] += 1
                continue
            yield m

        logger.info(f"MinHash LSH deduplication: removed {counter[0]} near-duplicate messages.")
