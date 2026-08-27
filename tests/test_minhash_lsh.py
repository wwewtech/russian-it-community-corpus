"""Tests for MinHash LSH deduplication (src/deduplication/minhash_lsh.py)."""

from __future__ import annotations

from src.deduplication.minhash_lsh import MinHashLSH
from src.ingestion.schema import CleanedMessage

LONG_TEXT_A = "Как настроить streaming replication в PostgreSQL с wal_level replica и слотами"
LONG_TEXT_B = "Совершенно другой разговор про python asyncio gil и блокировки потоков"


def _msg(msg_id: int, text: str) -> CleanedMessage:
    return CleanedMessage(
        msg_id=msg_id,
        chat_id=1,
        chat_name="community_node_01",
        timestamp="2026-01-01T00:00:00",
        unixtime=1767225600,
        author_raw="user",
        author_id_raw="user",
        author_anon="Developer_00001",
        author_id_anon="Developer_00001",
        text_clean=text,
    )


class TestShingles:
    def test_short_text_single_shingle(self):
        lsh = MinHashLSH(shingle_size=3)
        assert lsh._get_shingles("один два") == {"один два"}

    def test_empty_text(self):
        lsh = MinHashLSH(shingle_size=3)
        assert lsh._get_shingles("") == set()

    def test_shingle_count(self):
        lsh = MinHashLSH(shingle_size=3)
        shingles = lsh._get_shingles("один два три четыре пять")
        assert len(shingles) == 3  # 5 words - shingle_size + 1


class TestMinHashSignature:
    def test_signature_length(self):
        lsh = MinHashLSH(num_perm=32)
        sig = lsh.compute_minhash(LONG_TEXT_A)
        assert len(sig) == 32

    def test_empty_text_zero_signature(self):
        lsh = MinHashLSH(num_perm=16)
        assert lsh.compute_minhash("") == [0] * 16

    def test_identical_texts_same_signature(self):
        lsh = MinHashLSH(num_perm=32)
        assert lsh.compute_minhash(LONG_TEXT_A) == lsh.compute_minhash(LONG_TEXT_A)


class TestJaccardSimilarity:
    def test_identical_signatures(self):
        lsh = MinHashLSH(num_perm=32)
        sig = lsh.compute_minhash(LONG_TEXT_A)
        assert lsh.jaccard_similarity(sig, sig) == 1.0

    def test_empty_signatures(self):
        lsh = MinHashLSH(num_perm=32)
        assert lsh.jaccard_similarity([], []) == 0.0


class TestInsert:
    def test_short_text_skipped(self):
        lsh = MinHashLSH()
        is_dup, dup_of, sim = lsh.insert(1, "короткий")
        assert is_dup is False
        assert dup_of is None
        assert sim == 0.0
        assert 1 not in lsh.signatures  # not indexed

    def test_first_insert_not_duplicate(self):
        lsh = MinHashLSH()
        is_dup, dup_of, _ = lsh.insert(1, LONG_TEXT_A)
        assert is_dup is False
        assert dup_of is None
        assert 1 in lsh.signatures

    def test_identical_text_is_duplicate(self):
        lsh = MinHashLSH()
        lsh.insert(1, LONG_TEXT_A)
        is_dup, dup_of, sim = lsh.insert(2, LONG_TEXT_A)
        assert is_dup is True
        assert dup_of == 1
        assert sim >= lsh.threshold

    def test_different_text_not_duplicate(self):
        lsh = MinHashLSH()
        lsh.insert(1, LONG_TEXT_A)
        is_dup, dup_of, _ = lsh.insert(2, LONG_TEXT_B)
        assert is_dup is False
        assert dup_of is None


class TestDeduplicateMessages:
    def test_removes_duplicates(self):
        lsh = MinHashLSH()
        msgs = [_msg(1, LONG_TEXT_A), _msg(2, LONG_TEXT_B), _msg(3, LONG_TEXT_A)]
        unique, removed = lsh.deduplicate_messages(msgs)
        assert removed == 1
        assert [m.msg_id for m in unique] == [1, 2]

    def test_short_messages_always_kept(self):
        lsh = MinHashLSH()
        msgs = [_msg(1, "ок"), _msg(2, "ок")]
        unique, removed = lsh.deduplicate_messages(msgs)
        assert removed == 0
        assert len(unique) == 2

    def test_empty_list(self):
        lsh = MinHashLSH()
        unique, removed = lsh.deduplicate_messages([])
        assert unique == []
        assert removed == 0
