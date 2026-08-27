"""Tests for MinHash LSH deduplication (src/deduplication/minhash_lsh.py)."""

from __future__ import annotations

import pytest

from src.deduplication.minhash_lsh import DedupStats, MinHashLSHDeduplicator


class TestConstructor:
    def test_invalid_band_division_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            MinHashLSHDeduplicator(num_perm=100, bands=16)

    def test_valid_config(self):
        d = MinHashLSHDeduplicator(num_perm=128, bands=16)
        assert d.rows_per_band == 8
        assert d.stats.total_seen == 0


class TestIsDuplicate:
    def test_identical_text_is_duplicate(self):
        d = MinHashLSHDeduplicator(num_perm=64, bands=8)
        assert d.is_duplicate("Как настроить nginx reverse proxy в docker") is False
        assert d.is_duplicate("Как настроить nginx reverse proxy в docker") is True

    def test_different_text_is_kept(self):
        d = MinHashLSHDeduplicator(num_perm=64, bands=8)
        d.is_duplicate("Совершенно первый уникальный текст про postgres индексы")
        assert d.is_duplicate("Абсолютно другой разговор про python asyncio и gil") is False

    def test_stats_counters(self):
        d = MinHashLSHDeduplicator(num_perm=64, bands=8)
        d.is_duplicate("текст один")
        d.is_duplicate("текст один")
        d.is_duplicate("текст два совершенно иной")
        assert d.stats.total_seen == 3
        assert d.stats.duplicates_removed == 1
        assert d.stats.kept == 2


class TestDeduplicate:
    def test_keeps_first_occurrences(self):
        d = MinHashLSHDeduplicator(num_perm=64, bands=8)
        texts = ["уникальный текст alpha", "уникальный текст beta", "уникальный текст alpha"]
        kept = d.deduplicate(texts)
        assert kept == ["уникальный текст alpha", "уникальный текст beta"]

    def test_empty_list(self):
        d = MinHashLSHDeduplicator(num_perm=64, bands=8)
        assert d.deduplicate([]) == []


class TestSummary:
    def test_shape_and_values(self):
        d = MinHashLSHDeduplicator(num_perm=64, bands=8)
        d.is_duplicate("текст для сводки")
        s = d.summary()
        assert s["total_seen"] == 1
        assert s["kept"] == 1
        assert s["duplicates_removed"] == 0
        assert s["buckets"] > 0
        assert "elapsed_sec" in s


class TestDedupStats:
    def test_defaults(self):
        s = DedupStats()
        assert s.total_seen == 0
        assert s.duplicates_removed == 0
        assert s.kept == 0
        assert s.buckets == 0
        assert s.elapsed_sec == 0.0