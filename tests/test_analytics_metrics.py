"""Tests for statistical metrics in src/analytics/metrics.py."""

from __future__ import annotations

from collections import Counter

from src.analytics.metrics import (
    analyze_sentiment,
    compute_percentiles,
    compute_shannon_entropy,
    count_tokens,
)


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_returns_positive_for_text(self):
        assert count_tokens("Как настроить репликацию PostgreSQL?") > 0

    def test_fallback_multiplier(self):
        # With tiktoken unavailable the fallback is ~1.35 tokens per word.
        text = "one two three four"
        result = count_tokens(text)
        assert result >= len(text.split())  # never fewer tokens than words


class TestComputeShannonEntropy:
    def test_empty_counter(self):
        assert compute_shannon_entropy(Counter()) == 0.0

    def test_single_symbol_is_zero(self):
        assert compute_shannon_entropy(Counter({"a": 10})) == 0.0

    def test_uniform_two_symbols_is_one_bit(self):
        assert compute_shannon_entropy(Counter({"a": 1, "b": 1})) == 1.0

    def test_bounded_by_log2_k(self):
        counts = Counter({"a": 8, "b": 2})
        assert compute_shannon_entropy(counts) < 1.0

    def test_rounded_to_three_decimals(self):
        result = compute_shannon_entropy(Counter({"a": 3, "b": 1}))
        assert result == round(result, 3)


class TestComputePercentiles:
    def test_empty_list_returns_zeroes(self):
        stats = compute_percentiles([])
        assert stats["mean"] == 0.0
        assert stats["p99"] == 0.0
        assert len(stats) == 10

    def test_known_values(self):
        stats = compute_percentiles([1.0, 2.0, 3.0, 4.0])
        assert stats["min"] == 1.0
        assert stats["max"] == 4.0
        assert stats["mean"] == 2.5
        assert stats["median"] == 2.5

    def test_percentiles_monotonic(self):
        stats = compute_percentiles([float(i) for i in range(1, 101)])
        assert stats["p25"] <= stats["p75"] <= stats["p90"] <= stats["p99"]

    def test_single_value(self):
        stats = compute_percentiles([5.0])
        assert stats["mean"] == 5.0
        assert stats["std"] == 0.0


class TestAnalyzeSentiment:
    def test_empty_list(self):
        result = analyze_sentiment([])
        assert result["average"] == 0.0
        assert result["positive"] == 0

    def test_positive_message(self):
        result = analyze_sentiment(["Всё отлично работает, спасибо!"])
        assert result["positive"] == 1
        assert result["average"] > 0

    def test_negative_message(self):
        result = analyze_sentiment(["Ужас, всё падает и тормозит"])
        assert result["negative"] == 1
        assert result["average"] < 0

    def test_neutral_message(self):
        result = analyze_sentiment(["Обычное сообщение без эмоций"])
        assert result["neutral"] == 1
        assert result["average"] == 0.0

    def test_ratios_sum_to_hundred(self):
        texts = ["отлично", "ужас", "нейтральный текст"]
        result = analyze_sentiment(texts)
        total = result["pos_ratio"] + result["neg_ratio"] + result["neu_ratio"]
        assert total == 100.0
