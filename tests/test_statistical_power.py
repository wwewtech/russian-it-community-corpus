"""Tests for the dependency-free statistical power utilities."""

from __future__ import annotations

import pytest

from src.evaluation.statistical_power import (
    ci_half_width_pct,
    leak_rate_upper_bound,
    required_sample_size,
    wilson_interval,
    z_score,
)


class TestZScore:
    def test_known_values(self):
        assert abs(z_score(0.95) - 1.959964) < 1e-4
        assert abs(z_score(0.99) - 2.575829) < 1e-4
        assert abs(z_score(0.90) - 1.644854) < 1e-4

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            z_score(0.0)
        with pytest.raises(ValueError):
            z_score(1.0)


class TestWilsonInterval:
    def test_known_value(self):
        # 5/10 successes at 95%: Wilson interval is approximately (0.2366, 0.7634)
        low, high = wilson_interval(5, 10, 0.95)
        assert abs(low - 0.2366) < 1e-3
        assert abs(high - 0.7634) < 1e-3

    def test_zero_successes_lower_bound_is_zero(self):
        low, high = wilson_interval(0, 100, 0.95)
        assert low == 0.0
        assert 0 < high < 0.05

    def test_all_successes_upper_bound_is_one(self):
        low, high = wilson_interval(100, 100, 0.95)
        assert high == 1.0
        assert 0.95 < low < 1.0

    def test_bounds_within_unit_interval(self):
        for successes in (0, 1, 3, 7, 10):
            low, high = wilson_interval(successes, 10, 0.99)
            assert 0.0 <= low <= high <= 1.0

    def test_wider_interval_for_higher_confidence(self):
        low95, high95 = wilson_interval(5, 20, 0.95)
        low99, high99 = wilson_interval(5, 20, 0.99)
        assert high99 - low99 > high95 - low95

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            wilson_interval(5, 0)
        with pytest.raises(ValueError):
            wilson_interval(11, 10)


class TestLeakRateUpperBound:
    def test_zero_leaks_small_sample(self):
        # 0 leaks in 1000 messages at 99%: upper bound ~ 0.0046
        upper = leak_rate_upper_bound(0, 1000, 0.99)
        assert 0.003 < upper < 0.007

    def test_zero_leaks_large_sample_is_tight(self):
        upper_1k = leak_rate_upper_bound(0, 1_000, 0.99)
        upper_100k = leak_rate_upper_bound(0, 100_000, 0.99)
        assert upper_100k < upper_1k

    def test_more_leaks_raise_bound(self):
        assert leak_rate_upper_bound(5, 1000, 0.99) > leak_rate_upper_bound(0, 1000, 0.99)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            leak_rate_upper_bound(0, 0)
        with pytest.raises(ValueError):
            leak_rate_upper_bound(11, 10)


class TestRequiredSampleSize:
    def test_known_value(self):
        # p=0.5, margin=0.05, 95% -> n = 1.96^2 * 0.25 / 0.0025 = 384.16 -> 385
        assert required_sample_size(0.5, 0.05, 0.95) == 385

    def test_smaller_margin_needs_more_samples(self):
        assert required_sample_size(0.5, 0.01, 0.95) > required_sample_size(0.5, 0.05, 0.95)

    def test_extreme_rate_uses_conservative_p(self):
        # p=0 or p=1 falls back to p=0.5 (most conservative)
        assert required_sample_size(0.0, 0.05, 0.95) == 385
        assert required_sample_size(1.0, 0.05, 0.95) == 385

    def test_invalid_margin_raises(self):
        with pytest.raises(ValueError):
            required_sample_size(0.5, 0.0)


class TestCiHalfWidthPct:
    def test_returns_percentage_points(self):
        half = ci_half_width_pct(5, 10, 0.95)
        assert 0 < half < 50

    def test_larger_sample_narrower_ci(self):
        assert ci_half_width_pct(4, 40, 0.95) < ci_half_width_pct(1, 10, 0.95)
