"""
Statistical power utilities for benchmark reporting.

Provides dependency-free (stdlib-only) confidence-interval math so that
benchmark results published in reports/ carry honest uncertainty estimates
instead of bare point values. Previously the HumanEval/RuMMLU subsets were
N=8, which yields binomial confidence intervals of ±20% or wider — the
review flagged this as statistically meaningless. With the enlarged
subsets (N=40 / N=50) the Wilson intervals below are attached to every
published accuracy figure.
"""

from __future__ import annotations

import math


def z_score(confidence: float = 0.95) -> float:
    """Two-sided z-score for a given confidence level (e.g. 0.95 -> 1.95996...).

    Implemented via bisection over ``math.erf`` — no scipy dependency.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    target = 1.0 - (1.0 - confidence) / 2.0  # one-sided CDF target
    lo, hi = 0.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (lower, upper) as fractions in [0, 1]. Unlike the naive
    Wald interval, Wilson keeps the bounds inside [0, 1] and behaves
    correctly for small samples and extreme proportions (e.g. 0/N).
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be within [0, total]")

    z = z_score(confidence)
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def leak_rate_upper_bound(leaks: int, sampled: int, confidence: float = 0.99) -> float:
    """One-sided upper bound on the true leak rate after auditing a sample.

    Uses the one-sided normal approximation with a continuity-free
    variance term; for the tiny leak counts expected in a clean corpus
    this is conservative (the exact Clopper–Pearson bound is slightly
    tighter, so this never understates risk).
    """
    if sampled <= 0:
        raise ValueError("sampled must be positive")
    if not 0 <= leaks <= sampled:
        raise ValueError("leaks must be within [0, sampled]")

    # One-sided z: P(Z <= z) = confidence
    target = confidence
    lo, hi = 0.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < target:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2.0

    p = leaks / sampled
    z2 = z * z
    denom = 1.0 + z2 / sampled
    upper = (p + z2 / (2.0 * sampled) + z * math.sqrt(p * (1.0 - p) / sampled + z2 / (4.0 * sampled * sampled))) / denom
    return min(1.0, upper)


def required_sample_size(expected_rate: float, margin: float, confidence: float = 0.95) -> int:
    """Minimum sample size to estimate a proportion within +/- ``margin``.

    Standard Cochran formula n = z^2 * p * (1 - p) / E^2, rounded up.
    """
    if not 0.0 <= expected_rate <= 1.0:
        raise ValueError("expected_rate must be in [0, 1]")
    if margin <= 0.0:
        raise ValueError("margin must be positive")
    z = z_score(confidence)
    p = expected_rate if 0.0 < expected_rate < 1.0 else 0.5
    n = (z * z) * p * (1.0 - p) / (margin * margin)
    return max(1, math.ceil(n))


def ci_half_width_pct(successes: int, total: int, confidence: float = 0.95) -> float:
    """Half-width of the Wilson interval expressed in percentage points."""
    low, high = wilson_interval(successes, total, confidence)
    return round((high - low) / 2.0 * 100.0, 1)
