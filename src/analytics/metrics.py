"""
Statistical and NLP metrics calculations for conversational datasets.
"""

import math
import re
import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Any

import tiktoken
from tiktoken import Encoding

from src.config import SENTIMENT_DICT

# Try to initialize tiktoken cl100k_base for precise BPE token metrics
try:
    TIKTOKEN_ENC: Encoding | None = tiktoken.get_encoding("cl100k_base")
except Exception:
    TIKTOKEN_ENC = None


def count_tokens(text: str) -> int:
    """Accurately count BPE tokens using tiktoken (cl100k_base) or 1.35x word multiplier."""
    if not text:
        return 0
    if TIKTOKEN_ENC is not None:
        try:
            return len(TIKTOKEN_ENC.encode(text))
        except Exception:
            pass
    return max(1, int(len(text.split()) * 1.35))


def compute_shannon_entropy(word_counts: Counter[str]) -> float:
    """Calculate Shannon Lexical Diversity / Entropy: H = -sum(p * log2(p))."""
    total = sum(word_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in word_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 3)


def compute_percentiles(values: Sequence[float]) -> dict[str, float]:
    """Compute standard percentiles (p25, p50, p75, p90, p95, p99) and statistics."""
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    sorted_v = sorted(values)
    n = len(sorted_v)

    def _pct(p: float) -> float:
        idx = min(n - 1, max(0, int(p * n)))
        return sorted_v[idx]

    return {
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "std": round(statistics.stdev(values) if n > 1 else 0.0, 2),
        "min": round(sorted_v[0], 2),
        "max": round(sorted_v[-1], 2),
        "p25": round(_pct(0.25), 2),
        "p75": round(_pct(0.75), 2),
        "p90": round(_pct(0.90), 2),
        "p95": round(_pct(0.95), 2),
        "p99": round(_pct(0.99), 2),
    }


def analyze_sentiment(texts: list[str]) -> dict[str, Any]:
    """Calculate aggregate sentiment scores and sentiment distribution."""
    if not texts:
        return {"average": 0.0, "positive": 0, "negative": 0, "neutral": 0}

    scores = []
    pos = 0
    neg = 0
    neu = 0

    for t in texts:
        words = re.findall(r"\w+", t.lower())
        s = sum(SENTIMENT_DICT.get(w, 0) for w in words)
        scores.append(s)
        if s > 0:
            pos += 1
        elif s < 0:
            neg += 1
        else:
            neu += 1

    total = len(texts)
    return {
        "average": round(statistics.mean(scores), 3) if scores else 0.0,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "pos_ratio": round(pos / total * 100, 2) if total else 0.0,
        "neg_ratio": round(neg / total * 100, 2) if total else 0.0,
        "neu_ratio": round(neu / total * 100, 2) if total else 0.0,
    }
