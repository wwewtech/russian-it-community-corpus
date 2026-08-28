"""
Dataset Drift Monitoring for the RICC data platform.

Compares a reference dataset snapshot against the current one and
quantifies drift with three standard data-quality metrics:

1. **PSI (Population Stability Index)** over the message-length
   distribution — the industry-standard drift score for tabular data
   (PSI < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant).
2. **Jensen–Shannon divergence** over the domain-label distribution —
   detects topical shifts (e.g. a community node changing focus).
3. **Top-k vocabulary Jaccard overlap** — detects lexical drift in the
   corpus language itself.

The monitor is dependency-free (numpy/pandas only) and emits a
machine-readable JSON verdict consumed by CI and the Prefect flow.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# PSI interpretation thresholds (industry convention)
PSI_STABLE = 0.10
PSI_MODERATE = 0.25

# JS divergence thresholds (bits, base-2 log)
JS_STABLE = 0.05
JS_MODERATE = 0.20

_TOKEN_RE = re.compile(r"[a-zA-Zа-яё0-9]{3,}")


def _psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    """Population Stability Index between two discrete distributions."""
    expected = expected / max(expected.sum(), 1)
    actual = actual / max(actual.sum(), 1)
    expected = np.clip(expected, eps, None)
    actual = np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence (base-2, bounded by 1.0 bit)."""
    p = p / max(p.sum(), 1)
    q = q / max(q.sum(), 1)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _psi_bucket_edges(reference_lengths: np.ndarray, n_buckets: int = 10) -> np.ndarray:
    """Quantile-based bucket edges from the reference distribution."""
    quantiles = np.linspace(0.0, 100.0, n_buckets + 1)[1:-1]
    edges = np.percentile(reference_lengths, quantiles)
    return np.unique(edges)


class DatasetDriftMonitor:
    """Compare a reference snapshot against the current dataset snapshot."""

    def __init__(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        text_column: str = "text_clean",
        domain_column: str = "domain",
        length_buckets: int = 10,
        vocab_top_k: int = 500,
    ):
        self.reference = reference_df
        self.current = current_df
        self.text_column = text_column
        self.domain_column = domain_column
        self.length_buckets = length_buckets
        self.vocab_top_k = vocab_top_k

    # -- individual metrics ---------------------------------------------------

    def compute_length_psi(self) -> float:
        """PSI over the message character-length distribution."""
        ref_len = self.reference[self.text_column].fillna("").str.len().to_numpy(dtype=float)
        cur_len = self.current[self.text_column].fillna("").str.len().to_numpy(dtype=float)
        edges = _psi_bucket_edges(ref_len, self.length_buckets)
        if len(edges) == 0:
            return 0.0
        ref_hist, _ = np.histogram(ref_len, bins=np.concatenate(([-np.inf], edges, [np.inf])))
        cur_hist, _ = np.histogram(cur_len, bins=np.concatenate(([-np.inf], edges, [np.inf])))
        return round(_psi(ref_hist.astype(float), cur_hist.astype(float)), 4)

    def compute_domain_drift(self) -> dict[str, Any]:
        """JS divergence + per-domain share deltas over the domain column."""
        ref_counts = self.reference[self.domain_column].fillna("unknown").value_counts()
        cur_counts = self.current[self.domain_column].fillna("unknown").value_counts()
        domains = sorted(set(ref_counts.index) | set(cur_counts.index))
        ref_vec = np.array([ref_counts.get(d, 0) for d in domains], dtype=float)
        cur_vec = np.array([cur_counts.get(d, 0) for d in domains], dtype=float)

        js = round(_js_divergence(ref_vec, cur_vec), 4)
        ref_share = ref_vec / max(ref_vec.sum(), 1)
        cur_share = cur_vec / max(cur_vec.sum(), 1)
        deltas = {d: round(float(cur_share[i] - ref_share[i]) * 100, 3) for i, d in enumerate(domains)}
        biggest = max(deltas, key=lambda d: (abs(deltas[d]), deltas[d])) if deltas else None
        return {
            "js_divergence_bits": js,
            "biggest_share_shift_domain": biggest,
            "biggest_share_shift_pp": deltas.get(biggest, 0.0) if biggest else 0.0,
            "per_domain_share_delta_pp": deltas,
        }

    def compute_vocabulary_drift(self) -> dict[str, Any]:
        """Jaccard overlap of the top-k most frequent tokens."""
        ref_tokens: Counter[str] = Counter()
        for text in self.reference[self.text_column].fillna(""):
            ref_tokens.update(_TOKEN_RE.findall(text.lower()))
        cur_tokens: Counter[str] = Counter()
        for text in self.current[self.text_column].fillna(""):
            cur_tokens.update(_TOKEN_RE.findall(text.lower()))

        ref_top = {t for t, _ in ref_tokens.most_common(self.vocab_top_k)}
        cur_top = {t for t, _ in cur_tokens.most_common(self.vocab_top_k)}
        if not ref_top and not cur_top:
            return {"jaccard_overlap": 1.0, "new_top_tokens": 0, "dropped_top_tokens": 0}
        union = ref_top | cur_top
        jaccard = len(ref_top & cur_top) / len(union) if union else 1.0
        return {
            "jaccard_overlap": round(jaccard, 4),
            "new_top_tokens": len(cur_top - ref_top),
            "dropped_top_tokens": len(ref_top - cur_top),
        }

    # -- full report ------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Compute all drift metrics and produce a verdict report."""
        length_psi = self.compute_length_psi()
        domain = self.compute_domain_drift()
        vocab = self.compute_vocabulary_drift()

        def _psi_verdict(v: float) -> str:
            if v < PSI_STABLE:
                return "stable"
            if v < PSI_MODERATE:
                return "moderate_drift"
            return "significant_drift"

        def _js_verdict(v: float) -> str:
            if v < JS_STABLE:
                return "stable"
            if v < JS_MODERATE:
                return "moderate_drift"
            return "significant_drift"

        def _vocab_verdict(v: float) -> str:
            if v >= 0.90:
                return "stable"
            if v >= 0.75:
                return "moderate_drift"
            return "significant_drift"

        metrics = {
            "length_psi": {"value": length_psi, "verdict": _psi_verdict(length_psi)},
            "domain_distribution": {**domain, "verdict": _js_verdict(domain["js_divergence_bits"])},
            "vocabulary": {**vocab, "verdict": _vocab_verdict(vocab["jaccard_overlap"])},
        }
        any_significant = any(m["verdict"] == "significant_drift" for m in metrics.values())
        any_moderate = any(m["verdict"] == "moderate_drift" for m in metrics.values())
        overall = "significant_drift" if any_significant else ("moderate_drift" if any_moderate else "stable")

        return {
            "report_title": "DATASET DRIFT MONITORING REPORT",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "reference_rows": len(self.reference),
            "current_rows": len(self.current),
            "thresholds": {
                "psi": {"stable": PSI_STABLE, "moderate": PSI_MODERATE},
                "js_divergence_bits": {"stable": JS_STABLE, "moderate": JS_MODERATE},
                "vocab_jaccard": {"stable": 0.90, "moderate": 0.75},
            },
            "metrics": metrics,
            "overall_verdict": overall,
        }

    def generate_report(self, output_path: str | Path) -> Path:
        """Run the monitor and persist the JSON report."""
        report = self.run()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return out
