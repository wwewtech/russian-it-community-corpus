"""
Probabilistic PII Audit on real production data.

The existing red-team suite (ricc/validation/pii_redteam.py) runs a fixed
25,000-message sample and reports a binary pass/fail. The senior review
flagged this as "точечные проверки" — a point check with no statistical
guarantee. This module upgrades the audit to a *probabilistic* one:

1. **Stratified sampling** — messages are sampled proportionally from every
   community node (chat_name stratum), so no source is silently
   under-represented.
2. **Leak-rate estimation with confidence bounds** — for every PII category
   we report the observed rate, the Wilson interval, and a one-sided upper
   bound at 99% confidence ("with 99% confidence the true leak rate does
   not exceed X").
3. **Power analysis** — ``required_sample_size`` tells the operator how many
   messages must be audited to certify a target leak rate with a target
   margin, so the sample size is derived, not guessed.
4. **Verdict with statistical guarantee** — PASSED only if the 99% upper
   bound on the *total* leak rate is below the configured tolerance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ricc.evaluation.statistical_power import (
    leak_rate_upper_bound,
    required_sample_size,
    wilson_interval,
)
from ricc.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Leak detectors (self-contained; mirrors the deterministic scrubber patterns)
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(
    r"(?<![\w=])(?:\+?[78]|\+?380|\+?995|\+?374|\+?998|\+?375|\+?1)"
    r"[\s\-\(]*\d[\s]*\d[\s]*\d[\s\-\)]*\d[\s]*\d[\s]*\d[\s\-]*\d[\s]*\d[\s\-]*\d[\s]*\d\b"
    r"|(?<![\w=])(?:\+\d{1,3}[\s\-]?)?\(?\d{3,4}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b")
_CRYPTO_RE = re.compile(
    r"\b(?:1[a-km-zA-HJ-NP-Z1-9]{25,34}|3[a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-z0-9]{39,59})\b"
    r"|\b0x[a-fA-F0-9]{40}\b"
    r"|\bT[A-Za-z1-9]{33}\b"
    r"|\b(?:EQ|UQ)[a-zA-Z0-9_-]{46}\b"
)
_API_KEY_RE = re.compile(
    r"\bsk-(?:proj-)?[a-zA-Z0-9_\-]{20,}\b"
    r"|\b(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{20,50}\b"
    r"|\bgithub_pat_[a-zA-Z0-9_]{30,}\b"
    r"|\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b"
    r"|\b\d{8,12}:[a-zA-Z0-9_\-]{25,45}\b"
)
_JWT_RE = re.compile(r"\beyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b")
_SSH_KEY_RE = re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+PRIVATE KEY-----")
_PUBLIC_IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)
_REDACTED_TOKEN_RE = re.compile(r"\[(?:[A-Z_]+)_REDACTED\]|@user_anon|@user_\d+|\[PERSON_REDACTED\]")

# --- Categories the deterministic scrubber handles but this audit missed -----
# The scrubber (ricc/pii/regex_scrubber.py) cleans strictly more categories
# than the audit used to verify. A category that is scrubbed but never
# audited cannot fail the audit, so "PASSED" did not mean "clean". The
# detectors below close that gap. See docs/architecture/overview.md.

# `api_key=...`, `password=...`, `access_token=...` — assignment style secrets.
_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
    r"client[_-]?secret|password|passwd|pwd|token)\s*[:=]\s*"
    r"[\"']?[A-Za-z0-9_\-\.\/\+]{8,}[\"']?",
    re.IGNORECASE,
)
# Private Telegram invite links: t.me/+hash, t.me/joinchat/hash.
_PRIVATE_INVITE_RE = re.compile(r"t\.me/(?:\+|joinchat/)[A-Za-z0-9_\-]+", re.IGNORECASE)
# Database connection URLs with credentials: postgres://user:pass@host/db
_DB_URL_RE = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|amqp|mssql|clickhouse)://"
    r"[^\s:@/]+:[^\s:@/]+@[^\s/]+",
    re.IGNORECASE,
)
# Telegram forward headers: "Forwarded from X", "Переслано от X".
_FORWARD_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(?:forwarded\s+from|переслано\s+от|forward\s+from)\s*[:\-]?\s*(?P<who>[^\n]{2,80})",
    re.IGNORECASE,
)
# Raw @username mentions. Already-anonymised handles (@user_1, @user_anon)
# are NOT leaks and are excluded by the negative lookahead.
_USER_MENTION_RE = re.compile(r"(?<![\w@])@(?!user_\d+\b|user_anon\b)[A-Za-z][A-Za-z0-9_]{3,31}\b")

# Loopback / well-known IPs and semantic versions are NOT leaks.
_BENIGN_IPS = {"127.0.0.1", "0.0.0.0", "1.1.1.1", "8.8.8.8", "8.8.4.4"}  # nosec B104
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

DETECTORS: dict[str, re.Pattern[str]] = {
    "phones": _PHONE_RE,
    "emails": _EMAIL_RE,
    "crypto_wallets": _CRYPTO_RE,
    "api_keys": _API_KEY_RE,
    "jwt_tokens": _JWT_RE,
    "ssh_keys": _SSH_KEY_RE,
    "public_ips": _PUBLIC_IP_RE,
    # Categories previously scrubbed but never audited.
    "secret_assignments": _SECRET_ASSIGNMENT_RE,
    "private_invites": _PRIVATE_INVITE_RE,
    "db_urls": _DB_URL_RE,
    "forward_headers": _FORWARD_HEADER_RE,
    "user_mentions": _USER_MENTION_RE,
}


def find_leaks_in_text(text: str) -> dict[str, list[str]]:
    """Return per-category lists of raw leak matches found in one message.

    Matches that are already-redacted placeholders (e.g. ``@user_12``,
    ``[PHONE_REDACTED]``) are not leaks and are filtered out.
    """
    if not text:
        return {}
    found: dict[str, list[str]] = {}
    for category, pattern in DETECTORS.items():
        matches = []
        for m in pattern.finditer(text):
            token = m.group(0)
            if category == "phones" and _VERSION_RE.match(token.strip()):
                continue
            if category == "public_ips":
                if token in _BENIGN_IPS:
                    continue
                parts = [int(p) for p in token.split(".")]
                if parts[0] == 0 or all(p < 10 for p in parts):
                    continue  # semantic-version-like octet tuple
            if category == "user_mentions" and _REDACTED_TOKEN_RE.fullmatch(token):
                continue
            if category == "forward_headers":
                who = m.group("who").strip()
                if not who or _REDACTED_TOKEN_RE.search(who):
                    continue
                # A forward header whose author was anonymised is not a leak.
                token = who
            matches.append(token)
        if matches:
            found[category] = matches
    return found


class ProbabilisticPIIAuditor:
    """Stratified probabilistic audit of a production Parquet corpus."""

    def __init__(
        self,
        parquet_path: str | Path,
        text_column: str = "text_clean",
        strata_column: str = "chat_name",
        seed: int = 42,
    ):
        self.parquet_path = Path(parquet_path)
        self.text_column = text_column
        self.strata_column = strata_column
        self.seed = seed

    # -- power analysis -----------------------------------------------------

    @staticmethod
    def required_sample_size(
        expected_leak_rate: float = 0.001,
        margin: float = 0.0005,
        confidence: float = 0.99,
    ) -> int:
        """Messages to audit to estimate the leak rate within +/- margin."""
        return required_sample_size(expected_leak_rate, margin, confidence)

    # -- sampling ------------------------------------------------------------

    def _stratified_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Proportional allocation across strata with independent draws per stratum.

        Each stratum is drawn with its OWN derived RNG, so the selection is
        a genuine probability sample rather than a systematic sample that
        repeats the same relative positions in every stratum (which would
        bias the estimate and invalidate the confidence interval).
        """
        if len(df) <= sample_size:
            return df
        strata = df.groupby(self.strata_column, dropna=False)
        # Proportional allocation with largest-remainder rounding so the
        # total exactly equals sample_size.
        counts = strata.size()
        raw = counts / counts.sum() * sample_size
        alloc = raw.astype(int)
        remainder = sample_size - alloc.sum()
        if remainder > 0:
            frac = (raw - alloc).sort_values(ascending=False)
            for stratum in frac.index[:remainder]:
                alloc[stratum] += 1

        parts = []
        for stratum, n in alloc.items():
            group = strata.get_group(stratum)
            n = min(n, len(group))
            if n > 0:
                # Independent RNG per stratum: a distinct, reproducible
                # draw for every stratum instead of one shared seed.
                rng = np.random.default_rng(self._stratum_seed(stratum))
                idx = rng.choice(len(group), size=n, replace=False)
                parts.append(group.iloc[np.sort(idx)])
        return pd.concat(parts) if parts else df.head(0)

    def _stratum_seed(self, stratum: object) -> int:
        """Derive a stable per-stratum seed from the base seed.

        ``hash()`` is salted per process (PYTHONHASHSEED), so the stratum
        name is folded into a deterministic digest instead.
        """
        digest = hashlib.blake2b(str(stratum).encode("utf-8"), digest_size=8).digest()
        return (self.seed + int.from_bytes(digest, "big")) % (2**32)

    # -- audit ----------------------------------------------------------------

    def run_audit(
        self,
        sample_size: int | None = None,
        confidence: float | None = None,
        max_leak_tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Run the stratified probabilistic audit and return the report dict.

        The defaults come from ``settings.pii_audit``, which is loaded
        from ``params.yaml`` — the single source of the audit
        parameters. They used to be duplicated literals here, in the CLI
        and in the Prefect flow, so editing ``params.yaml`` changed
        nothing at runtime.
        """
        audit_settings = settings.pii_audit
        if sample_size is None:
            sample_size = audit_settings.sample_size
        if confidence is None:
            confidence = audit_settings.confidence
        if max_leak_tolerance is None:
            max_leak_tolerance = audit_settings.max_leak_tolerance

        if not self.parquet_path.exists():
            return {
                "status": "SKIPPED_DATASET_MISSING",
                "dataset_path": str(self.parquet_path),
                "note": "Production parquet not found; nothing to audit.",
            }

        df = pd.read_parquet(self.parquet_path)

        # --- Fail closed on an unusable corpus -----------------------------
        # Previously a missing text column produced an empty string for
        # every row, so zero leaks were "found" and the audit reported
        # PASSED with a confidence statement — a green light on a corpus
        # that was never actually inspected. A missing column (or an
        # unreadable corpus) must be SKIPPED/FAILED, never PASSED.
        missing: list[str] = [c for c in (self.text_column, self.strata_column) if c not in df.columns]
        if missing:
            return {
                "status": "FAILED_MISSING_COLUMNS",
                "dataset_path": str(self.parquet_path),
                "verdict": "FAILED",
                "missing_columns": missing,
                "available_columns": list(df.columns),
                "note": (
                    f"Corpus does not contain the audited column(s) {missing}. "
                    "The audit cannot inspect these rows — reporting PASSED here would "
                    "certify a corpus that was never read. Refusing to certify."
                ),
            }

        total_corpus = len(df)
        sample = self._stratified_sample(df, sample_size)
        sampled = len(sample)

        # Two distinct quantities, kept apart on purpose:
        #   category_leaks[cat]      -> number of MESSAGES in which the
        #                               category leaked (the Bernoulli
        #                               trial the Wilson interval needs)
        #   category_instances[cat]  -> total number of raw matches
        #                               (diagnostics only; it may exceed
        #                               the message count and MUST NOT be
        #                               passed to wilson_interval)
        category_leaks: dict[str, int] = {name: 0 for name in DETECTORS}
        category_instances: dict[str, int] = {name: 0 for name in DETECTORS}
        leak_examples: dict[str, list[str]] = {name: [] for name in DETECTORS}
        messages_with_leaks = 0
        per_stratum: dict[str, dict[str, int]] = {}

        for row in sample.itertuples(index=False):
            text = getattr(row, self.text_column, None)
            text = "" if text is None or not isinstance(text, str) else text
            stratum = str(getattr(row, self.strata_column, "unknown"))
            stratum_stats = per_stratum.setdefault(stratum, {"messages": 0, "leaks": 0})
            stratum_stats["messages"] += 1

            found = find_leaks_in_text(text)
            if found:
                messages_with_leaks += 1
                stratum_stats["leaks"] += 1
            for category, matches in found.items():
                # One message counts as ONE leak for this category ...
                category_leaks[category] += 1
                # ... while every raw match is tracked for diagnostics.
                category_instances[category] += len(matches)
                leak_examples[category].extend(matches[:3])

        # Statistical estimates per category
        category_stats: dict[str, Any] = {}
        total_leak_instances = 0
        for category, count in category_leaks.items():
            total_leak_instances += category_instances[category]
            low, high = wilson_interval(count, sampled, confidence)
            upper = leak_rate_upper_bound(count, sampled, confidence)
            category_stats[category] = {
                "messages_with_leak": count,
                "leak_instances": category_instances[category],
                "observed_rate": round(count / sampled, 8) if sampled else 0.0,
                "ci_lower": round(low, 8),
                "ci_upper": round(high, 8),
                "upper_bound_99": round(upper, 8),
                "examples": leak_examples[category][:5],
            }

        total_low, total_high = wilson_interval(messages_with_leaks, sampled, confidence)
        total_upper = leak_rate_upper_bound(messages_with_leaks, sampled, confidence)

        passed = total_upper <= max_leak_tolerance
        report = {
            "report_title": "PROBABILISTIC PII AUDIT (stratified, confidence-bounded)",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "dataset_path": str(self.parquet_path),
            "corpus_size": total_corpus,
            "sampled_messages": sampled,
            "sampling": {
                "method": "stratified_proportional",
                "strata_column": self.strata_column,
                "seed": self.seed,
                "strata_count": len(per_stratum),
            },
            "power_analysis": {
                "required_sample_for_1e-3_rate_at_5e-4_margin_99pct": self.required_sample_size(),
                "audited_sample_size": sampled,
                "sufficient": sampled >= self.required_sample_size(),
            },
            "results": {
                "messages_with_leaks": messages_with_leaks,
                "total_leak_instances": total_leak_instances,
                "message_leak_rate_observed": round(messages_with_leaks / sampled, 8) if sampled else 0.0,
                "message_leak_rate_ci": [round(total_low, 8), round(total_high, 8)],
                "message_leak_rate_upper_bound_99": round(total_upper, 8),
                "categories": category_stats,
            },
            "per_stratum": per_stratum,
            "audited_categories": sorted(DETECTORS),
            "tolerance": {
                "max_message_leak_rate": max_leak_tolerance,
                "confidence": confidence,
            },
            "verdict": "PASSED" if passed else "LEAKS_DETECTED",
            "statistical_guarantee": (
                f"With {confidence:.0%} confidence the true message-level leak rate "
                f"does not exceed {total_upper:.2e} (tolerance {max_leak_tolerance:.0e}). "
                f"Measured over {len(DETECTORS)} leak categories: {', '.join(sorted(DETECTORS))}. "
                "Personal names and locations are anonymised by the NER pass and are not "
                "re-detected by these pattern detectors; see reports/DATASET_AND_ANALYTICS.md."
            ),
        }
        logger.info(
            "Probabilistic PII audit: %s (%d/%d messages leaked)", report["verdict"], messages_with_leaks, sampled
        )
        return report

    def generate_report(self, output_path: str | Path, **audit_kwargs: Any) -> Path:
        """Run the audit and persist the JSON report."""
        report = self.run_audit(**audit_kwargs)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return out
