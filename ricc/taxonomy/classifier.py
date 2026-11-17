"""
Domain Classifier for Russian IT Community Messages and Dialogues.
"""

import logging
import re
from collections import defaultdict
from typing import Any

from ricc.settings import settings

logger = logging.getLogger(__name__)


class DomainClassifier:
    """
    High-throughput Domain Classifier using token set intersections.
    """

    def __init__(self, taxonomy: dict[str, dict[str, Any]] | None = None):
        self.taxonomy = taxonomy or settings.DOMAIN_TAXONOMY
        self.kw_to_domains: dict[str, list[str]] = defaultdict(list)
        for domain, info in self.taxonomy.items():
            for kw in info["keywords"]:
                self.kw_to_domains[kw.lower()].append(domain)
        self.all_keywords_set: set[str] = set(self.kw_to_domains.keys())
        # Multi-word keywords ("react native", "github actions", ...) can
        # never appear as a single token of the word tokenizer below; they
        # are matched as phrases over the lowercased text instead. Without
        # this, those keys were dead and the affected domains were
        # systematically under-counted.
        self._multiword_keywords: list[tuple[str, list[str]]] = [
            (kw, kw.split()) for kw in self.all_keywords_set if " " in kw
        ]

    def classify_text(self, text: str) -> tuple[str, float, dict[str, int]]:
        """
        Classify text and return (best_domain, confidence_score, domain_match_counts).
        """
        if not text:
            return "general_tech_chat", 0.0, {}

        lowered = text.lower()
        tokens = set(re.findall(r"[a-zA-Zа-яё0-9_\-\+\#\.]+", lowered))
        matched_kws = tokens.intersection(self.all_keywords_set)
        # Phrase pass for multi-word keywords: the phrase must appear as a
        # contiguous sequence, so "reactive native" never matches "react native".
        for kw, parts in self._multiword_keywords:
            if kw in matched_kws:
                continue
            if all(f" {part} " in f" {lowered} " for part in parts):
                matched_kws.add(kw)

        if not matched_kws:
            return "general_tech_chat", 0.0, {}

        scores: dict[str, int] = defaultdict(int)
        for kw in matched_kws:
            for domain in self.kw_to_domains[kw]:
                scores[domain] += 1

        # Deterministic ordering: higher score first, ties broken alphabetically by domain name.
        # Sorting on score alone would make the winner depend on dict iteration order,
        # which varies with PYTHONHASHSEED and breaks corpus reproducibility.
        sorted_domains = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        best_domain, best_count = sorted_domains[0]
        total_matches = sum(scores.values())
        confidence = round(best_count / total_matches, 3)

        return best_domain, confidence, dict(scores)
