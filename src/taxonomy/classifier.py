"""
Domain Classifier for Russian IT Community Messages and Dialogues.
"""

import logging
import re
from collections import defaultdict
from typing import Any

from src.config import DOMAIN_TAXONOMY

logger = logging.getLogger(__name__)


class DomainClassifier:
    """
    High-throughput Domain Classifier using token set intersections.
    """

    def __init__(self, taxonomy: dict[str, dict[str, Any]] = DOMAIN_TAXONOMY):
        self.taxonomy = taxonomy
        self.kw_to_domains: dict[str, list[str]] = defaultdict(list)
        for domain, info in taxonomy.items():
            for kw in info["keywords"]:
                self.kw_to_domains[kw.lower()].append(domain)
        self.all_keywords_set: set[str] = set(self.kw_to_domains.keys())

    def classify_text(self, text: str) -> tuple[str, float, dict[str, int]]:
        """
        Classify text and return (best_domain, confidence_score, domain_match_counts).
        """
        if not text:
            return "general_tech_chat", 0.0, {}

        tokens = set(re.findall(r"[a-zA-Zа-яё0-9_\-\+\#\.]+", text.lower()))
        matched_kws = tokens.intersection(self.all_keywords_set)

        if not matched_kws:
            return "general_tech_chat", 0.0, {}

        scores: dict[str, int] = defaultdict(int)
        for kw in matched_kws:
            for domain in self.kw_to_domains[kw]:
                scores[domain] += 1

        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_domain, best_count = sorted_domains[0]
        total_matches = sum(scores.values())
        confidence = round(best_count / total_matches, 3)

        return best_domain, confidence, dict(scores)
