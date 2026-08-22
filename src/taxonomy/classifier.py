"""
Domain Classifier for Russian IT Community Messages and Dialogues.
"""

import logging
import re
from collections import defaultdict

from src.config import DOMAIN_TAXONOMY

logger = logging.getLogger(__name__)


class DomainClassifier:
    """
    Classifies technical chat messages and dialogues into IT domains based on taxonomy dictionary.
    """

    def __init__(self, taxonomy: dict[str, dict[str, any]] = DOMAIN_TAXONOMY):
        self.taxonomy = taxonomy
        # Compile word-boundary regex patterns for fast matching
        self.domain_patterns: dict[str, list[re.Pattern]] = {}
        for domain, info in taxonomy.items():
            patterns = []
            for kw in info["keywords"]:
                # Escaped keyword with word boundaries or punctuation boundaries
                pat = re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)", re.IGNORECASE)
                patterns.append(pat)
            self.domain_patterns[domain] = patterns

    def classify_text(self, text: str) -> tuple[str, float, dict[str, int]]:
        """
        Classify text and return (best_domain, confidence_score, domain_match_counts).
        """
        if not text:
            return "general_tech_chat", 0.0, {}

        scores: dict[str, int] = defaultdict(int)

        for domain, patterns in self.domain_patterns.items():
            for pat in patterns:
                matches = len(pat.findall(text))
                if matches > 0:
                    scores[domain] += matches

        if not scores:
            return "general_tech_chat", 0.0, {}

        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_domain, best_count = sorted_domains[0]
        total_matches = sum(scores.values())
        confidence = round(best_count / total_matches, 3)

        return best_domain, confidence, dict(scores)
