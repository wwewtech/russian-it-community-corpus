"""
Multi-label technical keyword tagger and sentiment scoring analyzer.
"""

import logging
import re

from src.config import DOMAIN_TAXONOMY, SENTIMENT_DICT
from src.ingestion.schema import CleanedMessage
from src.taxonomy.classifier import DomainClassifier

logger = logging.getLogger(__name__)


class TechnicalTagger:
    """
    Extracts specific technical tags, assigns domain categories, and calculates sentiment scores.
    """

    def __init__(self):
        self.classifier = DomainClassifier()
        self.sentiment_dict = SENTIMENT_DICT

        # Flatten all known keywords for fast tagging
        self.all_keywords: dict[str, str] = {}
        for domain, info in DOMAIN_TAXONOMY.items():
            for kw in info["keywords"]:
                self.all_keywords[kw.lower()] = domain

    def compute_sentiment(self, text: str) -> int:
        """Calculate sentiment score based on lexicon matches."""
        if not text:
            return 0
        words = re.findall(r"\w+", text.lower())
        score = 0
        for w in words:
            if w in self.sentiment_dict:
                score += self.sentiment_dict[w]
        return score

    def extract_tags(self, text: str) -> list[str]:
        """Extract matched technical keyword tags from text."""
        if not text:
            return []
        text_lower = text.lower()
        words = set(re.findall(r"[a-zA-Zа-яё0-9_\-\+\#\.]+", text_lower))

        tags: set[str] = set()
        for kw in self.all_keywords:
            if kw in words or (len(kw) > 3 and f" {kw} " in f" {text_lower} "):
                tags.add(kw)

        return sorted(list(tags))

    def tag_message(self, msg: CleanedMessage) -> CleanedMessage:
        """
        Enrich a CleanedMessage with domain, tags, and sentiment score.
        """
        domain, conf, match_counts = self.classifier.classify_text(msg.text_clean)
        tags = self.extract_tags(msg.text_clean)
        sentiment = self.compute_sentiment(msg.text_clean)

        msg.domain = domain
        msg.tags = tags
        msg.sentiment_score = sentiment
        return msg

    def tag_batch(self, messages: list[CleanedMessage]) -> list[CleanedMessage]:
        """
        Tag a batch of messages.
        """
        for m in messages:
            self.tag_message(m)
        return messages
