"""
Multi-label technical keyword tagger and sentiment scoring analyzer.
"""

import logging
import re

from tqdm import tqdm

from src.config import DOMAIN_TAXONOMY, SENTIMENT_DICT
from src.ingestion.schema import CleanedMessage
from src.taxonomy.classifier import DomainClassifier

logger = logging.getLogger(__name__)


class TechnicalTagger:
    """
    High-speed extractor of technical keyword tags, domain assignment, and sentiment.
    """

    def __init__(self) -> None:
        self.classifier = DomainClassifier()
        self.sentiment_dict = SENTIMENT_DICT

        # Flatten all known keywords for fast set intersection
        self.all_keywords: dict[str, str] = {}
        for domain, info in DOMAIN_TAXONOMY.items():
            for kw in info["keywords"]:
                self.all_keywords[kw.lower()] = domain
        self.all_keywords_set: set[str] = set(self.all_keywords.keys())

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
        """Extract matched technical keyword tags from text via set intersection."""
        if not text:
            return []
        tokens = set(re.findall(r"[a-zA-Zа-яё0-9_\-\+\#\.]+", text.lower()))
        matched = tokens.intersection(self.all_keywords_set)
        return sorted(list(matched))

    def tag_message(self, msg: CleanedMessage) -> CleanedMessage:
        """
        Enrich a CleanedMessage with domain, tags, and sentiment score.
        """
        domain, _, _ = self.classifier.classify_text(msg.text_clean)
        tags = self.extract_tags(msg.text_clean)
        sentiment = self.compute_sentiment(msg.text_clean)

        msg.domain = domain
        msg.tags = tags
        msg.sentiment_score = sentiment
        return msg

    def tag_batch(self, messages: list[CleanedMessage]) -> list[CleanedMessage]:
        """
        Tag a batch of messages with a progress bar.
        """
        for m in tqdm(messages, desc="Domain & Tech Tagging", unit="msg"):
            self.tag_message(m)
        return messages
