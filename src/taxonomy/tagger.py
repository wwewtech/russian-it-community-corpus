"""
Multi-label technical keyword tagger and sentiment scoring analyzer.
"""

import logging
import re

from tqdm import tqdm

try:
    import ahocorasick

    HAS_AHO = True
except ImportError:
    HAS_AHO = False

from src.config import DOMAIN_TAXONOMY, SENTIMENT_DICT
from src.ingestion.schema import CleanedMessage
from src.taxonomy.classifier import DomainClassifier

logger = logging.getLogger(__name__)


class TechnicalTagger:
    """
    High-speed extractor of technical keyword tags, domain assignment, and sentiment.
    Uses Aho-Corasick algorithm for O(text_length + matches) multi-pattern matching.
    """

    def __init__(self) -> None:
        self.classifier = DomainClassifier()
        self.sentiment_dict = SENTIMENT_DICT

        # Build Aho-Corasick automaton for fast multi-keyword matching
        self.automaton = None
        if HAS_AHO:
            self.automaton = ahocorasick.Automaton()
            for domain, info in DOMAIN_TAXONOMY.items():
                for kw in info["keywords"]:
                    kw_lower = kw.lower()
                    # Store (keyword, domain) as value
                    self.automaton.add_word(kw_lower, (kw_lower, domain))
            self.automaton.make_automaton()
        else:
            # Fallback to set intersection
            self.all_keywords: dict[str, str] = {}
            for domain, info in DOMAIN_TAXONOMY.items():
                for kw in info["keywords"]:
                    self.all_keywords[kw.lower()] = domain
            self.all_keywords_set: set[str] = set(self.all_keywords.keys())
            logger.warning("ahocorasick not installed; falling back to set intersection (slower)")

    @property
    def all_keywords_set(self) -> set[str]:
        """Backward compatibility: return all keywords as a set."""
        if HAS_AHO and self.automaton is not None:
            # Rebuild from taxonomy for consistency
            return {kw.lower() for info in DOMAIN_TAXONOMY.values() for kw in info["keywords"]}
        return self._all_keywords_set_fallback

    @all_keywords_set.setter
    def all_keywords_set(self, value: set[str]) -> None:
        self._all_keywords_set_fallback = value

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
        """Extract matched technical keyword tags from text using Aho-Corasick."""
        if not text:
            return []
        text_lower = text.lower()

        if HAS_AHO and self.automaton is not None:
            matched = set()
            for end_index, (keyword, _domain) in self.automaton.iter(text_lower):
                start_index = end_index - len(keyword) + 1
                # Check word boundaries: character before and after must not be alphanumeric/_/-
                before_ok = (
                    start_index == 0
                    or not text_lower[start_index - 1].isalnum()
                    and text_lower[start_index - 1] not in "_-"
                )
                after_ok = (
                    end_index == len(text_lower) - 1
                    or not text_lower[end_index + 1].isalnum()
                    and text_lower[end_index + 1] not in "_-"
                )
                if before_ok and after_ok:
                    matched.add(keyword)
            return sorted(matched)

        # Fallback: set intersection
        tokens = set(re.findall(r"[a-zA-Zа-яё0-9_\-\+\#\.]+", text_lower))
        matched = tokens.intersection(self.all_keywords_set)
        return sorted(matched)

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
