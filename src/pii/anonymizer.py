"""
Unified PII Anonymizer combining Deterministic RegEx and NLP/NER scrubbers with consistent user pseudonymization.
"""

import logging

from src.ingestion.schema import CleanedMessage, NormalizedMessage
from src.pii.deep_anonymizer import DeepPIIAnonymizer

logger = logging.getLogger(__name__)


class UnifiedPIIAnonymizer(DeepPIIAnonymizer):
    """
    Unified PII Anonymizer inheriting full morphological declension,
    database URL masking, deterministic regex rules, and Natasha NER.
    """

    def __init__(self, enable_ner: bool = True):
        super().__init__(enable_ner=enable_ner)

    def get_user_pseudonym(self, raw_user_id: str, raw_author_name: str) -> tuple[str, str]:
        """Get or assign a consistent pseudonym for a user ID and display name."""
        return self._get_or_create_pseudonym(raw_user_id, raw_author_name)

    def anonymize_text(self, text: str) -> str:
        """Anonymize text using multi-pass scrubbers."""
        return self.scrub_text(text)

    def process_message(self, msg: NormalizedMessage) -> CleanedMessage:
        """Anonymize and clean a single NormalizedMessage."""
        return self.anonymize_message(msg)

    def process_batch(self, messages: list[NormalizedMessage]) -> list[CleanedMessage]:
        """Process a list of NormalizedMessages in batch with author name harvesting."""
        return self.process_all(messages)

    def get_stats_summary(self) -> dict[str, int]:
        """Get summary of anonymization stats."""
        return dict(self.stats)
