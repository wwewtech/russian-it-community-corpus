"""
Unified PII Anonymizer combining Deterministic RegEx and NLP/NER scrubbers with consistent user pseudonymization.
"""

import logging
from collections.abc import Iterable, Iterator

from ricc.ingestion.schema import CleanedMessage, NormalizedMessage
from ricc.pii.deep_anonymizer import DeepPIIAnonymizer

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

    def process_batch(self, messages: list[NormalizedMessage]) -> list[CleanedMessage]:
        """Process a list of NormalizedMessages in batch with author name harvesting."""
        return self.process_all(messages)

    def process_stream(
        self,
        messages: Iterable[NormalizedMessage],
        all_messages: Iterable[NormalizedMessage] | None = None,
    ) -> Iterator[CleanedMessage]:
        """Scrub a stream of messages without materialising it.

        The corpus is 2.8M messages; holding the input and the output
        lists at the same time costs several gigabytes. This variant
        yields the cleaned messages one by one.
        """
        return super().process_stream(messages, all_messages)

    def get_stats_summary(self) -> dict[str, int]:
        """Get summary of anonymization stats."""
        return dict(self.stats)
