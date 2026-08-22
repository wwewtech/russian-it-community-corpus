"""
Unified PII Anonymizer combining Deterministic RegEx and NLP/NER scrubbers with consistent user pseudonymization.
"""

import logging
import re

from src.ingestion.schema import CleanedMessage, NormalizedMessage
from src.pii.ner_scrubber import NERPIIScrubber
from src.pii.regex_scrubber import RegexPIIScrubber

logger = logging.getLogger(__name__)


class UnifiedPIIAnonymizer:
    """
    Two-tier PII anonymizer and user pseudonymization manager.
    """

    def __init__(self, enable_ner: bool = True):
        self.regex_scrubber = RegexPIIScrubber()
        self.ner_scrubber = NERPIIScrubber() if enable_ner else None

        # Consistent mapping from raw user identifier -> anonymized pseudonym
        self.user_id_map: dict[str, str] = {}
        self.author_name_map: dict[str, str] = {}
        self.username_handle_map: dict[str, str] = {}

        # Cumulative stats
        self.cumulative_stats: dict[str, int] = {
            "messages_processed": 0,
            "phones_scrubbed": 0,
            "emails_scrubbed": 0,
            "crypto_wallets_scrubbed": 0,
            "api_keys_scrubbed": 0,
            "jwt_tokens_scrubbed": 0,
            "ssh_keys_scrubbed": 0,
            "ip_addresses_scrubbed": 0,
            "private_invites_scrubbed": 0,
            "user_mentions_scrubbed": 0,
            "ner_persons_scrubbed": 0,
            "ner_locations_scrubbed": 0,
            "unique_authors_anonymized": 0,
        }

    def get_user_pseudonym(self, raw_user_id: str, raw_author_name: str) -> tuple[str, str]:
        """
        Get or assign a consistent pseudonym for a user ID and display name.
        """
        key = str(raw_user_id).strip()
        if not key or key.lower() in ("none", "unknown", "anonymous", "null"):
            key = str(raw_author_name).strip()

        if key not in self.user_id_map:
            new_idx = len(self.user_id_map) + 1
            anon_id = f"user_{new_idx:05d}"
            anon_name = f"Developer_{new_idx:05d}"
            self.user_id_map[key] = anon_id
            self.author_name_map[key] = anon_name
            self.cumulative_stats["unique_authors_anonymized"] += 1

        return self.user_id_map[key], self.author_name_map[key]

    def anonymize_text(self, text: str) -> str:
        """
        Anonymize text using regex and NER scrubbers.
        """
        if not text:
            return ""

        # Step 1: RegEx deterministic scrubbing
        clean_text, reg_stats = self.regex_scrubber.scrub(text, self.username_handle_map)

        # Update regex stats
        self.cumulative_stats["phones_scrubbed"] += reg_stats.get("phones", 0)
        self.cumulative_stats["emails_scrubbed"] += reg_stats.get("emails", 0)
        self.cumulative_stats["crypto_wallets_scrubbed"] += reg_stats.get("crypto_wallets", 0)
        self.cumulative_stats["api_keys_scrubbed"] += reg_stats.get("api_keys", 0)
        self.cumulative_stats["jwt_tokens_scrubbed"] += reg_stats.get("jwt_tokens", 0)
        self.cumulative_stats["ssh_keys_scrubbed"] += reg_stats.get("ssh_keys", 0)
        self.cumulative_stats["ip_addresses_scrubbed"] += reg_stats.get("ip_addresses", 0)
        self.cumulative_stats["private_invites_scrubbed"] += reg_stats.get("private_invites", 0)
        self.cumulative_stats["user_mentions_scrubbed"] += reg_stats.get("user_mentions", 0)

        # Step 2: NER scrubbing (if enabled and applicable)
        if self.ner_scrubber and self.ner_scrubber.enabled:
            clean_text, ner_stats = self.ner_scrubber.scrub(clean_text)
            self.cumulative_stats["ner_persons_scrubbed"] += ner_stats.get("ner_per", 0)
            self.cumulative_stats["ner_locations_scrubbed"] += ner_stats.get("ner_loc", 0)

        # Step 3: Whitespace normalization
        clean_text = re.sub(r"[ \t]+", " ", clean_text)
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
        return clean_text.strip()

    def process_message(self, msg: NormalizedMessage) -> CleanedMessage:
        """
        Anonymize and clean a single NormalizedMessage.
        """
        anon_id, anon_name = self.get_user_pseudonym(msg.author_id_raw, msg.author_raw)
        cleaned_text = self.anonymize_text(msg.text_raw)

        self.cumulative_stats["messages_processed"] += 1

        is_question = "?" in cleaned_text
        token_approx = int(len(cleaned_text.split()) * 1.35)

        return CleanedMessage(
            msg_id=msg.msg_id,
            chat_id=msg.chat_id,
            chat_name=msg.chat_name,
            timestamp=msg.timestamp.isoformat(),
            unixtime=msg.unixtime,
            author_anon=anon_name,
            author_id_anon=anon_id,
            text_clean=cleaned_text,
            reply_to_id=msg.reply_to_id,
            domain="general_tech_chat",
            tags=[],
            sentiment_score=0,
            token_count_approx=token_approx,
            is_question=is_question,
            thread_id=None,
        )

    def process_batch(self, messages: list[NormalizedMessage]) -> list[CleanedMessage]:
        """
        Process a list of NormalizedMessages in batch.
        """
        cleaned: list[CleanedMessage] = []
        for msg in messages:
            cleaned.append(self.process_message(msg))
        return cleaned

    def get_stats_summary(self) -> dict[str, int]:
        """Get summary of anonymization stats."""
        return dict(self.cumulative_stats)
