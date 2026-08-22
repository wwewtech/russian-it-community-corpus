"""
Ultra-Deep Morphological Case-Aware PII Anonymization Engine.
Performs dynamic name extraction, Russian grammatical declension (6 cases),
regex redaction, and Natasha NER with comprehensive whitelisting.
"""

import logging
import re
from collections import defaultdict

try:
    import pymorphy3 as pymorphy

    MORPH = pymorphy.MorphAnalyzer()
    HAS_MORPH = True
except Exception:
    try:
        import pymorphy2 as pymorphy

        MORPH = pymorphy.MorphAnalyzer()
        HAS_MORPH = True
    except Exception:
        MORPH = None
        HAS_MORPH = False

import contextlib

from src.ingestion.schema import CleanedMessage, NormalizedMessage
from src.pii.ner_scrubber import TECH_WHITELIST, NERPIIScrubber
from src.pii.regex_scrubber import RegexPIIScrubber

logger = logging.getLogger(__name__)

# Common Russian first names for dictionary matching
COMMON_RU_FIRST_NAMES: set[str] = {
    "александр",
    "алексей",
    "анатолий",
    "андрей",
    "антон",
    "артем",
    "артём",
    "артур",
    "богдан",
    "борис",
    "вадим",
    "валентин",
    "валерий",
    "василий",
    "виктор",
    "виталий",
    "владимир",
    "владислав",
    "вячеслав",
    "георгий",
    "глеб",
    "григорий",
    "даниил",
    "данил",
    "данила",
    "денис",
    "дмитрий",
    "евгений",
    "егор",
    "иван",
    "игорь",
    "илья",
    "кирилл",
    "константин",
    "лев",
    "леонид",
    "максим",
    "марк",
    "матвей",
    "михаил",
    "никита",
    "николай",
    "олег",
    "павел",
    "петр",
    "пётр",
    "роман",
    "ростислав",
    "руслан",
    "сергей",
    "станислав",
    "степан",
    "тимофей",
    "тимур",
    "федор",
    "фёдор",
    "филипп",
    "эдуард",
    "юрий",
    "ярослав",
    "алена",
    "алёна",
    "алина",
    "алиса",
    "алла",
    "анастасия",
    "анна",
    "антонина",
    "валентина",
    "валерия",
    "василиса",
    "вера",
    "вероника",
    "виктория",
    "галина",
    "дарья",
    "диана",
    "евгения",
    "екатерина",
    "елена",
    "елизавета",
    "жана",
    "жанна",
    "зинаида",
    "зоя",
    "инна",
    "ирина",
    "карина",
    "кира",
    "клавдия",
    "кристина",
    "ксения",
    "лариса",
    "лидия",
    "лилия",
    "любовь",
    "людмила",
    "маргарита",
    "марина",
    "мария",
    "марья",
    "надежда",
    "наталья",
    "наталия",
    "нина",
    "оксана",
    "олеся",
    "ольга",
    "полина",
    "райса",
    "раиса",
    "светлана",
    "снежана",
    "софия",
    "софья",
    "таисия",
    "тамара",
    "татьяна",
    "ульяна",
    "юлия",
    "яна",
}


class DeepPIIAnonymizer:
    """
    Production-grade PII Anonymizer with dynamic author name harvesting,
    Russian case inflections, deterministic regex rules, and Natasha NER.
    """

    def __init__(self, enable_ner: bool = True):
        self.regex_scrubber = RegexPIIScrubber()
        self.ner_scrubber = NERPIIScrubber() if enable_ner else None

        # User ID and pseudonym mappings
        self.user_id_to_anon: dict[str, str] = {}
        self.raw_name_to_anon: dict[str, str] = {}
        self.username_handles_map: dict[str, str] = {}

        # Set of all inflected name forms to redact in text
        self.name_forms_to_mask: set[str] = set()
        self.name_regex_patterns: list[re.Pattern] = []

        # Database connection strings pattern (e.g. postgresql://user:pass@host:5432/db)
        self.db_url_pattern = re.compile(
            r"\b(?:postgres|postgresql|mysql|mongodb|redis|amqp)://[^\s]+\b", re.IGNORECASE
        )

        # Telegram forward / quote pattern: > [Name]: or Forwarded from Name
        self.tg_forward_pattern = re.compile(r"(?i)(?:переслано от|forwarded from)\s+([A-Za-zА-Яа-яЁё0-9_\-\s]{2,30})")

        # Statistics
        self.stats: dict[str, int] = defaultdict(int)

    def register_authors(self, authors: list[tuple[str, str]]):
        """
        Pre-register all known authors across the corpus to harvest and inflect their names.
        authors is a list of (raw_author_id, raw_author_name)
        """
        logger.info(f"Harvesting and inflecting names for {len(authors)} unique authors...")

        for raw_id, raw_name in authors:
            self._get_or_create_pseudonym(raw_id, raw_name)
            self._harvest_name_inflections(raw_name)

        # Compile regex patterns for harvested names
        self._recompile_name_patterns()
        logger.info(f"Generated {len(self.name_forms_to_mask):,} inflected name forms for deep redaction.")

    def _get_or_create_pseudonym(self, raw_id: str, raw_name: str) -> tuple[str, str]:
        """Get or assign persistent pseudonym."""
        key = str(raw_id).strip()
        if not key or key.lower() in ("none", "unknown", "anonymous", "null"):
            key = str(raw_name).strip()

        if key not in self.user_id_to_anon:
            idx = len(self.user_id_to_anon) + 1
            anon_id = f"user_{idx:05d}"
            anon_name = f"Developer_{idx:05d}"
            self.user_id_to_anon[key] = anon_id
            self.raw_name_to_anon[str(raw_name).strip()] = anon_name
            self.stats["authors_anonymized"] += 1

        return self.user_id_to_anon[key], self.raw_name_to_anon.get(
            str(raw_name).strip(), f"Developer_{len(self.user_id_to_anon):05d}"
        )

    def _harvest_name_inflections(self, raw_name: str):
        """Extract first/last names from display name and generate all 6 Russian grammatical cases."""
        if not raw_name:
            return

        # Clean display name from emojis and special chars
        clean_name = re.sub(r"[^\w\sа-яёА-ЯЁa-zA-Z]", " ", raw_name).strip()
        words = clean_name.split()

        for w in words:
            w_lower = w.lower()
            if len(w_lower) < 3 or w_lower in TECH_WHITELIST or w_lower in ("chat", "channel", "admin", "bot"):
                continue

            # If it's a known first name or capitalized word
            if w_lower in COMMON_RU_FIRST_NAMES or (
                len(w) >= 3 and w[0].isupper() and any("\u0400" <= c <= "\u04ff" for c in w)
            ):
                self.name_forms_to_mask.add(w_lower)

                if HAS_MORPH and MORPH is not None:
                    try:
                        parsed = MORPH.parse(w_lower)
                        if parsed:
                            p = parsed[0]
                            # Generate all case forms: nomn, gent, datv, accs, ablt, loct
                            for case in ("nomn", "gent", "datv", "accs", "ablt", "loct"):
                                inflected = p.inflect({case})
                                if inflected and inflected.word not in TECH_WHITELIST and len(inflected.word) >= 3:
                                    self.name_forms_to_mask.add(inflected.word.lower())
                    except Exception:
                        pass

    def _recompile_name_patterns(self):
        """Compile regex patterns from harvested inflected names."""
        # Sort by length descending so longer compound names are matched first
        sorted_names = sorted(list(self.name_forms_to_mask), key=lambda x: len(x), reverse=True)
        if not sorted_names:
            return

        # Split into batches of 100 names per regex for performance
        self.name_regex_patterns = []
        batch_size = 100
        for i in range(0, len(sorted_names), batch_size):
            batch = sorted_names[i : i + batch_size]
            escaped = [re.escape(n) for n in batch]
            pat_str = r"(?i)(?<!\w)(" + "|".join(escaped) + r")(?!\w)"
            with contextlib.suppress(Exception):
                self.name_regex_patterns.append(re.compile(pat_str))

    def scrub_text(self, text: str) -> str:
        """
        Execute full deep multi-pass PII scrubbing on text.
        """
        if not text:
            return ""

        # Pass 1: Database connection strings with passwords
        def _sub_db(m):
            self.stats["db_urls"] += 1
            return "[DATABASE_URL_REDACTED]"

        text = self.db_url_pattern.sub(_sub_db, text)

        # Pass 2: Forwarded headers
        def _sub_fwd(m):
            self.stats["forward_headers"] += 1
            return "Forwarded from [PERSON_REDACTED]"

        text = self.tg_forward_pattern.sub(_sub_fwd, text)

        # Pass 3: Deterministic RegEx (Phones, Emails, Crypto, API Keys, IPs, Invites)
        text, reg_stats = self.regex_scrubber.scrub(text, self.username_handles_map)
        for k, v in reg_stats.items():
            self.stats[k] += v

        # Pass 4: Morphological Name Scrubber (all inflected case forms of known participants)
        for pat in self.name_regex_patterns:

            def _sub_name(m):
                matched = m.group(1).lower()
                if matched in TECH_WHITELIST:
                    return m.group(0)
                self.stats["morph_names"] += 1
                return "[PERSON_REDACTED]"

            text = pat.sub(_sub_name, text)

        # Pass 5: Natasha Neural NER (PER & LOC)
        if self.ner_scrubber and self.ner_scrubber.enabled:
            text, ner_stats = self.ner_scrubber.scrub(text)
            self.stats["ner_per"] += ner_stats.get("ner_per", 0)
            self.stats["ner_loc"] += ner_stats.get("ner_loc", 0)

        # Pass 6: Collapse repetitive redact tokens & clean formatting
        text = re.sub(r"(\[PERSON_REDACTED\]\s*){2,}", "[PERSON_REDACTED] ", text)
        text = re.sub(r"(\[PHONE_REDACTED\]\s*){2,}", "[PHONE_REDACTED] ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def anonymize_message(self, msg: NormalizedMessage) -> CleanedMessage:
        """Process a single NormalizedMessage into CleanedMessage."""
        anon_id, anon_name = self._get_or_create_pseudonym(msg.author_id_raw, msg.author_raw)
        clean_text = self.scrub_text(msg.text_raw)

        self.stats["messages_processed"] += 1
        is_q = "?" in clean_text
        tok_approx = int(len(clean_text.split()) * 1.35)

        return CleanedMessage(
            msg_id=msg.msg_id,
            chat_id=msg.chat_id,
            chat_name=msg.chat_name,
            timestamp=msg.timestamp.isoformat(),
            unixtime=msg.unixtime,
            author_anon=anon_name,
            author_id_anon=anon_id,
            text_clean=clean_text,
            reply_to_id=msg.reply_to_id,
            domain="general_tech_chat",
            tags=[],
            sentiment_score=0,
            token_count_approx=tok_approx,
            is_question=is_q,
            thread_id=None,
        )

    def process_all(self, messages: list[NormalizedMessage]) -> list[CleanedMessage]:
        """
        Process all messages in full: pre-registers all author names, inflects them, and scrubs text.
        """
        # Step 1: Pre-register all authors
        author_pairs = [(m.author_id_raw, m.author_raw) for m in messages]
        self.register_authors(author_pairs)

        # Step 2: Scrub all messages
        logger.info(f"Executing deep multi-pass PII scrubbing across {len(messages):,} messages...")
        cleaned = []
        for _i, m in enumerate(messages):
            cleaned.append(self.anonymize_message(m))

        logger.info(f"Deep PII Scrubbing completed: {self.stats}")
        return cleaned
