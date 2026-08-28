"""
Extended PII scrubbing tests: regex edge cases, deep morphological anonymizer,
NER scrubber control flow (with a fake Natasha Doc — no model downloads),
and the UnifiedPIIAnonymizer facade.
"""

import unittest
from datetime import datetime

from src.ingestion.schema import NormalizedMessage
from src.pii.anonymizer import UnifiedPIIAnonymizer
from src.pii.deep_anonymizer import DeepPIIAnonymizer
from src.pii.ner_scrubber import NERPIIScrubber
from src.pii.regex_scrubber import RegexPIIScrubber


def _make_msg(msg_id: int, author_id: str, author: str, text: str) -> NormalizedMessage:
    return NormalizedMessage(
        msg_id=msg_id,
        chat_id=10,
        chat_name="test_chat",
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        unixtime=1700000000,
        author_raw=author,
        author_id_raw=author_id,
        text_raw=text,
    )


def _make_ner_scrubber(enabled: bool = False) -> NERPIIScrubber:
    """Build a NERPIIScrubber without loading Natasha models (unit-test fast path)."""
    scrubber = object.__new__(NERPIIScrubber)
    scrubber.enabled = enabled
    return scrubber


class _FakeSpan:
    def __init__(self, start: int, stop: int, text: str, type: str):
        self.start = start
        self.stop = stop
        self.text = text
        self.type = type


def _make_fake_doc_cls(spans: list[_FakeSpan]):
    class _FakeDoc:
        def __init__(self, text: str):
            self.text = text
            self.spans = spans

        def segment(self, segmenter):
            pass

        def tag_ner(self, tagger):
            pass

    return _FakeDoc


class TestRegexScrubberEdgeCases(unittest.TestCase):
    def setUp(self):
        self.scrubber = RegexPIIScrubber()

    def test_empty_text(self):
        self.assertEqual(self.scrubber.scrub(""), ("", {}))

    def test_ton_wallet_masking(self):
        clean, stats = self.scrubber.scrub("Кошелек TON: EQ" + "A" * 46)
        self.assertIn("[CRYPTO_WALLET_TON]", clean)
        self.assertEqual(stats["crypto_wallets"], 1)

    def test_jwt_token_masking(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        clean, stats = self.scrubber.scrub(f"токен: {jwt}")
        self.assertIn("[JWT_TOKEN_REDACTED]", clean)
        self.assertEqual(stats["jwt_tokens"], 1)

    def test_aws_key_masking(self):
        clean, stats = self.scrubber.scrub("ключ AWS: AKIAIOSFODNN7EXAMPLE")
        self.assertIn("[AWS_KEY_REDACTED]", clean)
        self.assertEqual(stats["api_keys"], 1)

    def test_secret_assignment_masking(self):
        clean, _ = self.scrubber.scrub('config: api_key = "abcd1234efgh5678"')
        self.assertIn("[SECRET_REDACTED]", clean)
        self.assertNotIn("abcd1234efgh5678", clean)

    def test_ssh_private_key_masking(self):
        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA1234\n-----END RSA PRIVATE KEY-----"
        clean, stats = self.scrubber.scrub(f"вот ключ:\n{pem}\nпроверь")
        self.assertIn("[PRIVATE_KEY_REDACTED]", clean)
        self.assertEqual(stats["ssh_keys"], 1)

    def test_private_invite_link_masking(self):
        clean, stats = self.scrubber.scrub("заходи t.me/+AbCdEf_12345")
        self.assertIn("t.me/[INVITE_LINK_REDACTED]", clean)
        self.assertEqual(stats["private_invites"], 1)

    def test_mentions_with_pseudonym_map(self):
        clean, stats = self.scrubber.scrub("пингуй @ivan и @petya", mention_map={"ivan": "user_00001"})
        self.assertIn("@user_00001", clean)
        self.assertIn("@user_2", clean)  # unknown handle gets the next sequential pseudonym
        self.assertEqual(stats["user_mentions"], 2)

    def test_mentions_without_map_anonymized(self):
        clean, stats = self.scrubber.scrub("напиши @oleg пожалуйста")
        self.assertIn("@user_anon", clean)
        self.assertNotIn("oleg", clean)
        self.assertEqual(stats["user_mentions"], 1)

    def test_public_ip_masked_but_loopback_kept(self):
        clean, stats = self.scrubber.scrub("сервер 203.0.113.55, локально 127.0.0.1")
        self.assertIn("[IP_REDACTED]", clean)
        self.assertIn("127.0.0.1", clean)
        self.assertEqual(stats["ip_addresses"], 1)

    def test_semantic_version_ip_not_masked(self):
        clean, _stats = self.scrubber.scrub("обновись до 3.10.4 и проверь")
        self.assertIn("3.10.4", clean)
        self.assertNotIn("[IP_REDACTED]", clean)

    def test_community_name_masking(self):
        clean, stats = self.scrubber.scrub("это из wylsacom media пришло")
        self.assertIn("[COMMUNITY_REDACTED]", clean)
        self.assertEqual(stats["community_names"], 1)


class TestDeepAnonymizerCoverage(unittest.TestCase):
    def setUp(self):
        self.anonymizer = DeepPIIAnonymizer(enable_ner=False)

    def test_scrub_empty_text(self):
        self.assertEqual(self.anonymizer.scrub_text(""), "")

    def test_pseudonym_fallback_to_name_when_id_empty(self):
        anon_id, anon_name = self.anonymizer._get_or_create_pseudonym("", "Вася Пупкин")
        self.assertIn("Вася Пупкин", self.anonymizer.user_id_to_anon)
        self.assertTrue(anon_id.startswith("user_"))
        self.assertTrue(anon_name.startswith("Developer_"))

    def test_harvest_skips_empty_short_and_whitelist_words(self):
        self.anonymizer._harvest_name_inflections("")
        self.anonymizer._harvest_name_inflections("ab")
        self.anonymizer._harvest_name_inflections("Bot Admin")
        self.assertEqual(self.anonymizer.name_forms_to_mask, set())

    def test_recompile_with_no_names_yields_no_patterns(self):
        self.anonymizer._recompile_name_patterns()
        self.assertEqual(self.anonymizer.name_regex_patterns, [])

    def test_register_authors_and_morph_scrub(self):
        self.anonymizer.register_authors([("1", "Алексей"), ("2", "Мария")])
        self.assertTrue(self.anonymizer.name_regex_patterns)
        clean = self.anonymizer.scrub_text("Алексей написал Марии привет")
        self.assertIn("[PERSON_REDACTED]", clean)
        self.assertNotIn("алексей", clean.lower())
        self.assertGreaterEqual(self.anonymizer.stats["morph_names"], 2)

    def test_anonymize_message(self):
        msg = _make_msg(1, "42", "Иван", "Как настроить Docker?")
        cleaned = self.anonymizer.anonymize_message(msg)
        self.assertEqual(cleaned.msg_id, 1)
        self.assertEqual(cleaned.author_anon, "Developer_00001")
        self.assertEqual(cleaned.author_id_anon, "user_00001")
        self.assertTrue(cleaned.is_question)
        self.assertGreater(cleaned.token_count_approx, 0)
        self.assertEqual(self.anonymizer.stats["messages_processed"], 1)

    def test_process_all_batch(self):
        msgs = [
            _make_msg(1, "42", "Иван", "Привет, вопрос про FastAPI"),
            _make_msg(2, "43", "Ольга", "Ответ: используй middleware"),
        ]
        cleaned = self.anonymizer.process_all(msgs)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(self.anonymizer.stats["messages_processed"], 2)
        self.assertEqual(self.anonymizer.stats["authors_anonymized"], 2)
        self.assertNotIn("Иван", cleaned[0].text_clean)
        self.assertNotIn("Ольга", cleaned[1].text_clean)


class TestNERScrubberControlFlow(unittest.TestCase):
    """NER scrubber control-flow tests with a fake Doc — no Natasha model downloads."""

    def test_disabled_returns_text_unchanged(self):
        scrubber = _make_ner_scrubber(enabled=False)
        text = "Встретился с Иваном в Казани"
        out, stats = scrubber.scrub(text)
        self.assertEqual(out, text)
        self.assertEqual(stats, {"ner_per": 0, "ner_loc": 0})

    def test_empty_text(self):
        scrubber = _make_ner_scrubber(enabled=True)
        out, stats = scrubber.scrub("")
        self.assertEqual(out, "")
        self.assertEqual(stats, {"ner_per": 0, "ner_loc": 0})

    def test_prefilter_no_cyrillic(self):
        scrubber = _make_ner_scrubber(enabled=True)
        out, stats = scrubber.scrub("Hello Ivan Smith")
        self.assertEqual(out, "Hello Ivan Smith")
        self.assertEqual(stats, {"ner_per": 0, "ner_loc": 0})

    def test_prefilter_no_uppercase(self):
        scrubber = _make_ner_scrubber(enabled=True)
        out, stats = scrubber.scrub("привет мир, всё хорошо")
        self.assertEqual(out, "привет мир, всё хорошо")
        self.assertEqual(stats, {"ner_per": 0, "ner_loc": 0})

    def test_per_and_loc_replacement_with_whitelists(self):
        scrubber = _make_ner_scrubber(enabled=True)
        text = "Встретился с Иваном в Казани про Python и РФ"
        spans = [
            _FakeSpan(text.find("Иваном"), text.find("Иваном") + len("Иваном"), "Иваном", "PER"),
            _FakeSpan(text.find("Казани"), text.find("Казани") + len("Казани"), "Казани", "LOC"),
            _FakeSpan(text.find("Python"), text.find("Python") + len("Python"), "Python", "PER"),
            _FakeSpan(text.find("РФ"), text.find("РФ") + len("РФ"), "РФ", "LOC"),
        ]
        scrubber.Doc = _make_fake_doc_cls(spans)
        scrubber.segmenter = scrubber.ner_tagger = None

        out, stats = scrubber.scrub(text)
        self.assertIn("[PERSON_REDACTED]", out)
        self.assertIn("[LOCATION_REDACTED]", out)
        self.assertIn("Python", out)  # whitelisted tech term survives
        self.assertIn("РФ", out)  # whitelisted geo survives
        self.assertEqual(stats["ner_per"], 1)
        self.assertEqual(stats["ner_loc"], 1)

    def test_ner_exception_returns_original_text(self):
        scrubber = _make_ner_scrubber(enabled=True)

        class _ExplodingDoc:
            def __init__(self, text):
                raise RuntimeError("model failure")

        scrubber.Doc = _ExplodingDoc
        text = "Встретился с Иваном в Казани"
        out, stats = scrubber.scrub(text)
        self.assertEqual(out, text)
        self.assertEqual(stats, {"ner_per": 0, "ner_loc": 0})


class TestUnifiedAnonymizerFacade(unittest.TestCase):
    def setUp(self):
        self.anonymizer = UnifiedPIIAnonymizer(enable_ner=False)

    def test_anonymize_text(self):
        clean = self.anonymizer.anonymize_text("Позвони мне +7 999 123-45-67")
        self.assertIn("[PHONE_REDACTED]", clean)

    def test_process_batch_and_stats_summary(self):
        msgs = [
            _make_msg(1, "42", "Иван", "Вопрос про Django ORM?"),
            _make_msg(2, "43", "Анна", "Ответ: select_related"),
        ]
        cleaned = self.anonymizer.process_batch(msgs)
        self.assertEqual(len(cleaned), 2)
        self.assertTrue(all(hasattr(c, "text_clean") for c in cleaned))

        summary = self.anonymizer.get_stats_summary()
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary.get("messages_processed"), 2)

    def test_pseudonym_consistency_via_facade(self):
        first, _ = self.anonymizer.get_user_pseudonym("777", "Иван")
        again, _ = self.anonymizer.get_user_pseudonym("777", "Иван")
        self.assertEqual(first, again)


if __name__ == "__main__":
    unittest.main()
