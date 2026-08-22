"""
Unit tests for deterministic Regex and NER PII scrubbing.
"""

import unittest
from src.pii.regex_scrubber import RegexPIIScrubber
from src.pii.anonymizer import UnifiedPIIAnonymizer


class TestPIIScrubbing(unittest.TestCase):

    def setUp(self):
        self.scrubber = RegexPIIScrubber()
        self.anonymizer = UnifiedPIIAnonymizer(enable_ner=False)

    def test_phone_masking(self):
        texts = [
            "Позвони мне на +7 (999) 123-45-67 срочно",
            "Мой номер 89215554433, пиши в вотсап",
            "International: +1 (555) 234-5678",
        ]
        for t in texts:
            clean, stats = self.scrubber.scrub(t)
            self.assertIn("[PHONE_REDACTED]", clean)
            self.assertNotIn("999", clean)
            self.assertNotIn("89215554433", clean)

    def test_email_masking(self):
        t = "Отправь резюме на developer.senior@google.com или info@startup.ru"
        clean, stats = self.scrubber.scrub(t)
        self.assertIn("[EMAIL_REDACTED]", clean)
        self.assertNotIn("developer.senior@google.com", clean)
        self.assertNotIn("info@startup.ru", clean)

    def test_crypto_wallet_masking(self):
        t_btc = "Кидай оплату на 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
        clean_btc, _ = self.scrubber.scrub(t_btc)
        self.assertIn("[CRYPTO_WALLET_BTC]", clean_btc)

        t_eth = "Мой EVM адрес: 0x71C7656EC7ab88b098defB751B7401B5f6d8976F"
        clean_eth, _ = self.scrubber.scrub(t_eth)
        self.assertIn("[CRYPTO_WALLET_ETH]", clean_eth)

        t_tron = "Адрес USDT TRC20: TLsV52sRDL79HXGGm9yzwKibb6BeruhUzy"
        clean_tron, _ = self.scrubber.scrub(t_tron)
        self.assertIn("[CRYPTO_WALLET_TRON]", clean_tron)

    def test_api_keys_masking(self):
        t_openai = "Вот ключ: sk-proj-1234567890abcdef1234567890abcdef"
        clean_oa, _ = self.scrubber.scrub(t_openai)
        self.assertIn("[API_KEY_REDACTED]", clean_oa)

        t_gh = "Токен ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
        clean_gh, _ = self.scrubber.scrub(t_gh)
        self.assertIn("[API_KEY_REDACTED]", clean_gh)

        t_tg = "Бот токен: 123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ12345"
        clean_tg, _ = self.scrubber.scrub(t_tg)
        self.assertIn("[BOT_TOKEN_REDACTED]", clean_tg)

    def test_user_pseudonymization_consistency(self):
        u1_anon, _ = self.anonymizer.get_user_pseudonym("12345", "Ivan")
        u1_again, _ = self.anonymizer.get_user_pseudonym("12345", "Ivan")
        self.assertEqual(u1_anon, u1_again)

        u2_anon, _ = self.anonymizer.get_user_pseudonym("67890", "Petr")
        self.assertNotEqual(u1_anon, u2_anon)


if __name__ == "__main__":
    unittest.main()
