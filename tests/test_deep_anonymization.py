"""
Unit tests for Deep Case-Aware Morphological PII Anonymization.
"""

import unittest

from src.pii.deep_anonymizer import DeepPIIAnonymizer


class TestDeepAnonymization(unittest.TestCase):
    def setUp(self):
        self.anonymizer = DeepPIIAnonymizer(enable_ner=True)
        # Register test authors
        self.anonymizer.register_authors(
            [
                ("101", "Максим Кульгин"),
                ("102", "Денис"),
                ("103", "Александр"),
                ("104", "Илья Бугаев"),
                ("105", "Екатерина"),
            ]
        )

    def test_russian_name_declensions_masking(self):
        cases = [
            ("Максим, подскажи как настроить PostgreSQL?", "[PERSON_REDACTED]"),
            ("Я отправил код Максиму на ревью.", "[PERSON_REDACTED]"),
            ("Спроси у Кульгина про выбор серверов.", "[PERSON_REDACTED]"),
            ("Мы вчера долго спорили с Денисом про Next.js.", "[PERSON_REDACTED]"),
            ("Позвони Илье или напиши Екатерине.", "[PERSON_REDACTED]"),
        ]
        for text, expected in cases:
            cleaned = self.anonymizer.scrub_text(text)
            self.assertIn(expected, cleaned, f"Failed on text: {text} -> {cleaned}")

    def test_tech_whitelist_preserved(self):
        tech_text = "PostgreSQL, Docker, Kubernetes, DeepSeek, Cursor, FastAPI, Python, Hetzner, Selectel"
        cleaned = self.anonymizer.scrub_text(tech_text)
        self.assertIn("PostgreSQL", cleaned)
        self.assertIn("Docker", cleaned)
        self.assertIn("DeepSeek", cleaned)
        self.assertIn("Cursor", cleaned)
        self.assertIn("FastAPI", cleaned)
        self.assertIn("Python", cleaned)

    def test_database_url_masking(self):
        db_text = "Подключение к базе: postgresql://admin:secret123@10.0.0.1:5432/main_db"
        cleaned = self.anonymizer.scrub_text(db_text)
        self.assertIn("[DATABASE_URL_REDACTED]", cleaned)
        self.assertNotIn("secret123", cleaned)


if __name__ == "__main__":
    unittest.main()
