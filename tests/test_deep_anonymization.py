"""
Unit tests for Deep Case-Aware Morphological PII Anonymization.
"""

import unittest

from ricc.pii.deep_anonymizer import DeepPIIAnonymizer


class TestPseudonymCollisions(unittest.TestCase):
    """Two different authors with the same display name must stay distinct.

    The old name index stored a single pseudonym per name, so the second
    author overwrote the first and all of the first author's messages were
    exported under the second author's pseudonym.
    """

    def setUp(self):
        self.anonymizer = DeepPIIAnonymizer(enable_ner=False)
        self.anonymizer.register_authors(
            [
                ("id-1", "Дмитрий"),
                ("id-2", "Дмитрий"),
            ]
        )

    def test_same_name_authors_get_distinct_ids(self):
        self.assertNotEqual(self.anonymizer.user_id_to_anon["id-1"], self.anonymizer.user_id_to_anon["id-2"])

    def test_first_author_keeps_their_own_pseudonym(self):
        anon_id, anon_name = self.anonymizer._get_or_create_pseudonym("id-1", "Дмитрий")
        self.assertEqual(anon_id, self.anonymizer.user_id_to_anon["id-1"])
        # The display name resolves to THIS author's pseudonym, not the one
        # belonging to the later author who shared the name.
        self.assertEqual(anon_name, "Developer_00001")

    def test_second_author_keeps_their_own_pseudonym(self):
        anon_id, anon_name = self.anonymizer._get_or_create_pseudonym("id-2", "Дмитрий")
        self.assertEqual(anon_id, self.anonymizer.user_id_to_anon["id-2"])
        self.assertEqual(anon_name, "Developer_00002")

    def test_unknown_author_with_known_name_resolves_deterministically(self):
        """An author registered only via scrub_text still resolves to a
        recorded pseudonym, never to a fabricated colliding one."""
        _anon_id, anon_name = self.anonymizer._get_or_create_pseudonym("id-1", "Дмитрий")
        all_named = self.anonymizer.raw_name_to_anon["Дмитрий"]
        self.assertIn(anon_name, all_named)


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
