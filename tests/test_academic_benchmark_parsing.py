"""
Unit tests for the strict multiple-choice answer parser used by the
academic benchmark harness (src/evaluation/official_academic_benchmarks.py).

These tests guard against the false-positive regression where a substring
check (`key in response[:10]`) counted any occurrence of a letter A-D as a
correct answer, producing an implausible 100% RuMMLU accuracy.
"""

import unittest

from src.evaluation.official_academic_benchmarks import parse_mc_answer


class TestMultipleChoiceAnswerParsing(unittest.TestCase):
    def test_extracts_standalone_letter(self):
        self.assertEqual(parse_mc_answer("C"), "C")
        self.assertEqual(parse_mc_answer("Ответ: B"), "B")
        self.assertEqual(parse_mc_answer("B) Skip List"), "B")
        self.assertEqual(parse_mc_answer("Правильный ответ - D"), "D")

    def test_cyrillic_homoglyphs_need_an_explicit_marker(self):
        """A bare Cyrillic homoglyph is a Russian word, not an answer.

        The old parser mapped А/В/С to A/B/C at ANY position, so
        "Я не знаю в чём дело" was scored as "B" and "Это вариант с" as
        "C" — the same class of defect as the retracted substring bug,
        inflating the RuMMLU accuracy with answers the model never gave.
        """
        # Prose: no marker -> not an answer.
        self.assertIsNone(parse_mc_answer("Я не знаю в чём дело"))
        self.assertIsNone(parse_mc_answer("Это вариант с"))
        self.assertIsNone(parse_mc_answer("а и б"))
        self.assertIsNone(parse_mc_answer("А"))
        self.assertIsNone(parse_mc_answer("С"))

        # Explicit marker -> accepted and normalized.
        self.assertEqual(parse_mc_answer("Ответ: в"), "B")
        self.assertEqual(parse_mc_answer("Ответ: а"), "A")
        self.assertEqual(parse_mc_answer("Ответ (с)"), "C")
        self.assertEqual(parse_mc_answer("Answer: c"), "C")
        self.assertEqual(parse_mc_answer("вариант: В"), "B")

    def test_lowercase_latin_article_is_not_an_answer(self):
        """'Compose a docker file' must not be read as the answer 'A'."""
        self.assertIsNone(parse_mc_answer("Compose a docker file"))

    def test_no_false_positive_inside_words(self):
        # "Docker..." must NOT count as answer D; "Cat" must not count as A... etc.
        self.assertIsNone(parse_mc_answer("Docker compose решает проблему"))
        self.assertIsNone(parse_mc_answer("Kubernetes"))
        self.assertIsNone(parse_mc_answer("AB"))

    def test_first_standalone_letter_wins(self):
        self.assertEqual(parse_mc_answer("B или C? Думаю B."), "B")

    def test_empty_and_garbage_return_none(self):
        self.assertIsNone(parse_mc_answer(""))
        self.assertIsNone(parse_mc_answer("не знаю"))
        self.assertIsNone(parse_mc_answer(None if False else "..."))

    def test_rejects_letters_outside_valid_set(self):
        self.assertIsNone(parse_mc_answer("E"), None)


if __name__ == "__main__":
    unittest.main()
