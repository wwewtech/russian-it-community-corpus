"""Tests for the SFT dialogue quality regression monitor.

These tests pin the behaviour of :class:`SFTDialogueQualityMonitor` so a
future refactor cannot silently let the floors drift, and so that CI has a
fast, deterministic signal of "the SFT quality pipeline is still wired up".
"""

from __future__ import annotations

import unittest

import pandas as pd

from src.monitoring.sft_quality import SFTDialogueQualityMonitor


def _make_dialogue(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _make_long_dialogue(n_turns: int = 4, assistant_chars: int = 200) -> list[dict[str, str]]:
    """A realistic 4-turn Russian dialogue with non-trivial assistant replies.

    Turns alternate ``user → assistant → user → assistant`` so role-balance is
    healthy and adjacent turns are not identical.
    """
    body = "Подробный технический ответ на русском языке. " * (assistant_chars // 50)
    turns: list[dict[str, str]] = []
    for i in range(n_turns // 2):
        turns.append({"role": "user", "content": f"Вопрос номер {i} про настройку Nginx."})
        turns.append({"role": "assistant", "content": body})
    return turns


class TestSFTDialogueQualityMonitor(unittest.TestCase):
    def test_healthy_dataset_passes(self):
        dialogues = pd.DataFrame({"messages": [_make_long_dialogue() for _ in range(50)]})
        report = SFTDialogueQualityMonitor(dialogues).run()
        self.assertEqual(report["overall_verdict"], "pass")
        self.assertEqual(report["n_dialogues"], 50)
        for name, check in report["checks"].items():
            self.assertTrue(check["passed"], f"{name} should pass: {check}")

    def test_empty_assistant_turn_fails(self):
        # Half the dialogues have an empty assistant turn.
        good = [_make_long_dialogue() for _ in range(10)]
        bad = [_make_long_dialogue()] + [_make_dialogue("Вопрос?", "") for _ in range(10)]
        df = pd.DataFrame({"messages": good + bad})
        report = SFTDialogueQualityMonitor(df).run()
        self.assertEqual(report["overall_verdict"], "fail")
        self.assertFalse(report["checks"]["empty_response_ratio"]["passed"])

    def test_truncated_short_answers_fail(self):
        # All assistant replies are < 30 chars → below the 80-char floor.
        short_dialogues = [_make_dialogue("Вопрос?", "Кратко: да.") for _ in range(20)]
        df = pd.DataFrame({"messages": short_dialogues})
        report = SFTDialogueQualityMonitor(df).run()
        self.assertEqual(report["overall_verdict"], "fail")
        self.assertFalse(report["checks"]["median_assistant_turn_chars"]["passed"])

    def test_all_user_turns_fails_role_balance(self):
        only_user = [[{"role": "user", "content": "Только пользователь."} for _ in range(4)] for _ in range(20)]
        df = pd.DataFrame({"messages": only_user})
        report = SFTDialogueQualityMonitor(df).run()
        self.assertFalse(report["checks"]["role_balance"]["passed"])

    def test_english_corpus_fails_russian_ratio(self):
        en_dialogues = [
            [
                {"role": "user", "content": "How to configure nginx reverse proxy?"},
                {"role": "assistant", "content": "Use proxy_pass and set the headers."},
            ]
            for _ in range(20)
        ]
        df = pd.DataFrame({"messages": en_dialogues})
        report = SFTDialogueQualityMonitor(df).run()
        self.assertFalse(report["checks"]["russian_ratio"]["passed"])

    def test_duplicate_adjacent_turns_fail(self):
        # Two adjacent turns share the same content — an extraction bug.
        body = "Подробный ответ на русском языке. " * 10
        dup = [
            [
                {"role": "user", "content": "Вопрос 1"},
                {"role": "assistant", "content": body},
                {"role": "user", "content": "Вопрос 2"},
                {"role": "assistant", "content": body},
                # Two identical turns inserted at the end — adjacent duplicate.
                {"role": "user", "content": "Один и тот же текст"},
                {"role": "user", "content": "Один и тот же текст"},
            ]
            for _ in range(20)
        ]
        df = pd.DataFrame({"messages": dup})
        report = SFTDialogueQualityMonitor(df).run()
        self.assertFalse(report["checks"]["duplicate_turn_ratio"]["passed"])

    def test_empty_dataframe_returns_no_data(self):
        df = pd.DataFrame({"messages": []})
        report = SFTDialogueQualityMonitor(df).run()
        self.assertEqual(report["overall_verdict"], "no_data")
        self.assertEqual(report["n_dialogues"], 0)

    def test_missing_messages_column_raises(self):
        df = pd.DataFrame({"other": [[]]})
        with self.assertRaises(ValueError):
            SFTDialogueQualityMonitor(df, messages_col="messages").run()
