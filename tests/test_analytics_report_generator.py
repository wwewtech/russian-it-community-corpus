"""Tests for the analytics report generator (JSON -> markdown, single source of truth).

The markdown report must be a faithful rendering of the analytics summary dict —
these tests fail if the renderer drifts from the JSON payload.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.analytics.report_generator import ReportGenerator, generate_ascii_bar

MINIMAL_SUMMARY = {
    "report_metadata": {"generated_at": "2026-08-24T04:15:26.716205"},
    "volume_statistics": {
        "total_messages": 2816434,
        "unique_authors": 210887,
        "date_start": "2017-08-06 15:01:46",
        "date_end": "2026-08-22 18:48:36",
        "total_days_active": 3303,
        "messages_per_day": 852.69,
        "tokens_per_day": 14853.69,
        "total_words": 37242618,
        "total_tokens_estimated": 49061734,
        "vocabulary_unique_words": 43295,
        "character_length_distribution": {"mean": 83.07, "median": 43.0},
        "word_count_distribution": {"mean": 13.22, "median": 7.0},
        "token_count_distribution": {"mean": 17.42, "median": 9.0},
    },
    "temporal_dynamics": {
        "peak_hour": 21,
        "peak_weekday": "Вторник",
        "hourly_distribution": {"00:00": 110600, "21:00": 189815},
        "yearly_volume": {"2025": 918534},
    },
    "lexical_analytics": {"shannon_entropy": 12.212},
    "domain_slang_analytics": {
        "slang_terms_detected_count": 47,
        "top_slang_terms": [{"term": "баг", "count": 191}],
        "domain_message_distribution": {
            "general_tech_chat": {"count": 2688595, "percentage": 95.46},
        },
    },
}


class TestGenerateAsciiBar:
    def test_full_bar(self):
        assert generate_ascii_bar(10, 10, width=5) == "█████"

    def test_empty_bar(self):
        assert generate_ascii_bar(0, 10, width=5) == "░░░░░"

    def test_zero_max_returns_empty(self):
        assert generate_ascii_bar(5, 0) == ""

    def test_partial_fill(self):
        bar = generate_ascii_bar(5, 10, width=4)
        assert bar == "██░░"


class TestReportGeneratorExportJson:
    def test_roundtrip(self, tmp_path=None):
        out = Path("_test_export.json")
        try:
            gen = ReportGenerator(MINIMAL_SUMMARY)
            written = gen.export_json(out)
            assert written == out
            data = json.loads(out.read_text(encoding="utf-8"))
            assert data["volume_statistics"]["total_messages"] == 2816434
        finally:
            out.unlink(missing_ok=True)


class TestReportGeneratorExportMarkdown:
    def test_renders_header(self):
        md_path = Path("_test_render.md")
        try:
            ReportGenerator(MINIMAL_SUMMARY).export_markdown(md_path)
            md = md_path.read_text(encoding="utf-8")
            assert "АНАЛИТИЧЕСКИЙ ОТЧЁТ" in md
            assert "2026-08-24T04:15:26.716205" in md
        finally:
            md_path.unlink(missing_ok=True)

    def test_renders_volume_numbers(self):
        md_path = Path("_test_render.md")
        try:
            ReportGenerator(MINIMAL_SUMMARY).export_markdown(md_path)
            md = md_path.read_text(encoding="utf-8")
            assert "2,816,434" in md  # total messages with thousands separator
            assert "210,887" in md  # unique authors
            assert "43,295" in md  # vocabulary
        finally:
            md_path.unlink(missing_ok=True)

    def test_renders_temporal_and_domains(self):
        md_path = Path("_test_render.md")
        try:
            ReportGenerator(MINIMAL_SUMMARY).export_markdown(md_path)
            md = md_path.read_text(encoding="utf-8")
            assert "21:00" in md  # peak hour
            assert "Вторник" in md  # peak weekday
            assert "general_tech_chat" in md
            assert "95.5%" in md  # percentage formatted to 1 decimal
            assert "баг" in md  # slang table
        finally:
            md_path.unlink(missing_ok=True)

    def test_empty_data_does_not_crash(self):
        md_path = Path("_test_render_empty.md")
        try:
            ReportGenerator({}).export_markdown(md_path)
            md = md_path.read_text(encoding="utf-8")
            assert "АНАЛИТИЧЕСКИЙ ОТЧЁТ" in md
        finally:
            md_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
