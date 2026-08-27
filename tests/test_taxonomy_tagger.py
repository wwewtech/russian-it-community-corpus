"""Tests for the keyword tagger and sentiment scorer (src/taxonomy/tagger.py)."""

from __future__ import annotations

import pytest

from src.config import DOMAIN_TAXONOMY, SENTIMENT_DICT
from src.ingestion.schema import CleanedMessage
from src.taxonomy.tagger import TechnicalTagger

GENERAL_DOMAIN = "general_tech_chat"


@pytest.fixture(scope="module")
def tagger() -> TechnicalTagger:
    return TechnicalTagger()


def _msg(text: str, msg_id: int = 1) -> CleanedMessage:
    return CleanedMessage(
        msg_id=msg_id,
        chat_id=1,
        chat_name="community_node_01",
        timestamp="2026-01-01T00:00:00",
        unixtime=1767225600,
        author_raw="user",
        author_id_raw="user",
        author_anon="Developer_00001",
        author_id_anon="Developer_00001",
        text_clean=text,
    )


class TestExtractTags:
    def test_empty_text(self, tagger: TechnicalTagger):
        assert tagger.extract_tags("") == []

    def test_backend_keywords_matched(self, tagger: TechnicalTagger):
        tags = tagger.extract_tags("Как оптимизировать postgres запрос?")
        assert "postgres" in tags

    def test_tags_are_sorted(self, tagger: TechnicalTagger):
        tags = tagger.extract_tags("docker kubernetes postgres")
        assert tags == sorted(tags)

    def test_no_false_positives_on_plain_text(self, tagger: TechnicalTagger):
        assert tagger.extract_tags("привет, как дела?") == []


class TestComputeSentiment:
    def test_empty_text(self, tagger: TechnicalTagger):
        assert tagger.compute_sentiment("") == 0

    def test_positive(self, tagger: TechnicalTagger):
        assert tagger.compute_sentiment("всё отлично, спасибо") > 0

    def test_negative(self, tagger: TechnicalTagger):
        assert tagger.compute_sentiment("ужасно, всё тормозит") < 0

    def test_neutral(self, tagger: TechnicalTagger):
        assert tagger.compute_sentiment("обычный текст без эмоций") == 0


class TestTagMessage:
    def test_enriches_message(self, tagger: TechnicalTagger):
        msg = _msg("Как настроить docker и kubernetes в кластере?")
        result = tagger.tag_message(msg)
        assert result.domain == "devops_infra"
        assert "docker" in result.tags
        assert isinstance(result.sentiment_score, int)

    def test_returns_same_object(self, tagger: TechnicalTagger):
        msg = _msg("postgres транзакция")
        assert tagger.tag_message(msg) is msg


class TestTagBatch:
    def test_batch_returns_all(self, tagger: TechnicalTagger):
        msgs = [_msg("postgres индекс", 1), _msg("docker деплой", 2), _msg("привет", 3)]
        result = tagger.tag_batch(msgs)
        assert len(result) == 3
        assert result[0].domain == "backend_databases"


class TestConfigConsistency:
    def test_taxonomy_domains_have_keywords(self):
        for domain, info in DOMAIN_TAXONOMY.items():
            assert info["keywords"], f"domain {domain} has no keywords"

    def test_sentiment_scores_are_ints(self):
        assert all(isinstance(v, int) for v in SENTIMENT_DICT.values())
