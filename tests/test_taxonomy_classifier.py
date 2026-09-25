"""
Comprehensive unit tests for the Taxonomy layer:
  - src/taxonomy/classifier.py  -> DomainClassifier
  - src/taxonomy/tagger.py      -> TechnicalTagger

Covers classify_text, compute_sentiment, extract_tags, tag_message and tag_batch
including empty/None inputs, Russian/English mixing, confidence scoring,
case-insensitivity, deduplication, sorting and domain tie-breaking.
"""

from __future__ import annotations

import pytest

from src.config import DOMAIN_TAXONOMY, SENTIMENT_DICT
from src.ingestion.schema import CleanedMessage
from src.taxonomy.classifier import DomainClassifier
from src.taxonomy.tagger import TechnicalTagger

GENERAL_DOMAIN = "general_tech_chat"


def _make_classifier() -> DomainClassifier:
    return DomainClassifier()


def _make_tagger() -> TechnicalTagger:
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


# ---------------------------------------------------------------------------
# DomainClassifier.classify_text
# ---------------------------------------------------------------------------
class TestClassifyText:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.classifier = _make_classifier()

    def test_empty_string_returns_general(self):
        assert self.classifier.classify_text("") == (GENERAL_DOMAIN, 0.0, {})

    def test_none_text_returns_general(self):
        assert self.classifier.classify_text(None) == (GENERAL_DOMAIN, 0.0, {})

    def test_non_technical_text_returns_general(self):
        domain, confidence, scores = self.classifier.classify_text("привет, как дела? Сегодня солнечно.")
        assert domain == GENERAL_DOMAIN
        assert confidence == 0.0
        assert scores == {}

    def test_single_keyword_single_domain(self):
        domain, confidence, scores = self.classifier.classify_text("postgres")
        assert domain == "backend_databases"
        assert confidence == 1.0
        assert scores == {"backend_databases": 1}

    def test_multiple_keywords_same_domain(self):
        domain, confidence, scores = self.classifier.classify_text("postgres postgresql database")
        assert domain == "backend_databases"
        assert confidence == 1.0
        assert scores == {"backend_databases": 3}

    def test_multiple_domains_highest_confidence_wins(self):
        # 3 backend keywords vs 1 devops keyword -> backend wins at 0.75
        domain, confidence, scores = self.classifier.classify_text("postgres postgresql database docker")
        assert domain == "backend_databases"
        assert confidence == pytest.approx(0.75)
        assert scores["backend_databases"] == 3
        assert scores["devops_infra"] == 1

    def test_confidence_between_zero_and_one(self):
        for sample in (
            "docker",
            "postgres redis kafka celery graphql rest grpc orm sqlalchemy",
            "pytorch tensorflow lora python docker",
        ):
            _, confidence, _ = self.classifier.classify_text(sample)
            assert 0.0 < confidence <= 1.0

    def test_confidence_is_float(self):
        _, confidence, _ = self.classifier.classify_text("docker kubernetes ingress")
        assert isinstance(confidence, float)

    def test_returns_three_tuple(self):
        result = self.classifier.classify_text("docker")
        assert isinstance(result, tuple)
        assert len(result) == 3
        domain, confidence, scores = result
        assert isinstance(domain, str)
        assert isinstance(confidence, float)
        assert isinstance(scores, dict)

    def test_scores_dict_values_are_ints(self):
        _, _, scores = self.classifier.classify_text("docker docker")
        assert all(isinstance(v, int) for v in scores.values())

    def test_case_insensitive_matching(self):
        for text in ("POSTGRES", "PostgreSQL", "PoStGrEs"):
            domain, _, scores = self.classifier.classify_text(text)
            assert domain == "backend_databases"
            assert scores == {"backend_databases": 1}

    def test_case_insensitive_multiple_domains(self):
        domain, confidence, _ = self.classifier.classify_text("DOCKER")
        assert domain == "devops_infra"
        assert confidence == 1.0

    def test_dedup_tokens_same_keyword(self):
        # Repeating a token should count only once because of set intersection.
        domain, confidence, scores = self.classifier.classify_text("postgres postgres postgres")
        assert domain == "backend_databases"
        assert confidence == 1.0
        assert scores == {"backend_databases": 1}

    def test_tie_break_keeps_confidence_half(self):
        # One backend keyword + one devops keyword -> tie, confidence 0.5.
        # The winning domain is set-iteration dependent, so assert the value
        # and the candidate domains rather than a single winner.
        domain, confidence, scores = self.classifier.classify_text("postgres docker")
        assert confidence == pytest.approx(0.5)
        assert set(scores.keys()) == {"backend_databases", "devops_infra"}
        assert scores["backend_databases"] == 1
        assert scores["devops_infra"] == 1
        assert domain in {"backend_databases", "devops_infra"}

    def test_general_tech_chat_domain(self):
        domain, confidence, scores = self.classifier.classify_text("cursor vscode ide macbook")
        assert domain == GENERAL_DOMAIN
        assert confidence == 1.0
        assert scores == {GENERAL_DOMAIN: 4}

    def test_ai_ml_nlp_domain(self):
        domain, confidence, scores = self.classifier.classify_text("pytorch tensorflow lora")
        assert domain == "ai_ml_nlp"
        assert confidence == 1.0
        assert scores == {"ai_ml_nlp": 3}

    def test_frontend_ui_domain(self):
        domain, confidence, scores = self.classifier.classify_text("react vue javascript")
        assert domain == "frontend_ui"
        assert confidence == 1.0
        assert scores == {"frontend_ui": 3}

    def test_sysadmin_security_domain(self):
        domain, confidence, scores = self.classifier.classify_text("tls ssl firewall ssh")
        assert domain == "sysadmin_security"
        assert confidence == 1.0
        assert scores == {"sysadmin_security": 4}

    def test_business_legal_fintech_domain(self):
        domain, confidence, scores = self.classifier.classify_text("бизнес стартап саас")
        assert domain == "business_legal_fintech"
        assert confidence == 1.0
        assert scores == {"business_legal_fintech": 3}

    def test_career_team_management_domain(self):
        domain, confidence, scores = self.classifier.classify_text("собес резюме джун")
        assert domain == "career_team_management"
        assert confidence == 1.0
        assert scores == {"career_team_management": 3}

    def test_special_character_keywords(self):
        # c++, c#, .net, java are all backend_databases keywords matched as
        # whole tokens because +, # and . are part of the tokenizer charset.
        domain, confidence, scores = self.classifier.classify_text("c++ c# .net")
        assert domain == "backend_databases"
        assert confidence == 1.0
        assert scores == {"backend_databases": 3}

    def test_mixed_russian_and_english_keywords(self):
        # "postgres" + "база" (ru) are backend; "docker" is devops.
        # Tokens are deduplicated by the set intersection, yielding 2/3.
        domain, confidence, scores = self.classifier.classify_text("postgres база postgres база docker")
        assert domain == "backend_databases"
        assert confidence == pytest.approx(0.667)
        assert scores["backend_databases"] == 2
        assert scores["devops_infra"] == 1

    def test_does_not_mutate_input_text(self):
        original = "PostgreSQL транзакция"
        self.classifier.classify_text(original)
        assert original == "PostgreSQL транзакция"

    def test_custom_taxonomy_is_respected(self):
        custom = {
            "my_domain": {"title": "Custom", "keywords": ["customkw"]},
        }
        classifier = DomainClassifier(taxonomy=custom)
        domain, confidence, scores = classifier.classify_text("customkw customkw")
        assert domain == "my_domain"
        assert scores == {"my_domain": 1}
        assert confidence == 1.0

    def test_empty_taxonomy_falls_back_to_general(self):
        classifier = DomainClassifier(taxonomy={})
        assert classifier.classify_text("anything") == (GENERAL_DOMAIN, 0.0, {})


# ---------------------------------------------------------------------------
# TechnicalTagger.compute_sentiment
# ---------------------------------------------------------------------------
class TestComputeSentiment:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tagger = _make_tagger()

    def test_empty_string(self):
        assert self.tagger.compute_sentiment("") == 0

    def test_none_text(self):
        assert self.tagger.compute_sentiment(None) == 0

    def test_positive_word(self):
        assert self.tagger.compute_sentiment("хорошо") == SENTIMENT_DICT["хорошо"]

    def test_high_positive_word(self):
        assert self.tagger.compute_sentiment("отлично") == 2

    def test_negative_word(self):
        assert self.tagger.compute_sentiment("плохо") == -1

    def test_strong_negative_word(self):
        assert self.tagger.compute_sentiment("ужасно") == -2

    def test_neutral_word(self):
        # Words that exist in the dict with a 0 value.
        assert self.tagger.compute_sentiment("ок") == 0
        assert self.tagger.compute_sentiment("нормально") == 0

    def test_mixed_positive_and_negative(self):
        # "отлично" (+2) + "плохо" (-1) -> 1
        assert self.tagger.compute_sentiment("отлично плохо") == 1

    def test_case_insensitive(self):
        assert self.tagger.compute_sentiment("ХОРОШО") == 1
        assert self.tagger.compute_sentiment("Хорошо") == 1

    def test_accumulation_counting(self):
        assert self.tagger.compute_sentiment("хорошо отлично хорошо") == 1 + 2 + 1

    def test_unknown_words_are_neutral(self):
        assert self.tagger.compute_sentiment("просто техническая информация") == 0

    def test_returns_int(self):
        assert isinstance(self.tagger.compute_sentiment("хорошо"), int)


# ---------------------------------------------------------------------------
# TechnicalTagger.extract_tags
# ---------------------------------------------------------------------------
class TestExtractTags:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tagger = _make_tagger()

    def test_empty_string(self):
        assert self.tagger.extract_tags("") == []

    def test_none_text(self):
        assert self.tagger.extract_tags(None) == []

    def test_single_keyword(self):
        assert self.tagger.extract_tags("postgres запрос") == ["postgres"]

    def test_multiple_keywords(self):
        tags = self.tagger.extract_tags("docker kubernetes ingress")
        assert "docker" in tags
        assert "kubernetes" in tags

    def test_tags_are_sorted(self):
        tags = self.tagger.extract_tags("docker kubernetes postgres")
        assert tags == sorted(tags)

    def test_tags_sorted_exactly(self):
        tags = self.tagger.extract_tags("c++ c# .net java")
        assert tags == [".net", "c#", "c++", "java"]

    def test_deduplication(self):
        tags = self.tagger.extract_tags("postgres postgres postgres")
        assert tags == ["postgres"]
        assert len(tags) == 1

    def test_case_insensitive(self):
        tags = self.tagger.extract_tags("DOCKER DOCKER")
        assert "docker" in tags

    def test_no_false_positives_on_plain_text(self):
        assert self.tagger.extract_tags("привет, как дела?") == []

    def test_special_char_keywords_match(self):
        tags = self.tagger.extract_tags("c++ и c# и .net")
        assert set(tags) == {"c++", "c#", ".net"}

    def test_returns_list_type(self):
        assert isinstance(self.tagger.extract_tags("docker"), list)

    def test_mixed_russian_english(self):
        tags = self.tagger.extract_tags("postgresql и postgres в одном тексте")
        assert "postgresql" in tags
        assert "postgres" in tags

    def test_tags_are_lowercase(self):
        tags = self.tagger.extract_tags("DOCKER Kubernetes")
        assert all(t == t.lower() for t in tags)


# ---------------------------------------------------------------------------
# TechnicalTagger.tag_message
# ---------------------------------------------------------------------------
class TestTagMessage:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tagger = _make_tagger()

    def test_sets_domain(self):
        msg = _msg("docker kubernetes ingress")
        result = self.tagger.tag_message(msg)
        assert result.domain == "devops_infra"

    def test_sets_tags(self):
        msg = _msg("docker kubernetes ingress")
        result = self.tagger.tag_message(msg)
        assert "docker" in result.tags
        assert "kubernetes" in result.tags

    def test_sets_sentiment_score_int(self):
        msg = _msg("docker и всё отлично")
        result = self.tagger.tag_message(msg)
        assert isinstance(result.sentiment_score, int)
        assert result.sentiment_score == 2

    def test_returns_same_object(self):
        msg = _msg("postgres транзакция")
        assert self.tagger.tag_message(msg) is msg

    def test_defaults_for_empty_text(self):
        msg = _msg("")
        result = self.tagger.tag_message(msg)
        assert result.domain == GENERAL_DOMAIN
        assert result.tags == []
        assert result.sentiment_score == 0

    def test_defaults_for_none_text_clean(self):
        msg = _msg("")
        msg.text_clean = None  # type: ignore[assignment]
        result = self.tagger.tag_message(msg)
        assert result.domain == GENERAL_DOMAIN
        assert result.tags == []
        assert result.sentiment_score == 0

    def test_preserves_other_fields(self):
        msg = _msg("спасибо за docker помощь", msg_id=42)
        result = self.tagger.tag_message(msg)
        assert result.msg_id == 42
        assert result.text_clean == "спасибо за docker помощь"

    def test_sentiment_accumulation_across_tags(self):
        msg = _msg("docker и ужасно всё тормозит")
        result = self.tagger.tag_message(msg)
        assert result.sentiment_score == -2 - 1


# ---------------------------------------------------------------------------
# TechnicalTagger.tag_batch
# ---------------------------------------------------------------------------
class TestTagBatch:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tagger = _make_tagger()

    def test_batch_tags_all_messages(self):
        msgs = [_msg("postgres индекс", 1), _msg("docker деплой", 2), _msg("привет", 3)]
        result = self.tagger.tag_batch(msgs)
        assert len(result) == 3
        assert result[0].domain == "backend_databases"
        assert result[1].domain == "devops_infra"
        assert result[2].domain == GENERAL_DOMAIN

    def test_batch_returns_same_list_object(self):
        msgs = [_msg("postgres", 1), _msg("docker", 2)]
        result = self.tagger.tag_batch(msgs)
        assert result is msgs

    def test_batch_marks_each_message(self):
        msgs = [_msg("docker", 1), _msg("хорошо", 2), _msg("спасибо", 3)]
        result = self.tagger.tag_batch(msgs)
        assert result[0].domain == "devops_infra"
        assert "docker" in result[0].tags
        assert result[1].sentiment_score == 1
        assert result[2].sentiment_score == 1

    def test_empty_batch_returns_empty(self):
        assert self.tagger.tag_batch([]) == []

    def test_batch_preserves_order(self):
        msgs = [_msg("docker", 1), _msg("postgres", 2), _msg("ai", 3)]
        result = self.tagger.tag_batch(msgs)
        assert [m.domain for m in result] == ["devops_infra", "backend_databases", "ai_ml_nlp"]

    def test_batch_invokes_progress_bar(self, monkeypatch):
        import src.taxonomy.tagger as tagger_mod

        calls = []

        def fake_tqdm(iterable, **kwargs):
            calls.append(kwargs)
            return iterable

        monkeypatch.setattr(tagger_mod, "tqdm", fake_tqdm)
        msgs = [_msg("docker", 1), _msg("postgres", 2)]
        self.tagger.tag_batch(msgs)
        assert len(calls) == 1
        assert calls[0]["desc"] == "Domain & Tech Tagging"
        assert calls[0]["unit"] == "msg"


# ---------------------------------------------------------------------------
# Cross-module taxonomy/sentiment config consistency
# ---------------------------------------------------------------------------
class TestConfigConsistency:
    def test_all_domains_have_keywords(self):
        for domain, info in DOMAIN_TAXONOMY.items():
            assert info["keywords"], f"domain {domain} has no keywords"

    def test_sentiment_values_are_ints(self):
        assert all(isinstance(v, int) for v in SENTIMENT_DICT.values())

    def test_every_keyword_is_lowercased_and_indexable(self):
        classifier = _make_classifier()
        tagger = _make_tagger()
        for _domain, info in DOMAIN_TAXONOMY.items():
            for kw in info["keywords"]:
                lowered = kw.lower()
                if " " not in lowered:
                    # single-token keywords should be discoverable by both
                    # the classifier mapping and the tagger keyword set.
                    assert lowered in tagger.all_keywords_set
                    assert lowered in classifier.all_keywords_set
