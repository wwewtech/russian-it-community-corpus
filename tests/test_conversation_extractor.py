from __future__ import annotations

from unittest.mock import patch

import pytest

from src.config import MAX_SFT_TURNS
from src.graph.conversation_extractor import (
    AI_CONTAMINATION_PATTERNS,
    ConversationExtractor,
    is_llm_contaminated,
)
from src.ingestion.schema import CleanedMessage, RAGChunk, SFTDialogue

HIGH_QUALITY_REPLY = (
    "Для PostgreSQL настройте синхронную репликацию через primary_conninfo и standby_conninfo. "
    "Включите wal_level=replica, создайте standby командой pg_basebackup и проверяйте состояние "
    "через SELECT client_state, sent_lsn, write_lsn, replay_lsn FROM pg_stat_replication. "
    "В Kubernetes настройте readinessProbe, livenessProbe и rolling update, чтобы исключить "
    "split-brain при недоступности primary. Patroni автоматизирует failover PostgreSQL в production."
)

LOW_QUALITY_TURN = "Как настроить Docker?"


def make_message(
    msg_id: int,
    text: str,
    *,
    author_id: str = "user",
    author: str | None = None,
    timestamp: str | None = None,
    domain: str = "devops_infra",
    tags: list[str] | None = None,
    reply_to_id: int | None = None,
    chat_name: str = "Infrastructure Chat",
) -> CleanedMessage:
    return CleanedMessage(
        msg_id=msg_id,
        chat_id=700,
        chat_name=chat_name,
        timestamp=timestamp or f"2024-05-{(msg_id % 28) + 1:02d}T12:{msg_id % 60:02d}:00",
        unixtime=1_700_000_000 + msg_id,
        author_anon=author or f"Anonymous_{author_id}",
        author_id_anon=author_id,
        text_clean=text,
        reply_to_id=reply_to_id,
        domain=domain,
        tags=tags or [],
    )


def build_sft_thread(*, turns: int = MAX_SFT_TURNS + 4, thread_id: int = 1) -> dict[int, list[CleanedMessage]]:
    messages = []
    for index in range(turns):
        author_id = "user_a" if index % 2 == 0 else "user_b"
        text = f"Технический вопрос номер {index} про настройку PostgreSQL?"
        if index % 2:
            text = HIGH_QUALITY_REPLY
        messages.append(make_message(index + 1, text, author_id=author_id, reply_to_id=index or None))
    return {thread_id: messages}


class TestExtractSFTDialogues:
    def setup_method(self) -> None:
        self.extractor = ConversationExtractor(min_quality_score=2.0)

    def test_extracts_multi_turn_dialogue_with_two_authors_and_metadata(self) -> None:
        threads = {
            42: [
                make_message(1, "Как настроить PostgreSQL в Kubernetes?", author_id="user_a"),
                make_message(2, HIGH_QUALITY_REPLY, author_id="user_b", reply_to_id=1),
                make_message(
                    3,
                    "Нужно ли менять readinessProbe после rolling update?",
                    author_id="user_a",
                    reply_to_id=2,
                ),
                make_message(4, HIGH_QUALITY_REPLY, author_id="user_b", reply_to_id=3),
            ]
        }

        dialogues = self.extractor.extract_sft_dialogues(threads)

        assert len(dialogues) == 1
        dialogue = dialogues[0]
        assert isinstance(dialogue, SFTDialogue)
        assert dialogue.thread_id == 42
        assert dialogue.chat_name == "Infrastructure Chat"
        assert dialogue.topic_domain == "devops_infra"
        assert dialogue.turn_count == 4
        assert [turn.author for turn in dialogue.messages] == [
            "Anonymous_user_a",
            "Anonymous_user_b",
            "Anonymous_user_a",
            "Anonymous_user_b",
        ]

    def test_requires_at_least_two_distinct_authors(self) -> None:
        threads = {
            1: [
                make_message(1, "Как работает индекс PostgreSQL?", author_id="user_a"),
                make_message(2, HIGH_QUALITY_REPLY, author_id="user_a", reply_to_id=1),
            ]
        }

        assert self.extractor.extract_sft_dialogues(threads) == []

    @pytest.mark.parametrize("reaction", ["да", "ок", "спасибо", "лол"])
    def test_filters_trivial_reactions_from_quality_candidates(self, reaction: str) -> None:
        threads = {
            1: [
                make_message(1, LOW_QUALITY_TURN, author_id="user_a"),
                make_message(2, reaction, author_id="user_b", reply_to_id=1),
            ]
        }

        assert self.extractor.extract_sft_dialogues(threads) == []

    def test_rejects_dialogue_containing_llm_contamination(self) -> None:
        threads = {
            1: [
                make_message(1, "Как настроить PostgreSQL в Kubernetes?", author_id="user_a"),
                make_message(
                    2,
                    "Как искусственный интеллект, используйте индекс PostgreSQL и Kubernetes readinessProbe.",
                    author_id="user_b",
                    reply_to_id=1,
                ),
            ]
        }

        assert self.extractor.extract_sft_dialogues(threads) == []

    def test_caps_mega_thread_at_max_sft_turns(self) -> None:
        dialogues = self.extractor.extract_sft_dialogues(build_sft_thread(turns=MAX_SFT_TURNS + 20))

        assert len(dialogues) == 1
        assert dialogues[0].turn_count == MAX_SFT_TURNS
        assert len(dialogues[0].messages) == MAX_SFT_TURNS
        assert dialogues[0].messages[0].content.startswith("Технический вопрос номер 0")
        assert dialogues[0].messages[-1].content == HIGH_QUALITY_REPLY

    def test_roles_strictly_alternate_with_multiple_authors(self) -> None:
        threads = {
            1: [
                make_message(1, "Как настроить PostgreSQL?", author_id="user_a"),
                make_message(2, HIGH_QUALITY_REPLY, author_id="user_b", reply_to_id=1),
                make_message(
                    3,
                    "Как настроить Kubernetes deployment?",
                    author_id="user_c",
                    reply_to_id=1,
                ),
                make_message(4, HIGH_QUALITY_REPLY, author_id="user_b", reply_to_id=3),
            ]
        }

        dialogue = self.extractor.extract_sft_dialogues(threads)[0]

        assert [turn.role for turn in dialogue.messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]

    def test_two_turn_low_quality_dialogue_is_rejected(self) -> None:
        threads = {
            1: [
                make_message(1, LOW_QUALITY_TURN, author_id="user_a"),
                make_message(2, "Не знаю.", author_id="user_b", reply_to_id=1),
            ]
        }

        assert self.extractor.extract_sft_dialogues(threads) == []

    def test_four_turn_dialogue_is_allowed_below_quality_threshold(self) -> None:
        threads = {
            1: [
                make_message(1, LOW_QUALITY_TURN, author_id="user_a"),
                make_message(2, "Не знаю.", author_id="user_b", reply_to_id=1),
                make_message(3, LOW_QUALITY_TURN, author_id="user_a", reply_to_id=2),
                make_message(4, "Не знаю.", author_id="user_b", reply_to_id=3),
            ]
        }

        dialogues = self.extractor.extract_sft_dialogues(threads)

        assert len(dialogues) == 1
        assert dialogues[0].turn_count == 4
        assert dialogues[0].quality_score < 2.0

    def test_sorts_dialogues_by_quality_score_descending(self) -> None:
        high_thread = {
            1: [
                make_message(1, "Как настроить PostgreSQL в Kubernetes?", author_id="a"),
                make_message(2, HIGH_QUALITY_REPLY, author_id="b", reply_to_id=1),
            ]
        }
        low_thread = {
            2: [
                make_message(3, LOW_QUALITY_TURN, author_id="a"),
                make_message(4, "Не знаю.", author_id="b", reply_to_id=3),
                make_message(5, LOW_QUALITY_TURN, author_id="a", reply_to_id=4),
                make_message(6, "Не знаю.", author_id="b", reply_to_id=5),
            ]
        }

        dialogues = self.extractor.extract_sft_dialogues({**high_thread, **low_thread})

        assert [dialogue.thread_id for dialogue in dialogues] == [1, 2]
        assert dialogues[0].quality_score > dialogues[1].quality_score


class TestExtractDPOPairs:
    def setup_method(self) -> None:
        self.extractor = ConversationExtractor()

    def test_extracts_pair_from_root_and_distinct_reply_authors(self) -> None:
        root = make_message(
            1,
            "Как правильно настроить репликацию PostgreSQL в Kubernetes?",
            author_id="questioner",
            domain="backend_databases",
        )
        chosen = make_message(2, HIGH_QUALITY_REPLY, author_id="expert", reply_to_id=1)
        rejected = make_message(3, "Вроде достаточно.", author_id="newcomer", reply_to_id=1)

        pairs = self.extractor.extract_dpo_pairs({7: [root, chosen, rejected]})

        assert len(pairs) == 1
        pair = pairs[0]
        assert pair["thread_id"] == 7
        assert pair["prompt"] == root.text_clean
        assert pair["chosen"] == chosen.text_clean
        assert pair["rejected"] == rejected.text_clean
        assert pair["domain"] == "backend_databases"
        assert pair["chosen_quality"] >= 3.0
        assert pair["chosen_quality"] - pair["rejected_quality"] >= 1.5

    def test_rejects_quality_margin_below_one_and_half(self) -> None:
        words_30 = " ".join(["слово"] * 30) + "."
        words_40 = " ".join(["слово"] * 40) + "."
        root = make_message(
            1,
            "Какой вариант настройки PostgreSQL предпочтительнее?",
            author_id="questioner",
        )
        chosen = make_message(2, words_40, author_id="expert", reply_to_id=1)
        rejected = make_message(3, words_30, author_id="newcomer", reply_to_id=1)

        assert self.extractor.extract_dpo_pairs({1: [root, chosen, rejected]}) == []

    def test_rejects_best_reply_below_quality_floor(self) -> None:
        root = make_message(1, "Как настроить PostgreSQL в Kubernetes?", author_id="questioner")
        chosen = make_message(2, "Нет настроек.", author_id="expert", reply_to_id=1)
        rejected = make_message(3, "Не уверен.", author_id="newcomer", reply_to_id=1)

        assert self.extractor.extract_dpo_pairs({1: [root, chosen, rejected]}) == []

    def test_rejects_contaminated_top_scoring_reply(self) -> None:
        root = make_message(1, "Как настроить PostgreSQL в Kubernetes?", author_id="questioner")
        contaminated = make_message(
            2,
            "As an AI language model, configure PostgreSQL replication and Kubernetes probes.",
            author_id="bot",
            reply_to_id=1,
        )
        rejected = make_message(3, "Не уверен.", author_id="newcomer", reply_to_id=1)
        quality_by_text = {
            root.text_clean: 1.0,
            contaminated.text_clean: 10.0,
            rejected.text_clean: 2.0,
        }

        def quality(msg: CleanedMessage) -> float:
            return quality_by_text[msg.text_clean]

        with patch.object(self.extractor, "compute_message_quality", side_effect=quality):
            pairs = self.extractor.extract_dpo_pairs({1: [root, contaminated, rejected]})

        assert pairs == []


class TestExtractRAGChunks:
    def setup_method(self) -> None:
        self.extractor = ConversationExtractor()

    def test_builds_chunk_with_thread_metadata_and_author_prefixes(self) -> None:
        threads = {
            42: [
                make_message(
                    1,
                    "Как устроен алгоритм репликации PostgreSQL?",
                    author_id="user_a",
                    author="Author_A",
                    timestamp="2024-01-10T09:00:00",
                    domain="backend_databases",
                    tags=["postgresql", "replication"],
                ),
                make_message(
                    2,
                    "PostgreSQL передаёт WAL-данные и подтверждает фиксацию после quorum.",
                    author_id="user_b",
                    author="Author_B",
                    timestamp="2024-01-12T18:30:00",
                    domain="backend_databases",
                    tags=["wal", "production"],
                    reply_to_id=1,
                ),
            ]
        }

        chunks = self.extractor.extract_rag_chunks(threads, max_tokens_per_chunk=800)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, RAGChunk)
        assert chunk.chunk_id == "rag_kb_000042"
        assert chunk.thread_id == 42
        assert chunk.chat_name == "Infrastructure Chat"
        assert chunk.title == "Как устроен алгоритм репликации PostgreSQL?"
        assert chunk.date_range == "2024-01-10 — 2024-01-12"
        assert chunk.participants_count == 2
        assert chunk.message_count == 2
        assert chunk.topic_domain == "backend_databases"
        assert chunk.topic_tags == ["postgresql", "production", "replication", "wal"]
        assert chunk.content == (
            "[Author_A]: Как устроен алгоритм репликации PostgreSQL?\n"
            "[Author_B]: PostgreSQL передаёт WAL-данные и подтверждает фиксацию после quorum."
        )
        assert chunk.token_count == int(len(chunk.content.split()) * 1.35)

    def test_skips_thread_under_fifteen_estimated_tokens(self) -> None:
        threads = {
            1: [
                make_message(1, " ".join(["слово"] * 9), author_id="user_a"),
                make_message(2, "+", author_id="user_b", reply_to_id=1),
            ]
        }

        assert self.extractor.extract_rag_chunks(threads) == []

    def test_uses_single_date_when_messages_are_on_same_day(self) -> None:
        threads = {
            1: [
                make_message(1, "Подробный вопрос о PostgreSQL", timestamp="2024-03-01T09:00:00"),
                make_message(
                    2,
                    "Подробный ответ о PostgreSQL с примером запроса и описанием настройки",
                    timestamp="2024-03-01T10:00:00",
                ),
            ]
        }

        chunk = self.extractor.extract_rag_chunks(threads)[0]

        assert chunk.date_range == "2024-03-01"


class TestComputeMessageQuality:
    def setup_method(self) -> None:
        self.extractor = ConversationExtractor()

    def test_short_messages_score_zero(self) -> None:
        assert self.extractor.compute_message_quality(make_message(1, "")) == 0.0
        assert self.extractor.compute_message_quality(make_message(2, "ок")) == 0.0

    def test_base_and_word_count_bonus(self) -> None:
        two_words = self.extractor.compute_message_quality(make_message(1, "обычное сообщение"))
        capped_words = self.extractor.compute_message_quality(make_message(2, " ".join(["слово"] * 45)))

        assert two_words == 1.13
        assert capped_words == 4.0

    def test_technical_keyword_density_bonus_is_capped(self) -> None:
        text = " ".join(["Python", "PostgreSQL", "Redis", "Docker", "Kubernetes"] + ["слово"] * 40)
        message = make_message(1, f"{text}. Второе предложение.")

        assert self.extractor.compute_message_quality(message) == 7.5

    def test_punctuation_and_sentence_structure_bonus(self) -> None:
        message = make_message(1, "Обычное предложение. Второе предложение.")

        assert self.extractor.compute_message_quality(message) == 1.77

    def test_two_word_trivial_reaction_score(self) -> None:
        assert self.extractor.compute_message_quality(make_message(1, "не знаю")) == 0.2
        assert self.extractor.compute_message_quality(make_message(2, "Спасибо!")) == 0.0

    def test_llm_contamination_score(self) -> None:
        message = make_message(
            1,
            "Как языковая модель, я объясню настройку PostgreSQL и Kubernetes.",
        )

        assert self.extractor.compute_message_quality(message) == 0.1


AI_SELF_IDENTIFICATION_SAMPLES = [
    "Какая языковая модель отвечает на этот вопрос?",
    "Это искусственный интеллект, а не человек.",
    "Я — виртуальный ассистент и помогаю пользователям.",
    "As an AI, I cannot access private company systems.",
    "I'm a language model trained on public documents.",
    "Нас (модели Mistral) тренируют на разговорных данных.",
    "Мои разработчики из Anthropic ограничили доступ к файлам.",
    "Решение сформировано диалоговой системой.\nClaude 4.5 Sonnet",
]


class TestIsLLMContaminated:
    @pytest.mark.parametrize("text", AI_SELF_IDENTIFICATION_SAMPLES)
    def test_detects_russian_and_english_self_identification(self, text: str) -> None:
        assert is_llm_contaminated(text) is True
        assert any(pattern.search(text) for pattern in AI_CONTAMINATION_PATTERNS)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "Мы перенесли сервисы на FastAPI и Kubernetes.",
            "Используйте PostgreSQL, Redis и Docker.",
            "The model predicts latency from service metrics.",
        ],
    )
    def test_does_not_flag_human_technical_conversations(self, text: str) -> None:
        assert is_llm_contaminated(text) is False
        assert not any(pattern.search(text) for pattern in AI_CONTAMINATION_PATTERNS)
