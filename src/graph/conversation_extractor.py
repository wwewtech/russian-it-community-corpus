"""
Conversation Extractor for SFT, DPO, and RAG knowledge datasets.
"""

import logging
import re
from collections import defaultdict
from typing import Any

from src.config import DOMAIN_TAXONOMY, MIN_ANSWER_WORDS, MIN_QUESTION_WORDS
from src.ingestion.schema import CleanedMessage, RAGChunk, SFTDialogue, SFTTurn

logger = logging.getLogger(__name__)

# Noise / trivial reaction phrases to filter out from high-quality SFT answers
TRIVIAL_REACTIONS = {
    "+",
    "-",
    "++",
    "+++",
    "да",
    "нет",
    "ок",
    "ладно",
    "понял",
    "спасибо",
    "спс",
    "пасиб",
    "сяб",
    "благодарю",
    "норм",
    "ага",
    "угу",
    "лол",
    "кек",
    "хаха",
    "ахах",
    "пхах",
    "хз",
    "не знаю",
    "жиза",
    "база",
    "согласен",
    "точно",
    "up",
    "ап",
    "топ",
    "класс",
    "огонь",
    "лайк",
    "рил",
    "хмм",
    "мда",
    "хм",
}

CODE_INDICATORS = [
    "def ",
    "class ",
    "import ",
    "from ",
    "return ",
    "async ",
    "await ",
    "SELECT ",
    "FROM ",
    "WHERE ",
    "docker run",
    "kubectl",
    "curl ",
    "npm ",
    "git ",
    "pip install",
    "func ",
    "package ",
    "const ",
    "let ",
    "var ",
    "```",
    "{",
    "}",
    "->",
    "=>",
    "==",
    "!=",
    "http://",
    "https://",
]


class ConversationExtractor:
    """
    Extracts high-quality SFT dialogues, DPO pairs, and RAG knowledge chunks from conversation threads.
    """

    def __init__(
        self,
        min_question_words: int = MIN_QUESTION_WORDS,
        min_answer_words: int = MIN_ANSWER_WORDS,
        min_quality_score: float = 2.0,
    ):
        self.min_question_words = min_question_words
        self.min_answer_words = min_answer_words
        self.min_quality_score = min_quality_score

    def compute_message_quality(self, msg: CleanedMessage) -> float:
        """Calculate quality score for an individual message."""
        text = msg.text_clean
        words = text.split()
        word_count = len(words)
        if word_count < 2:
            return 0.0

        # Check trivial reactions
        if text.lower().strip("!.?,") in TRIVIAL_REACTIONS:
            return 0.2

        score = 1.0
        # Word count bonus (up to 3 points)
        score += min(3.0, word_count / 15.0)

        # Code & technical syntax bonus
        if any(ind in text for ind in CODE_INDICATORS):
            score += 2.0

        # Technical keyword density bonus
        tech_words_count = 0
        text_lower = text.lower()
        for _domain, info in DOMAIN_TAXONOMY.items():
            for kw in info["keywords"]:
                if kw in text_lower:
                    tech_words_count += 1
        score += min(3.0, tech_words_count * 0.6)

        # Punctuation & sentence structure bonus
        if len(re.findall(r"[.!?]", text)) >= 2:
            score += 0.5

        return round(score, 2)

    def extract_sft_dialogues(self, threads: dict[int, list[CleanedMessage]]) -> list[SFTDialogue]:
        """
        Extract multi-turn SFT dialogues from reconstructed threads.
        """
        sft_dialogues: list[SFTDialogue] = []

        for thread_id, msgs in threads.items():
            # Need at least 2 messages for a dialogue
            if len(msgs) < 2:
                continue

            # Check if there are at least two distinct participants
            authors = set(m.author_id_anon for m in msgs)
            if len(authors) < 2:
                continue

            # Group consecutive messages from the same author
            merged_turns: list[tuple[str, str, str, CleanedMessage]] = []  # (author_id, author_name, text, orig_msg)
            for m in msgs:
                if not m.text_clean or len(m.text_clean.strip()) < 3:
                    continue
                if merged_turns and merged_turns[-1][0] == m.author_id_anon:
                    # Append to previous turn
                    prev_author_id, prev_name, prev_text, orig = merged_turns[-1]
                    merged_turns[-1] = (prev_author_id, prev_name, prev_text + "\n" + m.text_clean, orig)
                else:
                    merged_turns.append((m.author_id_anon, m.author_anon, m.text_clean, m))

            if len(merged_turns) < 2:
                continue

            # Convert merged turns into alternating User / Assistant turns
            sft_turns: list[SFTTurn] = []
            total_tokens = 0
            dialogue_quality = 0.0

            first_author_id = merged_turns[0][0]

            for _idx, (author_id, author_name, turn_text, orig_msg) in enumerate(merged_turns):
                # Role assignment: Initial author is 'user', responders are 'assistant'
                # If subsequent questions from initial author, mark 'user', replies mark 'assistant'
                role = "user" if author_id == first_author_id else "assistant"

                # Ensure strict alternating roles if required by downstream LLMs
                if sft_turns and sft_turns[-1].role == role:
                    role = "assistant" if role == "user" else "user"

                approx_tokens = int(len(turn_text.split()) * 1.35)
                total_tokens += approx_tokens

                quality = self.compute_message_quality(orig_msg)
                dialogue_quality += quality

                sft_turns.append(
                    SFTTurn(
                        role=role,
                        author=author_name,
                        content=turn_text,
                        timestamp=orig_msg.timestamp,
                    )
                )

            # Ensure dialogue ends with an assistant response (standard SFT format)
            if sft_turns and sft_turns[-1].role == "user":
                sft_turns.pop()

            if len(sft_turns) >= 2:
                # Determine primary topic and tags across messages in this thread
                domain_counter: dict[str, int] = defaultdict(int)
                tags_collected: set[str] = set()

                for m in msgs:
                    domain_counter[m.domain] += 1
                    tags_collected.update(m.tags)

                primary_domain = (
                    max(domain_counter.items(), key=lambda x: x[1])[0] if domain_counter else "general_tech_chat"
                )
                avg_quality = round(dialogue_quality / len(sft_turns), 2)

                if avg_quality >= self.min_quality_score or len(sft_turns) >= 4:
                    sft_dialogues.append(
                        SFTDialogue(
                            thread_id=thread_id,
                            chat_name=msgs[0].chat_name,
                            topic_domain=primary_domain,
                            topic_tags=sorted(list(tags_collected))[:8],
                            messages=sft_turns,
                            quality_score=avg_quality,
                            turn_count=len(sft_turns),
                            total_tokens=total_tokens,
                        )
                    )

        # Sort dialogues by quality score descending
        sft_dialogues.sort(key=lambda d: (d.quality_score, d.turn_count), reverse=True)
        logger.info(f"Extracted {len(sft_dialogues)} curated high-quality SFT dialogues.")
        return sft_dialogues

    def extract_dpo_pairs(self, threads: dict[int, list[CleanedMessage]]) -> list[dict[str, Any]]:
        """
        Extract Direct Preference Optimization (DPO) pairs (prompt, chosen, rejected)
        when multiple responses exist for the same query.
        """
        dpo_pairs: list[dict[str, Any]] = []

        for thread_id, msgs in threads.items():
            if len(msgs) < 3:
                continue

            root_msg = msgs[0]
            if len(root_msg.text_clean.split()) < self.min_question_words:
                continue

            # Candidate replies from distinct authors
            replies = [
                m for m in msgs[1:] if m.author_id_anon != root_msg.author_id_anon and len(m.text_clean.strip()) > 3
            ]
            if len(replies) < 2:
                continue

            # Score each reply
            scored_replies = [(self.compute_message_quality(r), r) for r in replies]
            scored_replies.sort(key=lambda x: x[0], reverse=True)

            best_score, best_reply = scored_replies[0]
            worst_score, worst_reply = scored_replies[-1]

            # We need a significant quality margin to create a valid preference pair
            if (best_score - worst_score) >= 1.5 and best_score >= 3.0:
                dpo_pairs.append(
                    {
                        "thread_id": thread_id,
                        "prompt": root_msg.text_clean,
                        "chosen": best_reply.text_clean,
                        "rejected": worst_reply.text_clean,
                        "domain": root_msg.domain,
                        "chosen_quality": best_score,
                        "rejected_quality": worst_score,
                    }
                )

        logger.info(f"Extracted {len(dpo_pairs)} valid DPO preference pairs.")
        return dpo_pairs

    def extract_rag_chunks(
        self, threads: dict[int, list[CleanedMessage]], max_tokens_per_chunk: int = 800
    ) -> list[RAGChunk]:
        """
        Segment threads into structured RAG knowledge base chunks with rich metadata.
        """
        rag_chunks: list[RAGChunk] = []

        for thread_id, msgs in threads.items():
            if not msgs:
                continue

            # Determine thread metadata
            chat_name = msgs[0].chat_name
            authors = set(m.author_anon for m in msgs)
            domain_counter: dict[str, int] = defaultdict(int)
            all_tags: set[str] = set()
            for m in msgs:
                domain_counter[m.domain] += 1
                all_tags.update(m.tags)

            primary_domain = (
                max(domain_counter.items(), key=lambda x: x[1])[0] if domain_counter else "general_tech_chat"
            )
            date_start = msgs[0].timestamp[:10]
            date_end = msgs[-1].timestamp[:10]
            date_range = f"{date_start} — {date_end}" if date_start != date_end else date_start

            # Generate informative title from root message
            first_clean = msgs[0].text_clean.replace("\n", " ")
            title = first_clean[:90] + ("..." if len(first_clean) > 90 else "")

            # Build content stream with author context
            formatted_lines = []
            for m in msgs:
                if len(m.text_clean.strip()) > 1:
                    formatted_lines.append(f"[{m.author_anon}]: {m.text_clean}")

            full_thread_content = "\n".join(formatted_lines)
            tokens = int(len(full_thread_content.split()) * 1.35)

            if tokens < 15:
                continue

            chunk = RAGChunk(
                chunk_id=f"rag_kb_{thread_id:06d}",
                thread_id=thread_id,
                chat_name=chat_name,
                title=title if title else f"Technical discussion in {chat_name}",
                topic_domain=primary_domain,
                topic_tags=sorted(list(all_tags))[:6],
                content=full_thread_content,
                date_range=date_range,
                participants_count=len(authors),
                message_count=len(msgs),
                token_count=tokens,
            )
            rag_chunks.append(chunk)

        logger.info(f"Extracted {len(rag_chunks)} structured RAG knowledge base chunks.")
        return rag_chunks
