"""
Data models and schemas for chat messages and conversational threads.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class TextEntity(BaseModel):
    type: str
    text: str
    href: str | None = None


class NormalizedMessage(BaseModel):
    msg_id: int
    chat_id: int
    chat_name: str
    timestamp: datetime
    unixtime: int
    author_raw: str
    author_id_raw: str
    text_raw: str
    reply_to_id: int | None = None
    has_media: bool = False
    media_type: str | None = None
    is_service: bool = False
    forwarded_from: str | None = None


class CleanedMessage(BaseModel):
    msg_id: int
    chat_id: int
    chat_name: str
    timestamp: str  # ISO format string
    unixtime: int
    author_anon: str
    author_id_anon: str
    text_clean: str
    reply_to_id: int | None = None
    domain: str = "general_tech_chat"
    tags: list[str] = Field(default_factory=list)
    sentiment_score: int = 0
    token_count_approx: int = 0
    is_question: bool = False
    thread_id: int | None = None


class SFTTurn(BaseModel):
    role: str  # "user" or "assistant" (or "human" / "gpt")
    author: str
    content: str
    timestamp: str | None = None


class SFTDialogue(BaseModel):
    thread_id: int
    chat_name: str
    topic_domain: str
    topic_tags: list[str]
    messages: list[SFTTurn]
    quality_score: float = 0.0
    turn_count: int = 0
    total_tokens: int = 0


class RAGChunk(BaseModel):
    chunk_id: str
    thread_id: int
    chat_name: str
    title: str
    topic_domain: str
    topic_tags: list[str]
    content: str
    date_range: str
    participants_count: int
    message_count: int
    token_count: int
