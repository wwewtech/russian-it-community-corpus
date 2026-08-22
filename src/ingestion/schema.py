"""
Data models and schemas for chat messages and conversational threads.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class TextEntity(BaseModel):
    type: str
    text: str
    href: Optional[str] = None


class NormalizedMessage(BaseModel):
    msg_id: int
    chat_id: int
    chat_name: str
    timestamp: datetime
    unixtime: int
    author_raw: str
    author_id_raw: str
    text_raw: str
    reply_to_id: Optional[int] = None
    has_media: bool = False
    media_type: Optional[str] = None
    is_service: bool = False
    forwarded_from: Optional[str] = None


class CleanedMessage(BaseModel):
    msg_id: int
    chat_id: int
    chat_name: str
    timestamp: str  # ISO format string
    unixtime: int
    author_anon: str
    author_id_anon: str
    text_clean: str
    reply_to_id: Optional[int] = None
    domain: str = "general_tech_chat"
    tags: List[str] = Field(default_factory=list)
    sentiment_score: int = 0
    token_count_approx: int = 0
    is_question: bool = False
    thread_id: Optional[int] = None


class SFTTurn(BaseModel):
    role: str  # "user" or "assistant" (or "human" / "gpt")
    author: str
    content: str
    timestamp: Optional[str] = None


class SFTDialogue(BaseModel):
    thread_id: int
    chat_name: str
    topic_domain: str
    topic_tags: List[str]
    messages: List[SFTTurn]
    quality_score: float = 0.0
    turn_count: int = 0
    total_tokens: int = 0


class RAGChunk(BaseModel):
    chunk_id: str
    thread_id: int
    chat_name: str
    title: str
    topic_domain: str
    topic_tags: List[str]
    content: str
    date_range: str
    participants_count: int
    message_count: int
    token_count: int
