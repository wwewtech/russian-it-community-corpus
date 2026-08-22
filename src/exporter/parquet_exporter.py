"""
Apache Parquet exporter with zstd compression for high-performance ML dataset loading.
"""

import logging
from pathlib import Path
from typing import List, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.schema import CleanedMessage, RAGChunk, SFTDialogue

logger = logging.getLogger(__name__)


class ParquetExporter:
    """
    Exports clean messages, SFT dialogues, and RAG knowledge chunks to compressed Apache Parquet format.
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_messages(self, messages: List[CleanedMessage], file_name: str = "full_clean_messages.parquet") -> Path:
        """Export all cleaned messages to Parquet."""
        out_path = self.output_dir / file_name
        logger.info(f"Exporting {len(messages)} messages to Parquet at {out_path}...")

        records = [
            {
                "msg_id": m.msg_id,
                "chat_id": m.chat_id,
                "chat_name": m.chat_name,
                "timestamp": m.timestamp,
                "unixtime": m.unixtime,
                "author_anon": m.author_anon,
                "author_id_anon": m.author_id_anon,
                "text_clean": m.text_clean,
                "reply_to_id": m.reply_to_id,
                "domain": m.domain,
                "tags": m.tags,
                "sentiment_score": m.sentiment_score,
                "token_count_approx": m.token_count_approx,
                "is_question": m.is_question,
                "thread_id": m.thread_id,
            }
            for m in messages
        ]

        df = pd.DataFrame.from_records(records)
        df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
        logger.info(f"Successfully saved {out_path} (Size: {out_path.stat().st_size / (1024*1024):.2f} MB)")
        return out_path

    def export_sft_dialogues(self, dialogues: List[SFTDialogue], file_name: str = "sft_dialogues.parquet") -> Path:
        """Export SFT dialogues to Parquet."""
        out_path = self.output_dir / file_name
        logger.info(f"Exporting {len(dialogues)} SFT dialogues to Parquet at {out_path}...")

        records = [
            {
                "thread_id": d.thread_id,
                "chat_name": d.chat_name,
                "topic_domain": d.topic_domain,
                "topic_tags": d.topic_tags,
                "messages": [m.model_dump() for m in d.messages],
                "quality_score": d.quality_score,
                "turn_count": d.turn_count,
                "total_tokens": d.total_tokens,
            }
            for d in dialogues
        ]

        df = pd.DataFrame.from_records(records)
        df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
        logger.info(f"Successfully saved {out_path} (Size: {out_path.stat().st_size / (1024*1024):.2f} MB)")
        return out_path

    def export_rag_chunks(self, chunks: List[RAGChunk], file_name: str = "rag_knowledge_base.parquet") -> Path:
        """Export RAG chunks to Parquet."""
        out_path = self.output_dir / file_name
        logger.info(f"Exporting {len(chunks)} RAG chunks to Parquet at {out_path}...")

        records = [
            {
                "chunk_id": c.chunk_id,
                "thread_id": c.thread_id,
                "chat_name": c.chat_name,
                "title": c.title,
                "topic_domain": c.topic_domain,
                "topic_tags": c.topic_tags,
                "content": c.content,
                "date_range": c.date_range,
                "participants_count": c.participants_count,
                "message_count": c.message_count,
                "token_count": c.token_count,
            }
            for c in chunks
        ]

        df = pd.DataFrame.from_records(records)
        df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
        logger.info(f"Successfully saved {out_path} (Size: {out_path.stat().st_size / (1024*1024):.2f} MB)")
        return out_path
