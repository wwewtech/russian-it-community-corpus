"""
RAG Knowledge Base JSONL and Vector-ready chunk exporter.
"""

import json
import logging
from pathlib import Path
from typing import List, Union

from src.ingestion.schema import RAGChunk

logger = logging.getLogger(__name__)


class RAGExporter:
    """
    Exports structured RAG chunks for vector databases (Qdrant, Chroma, Pinecone, pgvector).
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_rag_jsonl(self, chunks: List[RAGChunk], file_name: str = "rag_chunks_kb.jsonl") -> Path:
        """Export RAG chunks to JSONL."""
        out_path = self.output_dir / file_name
        logger.info(f"Exporting {len(chunks)} RAG chunks to JSONL at {out_path}...")

        with open(out_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")

        logger.info(f"Saved RAG chunks JSONL at {out_path}")
        return out_path
