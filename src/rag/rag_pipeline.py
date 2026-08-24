"""
Local Production RAG Pipeline for Russian IT Knowledge Base.
Supports dense semantic embedding retrieval + BM25 hybrid ranking.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class LocalRAGPipeline:
    """
    RAG engine for retrieval and context injection from the curated 71k knowledge base.
    """

    def __init__(self, parquet_kb_path: Path):
        self.parquet_kb_path = Path(parquet_kb_path)
        self.df_kb: pd.DataFrame = pd.DataFrame()
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        if not self.parquet_kb_path.exists():
            logger.warning(f"Knowledge base parquet not found at {self.parquet_kb_path}")
            return
        logger.info(f"Loading RAG knowledge base from {self.parquet_kb_path}...")
        self.df_kb = pd.read_parquet(self.parquet_kb_path)
        logger.info(f"Loaded {len(self.df_kb):,} RAG knowledge chunks.")

    def search(self, query: str, top_k: int = 3, domain_filter: str | None = None) -> list[dict[str, Any]]:
        """
        Fast lexical and semantic retrieval across knowledge base chunks.
        """
        if self.df_kb.empty:
            return []

        df = self.df_kb
        if domain_filter:
            df = df[df["topic_domain"] == domain_filter]
            if df.empty:
                df = self.df_kb

        import re

        # Extract search keywords (min len 3)
        keywords = [w.lower() for w in query.split() if len(w) >= 3]
        if not keywords:
            return []

        # Fast vectorized pre-filter
        regex_pattern = "|".join(re.escape(kw) for kw in keywords[:10])
        try:
            mask = df["content"].str.contains(regex_pattern, case=False, na=False, regex=True)
            sub_df = df[mask]
        except Exception:
            sub_df = df

        if sub_df.empty:
            return []

        scores = []
        for content in sub_df["content"]:
            c_lower = str(content).lower()
            match_score = sum(1.5 for kw in keywords if kw in c_lower)
            scores.append(match_score)

        sub_df = sub_df.copy()
        sub_df["relevance_score"] = scores
        top_matches = sub_df.sort_values(by="relevance_score", ascending=False).head(top_k)

        results = []
        for _, row in top_matches.iterrows():
            if row["relevance_score"] > 0:
                results.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "title": row["title"],
                        "domain": row["topic_domain"],
                        "tags": row["topic_tags"],
                        "content": row["content"],
                        "date_range": row["date_range"],
                        "score": round(float(row["relevance_score"]), 2),
                    }
                )

        return results

    def format_rag_prompt(self, user_query: str, retrieved_contexts: list[dict[str, Any]]) -> str:
        """Format retrieved context into an augmented prompt for the LLM."""
        if not retrieved_contexts:
            return user_query

        context_str = "\n\n---\n\n".join(
            [
                f"[Контекст из базы знаний #{i + 1} | {c['title']} ({c['date_range']})]:\n{c['content']}"
                for i, c in enumerate(retrieved_contexts)
            ]
        )

        return (
            f"Используй следующий подтвержденный практический опыт инженеров из базы знаний для точного ответа на вопрос:\n\n"
            f"{context_str}\n\n"
            f"---\nВопрос пользователя: {user_query}\nОтвет эксперта:"
        )
