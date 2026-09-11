"""Hybrid retrieval pipeline for Russian IT knowledge base.

Lexical pre-filter + TF-IDF vector re-rank. Runs on CPU with only
``scikit-learn`` (already a core dependency), no external vector DB needed.

Design:
* Stage 1 — fast vectorized ``str.contains`` pre-filter (up to 5k candidates).
* Stage 2 — per-query TF-IDF fit on candidates + cosine re-rank.
  Per-query fit keeps memory flat even for 325k-chunk KBs (no global
  325k x 30k matrix in RAM) and degrades gracefully to lexical scores
  when sklearn is missing or candidates are tiny.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_PREFILTER_CAP = 5000
_TFIDF_MAX_FEATURES = 20000

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]{3,}", re.UNICODE)


def _extract_keywords(query: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(query.lower())][:10]


def _lexical_scores(contents: pd.Series, keywords: list[str]) -> list[float]:
    scores: list[float] = []
    for content in contents:
        c_lower = str(content).lower()
        scores.append(sum(1.5 for kw in keywords if kw in c_lower))
    return scores


class LocalRAGPipeline:
    """RAG engine for retrieval and context injection from knowledge chunks.

    Backward compatible with the previous lexical-only version:
    ``search(query, top_k, domain_filter)`` keeps its signature; a new
    ``mode`` argument selects ``"hybrid"`` (default), ``"lexical"`` or
    ``"tfidf"``.
    """

    def __init__(
        self,
        parquet_kb_path: Path | str,
        *,
        use_tfidf: bool = True,
        prefilter_cap: int = _PREFILTER_CAP,
    ) -> None:
        self.parquet_kb_path = Path(parquet_kb_path)
        self.df_kb: pd.DataFrame = pd.DataFrame()
        self.use_tfidf = use_tfidf
        self.prefilter_cap = prefilter_cap
        self.retriever: str = "lexical"
        self._load_knowledge_base()

    def _load_knowledge_base(self) -> None:
        if not self.parquet_kb_path.exists():
            logger.warning("Knowledge base parquet not found at %s", self.parquet_kb_path)
            return
        logger.info("Loading RAG knowledge base from %s...", self.parquet_kb_path)
        self.df_kb = pd.read_parquet(self.parquet_kb_path)
        logger.info("Loaded %s RAG knowledge chunks.", f"{len(self.df_kb):,}")

    @property
    def df(self) -> pd.DataFrame:
        """Alias for df_kb for backward compatibility."""
        return self.df_kb

    def __len__(self) -> int:
        return len(self.df_kb)

    def get_stats(self) -> dict[str, Any]:
        return {
            "chunks": len(self.df_kb),
            "retriever": self.retriever,
            "use_tfidf": self.use_tfidf,
            "domains": sorted(self.df_kb["topic_domain"].dropna().unique().tolist())
            if not self.df_kb.empty and "topic_domain" in self.df_kb.columns
            else [],
        }

    def search(
        self,
        query: str,
        top_k: int = 3,
        domain_filter: str | None = None,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Retrieve top-k chunks. Never raises on retrieval errors — returns []."""
        if self.df_kb.empty:
            return []
        keywords = _extract_keywords(query)
        if not keywords:
            return []

        df = self.df_kb
        if domain_filter and "topic_domain" in df.columns:
            filtered = df[df["topic_domain"] == domain_filter]
            if not filtered.empty:
                df = filtered

        try:
            regex_pattern = "|".join(re.escape(kw) for kw in keywords)
            mask = df["content"].str.contains(regex_pattern, case=False, na=False, regex=True)
            sub_df = df[mask]
        except Exception:
            sub_df = df

        if sub_df.empty:
            return []
        if len(sub_df) > self.prefilter_cap:
            # Keep lexical-best candidates before vector re-rank.
            tmp = sub_df.copy()
            tmp["_lex"] = _lexical_scores(tmp["content"], keywords)
            sub_df = tmp.sort_values(by="_lex", ascending=False).head(self.prefilter_cap).drop(columns=["_lex"])

        use_vector = self.use_tfidf and mode in ("hybrid", "tfidf") and len(sub_df) >= 2
        if use_vector:
            ranked = self._tfidf_rerank(sub_df, query, keywords, mode=mode)
            if ranked is not None:
                self.retriever = "hybrid-tfidf"
                return ranked[:top_k]
        self.retriever = "lexical"
        return self._lexical_top(sub_df, keywords, top_k)

    def _lexical_top(self, sub_df: pd.DataFrame, keywords: list[str], top_k: int) -> list[dict[str, Any]]:
        scored = sub_df.copy()
        scored["relevance_score"] = _lexical_scores(scored["content"], keywords)
        top = scored.sort_values(by="relevance_score", ascending=False).head(top_k)
        return [self._row_to_hit(row) for _, row in top.iterrows() if float(row["relevance_score"]) > 0]

    def _tfidf_rerank(
        self, sub_df: pd.DataFrame, query: str, keywords: list[str], *, mode: str
    ) -> list[dict[str, Any]] | None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except Exception as e:  # pragma: no cover - missing optional dep
            logger.warning("sklearn unavailable, lexical fallback: %s", e)
            return None
        try:
            contents = sub_df["content"].astype(str).tolist()
            vectorizer = TfidfVectorizer(
                max_features=_TFIDF_MAX_FEATURES,
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            doc_matrix = vectorizer.fit_transform(contents + [query])
            query_vec = doc_matrix[-1]
            doc_vecs = doc_matrix[:-1]
            cosines = cosine_similarity(doc_vecs, query_vec).ravel()
            lex = _lexical_scores(sub_df["content"], keywords)
            max_lex = max(lex) if max(lex) > 0 else 1.0
            ranked = sub_df.copy()
            if mode == "tfidf":
                ranked["relevance_score"] = [round(float(c) * 10, 2) for c in cosines]
            else:  # hybrid: 65% vector + 35% lexical (both 0..1, scaled to 0..10)
                ranked["relevance_score"] = [
                    round((0.65 * float(c) + 0.35 * (lx / max_lex)) * 10, 2) for c, lx in zip(cosines, lex, strict=True)
                ]
            ranked = ranked[ranked["relevance_score"] > 0].sort_values(by="relevance_score", ascending=False)
            if ranked.empty:
                return None
            return [self._row_to_hit(row) for _, row in ranked.iterrows()]
        except Exception as e:
            logger.warning("TF-IDF rerank failed, lexical fallback: %s", e)
            return None

    @staticmethod
    def _row_to_hit(row: pd.Series) -> dict[str, Any]:
        tags = row.get("topic_tags", [])
        if not isinstance(tags, (list, tuple)):
            tags = []
        return {
            "chunk_id": row.get("chunk_id", ""),
            "title": row.get("title", ""),
            "domain": row.get("topic_domain", "general"),
            "tags": list(tags),
            "content": str(row.get("content", "")),
            "date_range": row.get("date_range", ""),
            "score": round(float(row.get("relevance_score", 0.0)), 2),
        }

    def format_rag_prompt(self, user_query: str, retrieved_contexts: list[dict[str, Any]]) -> str:
        """Format retrieved context into an augmented prompt for the LLM."""
        if not retrieved_contexts:
            return user_query
        context_str = "\n\n---\n\n".join(
            f"[Контекст из базы знаний #{i + 1} | {c['title']} ({c['date_range']})]:\n{c['content']}"
            for i, c in enumerate(retrieved_contexts)
        )
        return (
            "Используй следующий подтвержденный практический опыт инженеров из базы знаний для точного ответа на вопрос:\n\n"
            f"{context_str}\n\n---\nВопрос пользователя: {user_query}\nОтвет эксперта:"
        )
