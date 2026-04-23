"""Cross-encoder reranker implementing the retrieve-and-rerank pattern.

A bi-encoder (FAISS dense retrieval) gives a fast first pass. The
cross-encoder here rescores the top candidates with full attention
between the query and each chunk, producing a more accurate ordering.
The model is loaded lazily on the first rerank call, so tests that
do not exercise reranking avoid the model download.

The low-confidence refusal continues to use the bi-encoder cosine
scores; the cross-encoder output is an unbounded logit and would
not share the same threshold.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_pipeline.models import RetrievalResult

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Lazy-loaded SBERT cross-encoder reranker."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _ensure_loaded(self) -> CrossEncoder:
        """Load the cross-encoder on first use, then cache."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Re-score and re-order results for query.

        Args:
            query: User's processed query string.
            results: Bi-encoder retrieval results to rescore.
            top_k: Optional truncation of the final list. None returns
                every reranked result.

        Returns:
            New RetrievalResult instances with the same chunk metadata
            but score replaced by the cross-encoder rerank score
            (an unbounded logit, typically in the range -10 to +15). The
            list is sorted descending by the rerank score and truncated
            to top_k if provided.

        Notes:
            The low-confidence refusal is checked on the pre-rerank
            bi-encoder scores. The rerank score is not in [0, 1], so
            the existing 0.4 threshold would not apply to it; the
            pipeline enforces this ordering.
        """
        if not results:
            return []

        model = self._ensure_loaded()
        pairs: list[tuple[str, str]] = [(query, r.text) for r in results]
        scores: np.ndarray = model.predict(pairs, show_progress_bar=False)

        rescored = [
            RetrievalResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=float(s),
                source=r.source,
                title=r.title,
                section=r.section,
                heading_path=r.heading_path,
            )
            for r, s in zip(results, scores, strict=True)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        if top_k is None:
            return rescored
        if top_k < 0:
            raise ValueError(
                f"top_k must be a non-negative integer or None, got {top_k}"
            )
        return rescored[:top_k]
