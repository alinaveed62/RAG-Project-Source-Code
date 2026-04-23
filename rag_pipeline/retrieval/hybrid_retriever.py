"""Hybrid retriever built on Reciprocal Rank Fusion.

Combines a dense bi-encoder retriever (FAISSRetriever) with a sparse
term-matching retriever (BM25Retriever) using Reciprocal Rank Fusion
(Cormack et al., 2009):

    score(d) = sum over runs r of 1 / (k_rrf + rank(d, r) + 1)

rank(d, r) is the zero-based rank of document d in run r, and k_rrf
is a constant (60 by convention) that damps the contribution from
lower-ranked items.

The evaluation experiment that compared retrieval modes calls into
this class so the production code path and the experiment use the
same implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_pipeline.models import RetrievalResult
from rag_pipeline.retrieval.bm25_retriever import BM25Retriever
from rag_pipeline.retrieval.retriever import FAISSRetriever

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Dense plus sparse retrieval fused via Reciprocal Rank Fusion.

    Callers must pass both the raw query string (for BM25) and the
    query embedding (for FAISS); this class does not encode queries
    itself. section_filter, when given, is propagated to both
    underlying retrievers.
    """

    def __init__(
        self,
        dense: FAISSRetriever,
        sparse: BM25Retriever,
        k_rrf: int = 60,
    ) -> None:
        if k_rrf < 1:
            raise ValueError(f"k_rrf must be >= 1, got {k_rrf}")
        self.dense = dense
        self.sparse = sparse
        self.k_rrf = k_rrf

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int | None = None,
        section_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """Run both retrievers and fuse their rankings.

        Args:
            query: Raw query string for BM25.
            query_embedding: Dense query embedding (1-D array).
            top_k: Number of fused results to return.
                Defaults to the dense retriever's config.top_k.
            section_filter: Optional section restriction; propagated to
                both underlying retrievers.

        Returns:
            A list of fused RetrievalResult sorted by RRF score
            (descending) and truncated to top_k.
        """
        effective_top_k = top_k if top_k is not None else self.dense.config.top_k
        dense_results = self.dense.retrieve(
            query_embedding,
            top_k=effective_top_k,
            section_filter=section_filter,
        )
        sparse_results = self.sparse.retrieve(
            query,
            top_k=effective_top_k,
            section_filter=section_filter,
        )
        return self._fuse(dense_results, sparse_results, effective_top_k)

    def _fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Combine two ranked lists into a single RRF-scored list.

        When the same chunk_id appears in both runs, the dense copy
        wins as the representative. The dense loop runs first and
        setdefault is a no-op on the sparse pass for IDs already seen.
        In production this is neutral: both retrievers are built from
        the same metadata list, so their non-score fields (text,
        source, title, section, heading_path) match per chunk_id.
        Only the fused RRF score is recomputed below.
        """
        scores: dict[str, float] = {}
        representative: dict[str, RetrievalResult] = {}

        for rank, r in enumerate(dense_results):
            contribution = 1.0 / (self.k_rrf + rank + 1)
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + contribution
            representative.setdefault(r.chunk_id, r)

        for rank, r in enumerate(sparse_results):
            contribution = 1.0 / (self.k_rrf + rank + 1)
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + contribution
            representative.setdefault(r.chunk_id, r)

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]

        fused: list[RetrievalResult] = []
        for cid in sorted_ids:
            r = representative[cid]
            fused.append(
                RetrievalResult(
                    chunk_id=cid,
                    text=r.text,
                    score=scores[cid],
                    source=r.source,
                    title=r.title,
                    section=r.section,
                    heading_path=r.heading_path,
                )
            )
        return fused
