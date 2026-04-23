"""FAISS-based retriever for finding relevant chunks."""

import logging

import faiss
import numpy as np

from rag_pipeline.config import RAGConfig
from rag_pipeline.models import RetrievalResult

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """Retrieves the most relevant chunks for a query using FAISS similarity search."""

    def __init__(
        self, index: faiss.Index, metadata: list[dict], config: RAGConfig
    ):
        self.index = index
        self.metadata = metadata
        self.config = config

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
        section_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """Search the FAISS index for chunks similar to the query.

        Args:
            query_embedding: 1-D array of shape (embedding_dim,)
            top_k: Number of results to return. Defaults to config.top_k.
            section_filter: If set, only return chunks from this section.
                Over-retrieves 3x then filters.

        Returns:
            List of RetrievalResult sorted by descending score.
        """
        top_k = top_k or self.config.top_k
        fetch_k = top_k * 3 if section_filter else top_k

        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < self.config.similarity_threshold:
                continue

            chunk_meta = self.metadata[idx]

            if section_filter and chunk_meta.get("section", "") != section_filter:
                continue

            results.append(
                RetrievalResult(
                    chunk_id=chunk_meta["id"],
                    text=chunk_meta["text"],
                    score=float(score),
                    source=chunk_meta.get("source", ""),
                    title=chunk_meta.get("title", ""),
                    section=chunk_meta.get("section", ""),
                    heading_path=chunk_meta.get("heading_path", []),
                )
            )

        results = results[:top_k]

        logger.info(
            "Retrieved %d results (top_k=%d, filter=%s)",
            len(results),
            top_k,
            section_filter,
        )
        return results
