"""BM25-based sparse retriever for baseline comparison."""

import logging
import re

from rank_bm25 import BM25Okapi

from rag_pipeline.config import RAGConfig
from rag_pipeline.models import RetrievalResult

logger = logging.getLogger(__name__)

# Unicode-aware word tokeniser. Strips trailing punctuation so a
# query like "lectures?" still matches documents containing
# "lectures"; a plain str.lower().split() would leave the ? attached
# and miss the match.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Retrieves the most relevant chunks using BM25 term matching.

    Provides a sparse retrieval baseline against which dense FAISS
    retrieval can be compared.
    """

    def __init__(self, chunks: list[dict], config: RAGConfig):
        self.metadata = chunks
        self.config = config

        corpus = [_tokenise(c["text"]) for c in chunks]
        # any(corpus) handles both an empty corpus and a corpus of
        # only empty token lists; a plain "if corpus" would build a
        # BM25Okapi over empty tokens and raise ZeroDivisionError when
        # it tried to compute the average document length.
        if any(corpus):
            self.bm25: BM25Okapi | None = BM25Okapi(corpus)
            logger.info("BM25 index built over %d chunks", len(chunks))
        else:
            self.bm25 = None
            logger.warning(
                "BM25 corpus is empty; retrieval will always return []."
            )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        section_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """Search chunks using BM25 term-matching.

        Args:
            query: Raw text query (not an embedding vector).
            top_k: Number of results to return. Defaults to config.top_k.
            section_filter: If set, only return chunks from this section.

        Returns:
            List of RetrievalResult sorted by descending BM25 score.
        """
        top_k = top_k or self.config.top_k

        if self.bm25 is None:
            return []

        tokenised_query = _tokenise(query)
        scores = self.bm25.get_scores(tokenised_query)

        # Pair each score with its chunk index, then sort descending.
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        results = []
        for idx, score in scored_indices:
            if score <= 0:
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

            if len(results) >= top_k:
                break

        logger.info(
            "BM25 retrieved %d results (top_k=%d, filter=%s)",
            len(results),
            top_k,
            section_filter,
        )
        return results
