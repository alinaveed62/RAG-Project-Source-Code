"""Semantic-boundary document chunker.

An alternative to the paragraph-and-sentence recursive splitter in
chunker.py. Instead of cutting at paragraph or token boundaries, the
semantic chunker embeds each sentence, measures cosine distance between
adjacent sentences, and inserts a chunk boundary wherever the distance is
in the top percentile. The intuition (Kamradt 2024; Khasanova and Suh
2025) is that a large drop in semantic similarity marks a topic shift, so
the resulting chunks are topically homogeneous and often retrieve more
cleanly than fixed-width chunks.

The chunker reuses extract_heading_path from the recursive chunker
so heading hierarchies are preserved identically across strategies, and
it accepts an injected embedder so unit tests can run without loading
the real sentence-transformers model.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Protocol

from keats_scraper.config import ChunkConfig
from keats_scraper.models.chunk import Chunk
from keats_scraper.models.document import Document
from keats_scraper.processors._heading_extractor import extract_heading_path

if TYPE_CHECKING:
    import tiktoken

logger = logging.getLogger(__name__)

# Same sentence splitter the recursive chunker uses so switching strategy
# does not introduce a second definition of "sentence" for the corpus.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class Embedder(Protocol):
    """Minimum interface the semantic chunker needs from its embedder.

    Kept as a Protocol rather than a concrete import so tests can
    inject any object whose encode returns an iterable of L2-normed
    vectors, and the real path uses
    sentence-transformers.SentenceTransformer without importing it at
    module load time (the scraper should not have to pay the cost of
    loading torch unless the semantic strategy is selected).
    """

    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        ...  # pragma: no cover - Protocol stub


class SemanticChunker:
    """Chunker that cuts at semantic-similarity breakpoints.

    Args:
        config: ChunkConfig carrying chunk_size,
            semantic_percentile_threshold, semantic_min_tokens,
            and preserve_headings.
        embedder: Optional injected embedder. When None the real
            sentence-transformers/multi-qa-MiniLM-L6-cos-v1 model is
            loaded lazily on first use; tests should pass a fake.
    """

    _DEFAULT_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

    def __init__(
        self,
        config: ChunkConfig | None = None,
        embedder: Embedder | None = None,
    ):
        self.config = config or ChunkConfig()
        self._embedder = embedder
        self._tokenizer: tiktoken.Encoding | str | None = None

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:  # pragma: no cover - real model load
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self._DEFAULT_MODEL)
        return self._embedder

    def _get_tokenizer(self) -> tiktoken.Encoding | str:
        """Lazy-load tiktoken or fall back to whitespace word counting.

        Mirrors Chunker._get_tokenizer so the semantic chunker's
        token budget tracks the recursive chunker's budget under the
        same install conditions.
        """
        if self._tokenizer is None:
            try:
                import tiktoken

                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except ImportError:  # pragma: no cover - tiktoken is shipped
                logger.warning(
                    "tiktoken not installed, falling back to word counts"
                )
                self._tokenizer = "word"
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        tokenizer = self._get_tokenizer()
        if tokenizer == "word":
            return len(text.split())
        return len(tokenizer.encode(text))

    def _split_sentences(self, text: str) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []
        parts = _SENTENCE_RE.split(stripped)
        return [p.strip() for p in parts if p.strip()]

    def _pairwise_distances(self, embeddings: list[list[float]]) -> list[float]:
        """Return 1 - cosine_similarity for each adjacent pair.

        Assumes embeddings are L2-normed (the real
        SentenceTransformer.encode(normalize_embeddings=True) path and
        the test fakes both honour this).
        """
        distances: list[float] = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i]
            b = embeddings[i + 1]
            sim = sum(x * y for x, y in zip(a, b))
            distances.append(1.0 - float(sim))
        return distances

    def _find_breakpoints(self, distances: list[float]) -> list[int]:
        """Pick sentence indices where the topic shifts.

        A breakpoint at index i means sentence i starts a new
        chunk. The function flags indices where the adjacent-pair
        distance sits at or above the configured percentile threshold
        AND is strictly positive. The d > 0 guard avoids the
        degenerate case where every adjacent pair has identical
        embeddings (zero distance): without it, the percentile threshold
        collapses to zero and every sentence boundary would be flagged.
        """
        if not distances:
            return []
        percentile = self.config.semantic_percentile_threshold
        sorted_distances = sorted(distances)
        n = len(sorted_distances)
        # Simple nearest-rank percentile: avoids a numpy dependency at
        # this level and matches np.percentile for our small sentence
        # counts (50-500 per document).
        idx = min(n - 1, max(0, round(percentile / 100.0 * (n - 1))))
        threshold = sorted_distances[idx]
        return [
            i + 1
            for i, d in enumerate(distances)
            if d >= threshold and d > 0.0
        ]

    def _group_sentences(
        self, sentences: list[str], breakpoints: list[int]
    ) -> list[list[str]]:
        """Form initial sentence groups at every breakpoint."""
        groups: list[list[str]] = []
        start = 0
        for bp in breakpoints:
            if bp > start:
                groups.append(sentences[start:bp])
                start = bp
        if start < len(sentences):
            groups.append(sentences[start:])
        return [g for g in groups if g]

    def _merge_small_and_split_large(
        self, groups: list[list[str]]
    ) -> list[str]:
        """Enforce the min/max token budget.

        Neighbouring groups below semantic_min_tokens are glued
        together so the retriever does not see fragmentary chunks; the
        result is then re-split at chunk_size tokens so no chunk
        exceeds the embedder's context window.
        """
        merged: list[list[str]] = []
        for group in groups:
            joined = " ".join(group)
            if merged and self._count_tokens(joined) < self.config.semantic_min_tokens:
                merged[-1].extend(group)
            else:
                merged.append(list(group))

        chunk_texts: list[str] = []
        for group in merged:
            text = " ".join(group)
            if self._count_tokens(text) <= self.config.chunk_size:
                chunk_texts.append(text)
                continue
            # Oversize group: fall back to token-window splitting so no
            # emitted chunk breaks the embedder's context budget. Both
            # backends text.split() and tiktoken.decode return
            # non-empty strings for every slice of a non-empty input, so
            # no whitespace-only filter is needed here.
            tokenizer = self._get_tokenizer()
            size = self.config.chunk_size
            if tokenizer == "word":
                words = text.split()
                for i in range(0, len(words), size):
                    chunk_texts.append(" ".join(words[i : i + size]))
            else:
                ids = tokenizer.encode(text)
                for i in range(0, len(ids), size):
                    chunk_texts.append(tokenizer.decode(ids[i : i + size]))
        return chunk_texts

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split document into semantic chunks, preserving heading path."""
        text = document.content
        if not text or not text.strip():
            logger.warning("Empty document: %s", document.id)
            return []

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            # Too short to find boundaries; emit the whole thing as one
            # chunk, still with heading path and optional heading prefix.
            chunk_texts = [text.strip()]
        else:
            embedder = self._get_embedder()
            embeddings = list(
                embedder.encode(sentences, normalize_embeddings=True)
            )
            distances = self._pairwise_distances(embeddings)
            breakpoints = self._find_breakpoints(distances)
            groups = self._group_sentences(sentences, breakpoints)
            chunk_texts = self._merge_small_and_split_large(groups)

        chunks: list[Chunk] = []
        total_chunks = len(chunk_texts)
        for i, chunk_text in enumerate(chunk_texts):
            # Approximate position by finding the chunk text in the
            # original document so the heading path is computed against
            # real positions. Search with up to the first 40 characters
            # of the chunk (or the full chunk if shorter) so short
            # chunks like one-sentence groups still locate their
            # position. find may still return -1 when the chunker
            # joined sentence fragments with whitespace that differs
            # from the original layout; fall back to the document start
            # so heading_path is always well-defined.
            search_key = chunk_text[: min(40, len(chunk_text))]
            position = text.find(search_key) if search_key else 0
            if position == -1:
                position = 0
            heading_path = extract_heading_path(text, position)

            if self.config.preserve_headings and heading_path:
                heading_context = " > ".join(heading_path)
                chunk_text = f"[Context: {heading_context}]\n\n{chunk_text}"

            chunks.append(
                Chunk.create(
                    text=chunk_text,
                    document_id=document.id,
                    document_title=document.metadata.title,
                    source_url=document.metadata.source_url,
                    chunk_index=i,
                    total_chunks=total_chunks,
                    section=document.metadata.section,
                    heading_path=heading_path,
                )
            )

        logger.info(
            "Semantic chunker produced %d chunks from document '%s'",
            len(chunks),
            document.metadata.title,
        )
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk every document in documents and concatenate results."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(
            "Semantic chunker produced %d chunks across %d documents",
            len(all_chunks),
            len(documents),
        )
        return all_chunks
