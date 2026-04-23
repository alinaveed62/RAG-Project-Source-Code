"""Document chunking for RAG pipeline."""

import re
from typing import TYPE_CHECKING

from keats_scraper.config import ChunkConfig
from keats_scraper.models.chunk import Chunk
from keats_scraper.models.document import Document
from keats_scraper.processors._heading_extractor import extract_heading_path
from keats_scraper.utils.logging_config import get_logger

if TYPE_CHECKING:
    import tiktoken

logger = get_logger()


class Chunker:
    """Splits documents into chunks suitable for RAG embedding."""

    def __init__(self, config: ChunkConfig | None = None):
        """
        Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkConfig()
        self._tokenizer = None

    def _get_tokenizer(self) -> "tiktoken.Encoding | str":
        """Lazy load tiktoken tokenizer."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                logger.warning("tiktoken not installed, using word-based chunking")
                self._tokenizer = "word"
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        tokenizer = self._get_tokenizer()
        if tokenizer == "word":
            return len(text.split())
        return len(tokenizer.encode(text))

    def _extract_heading_at_position(self, text: str, position: int) -> list[str]:
        """Extract the heading hierarchy at the given text position.

        Thin delegator to extract_heading_path. Kept as an instance
        method so existing tests that call
        chunker._extract_heading_at_position(...) continue to work
        after the logic was factored out for reuse by SemanticChunker.
        """
        return extract_heading_path(text, position)

    def _split_oversized(self, text: str) -> list[str]:
        """Split an oversized span into pieces bounded by chunk_size.

        Used when a single sentence (or bullet, or unbroken blob)
        already exceeds chunk_size and the sentence-level pass
        cannot honour the cap on its own. Prefers tiktoken buckets
        when available so pieces align with the embedding model's
        token budget; falls back to whitespace-word buckets when
        tiktoken is not installed.
        """
        tokenizer = self._get_tokenizer()
        size = self.config.chunk_size
        if isinstance(tokenizer, str):
            words = text.split()
            pieces = [
                " ".join(words[i : i + size]) for i in range(0, len(words), size)
            ]
            return [p for p in pieces if p.strip()]

        ids = tokenizer.encode(text)
        pieces = [
            tokenizer.decode(ids[i : i + size]) for i in range(0, len(ids), size)
        ]
        return [p for p in pieces if p.strip()]

    def _split_by_separators(self, text: str) -> list[tuple[str, int]]:
        """
        Split text by configured separators.

        Returns list of (chunk_text, start_position) tuples.
        """
        chunks = []
        current_chunk = []
        current_start = 0
        current_length = 0

        # Split into paragraphs first.
        paragraphs = re.split(r"\n\n+", text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self._count_tokens(para)

            # If the paragraph alone exceeds chunk_size, split it.
            if para_tokens > self.config.chunk_size:
                # Flush whatever is already in the buffer first.
                if current_chunk:
                    chunks.append(("\n\n".join(current_chunk), current_start))
                    current_chunk = []
                    current_length = 0

                # Split the long paragraph into sentences. para is
                # already stripped, so the split will not produce
                # empty strings.
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    sent_tokens = self._count_tokens(sentence)

                    # A single sentence can exceed chunk_size on its
                    # own (long tables rendered as one line, URL-heavy
                    # bullets, or PDF paragraphs without terminal
                    # punctuation). Split it at the token boundary so
                    # no emitted chunk breaks the size contract the
                    # embedder relies on.
                    if sent_tokens > self.config.chunk_size:
                        if current_chunk:
                            chunks.append(
                                ("\n\n".join(current_chunk), current_start)
                            )
                            current_chunk = []
                            current_length = 0
                        sentence_start = text.find(sentence, current_start)
                        if sentence_start == -1:  # pragma: no cover
                            # Defensive: the sentence was split out
                            # of text by re.split, so find should
                            # always succeed. This fallback only
                            # fires if the source text mutates
                            # between split and find, which the
                            # current pipeline does not do.
                            sentence_start = current_start
                        for piece in self._split_oversized(sentence):
                            chunks.append((piece, sentence_start))
                        current_start = sentence_start + len(sentence)
                        continue

                    if current_length + sent_tokens > self.config.chunk_size:
                        if current_chunk:  # pragma: no branch
                            chunks.append(("\n\n".join(current_chunk), current_start))
                        current_chunk = [sentence]
                        pos = text.find(sentence, current_start)
                        if pos != -1:  # pragma: no branch
                            current_start = pos
                        current_length = sent_tokens
                    else:
                        current_chunk.append(sentence)
                        current_length += sent_tokens

            elif current_length + para_tokens > self.config.chunk_size:
                # Flush the current chunk and start a new one.
                if current_chunk:  # pragma: no branch
                    chunks.append(("\n\n".join(current_chunk), current_start))
                current_chunk = [para]
                pos = text.find(para, current_start)
                if pos != -1:  # pragma: no branch
                    current_start = pos
                current_length = para_tokens
            else:
                # Append the paragraph to the current chunk.
                if not current_chunk:
                    pos = text.find(para, current_start)
                    if pos != -1:  # pragma: no branch
                        current_start = pos
                current_chunk.append(para)
                current_length += para_tokens

        # Emit the trailing buffer, if any.
        if current_chunk:
            chunks.append(("\n\n".join(current_chunk), current_start))

        return chunks

    def _add_overlap(
        self, chunks: list[tuple[str, int]]
    ) -> list[tuple[str, int]]:
        """Add overlap between chunks.

        chunk_overlap is measured in the same unit as chunk_size:
        cl100k_base tokens when tiktoken is installed, whitespace-
        delimited words when it is not. _count_tokens applies the
        same fallback, so size and overlap stay on the same scale in
        both modes.
        """
        if not chunks or self.config.chunk_overlap == 0:
            return chunks

        tokenizer = self._get_tokenizer()
        overlap_tokens = self.config.chunk_overlap

        result = []
        for i, (chunk_text, start_pos) in enumerate(chunks):
            if i > 0:
                prev_text = chunks[i - 1][0]

                if isinstance(tokenizer, str):
                    prev_words = prev_text.split()
                    overlap_words = (
                        prev_words[-overlap_tokens:]
                        if len(prev_words) > overlap_tokens
                        else prev_words
                    )
                    overlap_text = " ".join(overlap_words)
                else:
                    prev_ids = tokenizer.encode(prev_text)
                    tail_ids = (
                        prev_ids[-overlap_tokens:]
                        if len(prev_ids) > overlap_tokens
                        else prev_ids
                    )
                    overlap_text = tokenizer.decode(tail_ids)

                chunk_text = f"...{overlap_text}\n\n{chunk_text}"

            result.append((chunk_text, start_pos))

        return result

    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Split a document into chunks.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        text = document.content

        if not text or not text.strip():
            logger.warning(f"Empty document: {document.id}")
            return []

        # Split into chunks.
        raw_chunks = self._split_by_separators(text)

        # Add the configured overlap between adjacent chunks.
        chunks_with_overlap = self._add_overlap(raw_chunks)

        # Build the Chunk model objects.
        chunks = []
        total_chunks = len(chunks_with_overlap)

        for i, (chunk_text, start_pos) in enumerate(chunks_with_overlap):
            # Heading hierarchy that leads to this position.
            heading_path = self._extract_heading_at_position(text, start_pos)

            # Prepend a heading context line when configured.
            if self.config.preserve_headings and heading_path:
                heading_context = " > ".join(heading_path)
                chunk_text = f"[Context: {heading_context}]\n\n{chunk_text}"

            chunk = Chunk.create(
                text=chunk_text,
                document_id=document.id,
                document_title=document.metadata.title,
                source_url=document.metadata.source_url,
                chunk_index=i,
                total_chunks=total_chunks,
                section=document.metadata.section,
                heading_path=heading_path,
            )

            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks from document '{document.metadata.title}'")
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents to chunk

        Returns:
            Combined list of all chunks
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks
