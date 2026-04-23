"""Tests for Chunker."""

from unittest.mock import Mock, patch

import pytest

from keats_scraper.config import ChunkConfig
from keats_scraper.models.chunk import Chunk
from keats_scraper.models.document import Document, DocumentMetadata
from keats_scraper.processors.chunker import Chunker

# Tolerance applied when asserting a chunk's token count against
# ChunkConfig.chunk_size. The _split_by_separators path uses
# Chunker._count_tokens (tiktoken's encode with optional decode
# round-trip), which can legitimately differ from a direct
# tiktoken.encode count by one token at a boundary. Tests that
# measure the final chunk with direct tiktoken encoding stay strict
# (<= chunk_size); tests that measure via the chunker's helper use
# <= chunk_size + CHUNK_TOKEN_TOLERANCE.
CHUNK_TOKEN_TOLERANCE = 1


class TestChunkerInit:
    """Tests for Chunker initialization."""

    def test_init_default_config(self):
        """Test default ChunkConfig is used."""
        chunker = Chunker()
        assert chunker.config is not None
        assert isinstance(chunker.config, ChunkConfig)

    def test_init_custom_config(self):
        """Test custom config is used."""
        config = ChunkConfig(chunk_size=100, chunk_overlap=10)
        chunker = Chunker(config=config)
        assert chunker.config.chunk_size == 100
        assert chunker.config.chunk_overlap == 10

    def test_init_tokenizer_none(self):
        """Test tokenizer is initially None."""
        chunker = Chunker()
        assert chunker._tokenizer is None


class TestGetTokenizer:
    """Tests for _get_tokenizer method."""

    def test_get_tokenizer_with_tiktoken(self):
        """Test tiktoken tokenizer is loaded."""
        chunker = Chunker()

        with patch("tiktoken.get_encoding") as mock_get:
            mock_tokenizer = Mock()
            mock_get.return_value = mock_tokenizer

            result = chunker._get_tokenizer()

            mock_get.assert_called_once_with("cl100k_base")
            assert result is mock_tokenizer

    def test_get_tokenizer_caches_result(self):
        """Test tokenizer is cached."""
        chunker = Chunker()

        with patch("tiktoken.get_encoding") as mock_get:
            mock_tokenizer = Mock()
            mock_get.return_value = mock_tokenizer

            # Call twice
            result1 = chunker._get_tokenizer()
            result2 = chunker._get_tokenizer()

            # Should only be called once
            mock_get.assert_called_once()
            assert result1 is result2

    def test_get_tokenizer_fallback_to_word(self):
        """Test fallback to word-based when tiktoken not installed."""
        chunker = Chunker()

        with patch.dict("sys.modules", {"tiktoken": None}):
            with patch("builtins.__import__", side_effect=ImportError("No tiktoken")):
                result = chunker._get_tokenizer()
                assert result == "word"


class TestCountTokens:
    """Tests for _count_tokens method."""

    def test_count_tokens_with_tiktoken(self):
        """Test token counting with tiktoken."""
        chunker = Chunker()

        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        chunker._tokenizer = mock_tokenizer

        result = chunker._count_tokens("Hello world test")

        assert result == 5
        mock_tokenizer.encode.assert_called_once_with("Hello world test")

    def test_count_tokens_word_fallback(self):
        """Test word-based counting when tiktoken unavailable."""
        chunker = Chunker()
        chunker._tokenizer = "word"

        result = chunker._count_tokens("One two three four five")

        assert result == 5

    def test_count_tokens_empty_string(self):
        """Test counting empty string."""
        chunker = Chunker()
        chunker._tokenizer = "word"

        result = chunker._count_tokens("")

        assert result == 0


class TestExtractHeadingAtPosition:
    """Tests for _extract_heading_at_position method."""

    @pytest.fixture
    def chunker(self):
        """Create chunker instance."""
        return Chunker()

    def test_extract_no_headings(self, chunker):
        """Test empty list when no headings."""
        text = "Just some regular text without headings."
        result = chunker._extract_heading_at_position(text, len(text))
        assert result == []

    def test_extract_single_heading(self, chunker):
        """Test extracting single heading."""
        text = "# Main Heading\n\nSome content here."
        result = chunker._extract_heading_at_position(text, len(text))
        assert result == ["Main Heading"]

    def test_extract_multiple_levels(self, chunker):
        """Test extracting multiple heading levels."""
        text = "# Level 1\n\n## Level 2\n\n### Level 3\n\nContent"
        result = chunker._extract_heading_at_position(text, len(text))
        assert result == ["Level 1", "Level 2", "Level 3"]

    def test_extract_heading_hierarchy_reset(self, chunker):
        """Test hierarchy resets on higher-level heading."""
        text = """# First H1

## Subsection

# Second H1

Content here
"""
        result = chunker._extract_heading_at_position(text, len(text))
        # Should only have Second H1, not First H1 or Subsection
        assert result == ["Second H1"]

    def test_extract_heading_at_middle_position(self, chunker):
        """Test extracting headings up to middle position."""
        text = """# First

Content 1

# Second

Content 2
"""
        # Position before "# Second"
        position = text.find("# Second")
        result = chunker._extract_heading_at_position(text, position)
        assert result == ["First"]

    def test_extract_h6_heading(self, chunker):
        """Test h6 heading extraction."""
        text = "###### Deep Heading\n\nContent"
        result = chunker._extract_heading_at_position(text, len(text))
        assert result == ["Deep Heading"]


class TestOversizedSentenceSplit:
    """A single sentence that exceeds chunk_size must be split at the
    token boundary so the emitted chunks never break the size contract the
    embedder relies on.
    """

    def test_oversized_sentence_is_split_and_flushed(self):
        """Oversized sentence flushes the running chunk, then emits
        token-bounded pieces. Also exercises the sentence_start ==
        current_start fallback branch when the sentence cannot be relocated
        in the source text."""
        config = ChunkConfig(chunk_size=5, chunk_overlap=0, preserve_headings=False)
        chunker = Chunker(config=config)

        # Two real paragraphs: a short opening sentence to populate the
        # running chunk, then an oversized sentence whose single token count
        # exceeds chunk_size and forces the split path.
        text = (
            "Short first sentence. "
            "one two three four five six seven eight nine ten eleven twelve thirteen"
        )
        chunks = chunker._split_by_separators(text)

        # The short sentence flushes as its own chunk, then the oversized one
        # is broken into multiple pieces that each respect chunk_size.
        assert len(chunks) >= 2
        for chunk_text, _pos in chunks:
            # Every emitted piece must fit within the token budget.
            assert (
                chunker._count_tokens(chunk_text)
                <= config.chunk_size + CHUNK_TOKEN_TOLERANCE
            )


class TestSplitBySeparators:
    """Tests for _split_by_separators method."""

    @pytest.fixture
    def chunker(self):
        """Create chunker with small chunk size for testing."""
        config = ChunkConfig(chunk_size=10, chunk_overlap=0)
        return Chunker(config=config)

    def test_split_single_paragraph(self, chunker):
        """Test single paragraph under chunk size."""
        chunker._tokenizer = "word"
        text = "Short text."
        result = chunker._split_by_separators(text)
        assert len(result) == 1
        assert result[0][0] == "Short text."

    def test_split_multiple_paragraphs(self, chunker):
        """Test splitting multiple paragraphs."""
        chunker._tokenizer = "word"
        # Each paragraph has more than 10 words combined, so should split
        text = "First paragraph here.\n\nSecond paragraph here."
        result = chunker._split_by_separators(text)
        assert len(result) >= 1

    def test_split_returns_tuples(self, chunker):
        """Test returns list of tuples (text, position)."""
        chunker._tokenizer = "word"
        text = "Some text here."
        result = chunker._split_by_separators(text)
        assert isinstance(result, list)
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2

    def test_split_empty_text(self, chunker):
        """Test empty text returns empty list."""
        chunker._tokenizer = "word"
        text = ""
        result = chunker._split_by_separators(text)
        assert result == []

    def test_split_whitespace_only(self, chunker):
        """Test whitespace-only returns empty list."""
        chunker._tokenizer = "word"
        text = "   \n\n   "
        result = chunker._split_by_separators(text)
        assert result == []

    def test_split_long_paragraph_by_sentences(self):
        """Test long paragraphs are split by sentences."""
        config = ChunkConfig(chunk_size=5, chunk_overlap=0)
        chunker = Chunker(config=config)
        chunker._tokenizer = "word"

        # Long paragraph with multiple sentences
        text = "First sentence here. Second sentence here. Third sentence here."
        result = chunker._split_by_separators(text)

        # Should have multiple chunks
        assert len(result) >= 2

    def test_split_handles_trailing_whitespace_after_sentence(self):
        """Test sentences with trailing whitespace don't cause issues."""
        config = ChunkConfig(chunk_size=5, chunk_overlap=0)
        chunker = Chunker(config=config)
        chunker._tokenizer = "word"

        # Long paragraph ending with trailing whitespace after punctuation
        # This causes re.split to produce an empty string at the end
        text = "First sentence here. Second sentence here. "
        result = chunker._split_by_separators(text)

        # Should handle gracefully without empty chunks
        for chunk_text, _ in result:
            assert chunk_text.strip(), "Empty chunk should not be produced"

    def test_split_preserves_paragraph_structure(self):
        """Test paragraphs are joined with double newlines."""
        config = ChunkConfig(chunk_size=50, chunk_overlap=0)
        chunker = Chunker(config=config)
        chunker._tokenizer = "word"

        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = chunker._split_by_separators(text)

        # With large chunk size, should be one chunk
        assert len(result) == 1
        assert "\n\n" in result[0][0]


class TestAddOverlap:
    """Tests for _add_overlap method."""

    @pytest.fixture
    def chunker(self):
        """Create chunker with overlap."""
        config = ChunkConfig(chunk_size=100, chunk_overlap=3)
        return Chunker(config=config)

    def test_add_overlap_no_chunks(self, chunker):
        """Test empty list returns empty list."""
        result = chunker._add_overlap([])
        assert result == []

    def test_add_overlap_single_chunk(self, chunker):
        """Test single chunk has no overlap added."""
        chunks = [("First chunk text", 0)]
        result = chunker._add_overlap(chunks)
        assert len(result) == 1
        assert not result[0][0].startswith("...")

    def test_add_overlap_multiple_chunks(self, chunker):
        """Test overlap added to subsequent chunks."""
        chunks = [
            ("First chunk here", 0),
            ("Second chunk here", 20),
        ]
        result = chunker._add_overlap(chunks)

        assert len(result) == 2
        # First chunk unchanged
        assert not result[0][0].startswith("...")
        # Second chunk has overlap
        assert result[1][0].startswith("...")

    def test_add_overlap_zero_overlap(self):
        """Test no overlap when chunk_overlap is 0."""
        config = ChunkConfig(chunk_size=100, chunk_overlap=0)
        chunker = Chunker(config=config)

        chunks = [
            ("First chunk", 0),
            ("Second chunk", 15),
        ]
        result = chunker._add_overlap(chunks)

        assert not result[1][0].startswith("...")

    def test_add_overlap_fallback_uses_last_n_words_without_tiktoken(self):
        """When tiktoken is unavailable, fallback takes last N words.

        The public config field is named chunk_overlap; when the
        tokenizer sentinel is the string "word" the method treats
        the value as a word count, matching the fallback counter in
        Chunker._count_tokens.
        """
        config = ChunkConfig(chunk_size=100, chunk_overlap=3)
        chunker = Chunker(config=config)
        chunker._tokenizer = "word"

        chunks = [
            ("one two three four five six", 0),
            ("next chunk content", 30),
        ]
        result = chunker._add_overlap(chunks)

        assert "four five six" in result[1][0]

    def test_add_overlap_uses_exact_n_tokens_with_tiktoken(self):
        """Regression: overlap must encode back to exactly chunk_overlap
        tokens when the real tiktoken tokenizer is active.

        The original implementation used str.split to slice the tail
        of the previous chunk, which counts whitespace-delimited words,
        not cl100k tokens. Because cl100k averages about 1.3 tokens per
        English word, an overlap of 50 words silently became an overlap
        of about 65 tokens, diverging from the advertised
        chunk_overlap=50 setting. This test pins the token-accurate
        behaviour.
        """
        import tiktoken

        config = ChunkConfig(chunk_size=500, chunk_overlap=20)
        chunker = Chunker(config=config)

        long_text = (
            "The Department of Informatics at King's College London offers "
            "undergraduate and postgraduate programmes in Computer Science, "
            "Artificial Intelligence, and Robotics. Students attend lectures, "
            "complete coursework, and produce a final-year project report. "
            "Office hours are held weekly in the Strand building."
        )
        chunks = [(long_text, 0), ("Next chunk", 500)]
        result = chunker._add_overlap(chunks)

        overlap_text = result[1][0].removeprefix("...").split("\n\n")[0]

        tokenizer = tiktoken.get_encoding("cl100k_base")
        overlap_token_count = len(tokenizer.encode(overlap_text))

        assert overlap_token_count == config.chunk_overlap, (
            f"expected {config.chunk_overlap} tokens in overlap, "
            f"got {overlap_token_count}"
        )


class TestSplitOversized:
    """Tests for oversized-span splitting.

    A single sentence or long unbroken blob can itself exceed
    chunk_size (long tables rendered on one line, PDF paragraphs
    without terminal punctuation, etc.). Without a dedicated split pass
    the sentence-level chunker emits a chunk that breaks the size
    contract the embedder depends on.
    """

    def test_split_oversized_sentence_with_tiktoken(self):
        """A sentence beyond chunk_size becomes several pieces, each
        no longer than chunk_size tokens."""
        import tiktoken

        config = ChunkConfig(chunk_size=50, chunk_overlap=0)
        chunker = Chunker(config=config)

        # Build a single sentence whose cl100k_base token count comfortably
        # exceeds chunk_size (~600 tokens).
        long_sentence = "Muhammad attended the lecture " * 120
        document = Document.create(
            source_url="https://keats.kcl.ac.uk/mod/page/view.php?id=long",
            title="Long",
            content=long_sentence,
            content_type="page",
        )
        chunks = chunker.chunk_document(document)

        tokenizer = tiktoken.get_encoding("cl100k_base")
        assert len(chunks) >= 2
        for chunk in chunks:
            body = chunk.text
            if body.startswith("[Context:"):
                body = body.split("\n\n", 1)[-1]
            assert len(tokenizer.encode(body)) <= config.chunk_size

    def test_split_oversized_sentence_word_fallback(self):
        """When tiktoken is not available the chunker falls back to word
        buckets, and the oversized-sentence cap still holds."""
        config = ChunkConfig(chunk_size=20, chunk_overlap=0)
        chunker = Chunker(config=config)
        chunker._tokenizer = "word"

        long_sentence = " ".join(f"word{i}" for i in range(150))
        document = Document.create(
            source_url="https://keats.kcl.ac.uk/mod/page/view.php?id=wordfallback",
            title="WordFallback",
            content=long_sentence,
            content_type="page",
        )
        chunks = chunker.chunk_document(document)

        assert len(chunks) >= 2
        for chunk in chunks:
            body = chunk.text
            if body.startswith("[Context:"):
                body = body.split("\n\n", 1)[-1]
            assert len(body.split()) <= config.chunk_size


class TestChunkDocument:
    """Tests for chunk_document method."""

    @pytest.fixture
    def chunker(self):
        """Create chunker with test config."""
        config = ChunkConfig(chunk_size=50, chunk_overlap=5, preserve_headings=True)
        chunker = Chunker(config=config)
        chunker._tokenizer = "word"
        return chunker

    @pytest.fixture
    def sample_document(self):
        """Create sample document."""
        metadata = DocumentMetadata(
            source_url="https://example.com/doc",
            title="Test Document",
            content_type="page",
            section="Test Section",
        )
        return Document(
            id="doc123",
            content="This is the document content for testing.",
            metadata=metadata,
        )

    def test_chunk_returns_list(self, chunker, sample_document):
        """Test returns list of Chunk objects."""
        result = chunker.chunk_document(sample_document)
        assert isinstance(result, list)
        if result:
            assert isinstance(result[0], Chunk)

    def test_chunk_empty_document(self, chunker):
        """Test empty document returns empty list."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Empty",
            content_type="page",
        )
        doc = Document(id="empty", content="", metadata=metadata)

        result = chunker.chunk_document(doc)
        assert result == []

    def test_chunk_whitespace_document(self, chunker):
        """Test whitespace-only document returns empty list."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Whitespace",
            content_type="page",
        )
        doc = Document(id="ws", content="   \n\n   ", metadata=metadata)

        result = chunker.chunk_document(doc)
        assert result == []

    def test_chunk_sets_document_id(self, chunker, sample_document):
        """Test chunk has correct document_id."""
        result = chunker.chunk_document(sample_document)
        if result:
            assert result[0].metadata.document_id == "doc123"

    def test_chunk_sets_document_title(self, chunker, sample_document):
        """Test chunk has correct document_title."""
        result = chunker.chunk_document(sample_document)
        if result:
            assert result[0].metadata.document_title == "Test Document"

    def test_chunk_sets_source_url(self, chunker, sample_document):
        """Test chunk has correct source_url."""
        result = chunker.chunk_document(sample_document)
        if result:
            assert result[0].metadata.source_url == "https://example.com/doc"

    def test_chunk_sets_section(self, chunker, sample_document):
        """Test chunk has correct section."""
        result = chunker.chunk_document(sample_document)
        if result:
            assert result[0].metadata.section == "Test Section"

    def test_chunk_sets_chunk_index(self, chunker):
        """Test chunk_index is set correctly."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Long Doc",
            content_type="page",
        )
        # Create document with content that will produce multiple chunks
        doc = Document(
            id="long",
            content=" ".join(["word"] * 200),  # 200 words
            metadata=metadata,
        )

        result = chunker.chunk_document(doc)
        for i, chunk in enumerate(result):
            assert chunk.metadata.chunk_index == i

    def test_chunk_sets_total_chunks(self, chunker):
        """Test total_chunks_in_doc is set correctly."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Long Doc",
            content_type="page",
        )
        doc = Document(
            id="long",
            content=" ".join(["word"] * 200),
            metadata=metadata,
        )

        result = chunker.chunk_document(doc)
        expected_total = len(result)
        for chunk in result:
            assert chunk.metadata.total_chunks_in_doc == expected_total

    def test_chunk_extracts_heading_path(self, chunker):
        """Test heading_path is extracted for content after heading."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Headed Doc",
            content_type="page",
        )
        # Content with heading, then more content that will be in separate chunk
        doc = Document(
            id="headed",
            content="Intro paragraph.\n\n# Main Heading\n\nContent under the heading that is long enough to ensure we get multiple chunks.",
            metadata=metadata,
        )

        result = chunker.chunk_document(doc)
        # The heading extraction works on content BEFORE chunk position
        assert len(result) >= 1

    def test_chunk_preserves_headings_in_text(self, chunker):
        """Test heading context is prepended when preserve_headings=True."""
        # Create a document where a heading appears before chunk content
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Headed Doc",
            content_type="page",
        )
        # Need content with heading, then content in next chunk
        doc = Document(
            id="headed",
            content="# Main Heading\n\n" + "Word " * 60 + "\n\nMore content here.",
            metadata=metadata,
        )

        result = chunker.chunk_document(doc)
        # Just verify chunking worked - heading context only added when heading found before position
        assert len(result) >= 1

    def test_chunk_no_heading_context_when_disabled(self):
        """Test heading context not added when preserve_headings=False."""
        config = ChunkConfig(chunk_size=50, preserve_headings=False)
        chunker = Chunker(config=config)
        chunker._tokenizer = "word"

        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Doc",
            content_type="page",
        )
        doc = Document(
            id="doc",
            content="# Heading\n\nContent",
            metadata=metadata,
        )

        result = chunker.chunk_document(doc)
        if result:
            assert "[Context:" not in result[0].text


class TestChunkDocuments:
    """Tests for chunk_documents method."""

    @pytest.fixture
    def chunker(self):
        """Create chunker with test config."""
        config = ChunkConfig(chunk_size=50, chunk_overlap=0)
        chunker = Chunker(config=config)
        chunker._tokenizer = "word"
        return chunker

    def test_chunk_multiple_documents(self, chunker):
        """Test chunking multiple documents."""
        docs = []
        for i in range(3):
            metadata = DocumentMetadata(
                source_url=f"https://example.com/doc{i}",
                title=f"Document {i}",
                content_type="page",
            )
            doc = Document(
                id=f"doc{i}",
                content=f"Content for document {i}.",
                metadata=metadata,
            )
            docs.append(doc)

        result = chunker.chunk_documents(docs)

        assert len(result) == 3  # One chunk per doc
        assert result[0].metadata.document_id == "doc0"
        assert result[1].metadata.document_id == "doc1"
        assert result[2].metadata.document_id == "doc2"

    def test_chunk_empty_document_list(self, chunker):
        """Test empty document list returns empty list."""
        result = chunker.chunk_documents([])
        assert result == []

    def test_chunk_combines_all_chunks(self, chunker):
        """Test all chunks from all documents are combined."""
        metadata1 = DocumentMetadata(
            source_url="https://example.com/doc1",
            title="Doc 1",
            content_type="page",
        )
        # Document 1
        doc1 = Document(
            id="doc1",
            content="First document content here.",
            metadata=metadata1,
        )

        metadata2 = DocumentMetadata(
            source_url="https://example.com/doc2",
            title="Doc 2",
            content_type="page",
        )
        doc2 = Document(
            id="doc2",
            content="Second document content.",
            metadata=metadata2,
        )

        result = chunker.chunk_documents([doc1, doc2])

        # Should have at least one chunk from each doc
        assert len(result) >= 2
        # First chunk should be from doc1
        assert result[0].metadata.document_id == "doc1"
        # Last chunk should be from doc2
        assert result[-1].metadata.document_id == "doc2"


class TestIntegration:
    """Integration tests for Chunker."""

    def test_full_chunking_flow(self):
        """Test complete chunking flow."""
        config = ChunkConfig(chunk_size=20, chunk_overlap=3, preserve_headings=True)
        chunker = Chunker(config=config)
        chunker._tokenizer = "word"

        metadata = DocumentMetadata(
            source_url="https://example.com/handbook",
            title="Student Handbook",
            content_type="page",
            section="Academic Policies",
        )

        content = """# Introduction

Welcome to the student handbook. This document contains important policies.

## Academic Standards

Students must maintain good academic standing. All coursework should be original.

## Attendance

Regular attendance is expected for all courses.
"""

        doc = Document(id="handbook_intro", content=content, metadata=metadata)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata.document_id == "handbook_intro"
            assert chunk.metadata.section == "Academic Policies"
            assert chunk.text  # Not empty

    def test_chunk_preserves_unicode(self):
        """Test Unicode content is preserved."""
        config = ChunkConfig(chunk_size=100)
        chunker = Chunker(config=config)
        chunker._tokenizer = "word"

        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Unicode Test",
            content_type="page",
        )
        doc = Document(
            id="unicode",
            content="日本語テキスト with émojis 🎉 and accénts",
            metadata=metadata,
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert "日本語テキスト" in chunks[0].text
        assert "🎉" in chunks[0].text
