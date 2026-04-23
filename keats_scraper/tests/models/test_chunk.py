"""Tests for Chunk and related models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from keats_scraper.models.chunk import Chunk, ChunkMetadata


class TestChunkMetadata:
    """Tests for ChunkMetadata model."""

    def test_required_fields(self):
        """Test required fields must be provided."""
        with pytest.raises(ValidationError):
            ChunkMetadata()

    def test_valid_creation(self):
        """Test valid metadata creation."""
        metadata = ChunkMetadata(
            source_url="https://example.com",
            document_id="doc123",
            document_title="Doc Title",
            chunk_index=0,
            total_chunks_in_doc=5,
        )
        assert metadata.source_url == "https://example.com"
        assert metadata.document_id == "doc123"
        assert metadata.chunk_index == 0

    def test_default_section(self):
        """Test section defaults to empty string."""
        metadata = ChunkMetadata(
            source_url="https://example.com",
            document_id="doc123",
            document_title="Title",
            chunk_index=0,
            total_chunks_in_doc=1,
        )
        assert metadata.section == ""

    def test_default_heading_path(self):
        """Test heading_path defaults to empty list."""
        metadata = ChunkMetadata(
            source_url="https://example.com",
            document_id="doc123",
            document_title="Title",
            chunk_index=0,
            total_chunks_in_doc=1,
        )
        assert metadata.heading_path == []

    def test_default_char_count(self):
        """Test char_count defaults to 0."""
        metadata = ChunkMetadata(
            source_url="https://example.com",
            document_id="doc123",
            document_title="Title",
            chunk_index=0,
            total_chunks_in_doc=1,
        )
        assert metadata.char_count == 0

    def test_default_word_count(self):
        """Test word_count defaults to 0."""
        metadata = ChunkMetadata(
            source_url="https://example.com",
            document_id="doc123",
            document_title="Title",
            chunk_index=0,
            total_chunks_in_doc=1,
        )
        assert metadata.word_count == 0

    def test_extraction_date_auto_set(self):
        """Test extraction_date is auto-generated."""
        before = datetime.now(timezone.utc)
        metadata = ChunkMetadata(
            source_url="https://example.com",
            document_id="doc123",
            document_title="Title",
            chunk_index=0,
            total_chunks_in_doc=1,
        )
        after = datetime.now(timezone.utc)
        assert before <= metadata.extraction_date <= after

    def test_default_content_hash(self):
        """Test content_hash defaults to empty string."""
        metadata = ChunkMetadata(
            source_url="https://example.com",
            document_id="doc123",
            document_title="Title",
            chunk_index=0,
            total_chunks_in_doc=1,
        )
        assert metadata.content_hash == ""

    def test_all_fields(self):
        """Test all fields can be set."""
        now = datetime.now(timezone.utc)
        metadata = ChunkMetadata(
            source_url="https://example.com",
            document_id="doc123",
            document_title="Title",
            section="Section A",
            subsection="Subsection 1",
            heading_path=["H1", "H2"],
            chunk_index=2,
            total_chunks_in_doc=10,
            char_count=500,
            word_count=100,
            extraction_date=now,
            content_hash="abc123hash",
        )
        assert metadata.section == "Section A"
        assert metadata.subsection == "Subsection 1"
        assert metadata.heading_path == ["H1", "H2"]
        assert metadata.char_count == 500
        assert metadata.word_count == 100
        assert metadata.content_hash == "abc123hash"


class TestChunk:
    """Tests for Chunk model."""

    def test_required_fields(self):
        """Test required fields must be provided."""
        with pytest.raises(ValidationError):
            Chunk()

    def test_valid_creation(self):
        """Test valid chunk creation."""
        metadata = ChunkMetadata(
            source_url="https://example.com",
            document_id="doc123",
            document_title="Title",
            chunk_index=0,
            total_chunks_in_doc=1,
        )
        chunk = Chunk(
            id="chunk123",
            text="Chunk text content",
            metadata=metadata,
        )
        assert chunk.id == "chunk123"
        assert chunk.text == "Chunk text content"


class TestChunkCreate:
    """Tests for Chunk.create() factory method."""

    def test_creates_chunk(self):
        """Test create() returns a Chunk."""
        chunk = Chunk.create(
            text="Test chunk text",
            document_id="doc123",
            document_title="Doc Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=5,
        )
        assert isinstance(chunk, Chunk)

    def test_generates_id(self):
        """Test ID is generated as doc_id_chunk_index."""
        chunk = Chunk.create(
            text="Test",
            document_id="doc123",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=3,
            total_chunks=10,
        )
        assert chunk.id == "doc123_chunk_3"

    def test_calculates_content_hash(self):
        """Test content_hash is calculated from text."""
        chunk1 = Chunk.create(
            text="Same text",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )
        chunk2 = Chunk.create(
            text="Same text",
            document_id="doc2",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )
        # Same text should have same hash
        assert chunk1.metadata.content_hash == chunk2.metadata.content_hash

    def test_different_text_different_hash(self):
        """Test different text produces different hash."""
        chunk1 = Chunk.create(
            text="Text A",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )
        chunk2 = Chunk.create(
            text="Text B",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )
        assert chunk1.metadata.content_hash != chunk2.metadata.content_hash

    def test_calculates_char_count(self):
        """Test char_count is calculated from text."""
        text = "Hello world"
        chunk = Chunk.create(
            text=text,
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )
        assert chunk.metadata.char_count == len(text)

    def test_calculates_word_count(self):
        """Test word_count is calculated from text."""
        chunk = Chunk.create(
            text="One two three four five",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )
        assert chunk.metadata.word_count == 5

    def test_empty_text_counts(self):
        """Test counts for empty text."""
        chunk = Chunk.create(
            text="",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )
        assert chunk.metadata.char_count == 0
        assert chunk.metadata.word_count == 0

    def test_section_passed(self):
        """Test section is passed to metadata."""
        chunk = Chunk.create(
            text="Text",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
            section="Section A",
        )
        assert chunk.metadata.section == "Section A"

    def test_heading_path_passed(self):
        """Test heading_path is passed to metadata."""
        chunk = Chunk.create(
            text="Text",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
            heading_path=["H1", "H2", "H3"],
        )
        assert chunk.metadata.heading_path == ["H1", "H2", "H3"]

    def test_heading_path_none_becomes_empty_list(self):
        """Test heading_path None becomes empty list."""
        chunk = Chunk.create(
            text="Text",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
            heading_path=None,
        )
        assert chunk.metadata.heading_path == []

    def test_metadata_kwargs(self):
        """Test additional metadata kwargs are passed."""
        chunk = Chunk.create(
            text="Text",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
            subsection="Subsection X",
        )
        assert chunk.metadata.subsection == "Subsection X"


class TestChunkToDict:
    """Tests for Chunk.to_dict() method."""

    def test_returns_dict(self):
        """Test to_dict returns a dictionary."""
        chunk = Chunk.create(
            text="Test",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )
        result = chunk.to_dict()
        assert isinstance(result, dict)

    def test_contains_id(self):
        """Test dict contains id."""
        chunk = Chunk.create(
            text="Test",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )
        result = chunk.to_dict()
        assert "id" in result
        assert result["id"] == chunk.id

    def test_contains_text(self):
        """Test dict contains text."""
        chunk = Chunk.create(
            text="My text content",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )
        result = chunk.to_dict()
        assert "text" in result
        assert result["text"] == "My text content"

    def test_contains_metadata(self):
        """Test dict contains metadata."""
        chunk = Chunk.create(
            text="Test",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )
        result = chunk.to_dict()
        assert "metadata" in result
        assert isinstance(result["metadata"], dict)

    def test_metadata_fields_present(self):
        """Test metadata fields are present."""
        chunk = Chunk.create(
            text="Test",
            document_id="doc1",
            document_title="My Title",
            source_url="https://example.com/test",
            chunk_index=2,
            total_chunks=5,
            section="Section",
        )
        result = chunk.to_dict()
        assert result["metadata"]["document_id"] == "doc1"
        assert result["metadata"]["document_title"] == "My Title"
        assert result["metadata"]["source_url"] == "https://example.com/test"
        assert result["metadata"]["chunk_index"] == 2
        assert result["metadata"]["total_chunks_in_doc"] == 5
        assert result["metadata"]["section"] == "Section"


class TestChunkToEmbeddingFormat:
    """Tests for Chunk.to_embedding_format() method."""

    def test_returns_dict(self):
        """Test to_embedding_format returns a dictionary."""
        chunk = Chunk.create(
            text="Test",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )
        result = chunk.to_embedding_format()
        assert isinstance(result, dict)

    def test_contains_required_fields(self):
        """Test embedding format contains all required fields."""
        chunk = Chunk.create(
            text="Chunk text",
            document_id="doc1",
            document_title="Doc Title",
            source_url="https://example.com/page",
            chunk_index=0,
            total_chunks=1,
            section="My Section",
        )
        result = chunk.to_embedding_format()

        assert "id" in result
        assert "text" in result
        assert "source" in result
        assert "title" in result
        assert "section" in result
        assert "heading_path" in result

    def test_contains_heading_path(self):
        """heading_path must reach the embedding format so the RAG pipeline
        can surface handbook heading context in retrieval results."""
        chunk = Chunk.create(
            text="Chunk text",
            document_id="doc1",
            document_title="Doc Title",
            source_url="https://example.com/page",
            chunk_index=0,
            total_chunks=1,
            section="Teaching and Assessment",
            heading_path=["Teaching and Assessment", "Attendance"],
        )
        result = chunk.to_embedding_format()
        assert result["heading_path"] == [
            "Teaching and Assessment",
            "Attendance",
        ]

    def test_field_values(self):
        """Test embedding format field values are correct."""
        chunk = Chunk.create(
            text="Chunk text content",
            document_id="doc1",
            document_title="Document Title",
            source_url="https://example.com/page",
            chunk_index=0,
            total_chunks=1,
            section="Section A",
        )
        result = chunk.to_embedding_format()

        assert result["id"] == chunk.id
        assert result["text"] == "Chunk text content"
        assert result["source"] == "https://example.com/page"
        assert result["title"] == "Document Title"
        assert result["section"] == "Section A"

    def test_simpler_than_to_dict(self):
        """Test embedding format has fewer fields than to_dict."""
        chunk = Chunk.create(
            text="Test",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )
        embed_result = chunk.to_embedding_format()
        dict_result = chunk.to_dict()

        # Embedding format should be flatter
        assert "metadata" not in embed_result
        assert "metadata" in dict_result
