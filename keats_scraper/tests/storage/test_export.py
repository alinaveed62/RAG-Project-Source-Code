"""Tests for JSONLExporter."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from keats_scraper.models.chunk import Chunk
from keats_scraper.models.document import Document, DocumentMetadata
from keats_scraper.storage.export import JSONLExporter


class TestJSONLExporterInit:
    """Tests for JSONLExporter initialization."""

    def test_init_creates_directory(self, tmp_path):
        """Test output directory is created."""
        output_dir = tmp_path / "output"
        exporter = JSONLExporter(output_dir)
        assert output_dir.exists()

    def test_init_existing_directory(self, tmp_path):
        """Test init with existing directory doesn't fail."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        exporter = JSONLExporter(output_dir)
        assert output_dir.exists()

    def test_init_sets_output_dir(self, tmp_path):
        """Test output_dir is set correctly."""
        output_dir = tmp_path / "output"
        exporter = JSONLExporter(output_dir)
        assert exporter.output_dir == output_dir

    def test_init_nested_directory(self, tmp_path):
        """Test nested directory creation."""
        output_dir = tmp_path / "level1" / "level2" / "output"
        exporter = JSONLExporter(output_dir)
        assert output_dir.exists()


class TestExportDocuments:
    """Tests for export_documents method."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        docs = []
        for i in range(3):
            metadata = DocumentMetadata(
                source_url=f"https://example.com/doc{i}",
                title=f"Document {i}",
                content_type="page",
                section=f"Section {i}",
            )
            doc = Document(
                id=f"doc{i}",
                content=f"Content for document {i}",
                metadata=metadata,
            )
            docs.append(doc)
        return docs

    def test_export_creates_file(self, tmp_path, sample_documents):
        """Test JSONL file is created."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_documents(sample_documents)
        assert filepath.exists()

    def test_export_default_filename(self, tmp_path, sample_documents):
        """Test default filename is documents.jsonl."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_documents(sample_documents)
        assert filepath.name == "documents.jsonl"

    def test_export_custom_filename(self, tmp_path, sample_documents):
        """Test custom filename is used."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_documents(sample_documents, filename="custom.jsonl")
        assert filepath.name == "custom.jsonl"

    def test_export_returns_path(self, tmp_path, sample_documents):
        """Test returns a Path pointing at a file that was actually written."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_documents(sample_documents)
        assert isinstance(filepath, Path)
        assert filepath.exists()
        assert filepath.stat().st_size > 0

    def test_export_valid_jsonl(self, tmp_path, sample_documents):
        """Test output is valid JSONL."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_documents(sample_documents)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                assert "id" in data
                assert "content" in data
                assert "metadata" in data

    def test_export_line_count(self, tmp_path, sample_documents):
        """Test correct number of lines."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_documents(sample_documents)

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_export_empty_list(self, tmp_path):
        """Test export with empty list."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_documents([])

        assert filepath.exists()
        assert filepath.read_text() == ""

    def test_export_unicode_content(self, tmp_path):
        """Test Unicode content is preserved."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Título en español",
            content_type="page",
        )
        doc = Document(
            id="unicode_doc",
            content="日本語テキスト with émojis 🎉",
            metadata=metadata,
        )

        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_documents([doc])

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.loads(f.read().strip())
        assert data["content"] == "日本語テキスト with émojis 🎉"
        assert data["metadata"]["title"] == "Título en español"


class TestExportChunks:
    """Tests for export_chunks method."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        chunks = []
        for i in range(3):
            chunk = Chunk.create(
                text=f"Chunk content {i}",
                document_id="doc123",
                document_title="Test Document",
                source_url="https://example.com/doc",
                chunk_index=i,
                total_chunks=3,
                section=f"Section {i}",
            )
            chunks.append(chunk)
        return chunks

    def test_export_creates_file(self, tmp_path, sample_chunks):
        """Test JSONL file is created."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_chunks(sample_chunks)
        assert filepath.exists()

    def test_export_default_filename(self, tmp_path, sample_chunks):
        """Test default filename is handbook_chunks.jsonl."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_chunks(sample_chunks)
        assert filepath.name == "handbook_chunks.jsonl"

    def test_export_custom_filename(self, tmp_path, sample_chunks):
        """Test custom filename is used."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_chunks(sample_chunks, filename="my_chunks.jsonl")
        assert filepath.name == "my_chunks.jsonl"

    def test_export_valid_jsonl(self, tmp_path, sample_chunks):
        """Test output is valid JSONL."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_chunks(sample_chunks)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                assert "id" in data
                assert "text" in data
                assert "metadata" in data

    def test_export_line_count(self, tmp_path, sample_chunks):
        """Test correct number of lines."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_chunks(sample_chunks)

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_export_empty_list(self, tmp_path):
        """Test export with empty list."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_chunks([])

        assert filepath.exists()
        assert filepath.read_text() == ""

    def test_chunk_ids_preserved(self, tmp_path, sample_chunks):
        """Test chunk IDs are preserved in export."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_chunks(sample_chunks)

        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                assert data["id"] == sample_chunks[i].id


class TestExportEmbeddingFormat:
    """Tests for export_embedding_format method."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        chunks = []
        for i in range(2):
            chunk = Chunk.create(
                text=f"Text for embedding {i}",
                document_id=f"doc{i}",
                document_title=f"Title {i}",
                source_url=f"https://example.com/page{i}",
                chunk_index=0,
                total_chunks=1,
                section=f"Section {i}",
            )
            chunks.append(chunk)
        return chunks

    def test_export_creates_file(self, tmp_path, sample_chunks):
        """Test file is created."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_embedding_format(sample_chunks)
        assert filepath.exists()

    def test_export_default_filename(self, tmp_path, sample_chunks):
        """Test default filename is chunks_for_embedding.jsonl."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_embedding_format(sample_chunks)
        assert filepath.name == "chunks_for_embedding.jsonl"

    def test_export_custom_filename(self, tmp_path, sample_chunks):
        """Test custom filename is used."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_embedding_format(sample_chunks, filename="embed.jsonl")
        assert filepath.name == "embed.jsonl"

    def test_export_embedding_format_fields(self, tmp_path, sample_chunks):
        """Test exported data has correct embedding format fields."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_embedding_format(sample_chunks)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                assert "id" in data
                assert "text" in data
                assert "source" in data
                assert "title" in data
                assert "section" in data
                # Should not have full metadata
                assert "metadata" not in data

    def test_embedding_format_values(self, tmp_path, sample_chunks):
        """Test embedding format has correct values."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_embedding_format(sample_chunks)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())

        assert data["text"] == "Text for embedding 0"
        assert data["source"] == "https://example.com/page0"
        assert data["title"] == "Title 0"
        assert data["section"] == "Section 0"


class TestCreateIndex:
    """Tests for create_index method."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks with varied sections and documents."""
        chunks = []

        # Doc 1, Section A - 2 chunks
        for i in range(2):
            chunk = Chunk.create(
                text=f"Doc1 Section A chunk {i}",
                document_id="doc1",
                document_title="Document One",
                source_url="https://example.com/doc1",
                chunk_index=i,
                total_chunks=2,
                section="Section A",
            )
            chunks.append(chunk)

        # Doc 2, Section B - 1 chunk
        chunk = Chunk.create(
            text="Doc2 Section B chunk",
            document_id="doc2",
            document_title="Document Two",
            source_url="https://example.com/doc2",
            chunk_index=0,
            total_chunks=1,
            section="Section B",
        )
        chunks.append(chunk)

        # Doc 3, Section A - 1 chunk (same section as doc1)
        chunk = Chunk.create(
            text="Doc3 Section A chunk",
            document_id="doc3",
            document_title="Document Three",
            source_url="https://example.com/doc3",
            chunk_index=0,
            total_chunks=1,
            section="Section A",
        )
        chunks.append(chunk)

        return chunks

    def test_create_index_file(self, tmp_path, sample_chunks):
        """Test index file is created."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.create_index(sample_chunks)
        assert filepath.exists()

    def test_create_index_default_filename(self, tmp_path, sample_chunks):
        """Test default filename is chunk_index.json."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.create_index(sample_chunks)
        assert filepath.name == "chunk_index.json"

    def test_create_index_custom_filename(self, tmp_path, sample_chunks):
        """Test custom filename is used."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.create_index(sample_chunks, filename="my_index.json")
        assert filepath.name == "my_index.json"

    def test_index_is_valid_json(self, tmp_path, sample_chunks):
        """Test index is valid JSON."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.create_index(sample_chunks)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_index_has_created_at(self, tmp_path, sample_chunks):
        """Test index has created_at timestamp."""
        before = datetime.now(timezone.utc)
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.create_index(sample_chunks)
        after = datetime.now(timezone.utc)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "created_at" in data
        created = datetime.fromisoformat(data["created_at"])
        assert before <= created <= after

    def test_index_has_total_chunks(self, tmp_path, sample_chunks):
        """Test index has correct total_chunks count."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.create_index(sample_chunks)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["total_chunks"] == 4

    def test_index_chunks_by_section(self, tmp_path, sample_chunks):
        """Test chunks are indexed by section."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.create_index(sample_chunks)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "chunks_by_section" in data
        assert "Section A" in data["chunks_by_section"]
        assert "Section B" in data["chunks_by_section"]
        assert len(data["chunks_by_section"]["Section A"]) == 3
        assert len(data["chunks_by_section"]["Section B"]) == 1

    def test_index_chunks_by_document(self, tmp_path, sample_chunks):
        """Test chunks are indexed by document."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.create_index(sample_chunks)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "chunks_by_document" in data
        assert "doc1" in data["chunks_by_document"]
        assert "doc2" in data["chunks_by_document"]
        assert "doc3" in data["chunks_by_document"]

    def test_index_document_has_title(self, tmp_path, sample_chunks):
        """Test document index includes title."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.create_index(sample_chunks)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["chunks_by_document"]["doc1"]["title"] == "Document One"
        assert data["chunks_by_document"]["doc2"]["title"] == "Document Two"

    def test_index_document_has_chunk_ids(self, tmp_path, sample_chunks):
        """Test document index includes chunk IDs."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.create_index(sample_chunks)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["chunks_by_document"]["doc1"]["chunks"]) == 2
        assert len(data["chunks_by_document"]["doc2"]["chunks"]) == 1

    def test_index_empty_section_uses_unknown(self, tmp_path):
        """Test empty section is indexed as 'Unknown'."""
        chunk = Chunk.create(
            text="No section chunk",
            document_id="docX",
            document_title="Doc X",
            source_url="https://example.com/x",
            chunk_index=0,
            total_chunks=1,
            section="",  # Empty section
        )

        exporter = JSONLExporter(tmp_path)
        filepath = exporter.create_index([chunk])

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "Unknown" in data["chunks_by_section"]

    def test_index_empty_chunks_list(self, tmp_path):
        """Test index with empty chunks list."""
        exporter = JSONLExporter(tmp_path)
        filepath = exporter.create_index([])

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["total_chunks"] == 0
        assert data["chunks_by_section"] == {}
        assert data["chunks_by_document"] == {}


class TestLoadChunks:
    """Tests for load_chunks static method."""

    def test_load_chunks_returns_iterator(self, tmp_path):
        """Test load_chunks returns an iterator."""
        chunk = Chunk.create(
            text="Test chunk",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )

        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_chunks([chunk])

        result = JSONLExporter.load_chunks(filepath)
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_load_chunks_yields_chunk_objects(self, tmp_path):
        """Test load_chunks yields Chunk objects."""
        original = Chunk.create(
            text="Test chunk",
            document_id="doc1",
            document_title="Title",
            source_url="https://example.com",
            chunk_index=0,
            total_chunks=1,
        )

        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_chunks([original])

        loaded = list(JSONLExporter.load_chunks(filepath))
        assert len(loaded) == 1
        assert isinstance(loaded[0], Chunk)

    def test_load_chunks_data_preserved(self, tmp_path):
        """Test chunk data is preserved through export/load cycle."""
        original = Chunk.create(
            text="Test chunk content",
            document_id="doc123",
            document_title="My Document",
            source_url="https://example.com/page",
            chunk_index=5,
            total_chunks=10,
            section="Important Section",
        )

        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_chunks([original])

        loaded = list(JSONLExporter.load_chunks(filepath))[0]

        assert loaded.id == original.id
        assert loaded.text == original.text
        assert loaded.metadata.document_id == original.metadata.document_id
        assert loaded.metadata.section == original.metadata.section

    def test_load_multiple_chunks(self, tmp_path):
        """Test loading multiple chunks."""
        chunks = [
            Chunk.create(
                text=f"Chunk {i}",
                document_id="doc1",
                document_title="Title",
                source_url="https://example.com",
                chunk_index=i,
                total_chunks=5,
            )
            for i in range(5)
        ]

        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_chunks(chunks)

        loaded = list(JSONLExporter.load_chunks(filepath))
        assert len(loaded) == 5

    def test_load_empty_file(self, tmp_path):
        """Test loading empty JSONL file."""
        filepath = tmp_path / "empty.jsonl"
        filepath.write_text("")

        loaded = list(JSONLExporter.load_chunks(filepath))
        assert loaded == []


class TestLoadDocuments:
    """Tests for load_documents static method."""

    def test_load_documents_returns_iterator(self, tmp_path):
        """Test load_documents returns an iterator."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Title",
            content_type="page",
        )
        doc = Document(id="doc1", content="Content", metadata=metadata)

        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_documents([doc])

        result = JSONLExporter.load_documents(filepath)
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_load_documents_yields_document_objects(self, tmp_path):
        """Test load_documents yields Document objects."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Title",
            content_type="page",
        )
        doc = Document(id="doc1", content="Content", metadata=metadata)

        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_documents([doc])

        loaded = list(JSONLExporter.load_documents(filepath))
        assert len(loaded) == 1
        assert isinstance(loaded[0], Document)

    def test_load_documents_data_preserved(self, tmp_path):
        """Test document data is preserved through export/load cycle."""
        metadata = DocumentMetadata(
            source_url="https://example.com/mypage",
            title="My Document Title",
            content_type="pdf",
            section="Section A",
        )
        original = Document(
            id="doc123",
            content="This is my document content.",
            metadata=metadata,
            raw_html="<p>Test</p>",
        )

        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_documents([original])

        loaded = list(JSONLExporter.load_documents(filepath))[0]

        assert loaded.id == original.id
        assert loaded.content == original.content
        assert loaded.metadata.source_url == original.metadata.source_url
        assert loaded.metadata.title == original.metadata.title
        assert loaded.metadata.section == original.metadata.section

    def test_load_multiple_documents(self, tmp_path):
        """Test loading multiple documents."""
        docs = []
        for i in range(3):
            metadata = DocumentMetadata(
                source_url=f"https://example.com/page{i}",
                title=f"Title {i}",
                content_type="page",
            )
            docs.append(Document(id=f"doc{i}", content=f"Content {i}", metadata=metadata))

        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_documents(docs)

        loaded = list(JSONLExporter.load_documents(filepath))
        assert len(loaded) == 3

    def test_load_empty_file(self, tmp_path):
        """Test loading empty JSONL file."""
        filepath = tmp_path / "empty.jsonl"
        filepath.write_text("")

        loaded = list(JSONLExporter.load_documents(filepath))
        assert loaded == []


class TestRoundTrip:
    """Integration tests for full export/load round trips."""

    def test_document_round_trip(self, tmp_path):
        """Test documents survive full round trip."""
        metadata = DocumentMetadata(
            source_url="https://example.com/test",
            title="Round Trip Test",
            content_type="page",
            section="Test Section",
            word_count=5,
        )
        original = Document(
            id="roundtrip1",
            content="This is round trip content.",
            metadata=metadata,
        )

        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_documents([original])
        loaded = list(JSONLExporter.load_documents(filepath))[0]

        assert loaded.id == original.id
        assert loaded.content == original.content
        assert loaded.metadata.source_url == original.metadata.source_url

    def test_chunk_round_trip(self, tmp_path):
        """Test chunks survive full round trip."""
        original = Chunk.create(
            text="This is round trip chunk content.",
            document_id="docRT",
            document_title="Round Trip Doc",
            source_url="https://example.com/rt",
            chunk_index=2,
            total_chunks=5,
            section="RT Section",
            heading_path=["Heading 1", "Heading 2"],
        )

        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_chunks([original])
        loaded = list(JSONLExporter.load_chunks(filepath))[0]

        assert loaded.id == original.id
        assert loaded.text == original.text
        assert loaded.metadata.heading_path == original.metadata.heading_path
        assert loaded.metadata.chunk_index == original.metadata.chunk_index

    def test_unicode_round_trip(self, tmp_path):
        """Test Unicode content survives round trip."""
        metadata = DocumentMetadata(
            source_url="https://example.com/unicode",
            title="日本語タイトル",
            content_type="page",
        )
        original = Document(
            id="unicode1",
            content="Ελληνικά العربية 中文 🎉🎊",
            metadata=metadata,
        )

        exporter = JSONLExporter(tmp_path)
        filepath = exporter.export_documents([original])
        loaded = list(JSONLExporter.load_documents(filepath))[0]

        assert loaded.content == original.content
        assert loaded.metadata.title == original.metadata.title
