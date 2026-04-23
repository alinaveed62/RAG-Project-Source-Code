"""Tests for Document and related models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from keats_scraper.models.document import Document, DocumentMetadata, ResourceInfo


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    def test_required_fields(self):
        """Test required fields must be provided."""
        with pytest.raises(ValidationError):
            DocumentMetadata()

    def test_valid_creation(self):
        """Test valid metadata creation."""
        metadata = DocumentMetadata(
            source_url="https://example.com/page",
            title="Test Title",
            content_type="page",
        )
        assert metadata.source_url == "https://example.com/page"
        assert metadata.title == "Test Title"
        assert metadata.content_type == "page"

    def test_default_section(self):
        """Test section defaults to empty string."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Title",
            content_type="page",
        )
        assert metadata.section == ""

    def test_default_subsection(self):
        """Test subsection defaults to None."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Title",
            content_type="page",
        )
        assert metadata.subsection is None

    def test_default_word_count(self):
        """Test word_count defaults to 0."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Title",
            content_type="page",
        )
        assert metadata.word_count == 0

    def test_extraction_date_auto_set(self):
        """Test extraction_date is auto-generated."""
        before = datetime.now(timezone.utc)
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Title",
            content_type="page",
        )
        after = datetime.now(timezone.utc)

        assert before <= metadata.extraction_date <= after

    def test_last_modified_optional(self):
        """Test last_modified is optional."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Title",
            content_type="page",
        )
        assert metadata.last_modified is None

    def test_parent_id_optional(self):
        """Test parent_id is optional."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Title",
            content_type="page",
        )
        assert metadata.parent_id is None

    def test_all_fields(self):
        """Test all fields can be set."""
        now = datetime.now(timezone.utc)
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Title",
            section="Section 1",
            subsection="Subsection A",
            content_type="pdf",
            last_modified=now,
            extraction_date=now,
            word_count=100,
            parent_id="parent123",
        )
        assert metadata.section == "Section 1"
        assert metadata.subsection == "Subsection A"
        assert metadata.word_count == 100
        assert metadata.parent_id == "parent123"


class TestDocument:
    """Tests for Document model."""

    def test_required_fields(self):
        """Test required fields must be provided."""
        with pytest.raises(ValidationError):
            Document()

    def test_valid_creation(self):
        """Test valid document creation."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Title",
            content_type="page",
        )
        doc = Document(
            id="doc123",
            content="Test content",
            metadata=metadata,
        )
        assert doc.id == "doc123"
        assert doc.content == "Test content"

    def test_raw_html_optional(self):
        """Test raw_html is optional."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Title",
            content_type="page",
        )
        doc = Document(
            id="doc123",
            content="Test content",
            metadata=metadata,
        )
        assert doc.raw_html is None

    def test_raw_html_can_be_set(self):
        """Test raw_html can be set."""
        metadata = DocumentMetadata(
            source_url="https://example.com",
            title="Title",
            content_type="page",
        )
        doc = Document(
            id="doc123",
            content="Test content",
            raw_html="<html><body>Test</body></html>",
            metadata=metadata,
        )
        assert doc.raw_html == "<html><body>Test</body></html>"


class TestDocumentCreate:
    """Tests for Document.create() factory method."""

    def test_creates_document(self):
        """Test create() returns a Document."""
        doc = Document.create(
            source_url="https://example.com/page",
            title="Test Title",
            content="Test content here",
            content_type="page",
        )
        assert isinstance(doc, Document)

    def test_generates_id_from_url(self):
        """Test ID is generated from URL hash."""
        doc1 = Document.create(
            source_url="https://example.com/page1",
            title="Title",
            content="Content",
            content_type="page",
        )
        doc2 = Document.create(
            source_url="https://example.com/page2",
            title="Title",
            content="Content",
            content_type="page",
        )
        # Different URLs should have different IDs
        assert doc1.id != doc2.id

    def test_same_url_same_id(self):
        """Test same URL produces same ID."""
        doc1 = Document.create(
            source_url="https://example.com/same",
            title="Title 1",
            content="Content 1",
            content_type="page",
        )
        doc2 = Document.create(
            source_url="https://example.com/same",
            title="Title 2",
            content="Content 2",
            content_type="page",
        )
        assert doc1.id == doc2.id

    def test_id_is_12_chars(self):
        """Test ID is 12 characters (MD5 prefix)."""
        doc = Document.create(
            source_url="https://example.com",
            title="Title",
            content="Content",
            content_type="page",
        )
        assert len(doc.id) == 12

    def test_calculates_word_count(self):
        """Test word_count is calculated from content."""
        doc = Document.create(
            source_url="https://example.com",
            title="Title",
            content="One two three four five",
            content_type="page",
        )
        assert doc.metadata.word_count == 5

    def test_empty_content_word_count(self):
        """Test word_count is 0 for empty content."""
        doc = Document.create(
            source_url="https://example.com",
            title="Title",
            content="",
            content_type="page",
        )
        assert doc.metadata.word_count == 0

    def test_section_passed_to_metadata(self):
        """Test section is passed to metadata."""
        doc = Document.create(
            source_url="https://example.com",
            title="Title",
            content="Content",
            content_type="page",
            section="My Section",
        )
        assert doc.metadata.section == "My Section"

    def test_raw_html_stored(self):
        """Test raw_html is stored."""
        doc = Document.create(
            source_url="https://example.com",
            title="Title",
            content="Content",
            content_type="page",
            raw_html="<p>Test</p>",
        )
        assert doc.raw_html == "<p>Test</p>"

    def test_metadata_kwargs(self):
        """Test additional metadata kwargs are passed."""
        doc = Document.create(
            source_url="https://example.com",
            title="Title",
            content="Content",
            content_type="page",
            subsection="Subsection A",
            parent_id="parent123",
        )
        assert doc.metadata.subsection == "Subsection A"
        assert doc.metadata.parent_id == "parent123"


class TestDocumentToDict:
    """Tests for Document.to_dict() method."""

    def test_returns_dict(self):
        """Test to_dict returns a dictionary."""
        doc = Document.create(
            source_url="https://example.com",
            title="Title",
            content="Content",
            content_type="page",
        )
        result = doc.to_dict()
        assert isinstance(result, dict)

    def test_contains_id(self):
        """Test dict contains id."""
        doc = Document.create(
            source_url="https://example.com",
            title="Title",
            content="Content",
            content_type="page",
        )
        result = doc.to_dict()
        assert "id" in result
        assert result["id"] == doc.id

    def test_contains_content(self):
        """Test dict contains content."""
        doc = Document.create(
            source_url="https://example.com",
            title="Title",
            content="My content",
            content_type="page",
        )
        result = doc.to_dict()
        assert "content" in result
        assert result["content"] == "My content"

    def test_contains_metadata(self):
        """Test dict contains metadata."""
        doc = Document.create(
            source_url="https://example.com",
            title="Title",
            content="Content",
            content_type="page",
        )
        result = doc.to_dict()
        assert "metadata" in result
        assert isinstance(result["metadata"], dict)

    def test_metadata_serialized(self):
        """Test metadata is serialized correctly."""
        doc = Document.create(
            source_url="https://example.com",
            title="My Title",
            content="Content",
            content_type="pdf",
            section="Section X",
        )
        result = doc.to_dict()
        assert result["metadata"]["source_url"] == "https://example.com"
        assert result["metadata"]["title"] == "My Title"
        assert result["metadata"]["content_type"] == "pdf"
        assert result["metadata"]["section"] == "Section X"

    def test_datetime_serialized(self):
        """Test datetime fields are serialized to JSON-compatible format."""
        doc = Document.create(
            source_url="https://example.com",
            title="Title",
            content="Content",
            content_type="page",
        )
        result = doc.to_dict()
        # Should be string (ISO format)
        assert isinstance(result["metadata"]["extraction_date"], str)


class TestResourceInfo:
    """Tests for ResourceInfo model."""

    def test_required_fields(self):
        """Test required fields must be provided."""
        with pytest.raises(ValidationError):
            ResourceInfo()

    def test_valid_creation(self):
        """Test valid ResourceInfo creation."""
        info = ResourceInfo(
            url="https://example.com/page",
            title="Page Title",
            resource_type="page",
        )
        assert info.url == "https://example.com/page"
        assert info.title == "Page Title"
        assert info.resource_type == "page"

    def test_default_section(self):
        """Test section defaults to empty string."""
        info = ResourceInfo(
            url="https://example.com",
            title="Title",
            resource_type="page",
        )
        assert info.section == ""

    def test_default_section_index(self):
        """Test section_index defaults to 0."""
        info = ResourceInfo(
            url="https://example.com",
            title="Title",
            resource_type="page",
        )
        assert info.section_index == 0

    def test_all_fields(self):
        """Test all fields can be set."""
        info = ResourceInfo(
            url="https://example.com",
            title="Title",
            resource_type="pdf",
            section="Section 1",
            section_index=5,
        )
        assert info.section == "Section 1"
        assert info.section_index == 5
