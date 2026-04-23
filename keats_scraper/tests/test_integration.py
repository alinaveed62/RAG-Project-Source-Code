"""Integration tests for the scraping and processing pipeline."""

from unittest.mock import Mock

import pytest
from tests.fixtures.sample_html import (
    BOOK_CHAPTER_HTML,
    EMPTY_AFTER_CLEANING_HTML,
    PAGE_WITH_BLOCK_CONTENT_HTML,
    SIMPLE_PAGE_HTML,
)

from keats_scraper.config import ChunkConfig
from keats_scraper.models.document import Document
from keats_scraper.processors.chunker import Chunker
from keats_scraper.processors.content_validator import ContentValidator
from keats_scraper.processors.html_cleaner import HTMLCleaner
from keats_scraper.processors.text_normalizer import TextNormalizer
from keats_scraper.scraper.page_scraper import PageScraper
from keats_scraper.scraper.rate_limiter import RateLimiter


class TestBookChapterContentExtraction:
    """Tests that book chapter content is correctly extracted (the 154/161 empty bug)."""

    @pytest.fixture
    def page_scraper(self):
        mock_session = Mock()
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()
        return PageScraper(mock_session, mock_limiter)

    def test_book_chapter_produces_nonempty_content(self, page_scraper):
        """Book chapter HTML should produce non-empty content after extraction."""
        title, content_html = page_scraper.extract_content(
            BOOK_CHAPTER_HTML,
            "https://keats.kcl.ac.uk/mod/book/view.php?id=123&chapterid=1",
        )
        assert title == "Teaching & Assessment"
        assert len(content_html) > 100
        assert "Assessment Methods" in content_html
        assert "coursework" in content_html.lower()

    def test_book_chapter_cleaned_has_content(self, page_scraper):
        """Book chapter HTML should have meaningful text after HTML cleaning."""
        _, content_html = page_scraper.extract_content(
            BOOK_CHAPTER_HTML,
            "https://keats.kcl.ac.uk/mod/book/view.php?id=123&chapterid=1",
        )
        cleaner = HTMLCleaner()
        normalizer = TextNormalizer()

        cleaned = cleaner.clean(content_html)
        normalized = normalizer.normalize(cleaned)

        assert len(normalized) > 50
        assert "Assessment Methods" in normalized
        assert "coursework" in normalized.lower()

    def test_block_content_not_destroyed(self, page_scraper):
        """Content inside .block elements should NOT be removed."""
        title, content_html = page_scraper.extract_content(
            PAGE_WITH_BLOCK_CONTENT_HTML,
            "https://keats.kcl.ac.uk/mod/page/view.php?id=456",
        )
        assert "Student Support" in content_html
        assert "Personal Tutors" in content_html
        assert "Disability Support" in content_html


class TestPageContentExtraction:
    """Tests for standard page content extraction and cleaning."""

    @pytest.fixture
    def page_scraper(self):
        mock_session = Mock()
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()
        return PageScraper(mock_session, mock_limiter)

    def test_simple_page_extraction(self, page_scraper):
        """Simple page should extract correctly."""
        title, content_html = page_scraper.extract_content(
            SIMPLE_PAGE_HTML,
            "https://keats.kcl.ac.uk/mod/page/view.php?id=789",
        )
        assert title == "Attendance Policy"
        assert "Engagement Monitoring" in content_html

    def test_simple_page_full_pipeline(self, page_scraper):
        """Simple page through full clean + normalize pipeline."""
        _, content_html = page_scraper.extract_content(
            SIMPLE_PAGE_HTML,
            "https://keats.kcl.ac.uk/mod/page/view.php?id=789",
        )
        cleaner = HTMLCleaner()
        normalizer = TextNormalizer()

        text = normalizer.normalize(cleaner.clean(content_html))
        assert "attendance" in text.lower()
        assert "QR code" in text or "qr code" in text.lower()

    def test_navigation_removed(self, page_scraper):
        """Navigation elements should be stripped."""
        _, content_html = page_scraper.extract_content(
            SIMPLE_PAGE_HTML,
            "https://keats.kcl.ac.uk/mod/page/view.php?id=789",
        )
        assert "navbar" not in content_html.lower() or "<nav" not in content_html


class TestFullPipelineDocumentToChunks:
    """Tests for the complete document -> chunks pipeline."""

    def test_document_to_chunks(self):
        """Create a document and chunk it; verify chunks have content."""
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/mod/page/view.php?id=123",
            title="Test Handbook Page",
            content=(
                "# Teaching & Assessment\n\n"
                "## Assessment Methods\n\n"
                "The Department of Informatics uses a variety of assessment methods "
                "including coursework, examinations, and practical assignments. "
                "Each module specifies its own assessment criteria.\n\n"
                "## Coursework Submission\n\n"
                "All coursework must be submitted electronically via KEATS by the "
                "published deadline. Late submissions will incur penalties."
            ),
            content_type="page",
            section="Teaching & Assessment",
        )

        chunker = Chunker(ChunkConfig(chunk_size=50, chunk_overlap=5, preserve_headings=True))
        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.text.strip()
            assert chunk.metadata.document_id == doc.id
            assert chunk.metadata.section == "Teaching & Assessment"

    def test_empty_document_produces_no_chunks(self):
        """Empty documents should not produce any chunks."""
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/test",
            title="Empty",
            content="",
            content_type="page",
        )
        chunker = Chunker()
        chunks = chunker.chunk_document(doc)
        assert chunks == []


class TestEmptyContentNotSaved:
    """Tests that empty content is detected and rejected."""

    @pytest.fixture
    def page_scraper(self):
        mock_session = Mock()
        mock_limiter = Mock(spec=RateLimiter)
        mock_limiter.retry_on_rate_limit.side_effect = lambda func, **kwargs: func()
        return PageScraper(mock_session, mock_limiter)

    def test_empty_page_detection(self, page_scraper):
        """Pages with no meaningful content should be caught by content validation."""
        _, content_html = page_scraper.extract_content(
            EMPTY_AFTER_CLEANING_HTML,
            "https://keats.kcl.ac.uk/mod/page/view.php?id=999",
        )
        cleaner = HTMLCleaner()
        normalizer = TextNormalizer()

        text = normalizer.normalize(cleaner.clean(content_html))
        # The text should either be empty or very short (just boilerplate remnants)
        # Our main.py fix ensures these aren't saved
        assert len(text.strip()) < 50 or text.strip() == ""


class TestContentValidator:
    """Tests for content quality validation."""

    def test_valid_document_passes(self):
        """Documents with sufficient content should pass validation."""
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/test",
            title="Valid Page",
            content="This is a valid document with enough content to pass the minimum requirements for validation.",
            content_type="page",
            section="Test Section",
        )
        validator = ContentValidator()
        is_valid, issues = validator.validate_document(doc)
        assert is_valid is True

    def test_empty_document_fails(self):
        """Empty documents should fail validation."""
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/test",
            title="Empty",
            content="",
            content_type="page",
        )
        validator = ContentValidator()
        is_valid, issues = validator.validate_document(doc)
        assert is_valid is False
        assert any("Empty" in i for i in issues)

    def test_short_document_fails(self):
        """Very short documents should fail validation."""
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/test",
            title="Short",
            content="Hello",
            content_type="page",
        )
        validator = ContentValidator()
        is_valid, issues = validator.validate_document(doc)
        assert is_valid is False

    def test_quality_report_generation(self):
        """Quality report should summarize document collection."""
        docs = [
            Document.create(
                source_url="https://keats.kcl.ac.uk/test1",
                title="Good Doc",
                content="This is a valid document with enough words to pass validation checks for content quality.",
                content_type="page",
                section="Section A",
            ),
            Document.create(
                source_url="https://keats.kcl.ac.uk/test2",
                title="Empty Doc",
                content="",
                content_type="page",
                section="Section B",
            ),
        ]
        validator = ContentValidator()
        report = validator.generate_quality_report(docs)

        assert report["total_documents"] == 2
        assert report["empty_documents"] == 1
        assert report["valid_documents"] == 1
        assert "Section A" in report["sections_covered"]

    def test_missing_section_flagged(self):
        """Documents without section metadata should be flagged."""
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/test",
            title="No Section",
            content="This document has no section metadata assigned to it which should be flagged as an issue.",
            content_type="page",
            section="",
        )
        validator = ContentValidator()
        _, issues = validator.validate_document(doc)
        assert any("section" in i.lower() for i in issues)


class TestCleanAndValidateDocument:
    """Tests for the scrape loop's integration with ContentValidator.

    Verifies that main.clean_and_validate_document applies the validator
    as a soft filter (log warning, do not drop) for low-quality documents,
    and only drops documents that cleaning emptied out entirely.
    """

    @pytest.fixture
    def processors(self):
        return HTMLCleaner(), TextNormalizer(), ContentValidator()

    def test_empty_content_returns_none(self, processors):
        from main import clean_and_validate_document

        cleaner, normalizer, validator = processors
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/test",
            title="Empty",
            content="",
            content_type="page",
            section="Test Section",
        )
        result = clean_and_validate_document(doc, cleaner, normalizer, validator)
        assert result is None

    def test_low_quality_document_preserved_with_warning(self, processors, caplog):
        """A too-short document must be returned (not dropped) and a
        warning must be logged with the url and issues."""
        import logging

        from main import clean_and_validate_document

        cleaner, normalizer, validator = processors
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/stub",
            title="Stub",
            content="Only a handful of words here.",
            content_type="page",
            section="Test Section",
        )
        with caplog.at_level(logging.WARNING):
            result = clean_and_validate_document(doc, cleaner, normalizer, validator)

        assert result is not None
        assert result is doc
        assert any(
            "ContentValidator" in rec.message and "https://keats.kcl.ac.uk/stub" in rec.message
            for rec in caplog.records
        )

    def test_valid_document_passes_without_warning(self, processors, caplog):
        """A valid document must pass without any ContentValidator warning."""
        import logging

        from main import clean_and_validate_document

        cleaner, normalizer, validator = processors
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/good",
            title="Good",
            content=(
                "This is a long enough document with plenty of real handbook "
                "content covering attendance policy, engagement monitoring, "
                "and the reporting procedure for extenuating circumstances."
            ),
            content_type="page",
            section="Test Section",
        )
        with caplog.at_level(logging.WARNING):
            result = clean_and_validate_document(doc, cleaner, normalizer, validator)

        assert result is doc
        assert not any(
            "ContentValidator" in rec.message for rec in caplog.records
        )

    def test_raw_html_is_cleaned_before_validation(self, processors):
        from main import clean_and_validate_document

        cleaner, normalizer, validator = processors
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/html",
            title="Raw",
            content="",
            content_type="page",
            section="Test Section",
        )
        doc.raw_html = (
            "<div><h1>Attendance</h1><p>You are strongly advised to attend "
            "all classes, including lectures, seminars, and tutorials.</p></div>"
        )
        result = clean_and_validate_document(doc, cleaner, normalizer, validator)
        assert result is not None
        assert "attendance" in result.content.lower()
        assert "strongly advised" in result.content.lower()
