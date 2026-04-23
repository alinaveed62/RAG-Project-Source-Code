"""Tests for ContentValidator."""

from keats_scraper.models.document import Document
from keats_scraper.processors.content_validator import ContentValidator


class TestValidateDocument:
    """Tests for ContentValidator.validate_document."""

    def test_empty_content_is_invalid(self):
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/x",
            title="t",
            content="",
            content_type="page",
            section="s",
        )
        is_valid, issues = ContentValidator().validate_document(doc)
        assert is_valid is False
        assert "Empty content" in issues

    def test_whitespace_only_content_is_invalid(self):
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/x",
            title="t",
            content="   \n\t  ",
            content_type="page",
            section="s",
        )
        is_valid, issues = ContentValidator().validate_document(doc)
        assert is_valid is False
        assert "Empty content" in issues

    def test_short_content_flagged(self):
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/x",
            title="t",
            content="too short",
            content_type="page",
            section="s",
        )
        is_valid, issues = ContentValidator().validate_document(doc)
        assert is_valid is False
        assert any("Content too short" in i for i in issues)

    def test_too_few_words_flagged(self):
        # Content must clear MIN_CONTENT_LENGTH (50 chars) but fall under
        # MIN_WORD_COUNT (10 words). Use long, distinct words so the length
        # margin is unambiguous.
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/x",
            title="t",
            content=(
                "antidisestablishment postulated introductory hyperbolic "
                "recapitulation"
            ),
            content_type="page",
            section="s",
        )
        is_valid, issues = ContentValidator().validate_document(doc)
        # Content is well over 50 chars (70+) but only 5 words, so the
        # word-count floor is the only trigger.
        assert is_valid is False
        assert any("Too few words" in i for i in issues)
        assert not any("Content too short" in i for i in issues)

    def test_boilerplate_majority_flagged(self):
        """Three or more boilerplate phrases triggers the boilerplate flag."""
        content = (
            "Skip to main content. You are not logged in. Log in to your account. "
            "This document describes the KCL Informatics handbook contents "
            "and has enough words to pass the length thresholds easily."
        )
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/x",
            title="t",
            content=content,
            content_type="page",
            section="s",
        )
        is_valid, issues = ContentValidator().validate_document(doc)
        assert any("Likely boilerplate" in i for i in issues)
        # Boilerplate is advisory; the document still has content and words,
        # so is_valid remains True (not dropped).
        assert is_valid is True

    def test_missing_section_flagged(self):
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/x",
            title="t",
            content="This document has plenty of content to satisfy the length thresholds "
            "imposed by the validator, so only the missing section should be flagged.",
            content_type="page",
            section="",
        )
        is_valid, issues = ContentValidator().validate_document(doc)
        assert any("Missing section metadata" in i for i in issues)
        assert is_valid is True

    def test_valid_document_returns_true_with_no_issues(self):
        doc = Document.create(
            source_url="https://keats.kcl.ac.uk/x",
            title="t",
            content="This document has plenty of content to satisfy the length thresholds "
            "imposed by the validator. It also has a section metadata set.",
            content_type="page",
            section="Main",
        )
        is_valid, issues = ContentValidator().validate_document(doc)
        assert is_valid is True
        assert issues == []


class TestGenerateQualityReport:
    """Tests for ContentValidator.generate_quality_report."""

    def _doc(self, content: str, section: str = "Main") -> Document:
        return Document.create(
            source_url=f"https://keats.kcl.ac.uk/{hash(content)}",
            title="t",
            content=content,
            content_type="page",
            section=section,
        )

    def test_empty_list_returns_zero_totals(self):
        report = ContentValidator().generate_quality_report([])
        assert report["total_documents"] == 0
        assert report["total_words"] == 0
        assert report["section_count"] == 0

    def test_counts_empty_and_valid_docs(self):
        docs = [
            self._doc(""),
            self._doc("This document has enough words to pass all validator length thresholds."),
        ]
        report = ContentValidator().generate_quality_report(docs)
        assert report["total_documents"] == 2
        assert report["empty_documents"] == 1
        assert report["valid_documents"] == 1
        assert report["invalid_documents"] == 1

    def test_records_sections_and_word_stats(self):
        docs = [
            self._doc(
                "This document has enough words to pass all length thresholds in the validator.",
                section="Alpha",
            ),
            self._doc(
                "A second document also long enough for the validator to accept as valid content.",
                section="Beta",
            ),
        ]
        report = ContentValidator().generate_quality_report(docs)
        assert report["sections_covered"] == ["Alpha", "Beta"]
        assert report["section_count"] == 2
        assert report["total_words"] > 0
        assert report["min_words"] >= 1
        assert report["max_words"] >= report["min_words"]

    def test_records_issues_by_doc(self):
        short_doc = self._doc("short", section="")
        report = ContentValidator().generate_quality_report([short_doc])
        assert short_doc.id in report["documents_with_issues"]
        entry = report["documents_with_issues"][short_doc.id]
        assert "issues" in entry
        assert "title" in entry
