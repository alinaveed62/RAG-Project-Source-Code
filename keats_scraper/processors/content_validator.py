"""Content quality validation for scraped documents."""

from typing import Any

from keats_scraper.models.document import Document
from keats_scraper.utils.logging_config import get_logger

logger = get_logger()


class ContentValidator:
    """Validates document content quality and generates reports."""

    MIN_CONTENT_LENGTH = 50  # chars
    MIN_WORD_COUNT = 10

    BOILERPLATE_PHRASES = [
        "Skip to main content",
        "You are not logged in",
        "Log in to your account",
        "Turn editing on",
        "Turn editing off",
        "Navigation",
        "Home",
    ]

    def validate_document(self, doc: Document) -> tuple[bool, list[str]]:
        """
        Validate a single document's content quality.

        Args:
            doc: Document to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if not doc.content or not doc.content.strip():
            issues.append("Empty content")
            return False, issues

        content = doc.content.strip()

        if len(content) < self.MIN_CONTENT_LENGTH:
            issues.append(f"Content too short ({len(content)} chars, min {self.MIN_CONTENT_LENGTH})")

        word_count = len(content.split())
        if word_count < self.MIN_WORD_COUNT:
            issues.append(f"Too few words ({word_count}, min {self.MIN_WORD_COUNT})")

        # Check if content is mostly boilerplate
        boilerplate_count = sum(
            1 for phrase in self.BOILERPLATE_PHRASES
            if phrase.lower() in content.lower()
        )
        if boilerplate_count >= 3:
            issues.append(f"Likely boilerplate ({boilerplate_count} boilerplate phrases detected)")

        if not doc.metadata.section:
            issues.append("Missing section metadata")

        is_valid = not any(
            issue.startswith("Empty") or issue.startswith("Content too short") or issue.startswith("Too few words")
            for issue in issues
        )

        return is_valid, issues

    def generate_quality_report(self, documents: list[Document]) -> dict[str, Any]:
        """
        Generate a quality report for a collection of documents.

        Args:
            documents: List of documents to analyze

        Returns:
            Dictionary with quality statistics
        """
        total = len(documents)
        empty_count = 0
        valid_count = 0
        total_words = 0
        sections = set()
        issues_by_doc = {}
        word_counts = []

        for doc in documents:
            is_valid, issues = self.validate_document(doc)

            if not doc.content or not doc.content.strip():
                empty_count += 1
            else:
                words = len(doc.content.split())
                total_words += words
                word_counts.append(words)

            if is_valid:
                valid_count += 1

            if doc.metadata.section:
                sections.add(doc.metadata.section)

            if issues:
                issues_by_doc[doc.id] = {
                    "title": doc.metadata.title,
                    "issues": issues,
                }

        report = {
            "total_documents": total,
            "valid_documents": valid_count,
            "empty_documents": empty_count,
            "invalid_documents": total - valid_count,
            "total_words": total_words,
            "avg_words_per_doc": total_words / max(1, total - empty_count),
            "min_words": min(word_counts) if word_counts else 0,
            "max_words": max(word_counts) if word_counts else 0,
            "sections_covered": sorted(sections),
            "section_count": len(sections),
            "documents_with_issues": issues_by_doc,
        }

        logger.info(
            f"Quality report: {valid_count}/{total} valid, "
            f"{empty_count} empty, {len(sections)} sections"
        )

        return report
