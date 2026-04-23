"""Tests for the shared heading-hierarchy extractor."""

from __future__ import annotations

from keats_scraper.processors._heading_extractor import extract_heading_path


class TestExtractHeadingPath:
    def test_no_headings(self):
        text = "Just a paragraph with no headings at all."
        assert extract_heading_path(text, len(text)) == []

    def test_single_heading(self):
        text = "# Introduction\n\nSome text."
        assert extract_heading_path(text, len(text)) == ["Introduction"]

    def test_nested_hierarchy(self):
        text = "# Top\n\n## Middle\n\n### Leaf\n\nBody."
        assert extract_heading_path(text, len(text)) == [
            "Top",
            "Middle",
            "Leaf",
        ]

    def test_deeper_heading_drops_when_peer_arrives(self):
        # After '## Second Section', the earlier '### Detail' under the
        # first section must fall out of the path.
        text = (
            "# Intro\n\n"
            "## First Section\n\n"
            "### Detail\n\n"
            "## Second Section\n\n"
            "Body."
        )
        assert extract_heading_path(text, len(text)) == [
            "Intro",
            "Second Section",
        ]

    def test_position_before_any_heading(self):
        text = "# Intro\n\nBody."
        # Offset 0 is before the heading, so the path is empty.
        assert extract_heading_path(text, 0) == []

    def test_position_mid_document(self):
        text = (
            "# Alpha\n\nParagraph A.\n\n"
            "# Beta\n\nParagraph B.\n\n"
            "# Gamma\n\nParagraph C."
        )
        position = text.find("Paragraph B") + 5
        assert extract_heading_path(text, position) == ["Beta"]

    def test_title_whitespace_is_stripped(self):
        text = "#    Title with spaces   \n\nBody."
        assert extract_heading_path(text, len(text)) == ["Title with spaces"]
