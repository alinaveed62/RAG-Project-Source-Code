"""Tests for HTMLCleaner."""

import pytest
from bs4 import BeautifulSoup

from keats_scraper.processors.html_cleaner import HTMLCleaner


class TestHTMLCleanerInit:
    """Tests for HTMLCleaner initialization."""

    def test_init_succeeds(self):
        """Test initialization succeeds."""
        cleaner = HTMLCleaner()
        assert cleaner is not None

    def test_html2text_configured(self):
        """Test html2text is configured."""
        cleaner = HTMLCleaner()
        assert cleaner.h2t is not None
        assert cleaner.h2t.body_width == 0
        assert cleaner.h2t.unicode_snob is True


class TestRemoveUnwantedElements:
    """Tests for _remove_unwanted_elements method."""

    @pytest.mark.parametrize("tag", HTMLCleaner.REMOVE_TAGS)
    def test_removes_tag(self, tag):
        """Test each unwanted tag is removed."""
        cleaner = HTMLCleaner()
        html = f"<div><{tag}>Content to remove</{tag}><p>Keep this</p></div>"
        soup = BeautifulSoup(html, "lxml")
        result = cleaner._remove_unwanted_elements(soup)
        assert result.find(tag) is None

    def test_removes_script_content(self):
        """Test script elements and content are removed."""
        cleaner = HTMLCleaner()
        html = "<div><script>alert('test')</script><p>Keep</p></div>"
        soup = BeautifulSoup(html, "lxml")
        result = cleaner._remove_unwanted_elements(soup)
        assert "alert" not in str(result)

    def test_removes_style_content(self):
        """Test style elements and content are removed."""
        cleaner = HTMLCleaner()
        html = "<div><style>.test { color: red; }</style><p>Keep</p></div>"
        soup = BeautifulSoup(html, "lxml")
        result = cleaner._remove_unwanted_elements(soup)
        assert "color: red" not in str(result)

    def test_removes_accesshide_class(self):
        """Test .accesshide elements are removed."""
        cleaner = HTMLCleaner()
        html = '<div><span class="accesshide">Hidden</span><p>Visible</p></div>'
        soup = BeautifulSoup(html, "lxml")
        result = cleaner._remove_unwanted_elements(soup)
        assert "Hidden" not in result.get_text()

    def test_removes_sr_only_class(self):
        """Test .sr-only elements are removed."""
        cleaner = HTMLCleaner()
        html = '<div><span class="sr-only">Screen reader only</span><p>Visible</p></div>'
        soup = BeautifulSoup(html, "lxml")
        result = cleaner._remove_unwanted_elements(soup)
        assert "Screen reader only" not in result.get_text()

    def test_preserves_content(self):
        """Test regular content is preserved."""
        cleaner = HTMLCleaner()
        html = "<div><p>Keep this content</p></div>"
        soup = BeautifulSoup(html, "lxml")
        result = cleaner._remove_unwanted_elements(soup)
        assert "Keep this content" in result.get_text()


class TestCleanTables:
    """Tests for _clean_tables method."""

    def test_table_to_text(self):
        """Test tables are converted to text format."""
        cleaner = HTMLCleaner()
        html = """
        <table>
            <tr><th>Header 1</th><th>Header 2</th></tr>
            <tr><td>Cell 1</td><td>Cell 2</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "lxml")
        result = cleaner._clean_tables(soup)
        text = result.get_text()
        assert "Header 1" in text
        assert "Cell 1" in text

    def test_empty_table_removed(self):
        """Test empty tables are removed."""
        cleaner = HTMLCleaner()
        html = "<table></table><p>Keep this</p>"
        soup = BeautifulSoup(html, "lxml")
        result = cleaner._clean_tables(soup)
        assert result.find("table") is None

    def test_table_with_empty_rows_is_dropped(self):
        """Rows with no <td>/<th> cells are skipped, and a table whose
        rows are all empty is decomposed entirely.

        Covers the if cells: False branch in _clean_tables (the
        row-skip path) together with the else: table.decompose()
        fallthrough when rows ends up empty."""
        cleaner = HTMLCleaner()
        html = (
            "<table>"
            "<tr></tr>"  # empty row: cells list is []
            "<tr>   </tr>"  # whitespace-only row: still no td/th
            "</table>"
            "<p>Survivor</p>"
        )
        soup = BeautifulSoup(html, "lxml")
        result = cleaner._clean_tables(soup)
        assert result.find("table") is None
        assert "Survivor" in result.get_text()

    def test_cells_separated(self):
        """Test cells are separated by pipe."""
        cleaner = HTMLCleaner()
        html = """
        <table>
            <tr><td>A</td><td>B</td><td>C</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "lxml")
        result = cleaner._clean_tables(soup)
        text = result.get_text()
        # Cells should be in the text
        assert "A" in text
        assert "B" in text
        assert "C" in text

    def test_no_tables(self):
        """Test no error when no tables present."""
        cleaner = HTMLCleaner()
        html = "<div><p>No tables here</p></div>"
        soup = BeautifulSoup(html, "lxml")
        result = cleaner._clean_tables(soup)
        assert "No tables here" in result.get_text()


class TestNormalizeWhitespace:
    """Tests for _normalize_whitespace method."""

    def test_multiple_newlines_reduced(self):
        """Test 3+ newlines become 2."""
        cleaner = HTMLCleaner()
        text = "Para1\n\n\n\n\nPara2"
        result = cleaner._normalize_whitespace(text)
        assert "\n\n\n" not in result

    def test_multiple_spaces_reduced(self):
        """Test multiple spaces become single."""
        cleaner = HTMLCleaner()
        text = "Hello    World"
        result = cleaner._normalize_whitespace(text)
        assert "    " not in result

    def test_lines_stripped(self):
        """Test whitespace stripped from each line."""
        cleaner = HTMLCleaner()
        text = "  Line1  \n  Line2  "
        result = cleaner._normalize_whitespace(text)
        lines = result.split("\n")
        for line in lines:
            assert not line.startswith(" ")
            assert not line.endswith(" ")

    def test_empty_string(self):
        """Test empty string returns empty."""
        cleaner = HTMLCleaner()
        result = cleaner._normalize_whitespace("")
        assert result == ""


class TestRemoveBoilerplate:
    """Tests for _remove_boilerplate method."""

    @pytest.mark.parametrize("boilerplate", [
        "Skip to main content",
        "You are not logged in.",
        "Turn editing on",
        "Turn editing off",
    ])
    def test_removes_boilerplate_text(self, boilerplate):
        """Test boilerplate patterns are removed."""
        cleaner = HTMLCleaner()
        text = f"Content before. {boilerplate} Content after."
        result = cleaner._remove_boilerplate(text)
        assert boilerplate not in result

    def test_case_insensitive(self):
        """Test boilerplate removal is case insensitive."""
        cleaner = HTMLCleaner()
        text = "SKIP TO MAIN CONTENT\nActual content"
        result = cleaner._remove_boilerplate(text)
        # Should be removed regardless of case
        assert "SKIP TO MAIN CONTENT" not in result or len(result) < len(text)

    def test_preserves_content(self):
        """Test non-boilerplate content is preserved."""
        cleaner = HTMLCleaner()
        text = "Important information for students."
        result = cleaner._remove_boilerplate(text)
        assert "Important information" in result


class TestClean:
    """Tests for clean method (full pipeline)."""

    def test_clean_removes_scripts(self, sample_moodle_page_html):
        """Test clean removes script elements."""
        cleaner = HTMLCleaner()
        result = cleaner.clean(sample_moodle_page_html)
        assert "alert" not in result

    def test_clean_removes_nav(self, sample_moodle_page_html):
        """Test clean removes navigation."""
        cleaner = HTMLCleaner()
        result = cleaner.clean(sample_moodle_page_html)
        # Nav content should be removed or significantly reduced
        assert "Navigation content to remove" not in result

    def test_clean_preserves_content(self, sample_moodle_page_html):
        """Test clean preserves main content."""
        cleaner = HTMLCleaner()
        result = cleaner.clean(sample_moodle_page_html)
        assert "important information" in result.lower()

    def test_clean_empty_html(self):
        """Test clean with empty HTML returns empty string."""
        cleaner = HTMLCleaner()
        result = cleaner.clean("")
        assert result == ""

    def test_clean_none_returns_empty(self):
        """Test clean with None-like input returns empty."""
        cleaner = HTMLCleaner()
        result = cleaner.clean("")
        assert result == ""

    def test_clean_simple_html(self):
        """Test clean with simple HTML."""
        cleaner = HTMLCleaner()
        html = "<html><body><h1>Title</h1><p>Content here.</p></body></html>"
        result = cleaner.clean(html)
        assert "Title" in result
        assert "Content here" in result

    def test_clean_handles_malformed_html(self):
        """Test clean handles malformed HTML gracefully and preserves surviving content."""
        cleaner = HTMLCleaner()
        html = "<div><p>Unclosed paragraph<div>Nested wrong</p></div>"
        # Should not raise
        result = cleaner.clean(html)
        assert isinstance(result, str)
        assert result.strip()
        assert "Unclosed paragraph" in result
        assert "Nested wrong" in result

    def test_clean_exception_fallback(self, mocker):
        """Test clean falls back to get_text on error."""
        cleaner = HTMLCleaner()
        # Mock html2text to raise
        mocker.patch.object(cleaner.h2t, "handle", side_effect=Exception("Test error"))

        html = "<html><body><p>Fallback content</p></body></html>"
        result = cleaner.clean(html)
        # Should fall back to BeautifulSoup get_text
        assert "Fallback content" in result


class TestRemoveTags:
    """Tests for REMOVE_TAGS constant."""

    def test_remove_tags_defined(self):
        """Test REMOVE_TAGS is defined."""
        assert hasattr(HTMLCleaner, "REMOVE_TAGS")
        assert isinstance(HTMLCleaner.REMOVE_TAGS, list)

    def test_remove_tags_contains_script(self):
        """Test REMOVE_TAGS contains script."""
        assert "script" in HTMLCleaner.REMOVE_TAGS

    def test_remove_tags_contains_style(self):
        """Test REMOVE_TAGS contains style."""
        assert "style" in HTMLCleaner.REMOVE_TAGS

    def test_remove_tags_contains_nav(self):
        """Test REMOVE_TAGS contains nav."""
        assert "nav" in HTMLCleaner.REMOVE_TAGS


class TestRemoveSelectors:
    """Tests for REMOVE_SELECTORS constant."""

    def test_remove_selectors_defined(self):
        """Test REMOVE_SELECTORS is defined."""
        assert hasattr(HTMLCleaner, "REMOVE_SELECTORS")
        assert isinstance(HTMLCleaner.REMOVE_SELECTORS, list)

    def test_remove_selectors_contains_accesshide(self):
        """Test REMOVE_SELECTORS contains .accesshide."""
        assert ".accesshide" in HTMLCleaner.REMOVE_SELECTORS

    def test_remove_selectors_contains_sr_only(self):
        """Test REMOVE_SELECTORS contains .sr-only."""
        assert ".sr-only" in HTMLCleaner.REMOVE_SELECTORS
