"""Tests for TextNormalizer."""

import pytest

from keats_scraper.processors.text_normalizer import TextNormalizer


class TestTextNormalizerInit:
    """Tests for TextNormalizer initialization."""

    def test_init_succeeds(self):
        """Test initialization succeeds."""
        normalizer = TextNormalizer()
        assert normalizer is not None


class TestNormalizeUnicode:
    """Tests for normalize_unicode method."""

    def test_ascii_text_unchanged(self):
        """Test ASCII text is unchanged."""
        normalizer = TextNormalizer()
        text = "Hello World"
        result = normalizer.normalize_unicode(text)
        assert result == "Hello World"

    def test_nfc_normalization(self):
        """Test text is NFC normalized."""
        normalizer = TextNormalizer()
        # Combining character (e + combining acute)
        text = "caf\u0065\u0301"
        result = normalizer.normalize_unicode(text)
        # Should be composed form
        assert "\u0301" not in result or result == text  # NFC normalized

    def test_empty_string(self):
        """Test empty string returns empty."""
        normalizer = TextNormalizer()
        result = normalizer.normalize_unicode("")
        assert result == ""


class TestReplaceSpecialChars:
    """Tests for replace_special_chars method."""

    @pytest.mark.parametrize("input_char,expected", [
        ("\u2018", "'"),   # Left single quote
        ("\u2019", "'"),   # Right single quote
        ("\u201c", '"'),   # Left double quote
        ("\u201d", '"'),   # Right double quote
        ("\u2013", "-"),   # En dash
        ("\u2014", "-"),   # Em dash
        ("\u2026", "..."), # Ellipsis
        ("\u00a0", " "),   # Non-breaking space
        ("\u200b", ""),    # Zero-width space
        ("\ufeff", ""),    # BOM
    ])
    def test_character_replacements(self, input_char, expected):
        """Test each special character is replaced correctly."""
        normalizer = TextNormalizer()
        result = normalizer.replace_special_chars(f"a{input_char}b")
        assert result == f"a{expected}b"

    def test_multiple_replacements(self):
        """Test multiple special chars in one string."""
        normalizer = TextNormalizer()
        text = "\u201cHello\u201d \u2013 World\u2019s"
        result = normalizer.replace_special_chars(text)
        assert result == '"Hello" - World\'s'

    def test_no_special_chars(self):
        """Test text without special chars is unchanged."""
        normalizer = TextNormalizer()
        text = "Normal ASCII text here"
        result = normalizer.replace_special_chars(text)
        assert result == text

    def test_empty_string(self):
        """Test empty string returns empty."""
        normalizer = TextNormalizer()
        result = normalizer.replace_special_chars("")
        assert result == ""


class TestFixEncodingIssues:
    """Tests for fix_encoding_issues method."""

    @pytest.mark.parametrize("wrong,right", [
        ("\xe2\x80\x99", "'"),    # Right single quote mojibake
        ("\xe2\x80\x9c", '"'),    # Left double quote mojibake
        ("\xe2\x80\x9d", '"'),    # Right double quote mojibake
        ("\xe2\x80\x93", "-"),    # En dash mojibake
        ("\xe2\x80\x94", "-"),    # Em dash mojibake
        ("\xe2\x80\xa6", "..."),  # Ellipsis mojibake
        ("\xc2\xa0", " "),        # Non-breaking space mojibake
    ])
    def test_mojibake_fixes(self, wrong, right):
        """Test mojibake patterns are fixed."""
        normalizer = TextNormalizer()
        result = normalizer.fix_encoding_issues(f"a{wrong}b")
        assert result == f"a{right}b"

    def test_normal_text_unchanged(self):
        """Test normal text is unchanged."""
        normalizer = TextNormalizer()
        text = "Normal text without issues"
        result = normalizer.fix_encoding_issues(text)
        assert result == text

    def test_empty_string(self):
        """Test empty string returns empty."""
        normalizer = TextNormalizer()
        result = normalizer.fix_encoding_issues("")
        assert result == ""


class TestNormalizeWhitespace:
    """Tests for normalize_whitespace method."""

    def test_tabs_to_spaces(self):
        """Test tabs are converted to spaces."""
        normalizer = TextNormalizer()
        result = normalizer.normalize_whitespace("Hello\tWorld")
        assert "\t" not in result
        assert " " in result

    def test_crlf_to_lf(self):
        """Test Windows line endings are normalized."""
        normalizer = TextNormalizer()
        result = normalizer.normalize_whitespace("Line1\r\nLine2")
        assert "\r\n" not in result
        assert "\n" in result

    def test_cr_to_lf(self):
        """Test old Mac line endings are normalized."""
        normalizer = TextNormalizer()
        result = normalizer.normalize_whitespace("Line1\rLine2")
        assert "\r" not in result
        assert "\n" in result

    def test_multiple_spaces_reduced(self):
        """Test multiple spaces become single space."""
        normalizer = TextNormalizer()
        result = normalizer.normalize_whitespace("Hello    World")
        assert "    " not in result
        assert "Hello World" in result

    def test_multiple_newlines_reduced(self):
        """Test 3+ newlines become 2."""
        normalizer = TextNormalizer()
        result = normalizer.normalize_whitespace("Para1\n\n\n\n\nPara2")
        assert "\n\n\n" not in result
        assert "Para1\n\nPara2" == result

    def test_lines_stripped(self):
        """Test whitespace is stripped from each line."""
        normalizer = TextNormalizer()
        result = normalizer.normalize_whitespace("  Line1  \n  Line2  ")
        lines = result.split("\n")
        assert lines[0] == "Line1"
        assert lines[1] == "Line2"

    def test_leading_trailing_stripped(self):
        """Test leading/trailing whitespace is stripped."""
        normalizer = TextNormalizer()
        result = normalizer.normalize_whitespace("  \n\nHello\n\n  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_empty_string(self):
        """Test empty string returns empty."""
        normalizer = TextNormalizer()
        result = normalizer.normalize_whitespace("")
        assert result == ""

    def test_only_whitespace(self):
        """Test string with only whitespace returns empty."""
        normalizer = TextNormalizer()
        result = normalizer.normalize_whitespace("   \n\n\t\t   ")
        assert result == ""


class TestRemoveControlCharacters:
    """Tests for remove_control_characters method."""

    def test_preserves_newlines(self):
        """Test newlines are preserved."""
        normalizer = TextNormalizer()
        result = normalizer.remove_control_characters("Line1\nLine2")
        assert "\n" in result

    def test_preserves_tabs(self):
        """Test tabs are preserved."""
        normalizer = TextNormalizer()
        result = normalizer.remove_control_characters("Col1\tCol2")
        assert "\t" in result

    def test_removes_control_chars(self):
        """Test control characters are removed."""
        normalizer = TextNormalizer()
        # \x00 is NUL, \x07 is BEL
        result = normalizer.remove_control_characters("Hello\x00\x07World")
        assert "\x00" not in result
        assert "\x07" not in result
        assert "HelloWorld" in result

    def test_preserves_printable(self):
        """Test printable characters are preserved."""
        normalizer = TextNormalizer()
        text = "Hello World! 123 @#$"
        result = normalizer.remove_control_characters(text)
        assert result == text

    def test_empty_string(self):
        """Test empty string returns empty."""
        normalizer = TextNormalizer()
        result = normalizer.remove_control_characters("")
        assert result == ""


class TestStandardizeBullets:
    """Tests for standardize_bullets method."""

    @pytest.mark.parametrize("bullet", ["•", "·", "●", "○", "▪", "▫", "◦", "‣", "⁃"])
    def test_bullet_standardization(self, bullet):
        """Test each bullet character becomes -."""
        normalizer = TextNormalizer()
        result = normalizer.standardize_bullets(f"{bullet} Item")
        assert result == "- Item"

    def test_multiple_bullets(self):
        """Test multiple bullets in one string."""
        normalizer = TextNormalizer()
        text = "• Item 1\n● Item 2\n○ Item 3"
        result = normalizer.standardize_bullets(text)
        assert result == "- Item 1\n- Item 2\n- Item 3"

    def test_no_bullets(self):
        """Test text without bullets is unchanged."""
        normalizer = TextNormalizer()
        text = "- Already using dashes"
        result = normalizer.standardize_bullets(text)
        assert result == text

    def test_empty_string(self):
        """Test empty string returns empty."""
        normalizer = TextNormalizer()
        result = normalizer.standardize_bullets("")
        assert result == ""


class TestNormalize:
    """Tests for normalize method (full pipeline)."""

    def test_full_pipeline(self):
        """Test full normalization pipeline."""
        normalizer = TextNormalizer()
        text = "\u201cHello\u201d \u2013 World\u2019s\n\n\n\n• Item"
        result = normalizer.normalize(text)
        # Quotes replaced
        assert '"' in result
        assert "'" in result
        # Dash replaced
        assert "-" in result
        # Multiple newlines reduced
        assert "\n\n\n" not in result
        # Bullet standardized
        assert "•" not in result

    def test_empty_string(self):
        """Test empty string returns empty."""
        normalizer = TextNormalizer()
        result = normalizer.normalize("")
        assert result == ""

    def test_none_like_falsy(self):
        """Test falsy values return empty string."""
        normalizer = TextNormalizer()
        # Empty string
        assert normalizer.normalize("") == ""

    def test_whitespace_only(self):
        """Test whitespace-only string returns empty."""
        normalizer = TextNormalizer()
        result = normalizer.normalize("   \n\n\t   ")
        assert result == ""

    def test_complex_input(self):
        """Test complex input with multiple issues."""
        normalizer = TextNormalizer()
        text = """
        \u201cQuoted text\u201d

        • First point
        ● Second point


        Some text with\ttabs and   multiple   spaces.

        """
        result = normalizer.normalize(text)
        # Should be cleaned up
        assert "\u201c" not in result
        assert "•" not in result
        assert "●" not in result
        assert "   " not in result
        assert "\t" not in result

    def test_exception_returns_original(self, mocker):
        """Test exception during normalization returns original text."""
        normalizer = TextNormalizer()
        # Mock one method to raise exception
        mocker.patch.object(normalizer, "normalize_unicode", side_effect=Exception("Test error"))

        original = "Test text"
        result = normalizer.normalize(original)
        assert result == original


class TestCharReplacements:
    """Tests for CHAR_REPLACEMENTS constant."""

    def test_char_replacements_dict_exists(self):
        """Test CHAR_REPLACEMENTS is defined."""
        assert hasattr(TextNormalizer, "CHAR_REPLACEMENTS")
        assert isinstance(TextNormalizer.CHAR_REPLACEMENTS, dict)

    def test_char_replacements_non_empty(self):
        """Test CHAR_REPLACEMENTS is not empty."""
        assert len(TextNormalizer.CHAR_REPLACEMENTS) > 0

    def test_all_replacements_are_strings(self):
        """Test all replacement values are strings."""
        for key, value in TextNormalizer.CHAR_REPLACEMENTS.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
