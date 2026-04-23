"""Tests for custom exceptions."""

import pytest

from keats_scraper.utils.exceptions import (
    AuthenticationError,
    CheckpointError,
    ContentExtractionError,
    RateLimitError,
    ScraperException,
    SessionExpiredError,
)


class TestScraperException:
    """Tests for the base ScraperException."""

    def test_inherits_from_exception(self):
        """Test ScraperException inherits from Exception."""
        assert issubclass(ScraperException, Exception)

    def test_can_be_raised(self):
        """Test ScraperException can be raised."""
        with pytest.raises(ScraperException):
            raise ScraperException("Test error")

    def test_message_preserved(self):
        """Test exception message is preserved."""
        msg = "This is a test error message"
        exc = ScraperException(msg)
        assert str(exc) == msg

    def test_empty_message(self):
        """Test exception with empty message."""
        exc = ScraperException()
        assert str(exc) == ""


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_inherits_from_scraper_exception(self):
        """Test AuthenticationError inherits from ScraperException."""
        assert issubclass(AuthenticationError, ScraperException)

    def test_can_be_caught_as_scraper_exception(self):
        """Test AuthenticationError can be caught as ScraperException."""
        with pytest.raises(ScraperException):
            raise AuthenticationError("Auth failed")

    def test_can_be_caught_specifically(self):
        """Test AuthenticationError can be caught specifically."""
        with pytest.raises(AuthenticationError):
            raise AuthenticationError("Auth failed")

    def test_message_preserved(self):
        """Test exception message is preserved."""
        msg = "Authentication failed: invalid credentials"
        exc = AuthenticationError(msg)
        assert str(exc) == msg


class TestSessionExpiredError:
    """Tests for SessionExpiredError."""

    def test_inherits_from_scraper_exception(self):
        """Test SessionExpiredError inherits from ScraperException."""
        assert issubclass(SessionExpiredError, ScraperException)

    def test_can_be_caught_as_scraper_exception(self):
        """Test SessionExpiredError can be caught as ScraperException."""
        with pytest.raises(ScraperException):
            raise SessionExpiredError("Session expired")

    def test_can_be_caught_specifically(self):
        """Test SessionExpiredError can be caught specifically."""
        with pytest.raises(SessionExpiredError):
            raise SessionExpiredError("Session expired")

    def test_message_preserved(self):
        """Test exception message is preserved."""
        msg = "Session expired after 2 hours"
        exc = SessionExpiredError(msg)
        assert str(exc) == msg


class TestContentExtractionError:
    """Tests for ContentExtractionError."""

    def test_inherits_from_scraper_exception(self):
        """Test ContentExtractionError inherits from ScraperException."""
        assert issubclass(ContentExtractionError, ScraperException)

    def test_can_be_caught_as_scraper_exception(self):
        """Test ContentExtractionError can be caught as ScraperException."""
        with pytest.raises(ScraperException):
            raise ContentExtractionError("Extraction failed")

    def test_can_be_caught_specifically(self):
        """Test ContentExtractionError can be caught specifically."""
        with pytest.raises(ContentExtractionError):
            raise ContentExtractionError("Extraction failed")

    def test_message_preserved(self):
        """Test exception message is preserved."""
        msg = "Failed to extract content from page"
        exc = ContentExtractionError(msg)
        assert str(exc) == msg


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_inherits_from_scraper_exception(self):
        """Test RateLimitError inherits from ScraperException."""
        assert issubclass(RateLimitError, ScraperException)

    def test_can_be_caught_as_scraper_exception(self):
        """Test RateLimitError can be caught as ScraperException."""
        with pytest.raises(ScraperException):
            raise RateLimitError("Rate limited")

    def test_can_be_caught_specifically(self):
        """Test RateLimitError can be caught specifically."""
        with pytest.raises(RateLimitError):
            raise RateLimitError("Rate limited")

    def test_message_preserved(self):
        """Test exception message is preserved."""
        msg = "Rate limited: try again in 60 seconds"
        exc = RateLimitError(msg)
        assert str(exc) == msg


class TestCheckpointError:
    """Tests for CheckpointError."""

    def test_inherits_from_scraper_exception(self):
        """Test CheckpointError inherits from ScraperException."""
        assert issubclass(CheckpointError, ScraperException)

    def test_can_be_caught_as_scraper_exception(self):
        """Test CheckpointError can be caught as ScraperException."""
        with pytest.raises(ScraperException):
            raise CheckpointError("Checkpoint failed")

    def test_can_be_caught_specifically(self):
        """Test CheckpointError can be caught specifically."""
        with pytest.raises(CheckpointError):
            raise CheckpointError("Checkpoint failed")

    def test_message_preserved(self):
        """Test exception message is preserved."""
        msg = "Failed to save checkpoint"
        exc = CheckpointError(msg)
        assert str(exc) == msg


class TestExceptionHierarchy:
    """Tests for the complete exception hierarchy."""

    @pytest.mark.parametrize("exc_class", [
        AuthenticationError,
        SessionExpiredError,
        ContentExtractionError,
        RateLimitError,
        CheckpointError,
    ])
    def test_all_inherit_from_scraper_exception(self, exc_class):
        """Test all custom exceptions inherit from ScraperException."""
        assert issubclass(exc_class, ScraperException)

    @pytest.mark.parametrize("exc_class", [
        ScraperException,
        AuthenticationError,
        SessionExpiredError,
        ContentExtractionError,
        RateLimitError,
        CheckpointError,
    ])
    def test_all_inherit_from_exception(self, exc_class):
        """Test all custom exceptions inherit from base Exception."""
        assert issubclass(exc_class, Exception)

    def test_exception_chaining(self):
        """Test exception chaining works correctly."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise AuthenticationError("Auth failed") from e
        except AuthenticationError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
