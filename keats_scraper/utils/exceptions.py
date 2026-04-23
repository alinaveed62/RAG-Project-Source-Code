"""Custom exceptions for KEATS scraper."""


class ScraperException(Exception):
    """Base exception for scraper errors."""

    pass


class AuthenticationError(ScraperException):
    """Raised when authentication fails."""

    pass


class SessionExpiredError(ScraperException):
    """Raised when the session has expired."""

    pass


class ContentExtractionError(ScraperException):
    """Raised when content extraction fails."""

    pass


class RateLimitError(ScraperException):
    """Raised when rate limited by server."""

    pass


class CheckpointError(ScraperException):
    """Raised when checkpoint operations fail."""

    pass
