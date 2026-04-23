"""Rate limiting for respectful web scraping."""

import random
import threading
import time
from collections.abc import Callable
from typing import TypeVar

import requests

from keats_scraper.config import RateLimitConfig
from keats_scraper.utils.exceptions import RateLimitError
from keats_scraper.utils.logging_config import get_logger

T = TypeVar("T")

logger = get_logger()


class RateLimiter:
    """Implements rate limiting with random delays and exponential backoff."""

    def __init__(self, config: RateLimitConfig | None = None):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._last_request_time: float = 0
        self._request_count: int = 0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Wait appropriate time before next request.

        Holding self._lock across the elapsed/sleep/update block ensures
        that two concurrent workers cannot both pass the elapsed check and
        issue requests simultaneously, which would effectively halve the
        configured rate limit.
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time

            # Calculate minimum delay
            min_interval = 60.0 / self.config.requests_per_minute

            # Add random jitter
            delay = random.uniform(
                max(min_interval, self.config.min_delay_seconds),
                self.config.max_delay_seconds,
            )

            # Wait if needed
            if elapsed < delay:
                sleep_time = delay - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)

            self._last_request_time = time.time()
            self._request_count += 1

    def backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self.config.min_delay_seconds * (
            self.config.backoff_factor ** attempt
        )
        # Add jitter
        delay *= random.uniform(0.5, 1.5)
        return min(delay, 60.0)  # Cap at 60 seconds

    def reset(self) -> None:
        """Reset rate limiter state."""
        self._last_request_time = 0
        self._request_count = 0

    def retry_on_rate_limit(
        self,
        func: Callable[[], T],
        *,
        max_retries: int | None = None,
    ) -> T:
        """Invoke func and retry on HTTP 429, sleeping via backoff.

        The scraper is intentionally under the server rate limit
        (RateLimitConfig defaults to 20 req/min against a
        server that allows more), but KEATS can still bounce a single
        request with 429 during a traffic spike. When that happens the
        correct response is exponential back-off with jitter, not an
        immediate permanent failure. retry_on_rate_limit centralises
        that contract so every outbound request site shares one retry
        policy.

        Non-429 HTTPError responses (401, 403, 404, 5xx) propagate
        immediately -- they are not rate-limit problems and retrying
        would only amplify the issue. On retry exhaustion we raise the
        typed RateLimitError so callers can distinguish
        "server is throttling us" from "the document is gone".
        """
        attempts = max_retries if max_retries is not None else self.config.max_retries
        # Clamp to at least one attempt. A caller passing max_retries=0
        # (or mis-configuring RateLimitConfig.max_retries to 0) would
        # otherwise raise RateLimitError without ever invoking func,
        # which masks the real failure mode.
        attempts = max(1, attempts)
        last_url: str | None = None

        for attempt in range(attempts):  # pragma: no branch - loop-exit fall-through tested via max_retries exhausted path
            try:
                return func()
            except requests.HTTPError as exc:
                response = exc.response
                if response is None or response.status_code != 429:
                    raise
                last_url = getattr(response, "url", None)
                if attempt + 1 >= attempts:
                    break
                delay = self.backoff(attempt)
                logger.warning(
                    "Rate limited (429) on attempt %d/%d for %s -- "
                    "sleeping %.2fs before retry",
                    attempt + 1,
                    attempts,
                    last_url or "?",
                    delay,
                )
                time.sleep(delay)

        raise RateLimitError(
            f"Rate limited by server after {attempts} attempts "
            f"(url={last_url or '?'})"
        )

    @property
    def request_count(self) -> int:
        """Get total request count."""
        return self._request_count
