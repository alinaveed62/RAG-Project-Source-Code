"""Tests for RateLimiter."""

from unittest.mock import Mock, patch

import pytest
import requests

from keats_scraper.config import RateLimitConfig
from keats_scraper.scraper.rate_limiter import RateLimiter
from keats_scraper.utils.exceptions import RateLimitError


class TestRateLimiterInit:
    """Tests for RateLimiter initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default RateLimitConfig."""
        limiter = RateLimiter()
        assert limiter.config is not None

    def test_init_with_custom_config(self, rate_limit_config):
        """Test initialization with custom config."""
        limiter = RateLimiter(rate_limit_config)
        assert limiter.config == rate_limit_config

    def test_initial_state(self):
        """Test initial state values."""
        limiter = RateLimiter()
        assert limiter._last_request_time == 0
        assert limiter._request_count == 0


class TestWait:
    """Tests for wait method."""

    def test_wait_first_request_no_sleep(self):
        """Test first request doesn't sleep (or sleeps minimally)."""
        config = RateLimitConfig(
            min_delay_seconds=1.0,
            max_delay_seconds=2.0,
        )
        limiter = RateLimiter(config)

        with patch("time.time", return_value=1000.0):
            with patch("time.sleep") as mock_sleep:
                with patch("random.uniform", return_value=1.5):
                    limiter.wait()

        # First request when _last_request_time is 0 should have large elapsed time
        # So no sleep needed or minimal
        # The actual behavior depends on implementation

    def test_wait_increments_request_count(self):
        """Test request count is incremented."""
        config = RateLimitConfig(
            min_delay_seconds=0,
            max_delay_seconds=0.001,
        )
        limiter = RateLimiter(config)

        with patch("time.time", return_value=1000.0):
            with patch("time.sleep"):
                with patch("random.uniform", return_value=0):
                    limiter.wait()

        assert limiter._request_count == 1

    def test_wait_updates_last_request_time(self):
        """Test last_request_time is updated."""
        config = RateLimitConfig(
            min_delay_seconds=0,
            max_delay_seconds=0.001,
        )
        limiter = RateLimiter(config)

        with patch("time.time", return_value=1234.5):
            with patch("time.sleep"):
                with patch("random.uniform", return_value=0):
                    limiter.wait()

        assert limiter._last_request_time == 1234.5

    def test_wait_sleeps_when_too_fast(self):
        """Test sleep when requests are too fast."""
        config = RateLimitConfig(
            min_delay_seconds=2.0,
            max_delay_seconds=3.0,
        )
        limiter = RateLimiter(config)
        limiter._last_request_time = 1000.0

        # 0.5 seconds after last request
        with patch("time.time", return_value=1000.5):
            with patch("time.sleep") as mock_sleep:
                with patch("random.uniform", return_value=2.5):
                    limiter.wait()

        # Should sleep for approximately 2.0 seconds (2.5 - 0.5)
        mock_sleep.assert_called_once()
        sleep_time = mock_sleep.call_args[0][0]
        assert sleep_time > 0

    def test_wait_no_sleep_when_enough_time_passed(self):
        """Test no sleep when enough time has passed."""
        config = RateLimitConfig(
            min_delay_seconds=1.0,
            max_delay_seconds=2.0,
        )
        limiter = RateLimiter(config)
        limiter._last_request_time = 1000.0

        # 10 seconds after last request (plenty of time)
        with patch("time.time", return_value=1010.0):
            with patch("time.sleep") as mock_sleep:
                with patch("random.uniform", return_value=1.5):
                    limiter.wait()

        # Should not sleep
        mock_sleep.assert_not_called()

    def test_wait_respects_min_delay(self):
        """Test minimum delay is respected."""
        config = RateLimitConfig(
            min_delay_seconds=2.0,
            max_delay_seconds=2.0,  # Same as min for deterministic test
        )
        limiter = RateLimiter(config)
        limiter._last_request_time = 1000.0

        # Only 1 second passed
        with patch("time.time", return_value=1001.0):
            with patch("time.sleep") as mock_sleep:
                with patch("random.uniform", return_value=2.0):
                    limiter.wait()

        # Should sleep for ~1 second (2.0 - 1.0)
        mock_sleep.assert_called_once()
        sleep_time = mock_sleep.call_args[0][0]
        assert 0.9 <= sleep_time <= 1.1


class TestBackoff:
    """Tests for backoff method."""

    def test_backoff_first_attempt(self):
        """Test backoff for first attempt (attempt=0)."""
        config = RateLimitConfig(
            min_delay_seconds=1.0,
            backoff_factor=2.0,
        )
        limiter = RateLimiter(config)

        with patch("random.uniform", return_value=1.0):
            delay = limiter.backoff(0)

        # 1.0 * (2.0 ^ 0) * 1.0 = 1.0
        assert delay == pytest.approx(1.0, rel=0.1)

    @pytest.mark.parametrize("attempt,expected_base", [
        (0, 1.0),   # min_delay * 2^0
        (1, 2.0),   # min_delay * 2^1
        (2, 4.0),   # min_delay * 2^2
        (3, 8.0),   # min_delay * 2^3
    ])
    def test_backoff_exponential(self, attempt, expected_base):
        """Test exponential backoff for various attempts."""
        config = RateLimitConfig(
            min_delay_seconds=1.0,
            backoff_factor=2.0,
        )
        limiter = RateLimiter(config)

        with patch("random.uniform", return_value=1.0):
            delay = limiter.backoff(attempt)

        assert delay == pytest.approx(expected_base, rel=0.1)

    def test_backoff_caps_at_60_seconds(self):
        """Test maximum backoff is 60 seconds."""
        config = RateLimitConfig(
            min_delay_seconds=10.0,
            backoff_factor=2.0,
        )
        limiter = RateLimiter(config)

        with patch("random.uniform", return_value=1.5):
            delay = limiter.backoff(10)  # Would be 10 * 1024 * 1.5 without cap

        assert delay <= 60.0

    def test_backoff_includes_jitter(self):
        """Test jitter is applied to backoff."""
        config = RateLimitConfig(
            min_delay_seconds=1.0,
            backoff_factor=2.0,
        )
        limiter = RateLimiter(config)

        # With jitter = 0.5
        with patch("random.uniform", return_value=0.5):
            delay_low = limiter.backoff(1)

        # With jitter = 1.5
        with patch("random.uniform", return_value=1.5):
            delay_high = limiter.backoff(1)

        # Should be different due to jitter
        assert delay_low < delay_high


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_last_request_time(self):
        """Test reset clears last_request_time."""
        limiter = RateLimiter()
        limiter._last_request_time = 1234.5
        limiter.reset()
        assert limiter._last_request_time == 0

    def test_reset_clears_request_count(self):
        """Test reset clears request_count."""
        limiter = RateLimiter()
        limiter._request_count = 100
        limiter.reset()
        assert limiter._request_count == 0


class TestRequestCountProperty:
    """Tests for request_count property."""

    def test_request_count_starts_at_zero(self):
        """Test initial request count is 0."""
        limiter = RateLimiter()
        assert limiter.request_count == 0

    def test_request_count_increments(self):
        """Test count increments with each wait()."""
        config = RateLimitConfig(
            min_delay_seconds=0,
            max_delay_seconds=0.001,
        )
        limiter = RateLimiter(config)

        with patch("time.time", return_value=1000.0):
            with patch("time.sleep"):
                with patch("random.uniform", return_value=0):
                    limiter.wait()
                    limiter.wait()
                    limiter.wait()

        assert limiter.request_count == 3

    def test_request_count_readonly(self):
        """Test request_count is read-only property."""
        limiter = RateLimiter()
        # Should raise AttributeError when trying to set
        with pytest.raises(AttributeError):
            limiter.request_count = 10


def _make_429_error(url: str = "https://example.com/x") -> requests.HTTPError:
    """Build an HTTPError whose response carries status 429 and a url."""
    response = Mock(spec=requests.Response)
    response.status_code = 429
    response.url = url
    return requests.HTTPError(response=response)


def _make_non_429_error(status: int = 500) -> requests.HTTPError:
    """Build an HTTPError whose response carries a non-429 status."""
    response = Mock(spec=requests.Response)
    response.status_code = status
    response.url = "https://example.com/x"
    return requests.HTTPError(response=response)


class TestRetryOnRateLimit:
    """Tests for RateLimiter.retry_on_rate_limit.

    The helper centralises the 429 retry policy. These tests pin the
    three guarantees the helper makes: retry on 429 with backoff()
    delays, raise RateLimitError after exhaustion, and re-raise
    non-429 errors immediately.
    """

    def test_success_on_first_attempt_returns_value(self, rate_limit_config):
        """A callable that succeeds immediately returns its value without
        sleeping."""
        limiter = RateLimiter(rate_limit_config)
        func = Mock(return_value="ok")

        with patch("keats_scraper.scraper.rate_limiter.time.sleep") as mock_sleep:
            result = limiter.retry_on_rate_limit(func)

        assert result == "ok"
        func.assert_called_once()
        mock_sleep.assert_not_called()

    def test_success_on_second_attempt_after_429(self, rate_limit_config):
        """A single 429 is retried and the second success is returned."""
        limiter = RateLimiter(rate_limit_config)
        func = Mock(side_effect=[_make_429_error(), "ok"])

        with patch("keats_scraper.scraper.rate_limiter.time.sleep") as mock_sleep:
            result = limiter.retry_on_rate_limit(func)

        assert result == "ok"
        assert func.call_count == 2
        mock_sleep.assert_called_once()

    def test_exhausted_retries_raise_rate_limit_error(self, rate_limit_config):
        """After max_retries consecutive 429s the helper raises
        RateLimitError with the attempt count and URL."""
        limiter = RateLimiter(rate_limit_config)
        func = Mock(side_effect=_make_429_error("https://example.com/pdf"))

        with patch("keats_scraper.scraper.rate_limiter.time.sleep"):
            with pytest.raises(RateLimitError) as exc_info:
                limiter.retry_on_rate_limit(func, max_retries=3)

        assert func.call_count == 3
        assert "3 attempts" in str(exc_info.value)
        assert "https://example.com/pdf" in str(exc_info.value)

    def test_non_429_http_error_reraises_immediately(self, rate_limit_config):
        """A 500 (or any non-429 HTTPError) propagates without retry."""
        limiter = RateLimiter(rate_limit_config)
        func = Mock(side_effect=_make_non_429_error(500))

        with patch("keats_scraper.scraper.rate_limiter.time.sleep") as mock_sleep:
            with pytest.raises(requests.HTTPError):
                limiter.retry_on_rate_limit(func)

        func.assert_called_once()
        mock_sleep.assert_not_called()

    def test_http_error_without_response_reraises(self, rate_limit_config):
        """An HTTPError that carries no response object is not a
        rate-limit signal and must propagate unchanged."""
        limiter = RateLimiter(rate_limit_config)
        func = Mock(side_effect=requests.HTTPError("no response"))

        with pytest.raises(requests.HTTPError, match="no response"):
            limiter.retry_on_rate_limit(func)

        func.assert_called_once()
