"""Tests for configuration module."""

from pathlib import Path

from keats_scraper.config import (
    AuthConfig,
    ChunkConfig,
    KEATSConfig,
    RateLimitConfig,
    ScraperConfig,
)


class TestKEATSConfig:
    """Tests for KEATSConfig dataclass."""

    def test_course_url_format(self):
        """Test course URL has correct format."""
        config = KEATSConfig()
        assert "keats.kcl.ac.uk/course/view.php" in config.course_url

    def test_login_url(self):
        """Test login URL is set."""
        config = KEATSConfig()
        assert config.login_url == "https://keats.kcl.ac.uk/login/index.php"

    def test_base_url(self):
        """Test base URL is set."""
        config = KEATSConfig()
        assert config.base_url == "https://keats.kcl.ac.uk"

    def test_course_url_is_string(self):
        """Test course URL is a well-formed KEATS course URL."""
        config = KEATSConfig()
        assert isinstance(config.course_url, str)
        assert config.course_url.startswith("https://keats.kcl.ac.uk/")
        assert "course/view.php?id=" in config.course_url


class TestAuthConfig:
    """Tests for AuthConfig dataclass."""

    def test_default_cookie_file(self):
        """Test default cookie file path."""
        config = AuthConfig()
        assert config.cookie_file.name == ".cookies"

    def test_default_login_timeout(self):
        """Test default login timeout is 5 minutes."""
        config = AuthConfig()
        assert config.login_timeout == 300

    def test_session_check_url(self):
        """Test session check URL."""
        config = AuthConfig()
        assert config.session_check_url == "https://keats.kcl.ac.uk/my/"

    def test_encryption_key_is_string(self):
        """Test encryption key is a string."""
        config = AuthConfig()
        assert isinstance(config.encryption_key, str)

    def test_cookie_file_is_path(self):
        """Test cookie_file is an absolute Path pointing at .cookies."""
        config = AuthConfig()
        assert isinstance(config.cookie_file, Path)
        assert config.cookie_file.name == ".cookies"
        assert config.cookie_file.is_absolute()


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_requests_per_minute_is_int(self):
        """Test requests per minute is an integer."""
        config = RateLimitConfig()
        assert isinstance(config.requests_per_minute, int)
        assert config.requests_per_minute > 0

    def test_min_delay_is_float(self):
        """Test min delay is a float."""
        config = RateLimitConfig()
        assert isinstance(config.min_delay_seconds, float)
        assert config.min_delay_seconds >= 0

    def test_max_delay_is_float(self):
        """Test max delay is a float."""
        config = RateLimitConfig()
        assert isinstance(config.max_delay_seconds, float)
        assert config.max_delay_seconds >= config.min_delay_seconds

    def test_default_max_retries(self):
        """Test default max retries."""
        config = RateLimitConfig()
        assert config.max_retries == 3

    def test_default_backoff_factor(self):
        """Test default backoff factor."""
        config = RateLimitConfig()
        assert config.backoff_factor == 2.0


class TestChunkConfig:
    """Tests for ChunkConfig dataclass."""

    def test_chunk_size_is_int(self):
        """Test chunk size is an integer."""
        config = ChunkConfig()
        assert isinstance(config.chunk_size, int)
        assert config.chunk_size > 0

    def test_chunk_overlap_is_int(self):
        """Test chunk overlap is an integer."""
        config = ChunkConfig()
        assert isinstance(config.chunk_overlap, int)
        assert config.chunk_overlap >= 0

    def test_preserve_headings_default(self):
        """Test preserve_headings is True by default."""
        config = ChunkConfig()
        assert config.preserve_headings is True

    def test_chunk_overlap_less_than_size(self):
        """Test chunk overlap is less than chunk size."""
        config = ChunkConfig()
        assert config.chunk_overlap < config.chunk_size


class TestScraperConfig:
    """Tests for ScraperConfig dataclass."""

    def test_nested_configs_created(self):
        """Test all nested configs are created and carry their expected defaults."""
        config = ScraperConfig()
        assert isinstance(config.keats, KEATSConfig)
        assert isinstance(config.auth, AuthConfig)
        assert isinstance(config.rate_limit, RateLimitConfig)
        assert isinstance(config.chunk, ChunkConfig)
        assert config.keats.base_url == "https://keats.kcl.ac.uk"
        assert config.auth.login_timeout == 300
        assert config.rate_limit.max_retries == 3
        assert config.chunk.preserve_headings is True

    def test_data_directories_are_paths(self):
        """Test data directories are Path objects."""
        config = ScraperConfig()
        assert isinstance(config.data_dir, Path)
        assert isinstance(config.raw_dir, Path)
        assert isinstance(config.processed_dir, Path)
        assert isinstance(config.chunks_dir, Path)

    def test_log_file_is_path(self):
        """Test log file is a Path."""
        config = ScraperConfig()
        assert isinstance(config.log_file, Path)

    def test_log_level_is_string(self):
        """Test log level is a string."""
        config = ScraperConfig()
        assert isinstance(config.log_level, str)
        assert config.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class TestEnsureDirectories:
    """Tests for ScraperConfig.ensure_directories method."""

    def test_creates_all_directories(self, tmp_path):
        """Test ensure_directories creates all required directories."""
        config = ScraperConfig()
        config.raw_dir = tmp_path / "raw"
        config.processed_dir = tmp_path / "processed"
        config.chunks_dir = tmp_path / "chunks"

        config.ensure_directories()

        assert (tmp_path / "raw" / "html").exists()
        assert (tmp_path / "raw" / "pdf").exists()
        assert (tmp_path / "processed").exists()
        assert (tmp_path / "chunks").exists()

    def test_idempotent(self, tmp_path):
        """Test calling ensure_directories multiple times is safe."""
        config = ScraperConfig()
        config.raw_dir = tmp_path / "raw"
        config.processed_dir = tmp_path / "processed"
        config.chunks_dir = tmp_path / "chunks"

        # Call twice
        config.ensure_directories()
        config.ensure_directories()

        # Should still exist
        assert (tmp_path / "raw" / "html").exists()
        assert (tmp_path / "raw" / "pdf").exists()

    def test_creates_nested_directories(self, tmp_path):
        """Test creates nested directories with parents."""
        config = ScraperConfig()
        config.raw_dir = tmp_path / "deep" / "nested" / "raw"
        config.processed_dir = tmp_path / "deep" / "nested" / "processed"
        config.chunks_dir = tmp_path / "deep" / "nested" / "chunks"

        config.ensure_directories()

        assert (tmp_path / "deep" / "nested" / "raw" / "html").exists()
        assert (tmp_path / "deep" / "nested" / "raw" / "pdf").exists()
