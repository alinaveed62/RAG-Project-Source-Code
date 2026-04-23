"""Configuration management for KEATS scraper."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the .env file if present.
load_dotenv()

# Base paths.
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"


@dataclass
class KEATSConfig:
    """KEATS-specific configuration."""

    course_url: str = os.getenv(
        "KEATS_COURSE_URL", "https://keats.kcl.ac.uk/course/view.php?id=130212"
    )
    login_url: str = "https://keats.kcl.ac.uk/login/index.php"
    base_url: str = "https://keats.kcl.ac.uk"


@dataclass
class AuthConfig:
    """Authentication configuration."""

    cookie_file: Path = BASE_DIR / ".cookies"
    encryption_key: str = os.getenv("COOKIE_ENCRYPTION_KEY", "")
    login_timeout: int = 300  # 5 minutes for manual 2FA
    session_check_url: str = "https://keats.kcl.ac.uk/my/"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = int(os.getenv("REQUESTS_PER_MINUTE", "20"))
    min_delay_seconds: float = float(os.getenv("MIN_DELAY_SECONDS", "2"))
    max_delay_seconds: float = float(os.getenv("MAX_DELAY_SECONDS", "5"))
    max_retries: int = 3
    backoff_factor: float = 2.0


@dataclass
class ChunkConfig:
    """Chunking configuration for the RAG pipeline.

    chunk_strategy chooses between the default recursive chunker and
    the optional semantic-boundary chunker. The two extra semantic_*
    fields only apply when the semantic strategy is selected; under
    the default recursive strategy they are ignored, so existing
    runs keep producing byte-identical output.
    """

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    preserve_headings: bool = True
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "recursive")
    semantic_percentile_threshold: int = int(
        os.getenv("SEMANTIC_PERCENTILE_THRESHOLD", "80")
    )
    semantic_min_tokens: int = int(os.getenv("SEMANTIC_MIN_TOKENS", "100"))


@dataclass
class ScraperConfig:
    """Main scraper configuration."""

    keats: KEATSConfig = field(default_factory=KEATSConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)

    # Output paths.
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    chunks_dir: Path = CHUNKS_DIR

    # Logging.
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Path = BASE_DIR / "scraper.log"

    def ensure_directories(self):
        """Create all required directories."""
        for directory in [
            self.raw_dir / "html",
            self.raw_dir / "pdf",
            self.processed_dir,
            self.chunks_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Global config instance used throughout the scraper package.
config = ScraperConfig()
