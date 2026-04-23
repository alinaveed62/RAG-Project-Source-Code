"""Flask application configuration."""

import os
from pathlib import Path


class Config:
    """Base configuration."""

    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-key-change-in-production")
    FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", "flask_app/data"))
    INDEX_DIR = Path(os.environ.get("INDEX_DIR", "rag_pipeline/data/index"))
    CHUNKS_PATH = Path(
        os.environ.get(
            "CHUNKS_PATH",
            "keats_scraper/data/chunks/chunks_for_embedding.jsonl",
        )
    )
