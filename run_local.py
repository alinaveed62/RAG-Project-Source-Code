#!/usr/bin/env python3
"""Local startup script for the RAG chatbot.

Connects to Ollama for the language model, builds or loads the FAISS
index, then starts the Flask web interface. The default model is
gemma2:2b, which was the best performer in the evaluation.

Usage:
    python run_local.py                     # Start with defaults (gemma2:2b, loopback only)
    python run_local.py --build-index       # Rebuild the FAISS index first
    python run_local.py --model mistral     # Override the default Ollama model
    python run_local.py --host 0.0.0.0      # Expose on LAN (use with care)
"""

import argparse
import logging
import sys
from pathlib import Path

import requests

from flask_app.app import create_app
from rag_pipeline.config import RAGConfig
from rag_pipeline.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = PROJECT_ROOT / "keats_scraper" / "data" / "chunks" / "chunks_for_embedding.jsonl"
INDEX_DIR = PROJECT_ROOT / "rag_pipeline" / "data" / "index"


def check_ollama(base_url: str) -> bool:
    """Return True if Ollama responds at base_url.

    Catches the full RequestException hierarchy so a slow or partially
    responsive Ollama is reported as "not running" rather than raising.
    """
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the RAG chatbot locally")
    parser.add_argument(
        "--model",
        default="gemma2:2b",
        help="Ollama model name (default: gemma2:2b, the evaluation winner)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Network interface for Flask. Defaults to loopback; pass "
            "0.0.0.0 only if you intentionally want LAN access."
        ),
    )
    parser.add_argument("--port", type=int, default=5000, help="Flask port")
    parser.add_argument("--build-index", action="store_true", help="Rebuild FAISS index")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    args = parser.parse_args()

    # 1. Check Ollama
    if not check_ollama(args.ollama_url):
        logger.error("Ollama is not running at %s. Start with: ollama serve", args.ollama_url)
        sys.exit(1)
    logger.info("Ollama is running at %s", args.ollama_url)

    # 2. Check chunks exist
    if not CHUNKS_PATH.exists():
        logger.error(
            "Chunks file not found at %s. Run the scraper first:\n"
            "  cd keats_scraper && python main.py login && python main.py all",
            CHUNKS_PATH,
        )
        sys.exit(1)

    # 3. Create pipeline config
    config = RAGConfig(
        chunks_path=CHUNKS_PATH,
        index_dir=INDEX_DIR,
        inference_backend="ollama",
        ollama_model=args.model,
        ollama_base_url=args.ollama_url,
    )

    pipeline = RAGPipeline(config)

    # 4. Build or load index
    if args.build_index or not (INDEX_DIR / "index.faiss").exists():
        logger.info("Building FAISS index from %s ...", CHUNKS_PATH)
        pipeline.build_index()
    else:
        logger.info("Loading existing FAISS index from %s", INDEX_DIR)

    # 5. Set up pipeline (load index + connect to Ollama)
    pipeline.setup()

    # 6. Start Flask
    logger.info(
        "Starting Flask on %s:%d with model '%s'", args.host, args.port, args.model
    )
    app = create_app(pipeline=pipeline)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
