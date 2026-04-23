"""SentenceTransformer wrapper for encoding chunks and queries."""

import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_pipeline.config import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class ChunkEncoder:
    """Encodes text chunks and queries using a SentenceTransformer model."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def load_chunks(self, chunks_path: Path) -> list[dict]:
        """Load chunks from a JSONL file.

        Args:
            chunks_path: Path to chunks_for_embedding.jsonl.

        Returns:
            A list of chunk dicts. Each must contain id and text;
            source, title, section and heading_path are optional and
            RAGPipeline.build_index fills them with "", "", "" and []
            respectively.

        Raises:
            FileNotFoundError: If chunks_path does not exist. The chunker
                must be run before the encoder; run_local.py checks for
                this and prints a friendlier message upstream.
            json.JSONDecodeError: If any non-blank line in the JSONL file
                is not valid JSON, indicating a corrupt chunks file.
        """
        chunks = []
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)
        return chunks

    def encode_chunks(self, chunks: list[dict]) -> np.ndarray:
        """Encode chunk texts into embedding vectors.

        Args:
            chunks: List of chunk dicts, each must have a 'text' key.

        Returns:
            numpy array of shape (n_chunks, embedding_dim)
        """
        texts = [chunk["text"] for chunk in chunks]
        logger.info("Encoding %d chunks with %s...", len(texts), self.model_name)
        embeddings = self.model.encode(
            texts, show_progress_bar=True, normalize_embeddings=True
        )
        logger.info("Encoded embeddings shape: %s", embeddings.shape)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string.

        Args:
            query: The search query text.

        Returns:
            1-D numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            [query], normalize_embeddings=True
        )
        return embedding[0]
