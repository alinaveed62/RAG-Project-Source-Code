"""FAISS index building, saving, and loading."""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndexBuilder:
    """Builds and manages a FAISS index for similarity search."""

    def __init__(self, expected_dim: int | None = None):
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []
        self.expected_dim = expected_dim

    def build_index(
        self, embeddings: np.ndarray, metadata: list[dict]
    ) -> faiss.Index:
        """Build a FAISS IndexFlatIP from the given embeddings.

        Uses the inner product, which is equivalent to cosine
        similarity when embeddings are L2-normalised (the default
        behaviour of sentence-transformers).

        Args:
            embeddings: numpy array of shape (n, dim), L2-normalised.
            metadata: list of dicts aligned with the embedding rows,
                each carrying id, text, source, title and section.

        Returns:
            The built FAISS index.

        Raises:
            ValueError: If embeddings is not a 2-D array, or if
                expected_dim was set at construction time and the
                actual embedding dimension does not match it. This
                catches the case where the embedding model was
                changed without rebuilding the FAISS index.
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be a 2-D array of shape (n, dim); "
                f"got ndim={embeddings.ndim}, shape={embeddings.shape}."
            )
        dim = embeddings.shape[1]
        if self.expected_dim is not None and dim != self.expected_dim:
            raise ValueError(
                f"Embedding dim mismatch: got {dim}, expected "
                f"{self.expected_dim}. The index should be rebuilt after "
                "swapping the embedding model."
            )
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"metadata length {len(metadata)} does not match embeddings "
                f"rows {embeddings.shape[0]}; FAISS index and metadata must "
                "stay aligned or queries will return wrong chunks."
            )
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata
        logger.info(
            "Built FAISS index: %d vectors, %d dimensions", self.index.ntotal, dim
        )
        return self.index

    def save(self, index_dir: Path) -> None:
        """Save FAISS index and metadata to disk.

        Args:
            index_dir: Directory to write index.faiss and metadata.json.
        """
        if self.index is None:
            raise ValueError("No index built. Call build_index() first.")

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "index.faiss"))

        with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

        logger.info(
            "Saved FAISS index (%d vectors) to %s", self.index.ntotal, index_dir
        )

    def load(self, index_dir: Path) -> tuple[faiss.Index, list[dict]]:
        """Load a previously saved FAISS index and its metadata.

        Args:
            index_dir: Directory containing index.faiss and metadata.json.

        Returns:
            A tuple of (FAISS index, metadata list).

        Raises:
            FileNotFoundError: If either index.faiss or metadata.json is
                missing from index_dir. Callers typically check for the
                index beforehand and rebuild if absent (run_local.py).
            json.JSONDecodeError: If metadata.json is present but not
                valid JSON. A corrupt metadata file means the index
                cannot be trusted, so fail loudly rather than silently
                loading an empty list.
            ValueError: If index.faiss and metadata.json disagree on
                row count. save() writes them together, so a mismatch
                means the files have been modified independently.
                Loading a misaligned pair would silently return the
                wrong chunks at query time, or raise IndexError from
                inside FAISSRetriever; this check fails loudly instead.
        """
        # Load into local variables and validate before assigning to
        # self, so that a mismatched pair cannot leave the builder
        # half-loaded for a caller that catches the ValueError.
        index = faiss.read_index(str(index_dir / "index.faiss"))

        with open(index_dir / "metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)

        if index.ntotal != len(metadata):
            raise ValueError(
                f"FAISS index and metadata are out of sync in {index_dir}: "
                f"index.faiss has {index.ntotal} vectors, metadata.json "
                f"has {len(metadata)} entries. Rebuild the index."
            )

        self.index = index
        self.metadata = metadata

        logger.info(
            "Loaded FAISS index (%d vectors) from %s", self.index.ntotal, index_dir
        )
        return self.index, self.metadata
