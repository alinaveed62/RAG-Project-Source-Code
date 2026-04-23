"""Configuration for the RAG pipeline.

Validated with Pydantic v2 so misconfiguration fails at construction
time rather than at the first call into the pipeline. Every numeric
field carries an explicit range via Annotated[type, Field(...)] and
every string with a fixed set of valid values is typed as a Literal.
The inference_backend Literal accepts only "ollama" because no other
backend is shipped.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Shared default so ChunkEncoder's constructor signature stays in sync
# with RAGConfig.embedding_model and the two cannot drift apart.
DEFAULT_EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"


class RAGConfig(BaseModel):
    """Configuration for all RAG pipeline components.

    Default values match the empirical winners from the experiments
    in the evaluation chapter.
    """

    # validate_assignment=True extends Pydantic validation from
    # construction time to attribute-assignment time, so later writes
    # like config.top_k = -1 are rejected just as they would be in the
    # constructor. Without it, the Literal and Field constraints only
    # guard initial construction.
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True
    )

    # Paths
    chunks_path: Path = Field(
        default_factory=lambda: Path(
            "keats_scraper/data/chunks/chunks_for_embedding.jsonl"
        ),
        description="Path to the JSONL of chunks to embed.",
    )
    index_dir: Path = Field(
        default_factory=lambda: Path("rag_pipeline/data/index"),
        description="Directory where the FAISS index is saved.",
    )

    # Embedding
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        description=(
            "SentenceTransformer model name. Empirical winner on the "
            "embedding-model comparison; see "
            "evaluation/results/embedding_comparison.csv."
        ),
    )
    embedding_dim: Annotated[
        int,
        Field(
            ge=64,
            le=4096,
            description=(
                "Expected embedding dimension. Passed into "
                "FAISSIndexBuilder as a defensive sanity check to catch "
                "embedding-model / FAISS drift at index-build time "
                "(embedding model swapped without rebuilding the index)."
            ),
        ),
    ] = 384

    # Retrieval
    top_k: Annotated[int, Field(ge=1, le=50)] = 3
    similarity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.3
    enable_query_expansion: bool = True

    # Reranking defaults to False because dense-only retrieval beat
    # dense + rerank on MRR, P@1, P@5 and nDCG@5 in the reranking
    # ablation. It is still exposed as a per-request toggle because
    # the section-level ground truth is a coarse proxy and the
    # cross-encoder may help on finer-grained relevance.
    enable_reranking: bool = False
    rerank_fetch_k: Annotated[int, Field(ge=1, le=100)] = 20
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval mode: dense FAISS is the default; sparse (BM25) and
    # hybrid (RRF fusion) are selectable alternatives.
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "dense"
    rrf_k: Annotated[int, Field(ge=1, le=200)] = 60

    # Whether to inject [Source: cN] markers into the prompt so the
    # model can cite the chunk it drew from.
    enable_citation_injection: bool = True

    # Generation
    max_new_tokens: Annotated[int, Field(ge=1, le=4096)] = 512
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.3
    top_p: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9

    # Only the Ollama backend is shipped. The Literal blocks any
    # other value at construction time, so downstream code can assume
    # Ollama without re-checking.
    inference_backend: Literal["ollama"] = "ollama"
    ollama_model: str = "gemma2:2b"
    ollama_base_url: str = "http://localhost:11434"

    # Minimum mean retrieval score required before the pipeline will
    # answer at all; below this it refuses instead of guessing. The
    # gate is evaluated on the bi-encoder cosine score in [0, 1],
    # never on the unbounded cross-encoder rerank logits.
    low_confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.4
