"""Data models for the RAG pipeline."""

from pydantic import BaseModel, Field

from rag_pipeline.generation.citation_parser import Citation


class RetrievalResult(BaseModel):
    """A single retrieved chunk with its similarity score.

    score has three possible scales depending on the retrieval stage
    that produced the result, and callers must not compare scores
    across backends.

      Dense bi-encoder (pre-rerank): cosine similarity in [0, 1],
      because embeddings are L2-normalised by sentence-transformers.
      The low-confidence threshold is evaluated on this scale only.

      Cross-encoder (post-rerank): unbounded logit, typically in
      [-10, +15]. This is the reranker's predict() output.

      BM25 sparse: unbounded non-negative term-matching score.

    heading_path records the hierarchy of headings that led to this
    chunk in the original document (for example ["Teaching and
    Assessment", "Attendance"]). It defaults to an empty list so
    older chunk files without the field still deserialise.
    """

    chunk_id: str
    text: str
    score: float
    source: str
    title: str
    section: str
    heading_path: list[str] = Field(default_factory=list)


class RAGResponse(BaseModel):
    """Complete response from the RAG pipeline.

    rerank_time_ms defaults to 0.0 (no rerank was performed) and
    citations defaults to an empty list (citation parsing disabled or
    no markers found), so older serialised dicts that omit either
    field still deserialise cleanly through model_validate.
    """

    question: str
    answer: str
    sources: list[RetrievalResult]
    citations: list[Citation] = Field(default_factory=list)
    retrieval_time_ms: float
    rerank_time_ms: float = 0.0
    generation_time_ms: float
