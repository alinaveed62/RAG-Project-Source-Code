"""RAG pipeline for the KCL Informatics Student Handbook chatbot.

The re-exports below capture the surface that external packages
(flask_app, run_local.py, evaluation) actually consume. Deep module
imports from rag_pipeline.retrieval, rag_pipeline.generation and
rag_pipeline.embeddings continue to work; the re-exports are just
a convenience.
"""

from rag_pipeline.config import DEFAULT_EMBEDDING_MODEL, RAGConfig
from rag_pipeline.generation.citation_parser import (
    Citation,
    parse_citations,
    strip_citations,
)
from rag_pipeline.models import RAGResponse, RetrievalResult
from rag_pipeline.pipeline import (
    GENERATION_ERROR_ANSWER,
    LOW_CONFIDENCE_ANSWER,
    RAGPipeline,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "GENERATION_ERROR_ANSWER",
    "LOW_CONFIDENCE_ANSWER",
    "Citation",
    "RAGConfig",
    "RAGPipeline",
    "RAGResponse",
    "RetrievalResult",
    "parse_citations",
    "strip_citations",
]
