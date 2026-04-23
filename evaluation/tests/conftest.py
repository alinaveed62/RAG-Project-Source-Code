"""Shared fixtures and autouse setup for evaluation tests."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import matplotlib
import pytest

from rag_pipeline.config import RAGConfig
from rag_pipeline.models import RAGResponse, RetrievalResult


@pytest.fixture(autouse=True)
def _matplotlib_agg_backend() -> None:
    """Force the headless Agg backend so figure tests never hit a display."""
    matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _reset_section_cache() -> Generator[None, None, None]:
    """Reset the cached section → chunk-id index between tests."""
    from evaluation.metrics import retrieval_metrics

    retrieval_metrics._section_index_cache.clear()
    yield
    retrieval_metrics._section_index_cache.clear()


@pytest.fixture(autouse=True)
def _reset_nli_cache() -> Generator[None, None, None]:
    """Reset the cached NLI cross-encoder between tests so monkeypatched models don't leak."""
    from evaluation.metrics import sgf

    sgf._cached_default_model = None
    yield
    sgf._cached_default_model = None


@pytest.fixture
def make_retrieval_result():
    """Factory for RetrievalResult with sensible defaults."""

    def _make(
        chunk_id: str = "doc_a_chunk_0",
        text: str = "Example chunk text.",
        score: float = 0.8,
        source: str = "https://keats.kcl.ac.uk/mod/page/view.php?id=1",
        title: str = "Example Page",
        section: str = "Teaching & Assessment",
        heading_path: list[str] | None = None,
    ) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id,
            text=text,
            score=score,
            source=source,
            title=title,
            section=section,
            heading_path=heading_path or [],
        )

    return _make


@pytest.fixture
def make_rag_response(make_retrieval_result):
    """Factory for RAGResponse with sensible defaults."""

    def _make(
        question: str = "Do I have to attend all my lectures?",
        answer: str = "Yes, attendance is required.",
        sources: list[RetrievalResult] | None = None,
        retrieval_time_ms: float = 50.0,
        rerank_time_ms: float = 0.0,
        generation_time_ms: float = 200.0,
    ) -> RAGResponse:
        if sources is None:
            sources = [make_retrieval_result()]
        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            retrieval_time_ms=retrieval_time_ms,
            rerank_time_ms=rerank_time_ms,
            generation_time_ms=generation_time_ms,
        )

    return _make


@pytest.fixture
def sample_qa_pairs():
    """Factory for QA-pair lists."""

    def _make(n: int = 3, category: str = "attendance") -> list[dict[str, Any]]:
        return [
            {
                "id": f"q_{i:02d}",
                "question": f"Test question {i}?",
                "category": category,
                "expected_answer": f"Expected answer {i}.",
                "relevant_sections": ["Teaching & Assessment"],
                "difficulty": "easy",
            }
            for i in range(n)
        ]

    return _make


@pytest.fixture
def chunks_jsonl(tmp_path):
    """Write chunk dicts to a tmp JSONL and return the path."""

    def _write(chunks: list[dict[str, Any]]) -> Path:
        path = tmp_path / "chunks.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
        return path

    return _write


@pytest.fixture
def sample_chunk_dicts():
    """Plain dict chunks with id, text, section at minimum."""
    return [
        {
            "id": "doc_a_chunk_0",
            "text": "Students must attend all scheduled lectures.",
            "source": "url_a",
            "title": "Attendance",
            "section": "Teaching & Assessment",
        },
        {
            "id": "doc_a_chunk_1",
            "text": "Extenuating circumstances apply within seven days.",
            "source": "url_a",
            "title": "EC",
            "section": "Teaching & Assessment",
        },
        {
            "id": "doc_b_chunk_0",
            "text": "Student reps are elected each year.",
            "source": "url_b",
            "title": "Student Voice",
            "section": "Student Voice",
        },
    ]


@pytest.fixture
def rag_config(tmp_path, chunks_jsonl, sample_chunk_dicts) -> RAGConfig:
    """Minimal RAGConfig pointing at a tmp chunks file and index dir."""
    chunks_path = chunks_jsonl(sample_chunk_dicts)
    return RAGConfig(
        chunks_path=chunks_path,
        index_dir=tmp_path / "index",
    )


@pytest.fixture
def mock_pipeline(rag_config, make_rag_response):
    """MagicMock of RAGPipeline with sane defaults for .config, .answer, .setup."""
    pipeline = MagicMock()
    pipeline.config = rag_config
    pipeline.answer.return_value = make_rag_response()
    pipeline.setup.return_value = None
    pipeline.build_index.return_value = None
    pipeline.reload_generator.return_value = None
    return pipeline
