"""Tests for the FAISS retriever."""

import pytest

from rag_pipeline.config import RAGConfig
from rag_pipeline.embeddings.index_builder import FAISSIndexBuilder
from rag_pipeline.retrieval.retriever import FAISSRetriever


@pytest.fixture
def retriever_setup(sample_embeddings, sample_chunks, tmp_path):
    """Build an index and return a configured retriever."""
    builder = FAISSIndexBuilder()
    index = builder.build_index(sample_embeddings, sample_chunks)
    config = RAGConfig(
        chunks_path=tmp_path / "chunks.jsonl",
        index_dir=tmp_path / "index",
        top_k=3,
        similarity_threshold=0.0,  # Accept all for testing
    )
    retriever = FAISSRetriever(index, sample_chunks, config)
    return retriever, sample_embeddings


class TestFAISSRetrieverRetrieve:
    """Tests for retrieval functionality."""

    def test_retrieve_returns_results(self, retriever_setup):
        retriever, embeddings = retriever_setup
        results = retriever.retrieve(embeddings[0])
        assert len(results) > 0
        assert len(results) <= 3

    def test_retrieve_returns_retrieval_results(self, retriever_setup):
        retriever, embeddings = retriever_setup
        results = retriever.retrieve(embeddings[0])
        result = results[0]
        assert hasattr(result, "chunk_id")
        assert hasattr(result, "text")
        assert hasattr(result, "score")
        assert hasattr(result, "source")
        assert hasattr(result, "title")
        assert hasattr(result, "section")

    def test_retrieve_sorted_by_score(self, retriever_setup):
        retriever, embeddings = retriever_setup
        results = retriever.retrieve(embeddings[0])
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_custom_top_k(self, retriever_setup):
        retriever, embeddings = retriever_setup
        results = retriever.retrieve(embeddings[0], top_k=2)
        assert len(results) <= 2

    def test_retrieve_top_match_is_self(self, retriever_setup):
        retriever, embeddings = retriever_setup
        results = retriever.retrieve(embeddings[0])
        assert results[0].chunk_id == "doc_abc123_chunk_0"


class TestFAISSRetrieverFiltering:
    """Tests for section filtering and score thresholds."""

    def test_section_filter(self, retriever_setup):
        retriever, embeddings = retriever_setup
        results = retriever.retrieve(
            embeddings[0], top_k=5, section_filter="Programmes"
        )
        for r in results:
            assert r.section == "Programmes"

    def test_section_filter_no_matches(self, retriever_setup):
        retriever, embeddings = retriever_setup
        results = retriever.retrieve(
            embeddings[0], top_k=5, section_filter="NonexistentSection"
        )
        assert len(results) == 0

    def test_score_threshold_filters_low_scores(self, sample_embeddings, sample_chunks, tmp_path):
        builder = FAISSIndexBuilder()
        index = builder.build_index(sample_embeddings, sample_chunks)
        config = RAGConfig(
            chunks_path=tmp_path / "chunks.jsonl",
            index_dir=tmp_path / "index",
            top_k=5,
            similarity_threshold=0.99,  # Very high threshold
        )
        retriever = FAISSRetriever(index, sample_chunks, config)
        results = retriever.retrieve(sample_embeddings[0])
        # Only the exact match (score ~1.0) should pass
        assert len(results) <= 1


class TestFAISSRetrieverEdgeCases:
    """FAISS returns -1 indices when the index has fewer vectors than the
    caller asked for. retriever.retrieve() must skip them instead of
    dereferencing metadata[-1]. Covers retriever.py:49-50."""

    def test_retrieve_handles_faiss_minus_one_index(self, tmp_path):
        """Index with 2 vectors, request top_k=5 → FAISS fills the tail
        with -1 indices. retriever must skip them and return 2 results."""
        import numpy as np

        # Two near-identical vectors so both clear the default 0.3 threshold;
        # the test shape (2 vectors, top_k=5) is what forces FAISS to return
        # -1 indices for the last three slots.
        base = np.ones(384, dtype=np.float32)
        emb = np.stack([base, base * 0.95], axis=0)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        metadata = [
            {"id": "c0", "text": "alpha", "source": "", "title": "A", "section": "S"},
            {"id": "c1", "text": "beta", "source": "", "title": "B", "section": "S"},
        ]

        builder = FAISSIndexBuilder()
        index = builder.build_index(emb, metadata)
        config = RAGConfig(
            chunks_path=tmp_path / "chunks.jsonl",
            index_dir=tmp_path / "index",
            top_k=5,
            similarity_threshold=0.0,
        )
        retriever = FAISSRetriever(index, metadata, config)
        results = retriever.retrieve(emb[0])
        assert len(results) == 2
        assert {r.chunk_id for r in results} == {"c0", "c1"}
