"""Tests for HybridRetriever."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from rag_pipeline.config import RAGConfig
from rag_pipeline.models import RetrievalResult
from rag_pipeline.retrieval.hybrid_retriever import HybridRetriever


def _result(chunk_id: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        score=score,
        source=f"url_{chunk_id}",
        title=f"title_{chunk_id}",
        section=f"section_{chunk_id}",
    )


def _mock_retrievers(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    top_k: int = 3,
) -> tuple[MagicMock, MagicMock]:
    cfg = RAGConfig(top_k=top_k)
    dense = MagicMock()
    dense.config = cfg
    dense.retrieve.return_value = dense_results
    sparse = MagicMock()
    sparse.retrieve.return_value = sparse_results
    return dense, sparse


class TestConstruction:
    def test_requires_k_rrf_positive(self):
        dense = MagicMock()
        dense.config = RAGConfig()
        sparse = MagicMock()
        with pytest.raises(ValueError, match="k_rrf"):
            HybridRetriever(dense, sparse, k_rrf=0)


class TestRrfMath:
    def test_rrf_math_matches_hand_calculation(self):
        """Chunk at dense rank 0, sparse rank 2 gets 1/(60+1) + 1/(60+3)."""
        dense = [_result("a"), _result("b")]
        sparse = [_result("x"), _result("y"), _result("a")]
        dense_mock, sparse_mock = _mock_retrievers(dense, sparse, top_k=5)

        hybrid = HybridRetriever(dense_mock, sparse_mock, k_rrf=60)
        fused = hybrid.retrieve(
            query="q", query_embedding=np.zeros(4, dtype=np.float32), top_k=5
        )
        # Find chunk "a": rank 0 in dense, rank 2 in sparse.
        a_entry = next(r for r in fused if r.chunk_id == "a")
        expected_score = 1.0 / (60 + 0 + 1) + 1.0 / (60 + 2 + 1)
        assert a_entry.score == pytest.approx(expected_score, rel=1e-6)

    def test_handles_empty_dense_results(self):
        sparse = [_result("s1"), _result("s2")]
        dense_mock, sparse_mock = _mock_retrievers([], sparse, top_k=5)
        hybrid = HybridRetriever(dense_mock, sparse_mock, k_rrf=60)
        fused = hybrid.retrieve(
            query="q", query_embedding=np.zeros(4, dtype=np.float32), top_k=5
        )
        assert [r.chunk_id for r in fused] == ["s1", "s2"]
        # Sparse-only score for rank 0: 1/61.
        assert fused[0].score == pytest.approx(1 / 61)

    def test_handles_empty_sparse_results(self):
        dense = [_result("d1"), _result("d2")]
        dense_mock, sparse_mock = _mock_retrievers(dense, [], top_k=5)
        hybrid = HybridRetriever(dense_mock, sparse_mock, k_rrf=60)
        fused = hybrid.retrieve(
            query="q", query_embedding=np.zeros(4, dtype=np.float32), top_k=5
        )
        assert [r.chunk_id for r in fused] == ["d1", "d2"]

    def test_top_k_truncation(self):
        dense = [_result(f"d{i}") for i in range(5)]
        sparse = [_result(f"s{i}") for i in range(5)]
        dense_mock, sparse_mock = _mock_retrievers(dense, sparse, top_k=10)
        hybrid = HybridRetriever(dense_mock, sparse_mock)
        fused = hybrid.retrieve(
            query="q", query_embedding=np.zeros(4, dtype=np.float32), top_k=3
        )
        assert len(fused) == 3

    def test_dedup_when_same_chunk_in_both_lists(self):
        """Chunk that appears in both runs is represented once, not twice."""
        shared = _result("shared")
        dense = [shared, _result("d1")]
        sparse = [_result("s1"), _result("s2"), shared]
        dense_mock, sparse_mock = _mock_retrievers(dense, sparse, top_k=10)
        hybrid = HybridRetriever(dense_mock, sparse_mock, k_rrf=60)
        fused = hybrid.retrieve(
            query="q", query_embedding=np.zeros(4, dtype=np.float32), top_k=10
        )
        chunk_ids = [r.chunk_id for r in fused]
        assert chunk_ids.count("shared") == 1
        # Shared chunk's score should be the sum of both rank contributions.
        shared_entry = next(r for r in fused if r.chunk_id == "shared")
        expected = 1.0 / (60 + 0 + 1) + 1.0 / (60 + 2 + 1)
        assert shared_entry.score == pytest.approx(expected)


class TestSectionFilter:
    def test_section_filter_propagates_to_both_retrievers(self):
        dense = [_result("a")]
        sparse = [_result("b")]
        dense_mock, sparse_mock = _mock_retrievers(dense, sparse, top_k=3)
        hybrid = HybridRetriever(dense_mock, sparse_mock)
        hybrid.retrieve(
            query="q",
            query_embedding=np.zeros(4, dtype=np.float32),
            section_filter="Main",
        )
        assert dense_mock.retrieve.call_args[1]["section_filter"] == "Main"
        assert sparse_mock.retrieve.call_args[1]["section_filter"] == "Main"


class TestDefaultTopK:
    def test_uses_config_top_k_when_not_overridden(self):
        dense = [_result(f"d{i}") for i in range(10)]
        sparse: list[RetrievalResult] = []
        dense_mock, sparse_mock = _mock_retrievers(dense, sparse, top_k=3)
        hybrid = HybridRetriever(dense_mock, sparse_mock)
        fused = hybrid.retrieve(
            query="q", query_embedding=np.zeros(4, dtype=np.float32)
        )
        assert len(fused) == 3


class TestRepresentativeInvariant:
    """_fuse uses setdefault so when the same chunk_id appears in
    both runs the dense representative wins. In production both retrievers
    are constructed from the same metadata list in
    RAGPipeline.setup so their non-score fields are identical; the
    choice is neutral but the invariant must be locked so a future refactor
    doesn't silently change it."""

    def test_dense_representative_wins_when_metadata_diverges(self):
        dense_copy = RetrievalResult(
            chunk_id="shared",
            text="dense text",
            score=0.9,
            source="https://dense.example",
            title="dense title",
            section="dense section",
            heading_path=["Dense", "Path"],
        )
        sparse_copy = RetrievalResult(
            chunk_id="shared",
            text="sparse text",
            score=4.2,
            source="https://sparse.example",
            title="sparse title",
            section="sparse section",
            heading_path=["Sparse", "Path"],
        )
        dense_mock, sparse_mock = _mock_retrievers(
            [dense_copy], [sparse_copy], top_k=5
        )
        hybrid = HybridRetriever(dense_mock, sparse_mock, k_rrf=60)

        fused = hybrid.retrieve(
            query="q", query_embedding=np.zeros(4, dtype=np.float32), top_k=5
        )

        assert len(fused) == 1
        r = fused[0]
        # Non-score fields come from the dense copy.
        assert r.text == "dense text"
        assert r.source == "https://dense.example"
        assert r.title == "dense title"
        assert r.section == "dense section"
        assert r.heading_path == ["Dense", "Path"]
        # The score is the fused RRF score, not either input's raw score.
        assert r.score == pytest.approx(1 / 61 + 1 / 61)
