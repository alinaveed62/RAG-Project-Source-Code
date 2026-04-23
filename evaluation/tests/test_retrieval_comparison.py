"""Tests for the dense/sparse/hybrid retrieval-comparison experiment."""

from __future__ import annotations

import itertools
from unittest.mock import MagicMock, patch

from evaluation.experiments.retrieval_comparison import run_retrieval_comparison


def _chunks() -> list[dict]:
    return [
        {"id": "c1", "text": "t1", "source": "s1", "title": "T1", "section": "Teaching & Assessment"},
        {"id": "c2", "text": "t2", "source": "s2", "title": "T2", "section": "Teaching & Assessment"},
        {"id": "c3", "text": "t3", "source": "s3", "title": "T3", "section": "Programmes"},
    ]


def _mock_result(chunk_id: str):
    from rag_pipeline.models import RetrievalResult

    return RetrievalResult(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        score=0.8,
        source="s",
        title="T",
        section="Teaching & Assessment",
    )


def _patched(chunks, retrieved_ids=("c1", "c2")):
    encoder = MagicMock()
    encoder.load_chunks.return_value = chunks
    encoder.encode_chunks.return_value = [[0.1, 0.2]] * len(chunks)
    encoder.encode_query.return_value = [0.1, 0.2]

    builder = MagicMock()
    builder.build_index.return_value = MagicMock()

    dense = MagicMock()
    dense.retrieve.return_value = [_mock_result(cid) for cid in retrieved_ids]
    sparse = MagicMock()
    sparse.retrieve.return_value = [_mock_result(cid) for cid in retrieved_ids]
    hybrid = MagicMock()
    hybrid.retrieve.return_value = [_mock_result(cid) for cid in retrieved_ids]
    return encoder, builder, dense, sparse, hybrid


class TestRunRetrievalComparison:
    def test_three_strategies_three_rows(self, rag_config, sample_qa_pairs):
        chunks = _chunks()
        encoder, builder, dense, sparse, hybrid = _patched(chunks)
        with (
            patch("evaluation.experiments.retrieval_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.retrieval_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.retrieval_comparison.FAISSRetriever", return_value=dense),
            patch("evaluation.experiments.retrieval_comparison.BM25Retriever", return_value=sparse),
            patch("evaluation.experiments.retrieval_comparison.HybridRetriever", return_value=hybrid),
        ):
            df = run_retrieval_comparison(qa_pairs=sample_qa_pairs(2), config=rag_config)
        assert set(df["strategy"]) == {"dense_faiss", "sparse_bm25", "hybrid_rrf"}
        assert all(df["n_queries"] == 2)
        assert "mrr" in df.columns
        assert "mrr_ci_low" in df.columns
        # Each of the three strategies' retrieve() was called for the 2 QA pairs.
        assert dense.retrieve.call_count == 2
        assert sparse.retrieve.call_count == 2
        assert hybrid.retrieve.call_count == 2

    def test_empty_questions_skipped(self, rag_config):
        chunks = _chunks()
        encoder, builder, dense, sparse, hybrid = _patched(chunks, retrieved_ids=["c1"])
        qa = [
            {"id": "q1", "question": "  ", "relevant_sections": ["Teaching & Assessment"]},
            {"id": "q2", "question": "Real?", "relevant_sections": ["Teaching & Assessment"]},
        ]
        with (
            patch("evaluation.experiments.retrieval_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.retrieval_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.retrieval_comparison.FAISSRetriever", return_value=dense),
            patch("evaluation.experiments.retrieval_comparison.BM25Retriever", return_value=sparse),
            patch("evaluation.experiments.retrieval_comparison.HybridRetriever", return_value=hybrid),
        ):
            df = run_retrieval_comparison(qa_pairs=qa, config=rag_config)
        assert dense.retrieve.call_count == 1

    def test_all_empty_questions_yields_zero_n_queries_no_ci(self, rag_config):
        chunks = _chunks()
        encoder, builder, dense, sparse, hybrid = _patched(chunks, retrieved_ids=["c1"])
        qa = [
            {"id": "q1", "question": " ", "relevant_sections": []},
            {"id": "q2", "question": "", "relevant_sections": []},
        ]
        with (
            patch("evaluation.experiments.retrieval_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.retrieval_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.retrieval_comparison.FAISSRetriever", return_value=dense),
            patch("evaluation.experiments.retrieval_comparison.BM25Retriever", return_value=sparse),
            patch("evaluation.experiments.retrieval_comparison.HybridRetriever", return_value=hybrid),
        ):
            df = run_retrieval_comparison(qa_pairs=qa, config=rag_config)
        assert all(df["n_queries"] == 0)
        assert "mrr_ci_low" not in df.columns

    def test_latency_per_strategy_rounded(self, rag_config, sample_qa_pairs):
        chunks = _chunks()
        encoder, builder, dense, sparse, hybrid = _patched(chunks)
        time_seq = itertools.count(0.0, 0.00123)
        with (
            patch("evaluation.experiments.retrieval_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.retrieval_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.retrieval_comparison.FAISSRetriever", return_value=dense),
            patch("evaluation.experiments.retrieval_comparison.BM25Retriever", return_value=sparse),
            patch("evaluation.experiments.retrieval_comparison.HybridRetriever", return_value=hybrid),
            patch("evaluation.experiments.retrieval_comparison.time.perf_counter", side_effect=lambda: next(time_seq)),
        ):
            df = run_retrieval_comparison(qa_pairs=sample_qa_pairs(1), config=rag_config)
        assert "avg_retrieval_ms" in df.columns
        for val in df["avg_retrieval_ms"]:
            assert val == round(val, 2)
