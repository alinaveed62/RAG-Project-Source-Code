"""Tests for the reranking-comparison experiment."""

from __future__ import annotations

import itertools
from unittest.mock import MagicMock, patch

from evaluation.experiments.reranking_comparison import (
    _retrieve_bm25_rerank,
    _retrieve_dense_only,
    _retrieve_dense_rerank,
    run_reranking_comparison,
)


def _chunks() -> list[dict]:
    return [
        {"id": "c1", "text": "t1", "source": "s", "title": "T", "section": "Teaching & Assessment"},
        {"id": "c2", "text": "t2", "source": "s", "title": "T", "section": "Teaching & Assessment"},
    ]


def _result(chunk_id: str):
    from rag_pipeline.models import RetrievalResult

    return RetrievalResult(
        chunk_id=chunk_id, text=chunk_id, score=0.7, source="s",
        title="T", section="Teaching & Assessment",
    )


def _patched(chunks, retrieved_ids=("c1", "c2")):
    encoder = MagicMock()
    encoder.load_chunks.return_value = chunks
    encoder.encode_chunks.return_value = [[0.1, 0.2]] * len(chunks)
    encoder.encode_query.return_value = [0.1, 0.2]

    builder = MagicMock()
    builder.build_index.return_value = MagicMock()

    dense = MagicMock()
    dense.retrieve.return_value = [_result(cid) for cid in retrieved_ids]
    bm25 = MagicMock()
    bm25.retrieve.return_value = [_result(cid) for cid in retrieved_ids]
    reranker = MagicMock()
    reranker.rerank.return_value = [_result(cid) for cid in retrieved_ids]
    return encoder, builder, dense, bm25, reranker


class TestPrivateHelpers:
    def test_dense_only_returns_dense_results(self):
        dense = MagicMock()
        dense.retrieve.return_value = [_result("c1")]
        out = _retrieve_dense_only(dense, [0.1, 0.2], top_k=3)
        assert [r.chunk_id for r in out] == ["c1"]
        dense.retrieve.assert_called_once_with([0.1, 0.2], top_k=3)

    def test_dense_rerank_returns_reranked_and_ms(self):
        dense = MagicMock()
        dense.retrieve.return_value = [_result("c1"), _result("c2")]
        reranker = MagicMock()
        reranker.rerank.return_value = [_result("c2")]
        seq = itertools.count(0.0, 0.001)
        with patch(
            "evaluation.experiments.reranking_comparison.time.perf_counter",
            side_effect=lambda: next(seq),
        ):
            out, rerank_ms = _retrieve_dense_rerank(
                dense, reranker, "q", [0.1], fetch_k=10, top_k=3
            )
        assert [r.chunk_id for r in out] == ["c2"]
        assert rerank_ms == 1.0  # 0.001 s × 1000

    def test_bm25_rerank_returns_reranked_and_ms(self):
        bm25 = MagicMock()
        bm25.retrieve.return_value = [_result("c1"), _result("c2")]
        reranker = MagicMock()
        reranker.rerank.return_value = [_result("c1")]
        seq = itertools.count(0.0, 0.002)
        with patch(
            "evaluation.experiments.reranking_comparison.time.perf_counter",
            side_effect=lambda: next(seq),
        ):
            out, rerank_ms = _retrieve_bm25_rerank(
                bm25, reranker, "q", fetch_k=20, top_k=5
            )
        assert [r.chunk_id for r in out] == ["c1"]
        assert rerank_ms == 2.0


class TestRunRerankingComparison:
    def test_three_strategies_produce_three_rows(self, rag_config, sample_qa_pairs):
        chunks = _chunks()
        encoder, builder, dense, bm25, reranker = _patched(chunks)
        with (
            patch("evaluation.experiments.reranking_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.reranking_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.reranking_comparison.FAISSRetriever", return_value=dense),
            patch("evaluation.experiments.reranking_comparison.BM25Retriever", return_value=bm25),
            patch("evaluation.experiments.reranking_comparison.CrossEncoderReranker", return_value=reranker),
        ):
            df = run_reranking_comparison(qa_pairs=sample_qa_pairs(2), config=rag_config)
        assert set(df["strategy"]) == {"dense_only", "dense_rerank", "bm25_rerank"}
        assert all(df["n_queries"] == 2)
        # avg_rerank_ms must exist and equal 0 for dense_only.
        dense_only_row = df[df["strategy"] == "dense_only"].iloc[0]
        assert dense_only_row["avg_rerank_ms"] == 0.0

    def test_empty_questions_filtered(self, rag_config):
        chunks = _chunks()
        encoder, builder, dense, bm25, reranker = _patched(chunks, retrieved_ids=["c1"])
        qa = [
            {"id": "q1", "question": "   ", "relevant_sections": ["Teaching & Assessment"]},
            {"id": "q2", "question": "Real?", "relevant_sections": ["Teaching & Assessment"]},
        ]
        with (
            patch("evaluation.experiments.reranking_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.reranking_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.reranking_comparison.FAISSRetriever", return_value=dense),
            patch("evaluation.experiments.reranking_comparison.BM25Retriever", return_value=bm25),
            patch("evaluation.experiments.reranking_comparison.CrossEncoderReranker", return_value=reranker),
        ):
            df = run_reranking_comparison(qa_pairs=qa, config=rag_config)
        assert all(df["n_queries"] == 1)

    def test_all_empty_questions_drop_strategy_rows(self, rag_config):
        chunks = _chunks()
        encoder, builder, dense, bm25, reranker = _patched(chunks, retrieved_ids=["c1"])
        qa = [
            {"id": "q1", "question": " ", "relevant_sections": []},
            {"id": "q2", "question": "", "relevant_sections": []},
        ]
        with (
            patch("evaluation.experiments.reranking_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.reranking_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.reranking_comparison.FAISSRetriever", return_value=dense),
            patch("evaluation.experiments.reranking_comparison.BM25Retriever", return_value=bm25),
            patch("evaluation.experiments.reranking_comparison.CrossEncoderReranker", return_value=reranker),
        ):
            df = run_reranking_comparison(qa_pairs=qa, config=rag_config)
        # With no queries to evaluate, every strategy hits if not per_query_metrics: continue.
        assert df.empty or all(df["n_queries"] == 0)

    def test_timing_columns_rounded_to_two_decimals(self, rag_config, sample_qa_pairs):
        chunks = _chunks()
        encoder, builder, dense, bm25, reranker = _patched(chunks)
        seq = itertools.count(0.0, 0.00123)
        with (
            patch("evaluation.experiments.reranking_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.reranking_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.reranking_comparison.FAISSRetriever", return_value=dense),
            patch("evaluation.experiments.reranking_comparison.BM25Retriever", return_value=bm25),
            patch("evaluation.experiments.reranking_comparison.CrossEncoderReranker", return_value=reranker),
            patch("evaluation.experiments.reranking_comparison.time.perf_counter", side_effect=lambda: next(seq)),
        ):
            df = run_reranking_comparison(qa_pairs=sample_qa_pairs(1), config=rag_config)
        for col in ("avg_retrieval_ms", "avg_rerank_ms"):
            for val in df[col]:
                assert val == round(val, 2)

    def test_ci_added_per_metric(self, rag_config, sample_qa_pairs):
        chunks = _chunks()
        encoder, builder, dense, bm25, reranker = _patched(chunks)
        with (
            patch("evaluation.experiments.reranking_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.reranking_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.reranking_comparison.FAISSRetriever", return_value=dense),
            patch("evaluation.experiments.reranking_comparison.BM25Retriever", return_value=bm25),
            patch("evaluation.experiments.reranking_comparison.CrossEncoderReranker", return_value=reranker),
        ):
            df = run_reranking_comparison(qa_pairs=sample_qa_pairs(2), config=rag_config)
        assert "mrr_ci_low" in df.columns
        assert "mrr_ci_high" in df.columns
