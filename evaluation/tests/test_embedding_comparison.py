"""Tests for the embedding-model comparison experiment."""

from __future__ import annotations

import itertools
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from evaluation.experiments.embedding_comparison import (
    EMBEDDING_MODELS,
    run_embedding_comparison,
)


def _chunks() -> list[dict]:
    return [
        {"id": "c1", "text": "t1", "source": "s1", "title": "T1", "section": "Teaching & Assessment"},
        {"id": "c2", "text": "t2", "source": "s2", "title": "T2", "section": "Teaching & Assessment"},
        {"id": "c3", "text": "t3", "source": "s3", "title": "T3", "section": "Programmes"},
    ]


def _make_retrieval_result(chunk_id: str):
    from rag_pipeline.models import RetrievalResult

    return RetrievalResult(
        chunk_id=chunk_id,
        text=f"text of {chunk_id}",
        score=0.9,
        source="src",
        title="title",
        section="Teaching & Assessment",
    )


def _patch_encoder_builder_retriever(chunks, retrieved_ids):
    encoder = MagicMock()
    encoder.load_chunks.return_value = chunks
    encoder.encode_chunks.return_value = [[0.1, 0.2]] * len(chunks)
    encoder.encode_query.return_value = [0.1, 0.2]

    builder = MagicMock()
    builder.build_index.return_value = MagicMock()

    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_retrieval_result(cid) for cid in retrieved_ids]

    return encoder, builder, retriever


class TestRunEmbeddingComparison:
    def test_happy_path_one_model(self, rag_config, sample_qa_pairs):
        chunks = _chunks()
        encoder, builder, retriever = _patch_encoder_builder_retriever(chunks, ["c1", "c2"])
        with (
            patch("evaluation.experiments.embedding_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.embedding_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.embedding_comparison.FAISSRetriever", return_value=retriever),
        ):
            df = run_embedding_comparison(
                qa_pairs=sample_qa_pairs(2),
                config=rag_config,
                models=[{"name": "all-MiniLM-L6-v2", "dim": 384}],
            )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["model"] == "all-MiniLM-L6-v2"
        assert row["n_queries"] == 2
        assert "mrr" in df.columns
        assert "mrr_ci_low" in df.columns
        assert "mrr_ci_high" in df.columns
        assert encoder.load_chunks.called
        assert retriever.retrieve.call_count == 2

    def test_defaults_when_models_none(self, rag_config, sample_qa_pairs):
        chunks = _chunks()
        encoder, builder, retriever = _patch_encoder_builder_retriever(chunks, ["c1"])
        with (
            patch("evaluation.experiments.embedding_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.embedding_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.embedding_comparison.FAISSRetriever", return_value=retriever),
        ):
            df = run_embedding_comparison(qa_pairs=sample_qa_pairs(1), config=rag_config)
        assert len(df) == len(EMBEDDING_MODELS)

    def test_multiple_models_iterate(self, rag_config, sample_qa_pairs):
        chunks = _chunks()
        encoders = [
            _patch_encoder_builder_retriever(chunks, ["c1"])[0],
            _patch_encoder_builder_retriever(chunks, ["c2"])[0],
        ]
        with (
            patch("evaluation.experiments.embedding_comparison.ChunkEncoder", side_effect=encoders),
            patch("evaluation.experiments.embedding_comparison.FAISSIndexBuilder", return_value=MagicMock(build_index=MagicMock(return_value=MagicMock()))),
            patch(
                "evaluation.experiments.embedding_comparison.FAISSRetriever",
                return_value=MagicMock(retrieve=MagicMock(return_value=[_make_retrieval_result("c1")])),
            ),
        ):
            df = run_embedding_comparison(
                qa_pairs=sample_qa_pairs(1),
                config=rag_config,
                models=[
                    {"name": "m1", "dim": 384},
                    {"name": "m2", "dim": 384},
                ],
            )
        assert list(df["model"]) == ["m1", "m2"]

    def test_empty_question_skipped(self, rag_config):
        chunks = _chunks()
        encoder, builder, retriever = _patch_encoder_builder_retriever(chunks, ["c1"])
        qa = [
            {"id": "q1", "question": "   ", "relevant_sections": ["Teaching & Assessment"]},
            {"id": "q2", "question": "Real?", "relevant_sections": ["Teaching & Assessment"]},
        ]
        with (
            patch("evaluation.experiments.embedding_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.embedding_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.embedding_comparison.FAISSRetriever", return_value=retriever),
        ):
            df = run_embedding_comparison(
                qa_pairs=qa,
                config=rag_config,
                models=[{"name": "m", "dim": 384}],
            )
        assert retriever.retrieve.call_count == 1
        assert df.iloc[0]["n_queries"] == 1

    def test_all_empty_questions_no_ci_columns(self, rag_config):
        chunks = _chunks()
        encoder, builder, retriever = _patch_encoder_builder_retriever(chunks, ["c1"])
        qa = [
            {"id": "q1", "question": " ", "relevant_sections": []},
            {"id": "q2", "question": "", "relevant_sections": []},
        ]
        with (
            patch("evaluation.experiments.embedding_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.embedding_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.embedding_comparison.FAISSRetriever", return_value=retriever),
        ):
            df = run_embedding_comparison(
                qa_pairs=qa,
                config=rag_config,
                models=[{"name": "m", "dim": 384}],
            )
        assert df.iloc[0]["n_queries"] == 0
        assert "mrr_ci_low" not in df.columns

    def test_encoding_time_rounded_to_two_decimals(self, rag_config, sample_qa_pairs):
        chunks = _chunks()
        encoder, builder, retriever = _patch_encoder_builder_retriever(chunks, ["c1"])
        time_seq = itertools.count(0.0, 1.23456)
        with (
            patch("evaluation.experiments.embedding_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.embedding_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.embedding_comparison.FAISSRetriever", return_value=retriever),
            patch("evaluation.experiments.embedding_comparison.time.perf_counter", side_effect=lambda: next(time_seq)),
        ):
            df = run_embedding_comparison(
                qa_pairs=sample_qa_pairs(1),
                config=rag_config,
                models=[{"name": "m", "dim": 384}],
            )
        assert df.iloc[0]["encoding_time_s"] == pytest.approx(1.23)
