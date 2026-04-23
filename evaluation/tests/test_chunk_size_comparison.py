"""Tests for the chunk-size-comparison experiment."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from evaluation.experiments.chunk_size_comparison import (
    CHUNK_SIZES,
    run_chunk_size_comparison,
)


def _make_documents(tmp_path) -> Path:
    path: Path = tmp_path / "documents.jsonl"
    documents = [
        {
            "id": "d1",
            "content": "Students must attend lectures.",
            "raw_html": "<p>Students must attend lectures.</p>",
            "metadata": {
                "source_url": "u1",
                "document_title": "T1",
                "section": "Teaching & Assessment",
            },
        },
        {
            "id": "d2",
            "content": "Programmes are offered.",
            "raw_html": "<p>Programmes are offered.</p>",
            "metadata": {
                "source_url": "u2",
                "document_title": "T2",
                "section": "Programmes",
            },
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")
    return path


def _fake_chunk(chunk_id: str, section: str):
    return SimpleNamespace(
        id=chunk_id,
        text=f"chunk text {chunk_id}",
        metadata=SimpleNamespace(
            source_url="u",
            document_title="T",
            section=section,
        ),
    )


def _make_retrieval_result(chunk_id: str):
    from rag_pipeline.models import RetrievalResult

    return RetrievalResult(
        chunk_id=chunk_id, text=chunk_id, score=0.9, source="s",
        title="T", section="Teaching & Assessment",
    )


def _patched_deps(chunks_per_doc: int = 2):
    chunker = MagicMock()
    chunker.chunk_document.return_value = [
        _fake_chunk(f"c{i}", "Teaching & Assessment") for i in range(chunks_per_doc)
    ]

    encoder = MagicMock()
    encoder.encode_chunks.return_value = [[0.1, 0.2]] * (chunks_per_doc * 2)
    encoder.encode_query.return_value = [0.1, 0.2]

    builder = MagicMock()
    builder.build_index.return_value = MagicMock()

    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_retrieval_result("c0"), _make_retrieval_result("c1")]
    return chunker, encoder, builder, retriever


class TestRunChunkSizeComparison:
    def test_default_sizes_used(self, rag_config, tmp_path, sample_qa_pairs):
        chunker, encoder, builder, retriever = _patched_deps()
        docs_path = _make_documents(tmp_path)
        with (
            patch("evaluation.experiments.chunk_size_comparison.Chunker", return_value=chunker),
            patch("evaluation.experiments.chunk_size_comparison.Document", side_effect=lambda **kw: SimpleNamespace(**kw)),
            patch("evaluation.experiments.chunk_size_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.chunk_size_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.chunk_size_comparison.FAISSRetriever", return_value=retriever),
        ):
            df = run_chunk_size_comparison(
                qa_pairs=sample_qa_pairs(1),
                documents_path=docs_path,
                config=rag_config,
            )
        assert len(df) == len(CHUNK_SIZES)
        assert list(df["chunk_size"]) == CHUNK_SIZES

    def test_custom_sizes_and_overlap_10_percent(self, rag_config, tmp_path, sample_qa_pairs):
        chunker, encoder, builder, retriever = _patched_deps(chunks_per_doc=3)
        docs_path = _make_documents(tmp_path)
        with (
            patch("evaluation.experiments.chunk_size_comparison.Chunker", return_value=chunker),
            patch("evaluation.experiments.chunk_size_comparison.Document", side_effect=lambda **kw: SimpleNamespace(**kw)),
            patch("evaluation.experiments.chunk_size_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.chunk_size_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.chunk_size_comparison.FAISSRetriever", return_value=retriever),
        ):
            df = run_chunk_size_comparison(
                qa_pairs=sample_qa_pairs(1),
                documents_path=docs_path,
                config=rag_config,
                chunk_sizes=[512],
            )
        assert len(df) == 1
        row = df.iloc[0]
        assert row["chunk_size"] == 512
        assert row["chunk_overlap"] == 51  # 10 % of 512 truncated
        assert row["num_chunks"] == 6  # 3 chunks per doc × 2 docs

    def test_empty_question_skipped(self, rag_config, tmp_path):
        chunker, encoder, builder, retriever = _patched_deps()
        docs_path = _make_documents(tmp_path)
        qa = [
            {"id": "q1", "question": "   ", "relevant_sections": ["Teaching & Assessment"]},
            {"id": "q2", "question": "Real?", "relevant_sections": ["Teaching & Assessment"]},
        ]
        with (
            patch("evaluation.experiments.chunk_size_comparison.Chunker", return_value=chunker),
            patch("evaluation.experiments.chunk_size_comparison.Document", side_effect=lambda **kw: SimpleNamespace(**kw)),
            patch("evaluation.experiments.chunk_size_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.chunk_size_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.chunk_size_comparison.FAISSRetriever", return_value=retriever),
        ):
            df = run_chunk_size_comparison(
                qa_pairs=qa,
                documents_path=docs_path,
                config=rag_config,
                chunk_sizes=[512],
            )
        assert df.iloc[0]["n_queries"] == 1

    def test_all_questions_empty_no_ci_columns(self, rag_config, tmp_path):
        chunker, encoder, builder, retriever = _patched_deps()
        docs_path = _make_documents(tmp_path)
        qa = [
            {"id": "q1", "question": " ", "relevant_sections": []},
            {"id": "q2", "question": "", "relevant_sections": []},
        ]
        with (
            patch("evaluation.experiments.chunk_size_comparison.Chunker", return_value=chunker),
            patch("evaluation.experiments.chunk_size_comparison.Document", side_effect=lambda **kw: SimpleNamespace(**kw)),
            patch("evaluation.experiments.chunk_size_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.chunk_size_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.chunk_size_comparison.FAISSRetriever", return_value=retriever),
        ):
            df = run_chunk_size_comparison(
                qa_pairs=qa,
                documents_path=docs_path,
                config=rag_config,
                chunk_sizes=[256],
            )
        assert df.iloc[0]["n_queries"] == 0
        assert "mrr_ci_low" not in df.columns

    def test_blank_line_in_documents_jsonl_skipped(self, rag_config, tmp_path, sample_qa_pairs):
        chunker, encoder, builder, retriever = _patched_deps()
        docs_path = tmp_path / "docs.jsonl"
        with docs_path.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "id": "d1",
                        "content": "x",
                        "raw_html": "<p>x</p>",
                        "metadata": {"source_url": "u", "document_title": "T", "section": "S"},
                    }
                )
                + "\n"
            )
            f.write("\n")  # blank line
        with (
            patch("evaluation.experiments.chunk_size_comparison.Chunker", return_value=chunker),
            patch("evaluation.experiments.chunk_size_comparison.Document", side_effect=lambda **kw: SimpleNamespace(**kw)),
            patch("evaluation.experiments.chunk_size_comparison.ChunkEncoder", return_value=encoder),
            patch("evaluation.experiments.chunk_size_comparison.FAISSIndexBuilder", return_value=builder),
            patch("evaluation.experiments.chunk_size_comparison.FAISSRetriever", return_value=retriever),
        ):
            df = run_chunk_size_comparison(
                qa_pairs=sample_qa_pairs(1),
                documents_path=docs_path,
                config=rag_config,
                chunk_sizes=[512],
            )
        # 1 document × 2 chunks per doc
        assert df.iloc[0]["num_chunks"] == 2
