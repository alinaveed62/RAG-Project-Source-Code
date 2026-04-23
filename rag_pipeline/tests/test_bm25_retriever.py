"""Tests for the BM25 sparse retriever."""

import logging

import pytest

from rag_pipeline.config import RAGConfig
from rag_pipeline.models import RetrievalResult
from rag_pipeline.retrieval.bm25_retriever import BM25Retriever, _tokenise


@pytest.fixture
def bm25_config(tmp_path):
    return RAGConfig(
        chunks_path=tmp_path / "chunks.jsonl",
        index_dir=tmp_path / "index",
        top_k=3,
    )


@pytest.fixture
def bm25_retriever(sample_chunks, bm25_config):
    return BM25Retriever(sample_chunks, bm25_config)


class TestBM25RetrieverInit:
    """Tests for BM25 index construction."""

    def test_builds_index_from_chunks(self, sample_chunks, bm25_config):
        retriever = BM25Retriever(sample_chunks, bm25_config)
        assert retriever.bm25 is not None
        assert len(retriever.metadata) == len(sample_chunks)

    def test_empty_chunks(self, bm25_config):
        retriever = BM25Retriever([], bm25_config)
        assert len(retriever.metadata) == 0
        # retrieve() must take the empty-corpus short-circuit rather than
        # call into the missing BM25Okapi instance.
        assert retriever.retrieve("anything") == []

    def test_empty_chunks_logs_warning(self, bm25_config, caplog):
        """A silent empty corpus hides build-pipeline bugs: queries would
        return [] forever with no log trace. Emitting a warning at
        construction pins the diagnostic to the actual root cause."""
        with caplog.at_level(
            logging.WARNING, logger="rag_pipeline.retrieval.bm25_retriever"
        ):
            BM25Retriever([], bm25_config)
        assert any(
            "BM25 corpus is empty" in record.message for record in caplog.records
        ), f"expected empty-corpus warning, got records={caplog.records!r}"


class TestBM25RetrieverRetrieve:
    """Tests for BM25 retrieval."""

    def test_retrieve_returns_retrieval_results(self, bm25_retriever):
        results = bm25_retriever.retrieve("attendance lectures")
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_ranks_by_relevance(self, bm25_retriever):
        results = bm25_retriever.retrieve("attendance lectures")
        assert len(results) > 0
        # First result should be about attendance
        assert "attend" in results[0].text.lower()

    def test_retrieve_respects_top_k(self, bm25_retriever):
        results = bm25_retriever.retrieve("student", top_k=2)
        assert len(results) <= 2

    def test_retrieve_default_top_k_from_config(self, bm25_retriever):
        results = bm25_retriever.retrieve("student")
        assert len(results) <= 3  # config.top_k = 3

    def test_retrieve_scores_are_positive(self, bm25_retriever):
        results = bm25_retriever.retrieve("lectures attendance")
        for r in results:
            assert r.score > 0

    def test_retrieve_scores_descending(self, bm25_retriever):
        results = bm25_retriever.retrieve("student")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_with_section_filter(self, bm25_retriever):
        results = bm25_retriever.retrieve(
            "student", section_filter="Teaching & Assessment"
        )
        for r in results:
            assert r.section == "Teaching & Assessment"

    def test_retrieve_no_matching_query(self, bm25_retriever):
        results = bm25_retriever.retrieve("xyznonexistent")
        assert len(results) == 0

    def test_retrieve_early_break_on_top_k(self, bm25_config):
        """With top_k smaller than the number of matching chunks, the loop
        must early-exit via the break rather than scoring every chunk.
        Two documents match the query so BM25 gives them positive IDF; a
        top_k of 1 forces the break path that long queries on large
        corpora would otherwise leave uncovered on this small fixture."""
        chunks = [
            {"id": "c0", "text": "target document text", "source": "",
             "title": "", "section": ""},
            {"id": "c1", "text": "another target mention", "source": "",
             "title": "", "section": ""},
            {"id": "c2", "text": "unrelated filler alpha", "source": "",
             "title": "", "section": ""},
            {"id": "c3", "text": "unrelated filler beta", "source": "",
             "title": "", "section": ""},
            {"id": "c4", "text": "unrelated filler gamma", "source": "",
             "title": "", "section": ""},
        ]
        retriever = BM25Retriever(chunks, bm25_config)
        results = retriever.retrieve("target", top_k=1)
        assert len(results) == 1

    def test_retrieve_populates_metadata(self, bm25_retriever):
        results = bm25_retriever.retrieve("attendance lectures")
        assert results, "expected at least one BM25 match for the query"
        r = results[0]
        assert isinstance(r.chunk_id, str) and r.chunk_id
        assert isinstance(r.text, str) and r.text
        assert isinstance(r.title, str) and r.title
        assert isinstance(r.section, str) and r.section


class TestBM25Tokeniser:
    """Regression tests for the tokeniser: a plain text.lower().split()
    leaves punctuation attached, so queries such as "lectures?" would fail
    to match documents containing "lectures". The regex tokeniser strips
    trailing punctuation and preserves Unicode word characters."""

    def test_strips_trailing_question_mark(self):
        assert _tokenise("lectures?") == ["lectures"]

    def test_strips_period_and_comma(self):
        assert _tokenise("lectures. commas, everywhere.") == [
            "lectures",
            "commas",
            "everywhere",
        ]

    def test_lowercases_tokens(self):
        assert _tokenise("HELLO World") == ["hello", "world"]

    def test_handles_unicode_word_characters(self):
        # "naive" written with a combining diaeresis should still tokenise.
        assert _tokenise("naïve") == ["naïve"]

    def test_empty_returns_empty_list(self):
        assert _tokenise("") == []

    def test_punctuation_only_returns_empty_list(self):
        assert _tokenise("?!,.;:") == []

    def test_query_lectures_question_mark_retrieves_lectures_chunk(
        self, bm25_retriever
    ):
        with_punct = bm25_retriever.retrieve("lectures?")
        without_punct = bm25_retriever.retrieve("lectures")

        assert len(with_punct) > 0, "query 'lectures?' returned no hits"
        assert len(without_punct) > 0, "query 'lectures' returned no hits"

        # Both queries should surface the same top result.
        assert with_punct[0].chunk_id == without_punct[0].chunk_id
