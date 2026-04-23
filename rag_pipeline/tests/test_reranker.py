"""Tests for the cross-encoder reranker."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_pipeline.models import RetrievalResult
from rag_pipeline.retrieval.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker


def _result(chunk_id: str, score: float, text: str = "") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text or f"chunk text for {chunk_id}",
        score=score,
        source="https://example.test",
        title="Title",
        section="Main",
    )


class TestRerankerLazyLoad:
    def test_construct_does_not_load(self):
        reranker = CrossEncoderReranker()
        assert reranker._model is None
        assert reranker.model_name == DEFAULT_RERANKER_MODEL

    def test_first_rerank_loads_model(self):
        reranker = CrossEncoderReranker()
        with patch("sentence_transformers.CrossEncoder") as MockCE:
            instance = MockCE.return_value
            instance.predict.return_value = np.array([0.5], dtype=np.float32)
            reranker.rerank("query", [_result("c1", 0.9)])
            MockCE.assert_called_once_with(DEFAULT_RERANKER_MODEL)
        assert reranker._model is not None

    def test_second_rerank_does_not_reload(self):
        reranker = CrossEncoderReranker()
        with patch("sentence_transformers.CrossEncoder") as MockCE:
            instance = MockCE.return_value
            instance.predict.return_value = np.array([0.5], dtype=np.float32)
            reranker.rerank("query", [_result("c1", 0.9)])
            reranker.rerank("query2", [_result("c2", 0.8)])
            MockCE.assert_called_once()


class TestRerankerBehaviour:
    def test_empty_results_returns_empty(self):
        reranker = CrossEncoderReranker()
        # Should not load the model for empty input.
        with patch("sentence_transformers.CrossEncoder") as MockCE:
            assert reranker.rerank("query", []) == []
            MockCE.assert_not_called()

    def test_reorders_when_scores_disagree(self):
        reranker = CrossEncoderReranker()
        bi_results = [
            _result("c_high_bi", 0.9, text="text high bi"),
            _result("c_low_bi", 0.4, text="text low bi"),
        ]
        # Cross-encoder flips the ranking.
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 9.5], dtype=np.float32)
        reranker._model = mock_model

        out = reranker.rerank("q", bi_results)
        assert [r.chunk_id for r in out] == ["c_low_bi", "c_high_bi"]
        assert out[0].score == pytest.approx(9.5)
        assert out[1].score == pytest.approx(0.1)

    def test_preserves_chunk_metadata(self):
        reranker = CrossEncoderReranker()
        original = RetrievalResult(
            chunk_id="cX",
            text="full chunk text",
            score=0.5,
            source="https://example.test/path",
            title="Some Title",
            section="Some Section",
        )
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([7.0], dtype=np.float32)
        reranker._model = mock_model

        out = reranker.rerank("q", [original])
        assert len(out) == 1
        assert out[0].chunk_id == "cX"
        assert out[0].text == "full chunk text"
        assert out[0].source == "https://example.test/path"
        assert out[0].title == "Some Title"
        assert out[0].section == "Some Section"
        assert out[0].score == pytest.approx(7.0)

    def test_top_k_truncation(self):
        reranker = CrossEncoderReranker()
        bi_results = [_result(f"c{i}", 0.5) for i in range(5)]
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32
        )
        reranker._model = mock_model

        out = reranker.rerank("q", bi_results, top_k=3)
        assert len(out) == 3
        # Highest-scoring three: c4 (5.0), c3 (4.0), c2 (3.0).
        assert [r.chunk_id for r in out] == ["c4", "c3", "c2"]

    def test_top_k_none_returns_all(self):
        reranker = CrossEncoderReranker()
        bi_results = [_result(f"c{i}", 0.5) for i in range(4)]
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(
            [4.0, 3.0, 2.0, 1.0], dtype=np.float32
        )
        reranker._model = mock_model

        out = reranker.rerank("q", bi_results, top_k=None)
        assert len(out) == 4
        # Already in descending order from the mock.
        assert [r.chunk_id for r in out] == ["c0", "c1", "c2", "c3"]

    def test_passes_query_text_pairs_to_model(self):
        reranker = CrossEncoderReranker()
        bi_results = [
            _result("c1", 0.5, text="alpha text"),
            _result("c2", 0.5, text="beta text"),
        ]
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.2], dtype=np.float32)
        reranker._model = mock_model

        reranker.rerank("the query", bi_results)
        called_pairs = mock_model.predict.call_args[0][0]
        assert called_pairs == [("the query", "alpha text"), ("the query", "beta text")]
        # show_progress_bar is False so test runs quietly.
        assert mock_model.predict.call_args[1].get("show_progress_bar") is False

    def test_negative_top_k_raises(self):
        """Pydantic validates config.top_k in [1, 50] at construction
        time, but rerank is also part of the public API and a direct
        caller could pass a negative top_k. Python list slicing silently
        accepts negative indices ( xs[:-1] drops the tail), which would
        return wrong-sized results; raise explicitly instead."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.2], dtype=np.float32)
        reranker._model = mock_model

        with pytest.raises(
            ValueError, match=r"top_k must be a non-negative integer or None"
        ):
            reranker.rerank("q", [_result("c1", 0.5), _result("c2", 0.5)], top_k=-1)

    def test_strict_zip_catches_mismatched_predict_length(self):
        """reranker.py wraps the results/scores pair in zip(...,
        strict=True) so a future sentence_transformers release
        (or a subclass that pre-filters) that returns fewer scores
        than input pairs is caught immediately rather than silently
        dropping the tail chunks. The real CrossEncoder.predict
        always returns the right length, so the test uses a mock
        model that returns one fewer score than there are pairs
        and checks the guard fires.
        """
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        # Two query-doc pairs go in, only one score comes back out.
        mock_model.predict.return_value = np.array([0.1], dtype=np.float32)
        reranker._model = mock_model

        with pytest.raises(ValueError):
            reranker.rerank(
                "q", [_result("c1", 0.5), _result("c2", 0.5)]
            )
