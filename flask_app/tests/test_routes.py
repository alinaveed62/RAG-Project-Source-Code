"""Tests for Flask routes."""

import json
from unittest.mock import MagicMock

import pytest

from flask_app.app import create_app
from flask_app.config import Config
from rag_pipeline.models import RAGResponse, RetrievalResult


class TestConfig(Config):
    TESTING = True


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.answer.return_value = RAGResponse(
        question="test?",
        answer="The answer.",
        sources=[
            RetrievalResult(
                chunk_id="c1",
                text="Source text.",
                score=0.9,
                source="https://keats.kcl.ac.uk/test",
                title="Test Title",
                section="Main",
                heading_path=["Main", "Intro"],
            )
        ],
        retrieval_time_ms=10.0,
        generation_time_ms=200.0,
    )
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    app = create_app(TestConfig, pipeline=mock_pipeline)
    with app.test_client() as client:
        yield client


@pytest.fixture
def client_no_pipeline():
    app = create_app(TestConfig, pipeline=None)
    with app.test_client() as client:
        yield client


class TestIndexRoute:
    def test_get_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_get_contains_form(self, client):
        response = client.get("/")
        assert b"question" in response.data

    def test_get_contains_disclaimer(self, client):
        response = client.get("/")
        assert b"AI system" in response.data
        assert b"may contain errors" in response.data

    def test_post_with_question(self, client, mock_pipeline):
        response = client.post("/", data={"question": "What is EC?"})
        assert response.status_code == 200
        assert b"The answer." in response.data
        mock_pipeline.answer.assert_called_once_with("What is EC?")

    def test_post_empty_question(self, client):
        response = client.post("/", data={"question": ""})
        assert response.status_code == 200
        assert b"Please enter a question" in response.data

    def test_post_whitespace_question(self, client):
        response = client.post("/", data={"question": "   "})
        assert response.status_code == 200
        assert b"Please enter a question" in response.data

    def test_post_without_pipeline(self, client_no_pipeline):
        response = client_no_pipeline.post("/", data={"question": "test?"})
        assert response.status_code == 200
        assert b"not loaded" in response.data

    def test_answer_shows_sources(self, client):
        response = client.post("/", data={"question": "test?"})
        assert b"Test Title" in response.data
        assert b"Main" in response.data

    def test_answer_shows_heading_path(self, client):
        """heading_path must be rendered so the reader sees the handbook
        location of the cited chunk, not just the top-level section."""
        response = client.post("/", data={"question": "test?"})
        assert b"source-path" in response.data
        assert b"Intro" in response.data

    def test_answer_shows_timing(self, client):
        response = client.post("/", data={"question": "test?"})
        assert b"Retrieval:" in response.data
        assert b"Generation:" in response.data

    def test_answer_hides_rerank_timing_when_not_performed(self, client):
        """When the mock response has rerank_time_ms == 0 (reranker
        disabled), the template must not render a Rerank: 0ms label,
        so non-reranked answers do not show a noisy zero timing.
        """
        response = client.post("/", data={"question": "test?"})
        assert b"Rerank:" not in response.data

    def test_answer_shows_rerank_timing_when_performed(self, mock_pipeline):
        """When the pipeline returns a RAGResponse with a non-zero
        rerank_time_ms, the template must show it alongside the retrieval
        and generation timings so the user sees the full latency breakdown.
        """
        mock_pipeline.answer.return_value = RAGResponse(
            question="test?",
            answer="reranked answer",
            sources=[
                RetrievalResult(
                    chunk_id="c1",
                    text="source text",
                    score=7.5,
                    source="https://keats.kcl.ac.uk/test",
                    title="T",
                    section="S",
                )
            ],
            retrieval_time_ms=12.0,
            rerank_time_ms=34.0,
            generation_time_ms=210.0,
        )
        app = create_app(TestConfig, pipeline=mock_pipeline)
        with app.test_client() as client:
            response = client.post("/", data={"question": "test?"})
        assert b"Rerank:" in response.data
        assert b"34ms" in response.data


class TestFeedbackRoute:
    def test_feedback_returns_200(self, client):
        response = client.post(
            "/feedback",
            data={
                "question": "test?",
                "answer_rating": "4",
                "source_rating": "3",
                "comments": "Good answer",
            },
        )
        assert response.status_code == 200
        assert b"Thank you" in response.data

    def test_feedback_writes_jsonl(self, client, tmp_path):
        # Override feedback dir for this test
        client.application.config["FEEDBACK_DIR"] = tmp_path

        client.post(
            "/feedback",
            data={
                "question": "test?",
                "answer_rating": "5",
                "source_rating": "4",
                "comments": "",
            },
        )

        feedback_path = tmp_path / "feedback.jsonl"
        assert feedback_path.exists()
        with open(feedback_path) as f:
            entry = json.loads(f.readline())
        assert entry["question"] == "test?"
        assert entry["answer_rating"] == 5
        assert "timestamp" in entry


class TestFeedbackRatingBounds:
    """Ratings outside 1 to 5 must be coerced to 0 before they reach the
    feedback log, so the store only contains valid values. The
    _parse_rating helper handles integers, out-of-range numbers, and
    non-numeric strings alike.
    """

    @pytest.mark.parametrize("raw", ["0", "-1", "6", "99", "abc", "", "3.5"])
    def test_out_of_range_rating_recorded_as_zero(self, client, tmp_path, raw):
        client.application.config["FEEDBACK_DIR"] = tmp_path

        client.post(
            "/feedback",
            data={
                "question": "test?",
                "answer_rating": raw,
                "source_rating": raw,
                "comments": "",
            },
        )

        feedback_path = tmp_path / "feedback.jsonl"
        assert feedback_path.exists()
        with open(feedback_path) as f:
            entry = json.loads(f.readline())
        assert entry["answer_rating"] == 0
        assert entry["source_rating"] == 0

    @pytest.mark.parametrize("raw", ["1", "2", "3", "4", "5"])
    def test_in_range_rating_preserved(self, client, tmp_path, raw):
        client.application.config["FEEDBACK_DIR"] = tmp_path

        client.post(
            "/feedback",
            data={
                "question": "test?",
                "answer_rating": raw,
                "source_rating": raw,
                "comments": "",
            },
        )

        feedback_path = tmp_path / "feedback.jsonl"
        with open(feedback_path) as f:
            entry = json.loads(f.readline())
        assert entry["answer_rating"] == int(raw)
        assert entry["source_rating"] == int(raw)

    def test_missing_rating_recorded_as_zero(self, client, tmp_path):
        client.application.config["FEEDBACK_DIR"] = tmp_path

        client.post(
            "/feedback",
            data={
                "question": "test?",
                # answer_rating and source_rating intentionally omitted.
                "comments": "",
            },
        )

        feedback_path = tmp_path / "feedback.jsonl"
        with open(feedback_path) as f:
            entry = json.loads(f.readline())
        assert entry["answer_rating"] == 0
        assert entry["source_rating"] == 0
