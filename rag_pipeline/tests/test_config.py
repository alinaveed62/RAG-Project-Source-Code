"""Tests for the Pydantic v2 RAGConfig validators."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_pipeline.config import RAGConfig
from rag_pipeline.models import RAGResponse


class TestRAGConfigDefaults:
    """Defaults match the empirical winners from the experiments."""

    def test_construct_with_no_arguments_succeeds(self):
        config = RAGConfig()
        assert config.embedding_model == "multi-qa-MiniLM-L6-cos-v1"
        assert config.embedding_dim == 384
        assert config.ollama_model == "gemma2:2b"
        assert config.top_k == 3
        assert config.inference_backend == "ollama"

    def test_default_paths_are_pathlib(self):
        config = RAGConfig()
        assert isinstance(config.chunks_path, Path)
        assert isinstance(config.index_dir, Path)

    def test_reranking_defaults(self):
        config = RAGConfig()
        # Default False because the reranking ablation showed that
        # dense-only retrieval beat dense-plus-rerank on every metric
        # under the section-level ground truth. The reranker is still
        # exposed as a per-request toggle.
        assert config.enable_reranking is False
        assert config.rerank_fetch_k == 20
        assert config.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_retrieval_mode_default(self):
        config = RAGConfig()
        assert config.retrieval_mode == "dense"
        assert config.rrf_k == 60

    def test_low_confidence_threshold_default(self):
        config = RAGConfig()
        assert config.low_confidence_threshold == 0.4


class TestRAGConfigOverrides:
    """Plain keyword overrides still work."""

    def test_override_top_k(self):
        config = RAGConfig(top_k=5)
        assert config.top_k == 5

    def test_override_embedding_model(self):
        config = RAGConfig(embedding_model="all-MiniLM-L6-v2")
        assert config.embedding_model == "all-MiniLM-L6-v2"

    def test_override_paths(self, tmp_path):
        config = RAGConfig(
            chunks_path=tmp_path / "chunks.jsonl",
            index_dir=tmp_path / "index",
        )
        assert config.chunks_path == tmp_path / "chunks.jsonl"
        assert config.index_dir == tmp_path / "index"


class TestRAGConfigValidationFailures:
    """Negative paths must raise ValidationError, not silently accept bad input."""

    def test_top_k_zero_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(top_k=0)

    def test_top_k_negative_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(top_k=-1)

    def test_top_k_above_50_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(top_k=51)

    def test_similarity_threshold_above_one_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(similarity_threshold=1.5)

    def test_similarity_threshold_negative_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(similarity_threshold=-0.1)

    def test_temperature_above_two_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(temperature=2.5)

    def test_temperature_negative_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(temperature=-0.5)

    def test_top_p_above_one_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(top_p=1.5)

    def test_low_confidence_threshold_above_one_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(low_confidence_threshold=2.0)

    def test_low_confidence_threshold_negative_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(low_confidence_threshold=-0.1)

    def test_max_new_tokens_zero_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(max_new_tokens=0)

    def test_rerank_fetch_k_zero_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(rerank_fetch_k=0)

    def test_rerank_fetch_k_above_100_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(rerank_fetch_k=101)

    def test_inference_backend_huggingface_raises(self):
        """The HuggingFace backend is not supported, so construction
        with anything other than 'ollama' must fail loudly. The
        Literal on inference_backend makes any non-Ollama value
        impossible by construction."""
        with pytest.raises(ValidationError):
            RAGConfig(inference_backend="huggingface")  # type: ignore[arg-type]

    def test_retrieval_mode_invalid_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(retrieval_mode="approximate")  # type: ignore[arg-type]

    def test_rrf_k_zero_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(rrf_k=0)

    def test_embedding_dim_too_small_raises(self):
        with pytest.raises(ValidationError):
            RAGConfig(embedding_dim=32)


class TestRAGConfigValidateAssignment:
    """validate_assignment=True extends validation from construction
    time to attribute-assignment time. Without it, later writes like
    config.top_k = -1 would silently succeed and the Field and
    Literal constraints would only protect the constructor.
    """

    def test_invalid_assignment_raises(self):
        config = RAGConfig()
        with pytest.raises(ValidationError):
            config.top_k = -1  # outside the valid range of [1, 50]

    def test_invalid_literal_assignment_raises(self):
        config = RAGConfig()
        with pytest.raises(ValidationError):
            config.inference_backend = "huggingface"  # type: ignore[assignment]

    def test_valid_assignment_succeeds(self):
        """Assignment with a valid value must not raise and must actually
        persist the new value (not silently roll back)."""
        config = RAGConfig()
        config.top_k = 5
        assert config.top_k == 5


class TestRAGResponseBackwardsCompat:
    """RAGResponse.citations and rerank_time_ms carry neutral defaults
    so older serialised dicts (a cached evaluator response, a
    hand-written fixture) still deserialise via model_validate with
    the new fields filled in from those defaults.
    """

    def test_older_dict_without_citations_deserialises(self):
        data = {
            "question": "Do I have to attend lectures?",
            "answer": "Yes, attendance is recorded.",
            "sources": [],
            "retrieval_time_ms": 0.0,
            "generation_time_ms": 0.0,
        }
        response = RAGResponse.model_validate(data)
        assert response.citations == []
        assert response.rerank_time_ms == 0.0

    def test_newer_dict_round_trips_through_model_dump(self):
        original = RAGResponse(
            question="q",
            answer="a",
            sources=[],
            retrieval_time_ms=1.0,
            rerank_time_ms=2.0,
            generation_time_ms=3.0,
        )
        dumped = original.model_dump()
        restored = RAGResponse.model_validate(dumped)
        assert restored == original
