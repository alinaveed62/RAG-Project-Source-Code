"""Tests for the end-to-end RAG pipeline."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_pipeline.config import RAGConfig
from rag_pipeline.models import RAGResponse, RetrievalResult
from rag_pipeline.pipeline import (
    GENERATION_ERROR_ANSWER,
    LOW_CONFIDENCE_ANSWER,
    RAGPipeline,
)


@pytest.fixture
def mock_pipeline(tmp_path):
    """Create a RAGPipeline with all components mocked.

    RAGConfig enforces inference_backend == "ollama" via its Literal,
    so the test still has to avoid loading the real Ollama generator
    and encoder; both _load_generator and ChunkEncoder are stubbed.
    """
    config = RAGConfig(
        chunks_path=tmp_path / "chunks.jsonl",
        index_dir=tmp_path / "index",
    )

    with patch("rag_pipeline.pipeline.ChunkEncoder") as MockEncoder, \
         patch("rag_pipeline.pipeline.FAISSIndexBuilder") as MockBuilder:

        mock_encoder_instance = MockEncoder.return_value
        mock_encoder_instance.encode_query.return_value = np.zeros(384, dtype=np.float32)
        mock_encoder_instance.load_chunks.return_value = [
            {"id": "c1", "text": "test", "source": "", "title": "T", "section": "S"}
        ]
        mock_encoder_instance.encode_chunks.return_value = np.zeros((1, 384), dtype=np.float32)

        mock_builder_instance = MockBuilder.return_value
        mock_index = MagicMock()
        mock_index.ntotal = 1
        mock_builder_instance.load.return_value = (mock_index, [
            {"id": "c1", "text": "test text", "source": "url", "title": "T", "section": "S"}
        ])

        pipeline = RAGPipeline(config)

        # Manually set up retriever and generator for answer() tests
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RetrievalResult(
                chunk_id="c1",
                text="test text",
                score=0.85,
                source="https://keats.kcl.ac.uk/test",
                title="Test Title",
                section="Main",
            )
        ]

        mock_gen = MagicMock()
        mock_gen.generate.return_value = "The answer is 42."

        yield pipeline, mock_retriever, mock_gen, mock_encoder_instance, mock_builder_instance


class TestRAGPipelineBuildIndex:
    """Tests for index building."""

    def test_build_index_calls_encoder_and_builder(self, mock_pipeline, tmp_path):
        pipeline, _, _, mock_encoder, mock_builder = mock_pipeline
        chunks_path = tmp_path / "chunks.jsonl"
        chunks_path.write_text('{"id":"c1","text":"t","source":"","title":"","section":""}\n')

        pipeline.build_index(chunks_path)

        mock_encoder.load_chunks.assert_called_once_with(chunks_path)
        mock_encoder.encode_chunks.assert_called_once()
        mock_builder.build_index.assert_called_once()
        mock_builder.save.assert_called_once()


class TestRAGPipelineSetup:
    """Tests for pipeline setup."""

    @patch("rag_pipeline.pipeline.RAGPipeline._load_generator")
    def test_setup_loads_index_and_generator(self, mock_load_gen, mock_pipeline):
        pipeline, _, _, _, mock_builder = mock_pipeline

        pipeline.setup()

        mock_builder.load.assert_called_once()
        mock_load_gen.assert_called_once()
        assert pipeline.retriever is not None


class TestRAGPipelineAnswer:
    """Tests for the answer method."""

    def test_answer_returns_rag_response(self, mock_pipeline):
        pipeline, mock_retriever, mock_gen, _, _ = mock_pipeline
        pipeline.retriever = mock_retriever
        pipeline.generator = mock_gen

        response = pipeline.answer("What is the attendance policy?")

        assert isinstance(response, RAGResponse)
        assert response.question == "What is the attendance policy?"
        assert response.answer == "The answer is 42."
        assert len(response.sources) == 1
        # Timer captured a float in a sensible range rather than the
        # >= 0 tautology (perf_counter deltas are always non-negative).
        assert isinstance(response.retrieval_time_ms, float)
        assert 0.0 <= response.retrieval_time_ms < 5000.0
        assert isinstance(response.generation_time_ms, float)
        assert 0.0 <= response.generation_time_ms < 5000.0

    def test_answer_without_setup_raises(self, mock_pipeline):
        pipeline, _, _, _, _ = mock_pipeline
        with pytest.raises(RuntimeError, match="not set up"):
            pipeline.answer("test?")

    def test_answer_with_section_filter(self, mock_pipeline):
        pipeline, mock_retriever, mock_gen, _, _ = mock_pipeline
        pipeline.retriever = mock_retriever
        pipeline.generator = mock_gen

        pipeline.answer("test?", section_filter="Main")

        mock_retriever.retrieve.assert_called_once()
        call_kwargs = mock_retriever.retrieve.call_args[1]
        assert call_kwargs["section_filter"] == "Main"

    def test_answer_processes_query(self, mock_pipeline):
        pipeline, mock_retriever, mock_gen, mock_encoder, _ = mock_pipeline
        pipeline.retriever = mock_retriever
        pipeline.generator = mock_gen

        pipeline.answer("What is the EC policy?")

        # The encoder should receive the processed (expanded) query
        mock_encoder.encode_query.assert_called_once()
        query_arg = mock_encoder.encode_query.call_args[0][0]
        assert "Extenuating Circumstances" in query_arg


class TestLowConfidenceRefusal:
    """Tests for the low-confidence refusal logic."""

    def test_empty_results_returns_refusal(self, mock_pipeline):
        pipeline, mock_retriever, mock_gen, _, _ = mock_pipeline
        pipeline.retriever = mock_retriever
        pipeline.generator = mock_gen
        mock_retriever.retrieve.return_value = []

        response = pipeline.answer("What is quantum physics?")

        assert response.answer == LOW_CONFIDENCE_ANSWER
        assert response.generation_time_ms == 0.0
        mock_gen.generate.assert_not_called()

    def test_low_average_score_returns_refusal(self, mock_pipeline):
        pipeline, mock_retriever, mock_gen, _, _ = mock_pipeline
        pipeline.retriever = mock_retriever
        pipeline.generator = mock_gen

        # Average score 0.35 < default low_confidence_threshold 0.4
        mock_retriever.retrieve.return_value = [
            RetrievalResult(
                chunk_id="c1", text="t", score=0.35,
                source="", title="", section="",
            ),
            RetrievalResult(
                chunk_id="c2", text="t", score=0.35,
                source="", title="", section="",
            ),
        ]

        response = pipeline.answer("Something irrelevant?")

        assert response.answer == LOW_CONFIDENCE_ANSWER
        mock_gen.generate.assert_not_called()

    def test_high_confidence_proceeds_normally(self, mock_pipeline):
        pipeline, mock_retriever, mock_gen, _, _ = mock_pipeline
        pipeline.retriever = mock_retriever
        pipeline.generator = mock_gen

        # Score 0.85 > 0.4 threshold
        mock_retriever.retrieve.return_value = [
            RetrievalResult(
                chunk_id="c1", text="test text", score=0.85,
                source="url", title="T", section="S",
            )
        ]

        response = pipeline.answer("What is the attendance policy?")

        assert response.answer == "The answer is 42."
        mock_gen.generate.assert_called_once()

    def test_is_low_confidence_with_no_results(self, mock_pipeline):
        pipeline, _, _, _, _ = mock_pipeline
        assert pipeline._is_low_confidence([]) is True

    def test_is_low_confidence_with_high_scores(self, mock_pipeline):
        pipeline, _, _, _, _ = mock_pipeline
        results = [
            RetrievalResult(
                chunk_id="c1", text="t", score=0.9,
                source="", title="", section="",
            )
        ]
        assert pipeline._is_low_confidence(results) is False


class TestRerankIntegration:
    """pipeline.answer() must call the reranker after the bi-encoder
    pass when reranking is enabled, and the low-confidence refusal
    must be evaluated on the pre-rerank bi-encoder scores."""

    def test_answer_calls_reranker_when_enabled(self, mock_pipeline):
        pipeline, mock_retriever, mock_gen, _, _ = mock_pipeline
        pipeline.retriever = mock_retriever
        pipeline.generator = mock_gen

        # Reranker mock that reverses its input AND sleeps 1ms so the rerank
        # timer captures a measurable delta. Without the sleep, a fast CPU
        # can produce a mocked-rerank time too small to distinguish from the
        # "reranker disabled" zero.
        def _slow_rerank(q, results, top_k=None):
            time.sleep(0.001)
            return list(reversed(results))[: top_k or len(results)]

        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = _slow_rerank
        pipeline.reranker = mock_reranker

        # Bi-encoder returns two high-confidence results.
        mock_retriever.retrieve.return_value = [
            RetrievalResult(chunk_id="c1", text="t1", score=0.85, source="", title="", section=""),
            RetrievalResult(chunk_id="c2", text="t2", score=0.80, source="", title="", section=""),
        ]

        response = pipeline.answer("question?")

        mock_reranker.rerank.assert_called_once()
        # The reranker reversed the order; the response sources should be
        # [c2, c1] truncated to top_k.
        assert [r.chunk_id for r in response.sources] == ["c2", "c1"][: pipeline.config.top_k]
        # Positive assertion that the timer captured the rerank
        # step. The disabled-rerank path has its own == 0.0 test
        # below, so the two together cover both branches.
        assert response.rerank_time_ms > 0.0

    def test_answer_skips_reranker_when_disabled(self, mock_pipeline):
        pipeline, mock_retriever, mock_gen, _, _ = mock_pipeline
        pipeline.retriever = mock_retriever
        pipeline.generator = mock_gen
        pipeline.reranker = None  # Disabled.

        mock_retriever.retrieve.return_value = [
            RetrievalResult(chunk_id="c1", text="t", score=0.85, source="", title="", section=""),
        ]
        response = pipeline.answer("question?")
        # No rerank means rerank_time_ms is 0.0.
        assert response.rerank_time_ms == 0.0

    def test_fr9_uses_pre_rerank_bi_encoder_scores(self, mock_pipeline):
        """When the bi-encoder average score falls below the refusal
        threshold, the reranker must not be called and a refusal must
        be returned. This is the safety property of the rerank wiring.
        """
        pipeline, mock_retriever, mock_gen, _, _ = mock_pipeline
        pipeline.retriever = mock_retriever
        pipeline.generator = mock_gen
        mock_reranker = MagicMock()
        pipeline.reranker = mock_reranker

        # Average of 0.30 < default low_confidence_threshold of 0.40.
        mock_retriever.retrieve.return_value = [
            RetrievalResult(chunk_id="c1", text="t", score=0.30, source="", title="", section=""),
            RetrievalResult(chunk_id="c2", text="t", score=0.30, source="", title="", section=""),
        ]
        response = pipeline.answer("borderline?")

        assert response.answer == LOW_CONFIDENCE_ANSWER
        mock_reranker.rerank.assert_not_called()
        mock_gen.generate.assert_not_called()


class TestCitationParsingIntegration:
    """pipeline.answer() must extract validated citations from the
    model output when citation injection is enabled, and must drop
    any citation that references a chunk id not actually retrieved.
    """

    def test_answer_attaches_validated_citations(self, mock_pipeline):
        pipeline, mock_retriever, mock_gen, _, _ = mock_pipeline
        pipeline.retriever = mock_retriever
        pipeline.generator = mock_gen
        pipeline.reranker = None
        # The retrieved chunk has chunk_id = "c1".
        mock_retriever.retrieve.return_value = [
            RetrievalResult(chunk_id="c1", text="t", score=0.85, source="", title="", section=""),
        ]
        # LLM cites the real chunk and a hallucinated one.
        mock_gen.generate.return_value = (
            "Lectures are recommended [Source: c1]. Coffee helps [Source: invented]."
        )
        response = pipeline.answer("attendance?")
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == "c1"

    def test_answer_skips_citation_parsing_when_disabled(self, mock_pipeline, tmp_path):
        from rag_pipeline.config import RAGConfig

        config = RAGConfig(
            chunks_path=tmp_path / "chunks.jsonl",
            index_dir=tmp_path / "index",
            enable_citation_injection=False,
        )
        pipeline, mock_retriever, mock_gen, _, _ = mock_pipeline
        pipeline.config = config
        pipeline.retriever = mock_retriever
        pipeline.generator = mock_gen
        pipeline.reranker = None

        mock_retriever.retrieve.return_value = [
            RetrievalResult(chunk_id="c1", text="t", score=0.85, source="", title="", section=""),
        ]
        mock_gen.generate.return_value = "Plain answer with no markers."
        response = pipeline.answer("q?")
        assert response.citations == []


class TestRAGPipelineRetrievalModes:
    """pipeline.answer() must route to the sparse or hybrid retriever when
    the configured retrieval_mode asks for it. Covers pipeline.py:260, 264."""

    def test_answer_sparse_retrieval_mode(self, mock_pipeline, tmp_path):
        pipeline, _, mock_gen, _, _ = mock_pipeline
        pipeline.generator = mock_gen
        pipeline.config = RAGConfig(
            chunks_path=tmp_path / "chunks.jsonl",
            index_dir=tmp_path / "index",
            retrieval_mode="sparse",
        )
        pipeline.retriever = MagicMock()  # dense, should not be called
        sparse_mock = MagicMock()
        sparse_mock.retrieve.return_value = [
            RetrievalResult(
                chunk_id="c1", text="t", score=0.85,
                source="", title="T", section="S",
            )
        ]
        pipeline.bm25_retriever = sparse_mock

        response = pipeline.answer("attendance?")

        sparse_mock.retrieve.assert_called_once()
        pipeline.retriever.retrieve.assert_not_called()
        assert response.answer == "The answer is 42."

    def test_answer_hybrid_retrieval_mode(self, mock_pipeline, tmp_path):
        pipeline, mock_retriever, mock_gen, _, _ = mock_pipeline
        pipeline.retriever = mock_retriever  # needed for type check
        pipeline.generator = mock_gen
        pipeline.config = RAGConfig(
            chunks_path=tmp_path / "chunks.jsonl",
            index_dir=tmp_path / "index",
            retrieval_mode="hybrid",
        )
        pipeline.bm25_retriever = MagicMock()
        hybrid_mock = MagicMock()
        hybrid_mock.retrieve.return_value = [
            RetrievalResult(
                chunk_id="c1", text="t", score=0.85,
                source="", title="T", section="S",
            )
        ]
        pipeline.hybrid_retriever = hybrid_mock

        response = pipeline.answer("attendance?")

        hybrid_mock.retrieve.assert_called_once()
        kwargs = hybrid_mock.retrieve.call_args.kwargs
        assert "query_embedding" in kwargs
        mock_retriever.retrieve.assert_not_called()
        assert response.answer == "The answer is 42."


class TestFR9RetrievalModeGate:
    """The aggregate low-confidence threshold is tied to the dense
    cosine scale, so it must apply only when retrieval_mode == "dense".
    BM25 scores and RRF fused scores sit on different ranges where the
    0.4 threshold is not meaningful. The empty-results branch must
    still refuse whatever the mode.
    """

    def _sparse_pipeline(self, mock_pipeline, tmp_path, retrieval_mode):
        pipeline, _, mock_gen, _, _ = mock_pipeline
        pipeline.generator = mock_gen
        pipeline.config = RAGConfig(
            chunks_path=tmp_path / "chunks.jsonl",
            index_dir=tmp_path / "index",
            retrieval_mode=retrieval_mode,
        )
        pipeline.retriever = MagicMock()
        return pipeline, mock_gen

    def test_sparse_mode_low_score_does_not_refuse(self, mock_pipeline, tmp_path):
        pipeline, mock_gen = self._sparse_pipeline(
            mock_pipeline, tmp_path, "sparse"
        )
        sparse = MagicMock()
        # BM25 raw scores that are well below the 0.4 cosine
        # threshold but still positive. The sparse retriever has
        # already dropped zero-score matches, so a non-empty list is
        # the signal that the retriever trusts the result.
        sparse.retrieve.return_value = [
            RetrievalResult(
                chunk_id="c1", text="t", score=0.05,
                source="", title="T", section="S",
            )
        ]
        pipeline.bm25_retriever = sparse

        response = pipeline.answer("lectures?")

        assert response.answer == "The answer is 42."
        mock_gen.generate.assert_called_once()

    def test_hybrid_mode_low_score_does_not_refuse(self, mock_pipeline, tmp_path):
        pipeline, mock_gen = self._sparse_pipeline(
            mock_pipeline, tmp_path, "hybrid"
        )
        pipeline.bm25_retriever = MagicMock()
        hybrid = MagicMock()
        # RRF fused scores peak near 2/(60+1) = 0.033; far below 0.4 but
        # still the ranked-top result.
        hybrid.retrieve.return_value = [
            RetrievalResult(
                chunk_id="c1", text="t", score=0.033,
                source="", title="T", section="S",
            )
        ]
        pipeline.hybrid_retriever = hybrid

        response = pipeline.answer("lectures?")

        assert response.answer == "The answer is 42."
        mock_gen.generate.assert_called_once()

    def test_sparse_mode_still_refuses_on_empty_results(
        self, mock_pipeline, tmp_path
    ):
        pipeline, mock_gen = self._sparse_pipeline(
            mock_pipeline, tmp_path, "sparse"
        )
        sparse = MagicMock()
        sparse.retrieve.return_value = []
        pipeline.bm25_retriever = sparse

        response = pipeline.answer("unrelated?")

        assert response.answer == LOW_CONFIDENCE_ANSWER
        mock_gen.generate.assert_not_called()

    def test_is_low_confidence_sparse_mode_with_nonzero_scores(
        self, mock_pipeline, tmp_path
    ):
        pipeline, _ = self._sparse_pipeline(mock_pipeline, tmp_path, "sparse")
        results = [
            RetrievalResult(
                chunk_id="c1", text="t", score=0.05,
                source="", title="", section="",
            )
        ]
        assert pipeline._is_low_confidence(results) is False

    def test_is_low_confidence_dense_mode_below_threshold(
        self, mock_pipeline, tmp_path
    ):
        """Regression: the dense-mode cosine threshold still fires after the
        retrieval_mode guard was added."""
        pipeline, _ = self._sparse_pipeline(mock_pipeline, tmp_path, "dense")
        results = [
            RetrievalResult(
                chunk_id="c1", text="t", score=0.30,
                source="", title="", section="",
            ),
            RetrievalResult(
                chunk_id="c2", text="t", score=0.30,
                source="", title="", section="",
            ),
        ]
        assert pipeline._is_low_confidence(results) is True


class TestRAGPipelineSetupBranches:
    """Setup() must wire sparse, hybrid, and rerank components when the
    config asks for them. Covers pipeline.py:135-144."""

    def _make_pipeline(self, tmp_path, **overrides):
        config = RAGConfig(
            chunks_path=tmp_path / "chunks.jsonl",
            index_dir=tmp_path / "index",
            **overrides,
        )
        with patch("rag_pipeline.pipeline.ChunkEncoder"):
            pipeline = RAGPipeline(config)
        mock_index = MagicMock()
        mock_index.ntotal = 1
        pipeline.index_builder = MagicMock()
        pipeline.index_builder.load.return_value = (
            mock_index,
            [{"id": "c1", "text": "t", "source": "", "title": "T", "section": "S"}],
        )
        return pipeline

    def test_setup_initialises_sparse_retriever(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path, retrieval_mode="sparse")
        with patch("rag_pipeline.pipeline.BM25Retriever") as MockBM25, \
             patch("rag_pipeline.pipeline.RAGPipeline._load_generator"):
            pipeline.setup()
        MockBM25.assert_called_once()
        assert pipeline.bm25_retriever is not None

    def test_setup_initialises_hybrid_retriever(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path, retrieval_mode="hybrid")
        with patch("rag_pipeline.pipeline.BM25Retriever") as MockBM25, \
             patch("rag_pipeline.pipeline.HybridRetriever") as MockHybrid, \
             patch("rag_pipeline.pipeline.RAGPipeline._load_generator"):
            pipeline.setup()
        MockBM25.assert_called_once()
        MockHybrid.assert_called_once()
        assert pipeline.hybrid_retriever is not None

    def test_setup_initialises_reranker(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path, enable_reranking=True)
        with patch("rag_pipeline.pipeline.CrossEncoderReranker") as MockRerank, \
             patch("rag_pipeline.pipeline.RAGPipeline._load_generator"):
            pipeline.setup()
        MockRerank.assert_called_once_with(pipeline.config.reranker_model)
        assert pipeline.reranker is not None

    def test_setup_skips_reranker_when_disabled(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path, enable_reranking=False)
        with patch("rag_pipeline.pipeline.CrossEncoderReranker") as MockRerank, \
             patch("rag_pipeline.pipeline.RAGPipeline._load_generator"):
            pipeline.setup()
        MockRerank.assert_not_called()
        assert pipeline.reranker is None

    def test_load_generator_instantiates_ollama(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        with patch(
            "rag_pipeline.generation.ollama_generator.OllamaGenerator"
        ) as MockGen:
            mock_instance = MockGen.return_value
            pipeline._load_generator()
        MockGen.assert_called_once_with(pipeline.config)
        mock_instance.load_model.assert_called_once()
        assert pipeline.generator is mock_instance


class TestGeneratorFailureHandling:
    """pipeline.answer() must return a defined RAGResponse with
    GENERATION_ERROR_ANSWER when the Ollama generator raises RuntimeError,
    never bubble a raw exception into the Flask request handler."""

    def test_answer_returns_error_response_on_generator_failure(
        self, mock_pipeline
    ):
        pipeline, mock_retriever, mock_gen, _, _ = mock_pipeline
        pipeline.retriever = mock_retriever
        pipeline.generator = mock_gen
        pipeline.reranker = None

        mock_retriever.retrieve.return_value = [
            RetrievalResult(
                chunk_id="c1", text="t", score=0.85,
                source="url", title="T", section="S",
            )
        ]
        mock_gen.generate.side_effect = RuntimeError("ollama down")

        response = pipeline.answer("attendance?")

        assert response.answer == GENERATION_ERROR_ANSWER
        assert response.citations == []
        assert [s.chunk_id for s in response.sources] == ["c1"]
        # Timing fields populated even on the error path so metrics remain
        # comparable with the happy path. Types + sanity bounds are asserted
        # rather than the tautological >= 0.
        assert isinstance(response.retrieval_time_ms, float)
        assert 0.0 <= response.retrieval_time_ms < 5000.0
        assert response.rerank_time_ms == 0.0  # reranker disabled in this test
        assert isinstance(response.generation_time_ms, float)
        assert 0.0 <= response.generation_time_ms < 5000.0

    def test_answer_reranker_exception_bubbles_by_design(self, mock_pipeline):
        """A reranker crash is a fatal bug, not an operator-visible event
        like Ollama going down. The pipeline deliberately does NOT wrap
        the reranker call, so the exception surfaces for debugging."""
        pipeline, mock_retriever, mock_gen, _, _ = mock_pipeline
        pipeline.retriever = mock_retriever
        pipeline.generator = mock_gen

        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = RuntimeError("cross-encoder crashed")
        pipeline.reranker = mock_reranker

        mock_retriever.retrieve.return_value = [
            RetrievalResult(
                chunk_id="c1", text="t", score=0.85,
                source="", title="", section="",
            )
        ]

        with pytest.raises(RuntimeError, match="cross-encoder crashed"):
            pipeline.answer("question?")


class TestReloadGeneratorBackendGuard:
    """With the inference_backend Literal making non-Ollama backends
    impossible by construction, reload_generator always takes the
    ollama branch. The construction-side guard is covered in
    test_config.py; this class only exercises the happy-path reload.
    """

    def test_reload_generator_on_ollama_backend_calls_load_generator(
        self, mock_pipeline
    ):
        pipeline, _, _, _, _ = mock_pipeline
        assert pipeline.config.inference_backend == "ollama"

        with patch.object(pipeline, "_load_generator") as mock_load:
            pipeline.reload_generator("llama3.2")

        assert pipeline.config.ollama_model == "llama3.2"
        mock_load.assert_called_once()

    def test_reload_generator_raises_on_non_ollama_backend(self, mock_pipeline):
        """Defence-in-depth: RAGConfig's Literal + validate_assignment=True
        already reject inference_backend != "ollama" at both construction
        and assignment time, so the only way to reach the guard inside
        reload_generator is to bypass Pydantic validation entirely. We
        simulate that here with object.__setattr__ to prove the guard
        still fires if future refactors introduce a non-validated mutation
        path (e.g. model_construct-based factories)."""
        pipeline, _, _, _, _ = mock_pipeline
        object.__setattr__(pipeline.config, "inference_backend", "huggingface")
        with pytest.raises(ValueError, match="only supported with the ollama"):
            pipeline.reload_generator("whatever")
