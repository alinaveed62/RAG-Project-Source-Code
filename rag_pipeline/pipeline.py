"""End-to-end RAG pipeline that ties encoding, retrieval and generation together.

The default dense-retrieval flow is:

    query expansion, encode, FAISS bi-encoder retrieve (over-fetch
    when the reranker is enabled), low-confidence check on the
    bi-encoder cosine scores, optional cross-encoder rerank,
    citation-aware prompt, Ollama generation, citation parsing,
    response.

The same answer() method supports two alternative retrieval modes
via RAGConfig.retrieval_mode: "sparse" routes through BM25Retriever
only, and "hybrid" fuses dense and sparse results with HybridRetriever
using reciprocal rank fusion.

The low-confidence refusal is two-stage on the dense path. First a
per-result similarity_threshold filter inside the retriever drops
individual weak matches. Then an aggregate mean-score gate here
compares against low_confidence_threshold (0.4). The aggregate gate
is tied to the bi-encoder cosine scale in [0, 1]; BM25 scores are
unbounded non-negative and RRF fused scores peak near
2 / (k_rrf + 1) which is about 0.033, so the aggregate gate is
skipped for sparse and hybrid modes. The empty-results refusal
still fires for every mode. Cross-encoder rerank logits are in a
different range again (roughly -10 to +15), which is why the rerank
step always runs after the refusal gate, never before.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rag_pipeline.config import RAGConfig
from rag_pipeline.embeddings.encoder import ChunkEncoder
from rag_pipeline.embeddings.index_builder import FAISSIndexBuilder
from rag_pipeline.generation.citation_parser import Citation, parse_citations
from rag_pipeline.generation.prompt_templates import build_rag_prompt
from rag_pipeline.models import RAGResponse, RetrievalResult
from rag_pipeline.retrieval.bm25_retriever import BM25Retriever
from rag_pipeline.retrieval.hybrid_retriever import HybridRetriever
from rag_pipeline.retrieval.query_processor import QueryProcessor
from rag_pipeline.retrieval.reranker import CrossEncoderReranker
from rag_pipeline.retrieval.retriever import FAISSRetriever

if TYPE_CHECKING:
    from rag_pipeline.generation.ollama_generator import OllamaGenerator

logger = logging.getLogger(__name__)

# Refusal message returned when retrieval confidence is too low.
# Module-level constant so the evaluation scripts can compare against
# a stable string to exclude refusals from answer-quality metrics.
LOW_CONFIDENCE_ANSWER = (
    "I don't have enough reliable information from the Student Handbook "
    "to answer this question confidently. Please check the handbook directly "
    "on KEATS or contact the Department of Informatics for guidance."
)

# User-facing message returned when the language model fails
# mid-request. Kept distinct from LOW_CONFIDENCE_ANSWER so downstream
# code (the evaluator, the Flask UI) can tell a retrieval refusal
# apart from a generator error without parsing message text.
GENERATION_ERROR_ANSWER = (
    "An error occurred while generating an answer. The retrieved sources "
    "below may still be useful; please try again in a moment or consult "
    "the Student Handbook directly on KEATS."
)


class RAGPipeline:
    """Full RAG pipeline: query processing, retrieval, optional rerank, generation."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.encoder = ChunkEncoder(config.embedding_model)
        self.index_builder = FAISSIndexBuilder(expected_dim=config.embedding_dim)
        self.retriever: FAISSRetriever | None = None
        self.bm25_retriever: BM25Retriever | None = None
        self.hybrid_retriever: HybridRetriever | None = None
        self.reranker: CrossEncoderReranker | None = None
        self.generator: OllamaGenerator | None = None
        self.query_processor = QueryProcessor(
            enable_expansion=config.enable_query_expansion
        )

    def build_index(self, chunks_path: Path | None = None) -> None:
        """Encode all chunks and build a FAISS index.

        This is a one-time operation. The index is saved to config.index_dir.

        Args:
            chunks_path: Path to chunks_for_embedding.jsonl.
                Defaults to config.chunks_path.
        """
        chunks_path = chunks_path or self.config.chunks_path
        chunks = self.encoder.load_chunks(chunks_path)

        # Build the metadata list aligned with the FAISS row indices.
        # heading_path defaults to an empty list so chunk files produced
        # before the field was added still load.
        metadata = [
            {
                "id": c["id"],
                "text": c["text"],
                "source": c.get("source", ""),
                "title": c.get("title", ""),
                "section": c.get("section", ""),
                "heading_path": c.get("heading_path", []),
            }
            for c in chunks
        ]

        embeddings = self.encoder.encode_chunks(chunks)
        self.index_builder.build_index(embeddings, metadata)
        self.index_builder.save(self.config.index_dir)

        logger.info("Index built and saved to %s", self.config.index_dir)

    def setup(self) -> None:
        """Load a pre-built FAISS index and the LLM model.

        Call this once before calling answer().
        """
        index, metadata = self.index_builder.load(self.config.index_dir)
        self.retriever = FAISSRetriever(index, metadata, self.config)
        logger.info("Retriever ready with %d vectors", index.ntotal)

        # Build the alternative retrievers when the chosen mode needs them.
        if self.config.retrieval_mode in {"sparse", "hybrid"}:
            self.bm25_retriever = BM25Retriever(metadata, self.config)
        if self.config.retrieval_mode == "hybrid":
            assert self.bm25_retriever is not None  # noqa: S101 - type narrowing for mypy; set by the preceding if branch
            self.hybrid_retriever = HybridRetriever(
                dense=self.retriever,
                sparse=self.bm25_retriever,
                k_rrf=self.config.rrf_k,
            )
            logger.info("Hybrid retriever enabled (k_rrf=%d)", self.config.rrf_k)

        if self.config.enable_reranking:
            self.reranker = CrossEncoderReranker(self.config.reranker_model)
            logger.info("Reranker enabled (model=%s)", self.config.reranker_model)
        else:
            self.reranker = None

        self._load_generator()
        logger.info("Generator ready (backend=%s)", self.config.inference_backend)

    def _load_generator(self) -> None:
        """Construct the Ollama generator and load its model.

        Only Ollama is supported; RAGConfig validates
        inference_backend == "ollama" at construction time, so no
        other branch is reachable here.
        """
        from rag_pipeline.generation.ollama_generator import OllamaGenerator

        self.generator = OllamaGenerator(self.config)
        self.generator.load_model()

    def reload_generator(self, model_name: str) -> None:
        """Reload the generator with a different Ollama model.

        The LLM comparison experiment uses this to swap between
        models (mistral, llama3.2, phi3, gemma2) without rebuilding
        the whole pipeline. The RAGConfig.inference_backend Literal
        is enforced on both construction and assignment, so the check
        below is defence in depth; it also documents the invariant at
        the point where someone might accidentally add another branch.

        Args:
            model_name: Ollama model name (for example "gemma2:2b").

        Raises:
            ValueError: If the configured inference backend is not ollama.
        """
        if self.config.inference_backend != "ollama":
            raise ValueError(
                "reload_generator is only supported with the ollama backend; "
                f"current backend is {self.config.inference_backend!r}."
            )
        self.config.ollama_model = model_name
        self._load_generator()
        logger.info("Generator reloaded with model: %s", model_name)

    def _is_low_confidence(self, results: list[RetrievalResult]) -> bool:
        """Return True if the retrieval results are too weak to answer from.

        Empty results always count as low confidence, whatever the
        retrieval mode: "nothing retrieved" is the unambiguous signal.

        The aggregate mean-score threshold only applies in dense mode.
        Dense bi-encoder scores are cosine similarities in [0, 1]
        because sentence-transformers L2-normalises its embeddings,
        so config.low_confidence_threshold = 0.4 is directly
        interpretable on that scale. BM25 scores are unbounded
        non-negative, and RRF fused scores peak near
        2 / (config.rrf_k + 1) which is about 0.033; reusing the 0.4
        cosine threshold on either scale would make sparse never
        refuse and hybrid always refuse. Cross-encoder rerank logits
        are in yet another range (roughly -10 to +15), which is why
        the rerank step runs after this gate on the dense path.

        The dense gate is two-stage. The retriever first drops any
        per-result score below config.similarity_threshold. The
        results that survive that cut are averaged here, and the
        pipeline refuses when the average is below
        config.low_confidence_threshold. The per-result filter stops a
        single weak chunk from entering the prompt; the aggregate
        filter rejects queries whose entire top-k is only marginally
        relevant.

        For sparse and hybrid modes the aggregate gate is skipped:
        those retrievers already drop zero-score matches, so a
        non-empty list is itself the confidence signal.
        """
        if not results:
            return True
        if self.config.retrieval_mode != "dense":
            return False
        avg_score = sum(r.score for r in results) / len(results)
        return avg_score < self.config.low_confidence_threshold

    def answer(
        self,
        question: str,
        section_filter: str | None = None,
    ) -> RAGResponse:
        """Answer a question using the full RAG pipeline.

        Args:
            question: The user's question.
            section_filter: Optional section name to restrict retrieval.

        Returns:
            RAGResponse with answer, sources, citations, and timing info.
        """
        if self.retriever is None or self.generator is None:
            raise RuntimeError("Pipeline not set up. Call setup() first.")

        # Process query
        processed_query = self.query_processor.process(question)

        # Retrieve. When the reranker is enabled we over-fetch a wider
        # candidate set so it has room to reorder; the refusal gate
        # below still runs on the bi-encoder scores from this pass.
        t0 = time.perf_counter()
        query_embedding = self.encoder.encode_query(processed_query)
        fetch_k = self.config.rerank_fetch_k if self.reranker is not None else None

        if self.config.retrieval_mode == "sparse" and self.bm25_retriever is not None:
            bi_results = self.bm25_retriever.retrieve(
                processed_query, top_k=fetch_k, section_filter=section_filter
            )
        elif self.config.retrieval_mode == "hybrid" and self.hybrid_retriever is not None:
            bi_results = self.hybrid_retriever.retrieve(
                processed_query,
                query_embedding=query_embedding,
                top_k=fetch_k,
                section_filter=section_filter,
            )
        else:
            bi_results = self.retriever.retrieve(
                query_embedding,
                top_k=fetch_k,
                section_filter=section_filter,
            )
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # If retrieval confidence is too low, return the refusal
        # message rather than hallucinating. This check always looks
        # at the bi-encoder cosine scores in [0, 1], never the
        # unbounded cross-encoder logits.
        if self._is_low_confidence(bi_results):
            logger.info(
                "Low confidence for query '%s', returning refusal", question[:50]
            )
            return RAGResponse(
                question=question,
                answer=LOW_CONFIDENCE_ANSWER,
                sources=bi_results[: self.config.top_k],
                citations=[],
                retrieval_time_ms=round(retrieval_ms, 2),
                rerank_time_ms=0.0,
                generation_time_ms=0.0,
            )

        # Optional cross-encoder rerank, then truncate to config.top_k.
        rerank_ms = 0.0
        if self.reranker is not None:
            t1 = time.perf_counter()
            results = self.reranker.rerank(
                processed_query, bi_results, top_k=self.config.top_k
            )
            rerank_ms = (time.perf_counter() - t1) * 1000
        else:
            results = bi_results[: self.config.top_k]

        # Generate. Wrap in a narrow try/except so an Ollama failure
        # returns a defined RAGResponse (carrying GENERATION_ERROR_ANSWER)
        # rather than crashing the Flask request. OllamaGenerator
        # raises RuntimeError for all of its failure modes, so that is
        # the only type caught here; KeyboardInterrupt and unexpected
        # bugs continue to propagate.
        t2 = time.perf_counter()
        prompt = build_rag_prompt(
            question,
            results,
            enable_citation_injection=self.config.enable_citation_injection,
        )
        try:
            answer_text = self.generator.generate(prompt)
        except RuntimeError:
            logger.exception(
                "Generation failed for query %r; returning error response",
                question[:50],
            )
            generation_ms = (time.perf_counter() - t2) * 1000
            return RAGResponse(
                question=question,
                answer=GENERATION_ERROR_ANSWER,
                sources=results,
                citations=[],
                retrieval_time_ms=round(retrieval_ms, 2),
                rerank_time_ms=round(rerank_ms, 2),
                generation_time_ms=round(generation_ms, 2),
            )
        generation_ms = (time.perf_counter() - t2) * 1000

        # Parse and validate the inline citations, dropping any that
        # reference chunk IDs the pipeline did not actually retrieve.
        citations: list[Citation] = []
        if self.config.enable_citation_injection:
            citations = parse_citations(
                answer_text, valid_chunk_ids=[r.chunk_id for r in results]
            )

        return RAGResponse(
            question=question,
            answer=answer_text,
            sources=results,
            citations=citations,
            retrieval_time_ms=round(retrieval_ms, 2),
            rerank_time_ms=round(rerank_ms, 2),
            generation_time_ms=round(generation_ms, 2),
        )
