"""Re-run the retrieval-only comparison experiments (retrieval,
reranking) with per-query metric emission so
``significance_tests.py`` can compute paired Wilcoxon + Cohen's d +
Holm-Bonferroni across every pair of systems within each experiment.

The aggregate comparison scripts under this directory compute their
bootstrap CIs from per-query vectors already, but they do not
persist those vectors. This script reuses the same pipeline code
(FAISS retriever, BM25 retriever, hybrid retriever, cross-encoder
reranker) so the emitted per-query metrics come from the same code
paths that produced the baseline numerics.

LLM-based per-query data would also require running llm_comparison
across all four Ollama models; that run is expensive and kicked off
separately. The retrieval-only experiments fit in a few seconds and
are enough to populate significance tests for the retrieval and
reranking ablations.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from evaluation.metrics.retrieval_metrics import (
    evaluate_retrieval,
    sections_to_chunk_ids,
)
from rag_pipeline.config import RAGConfig
from rag_pipeline.embeddings.encoder import ChunkEncoder
from rag_pipeline.embeddings.index_builder import FAISSIndexBuilder
from rag_pipeline.models import RetrievalResult
from rag_pipeline.retrieval.bm25_retriever import BM25Retriever
from rag_pipeline.retrieval.hybrid_retriever import HybridRetriever
from rag_pipeline.retrieval.query_processor import QueryProcessor
from rag_pipeline.retrieval.reranker import CrossEncoderReranker
from rag_pipeline.retrieval.retriever import FAISSRetriever

logger = logging.getLogger(__name__)

DEFAULT_PER_QUERY_DIR = Path("evaluation/results/per_query")


def _qa_id(qa: dict[str, Any], fallback_index: int) -> str:
    """Return the QA pair's id, falling back to a positional id.

    The bundled qa_pairs.json carries explicit ids per pair; the
    positional fallback is a defensive guard so the emitted JSON is
    still well-formed if a future test set drops the id column.
    """
    return str(qa.get("id") or f"qa_{fallback_index:03d}")


def _per_query_dense_vs_sparse_vs_hybrid(
    qa_pairs: list[dict[str, Any]],
    config: RAGConfig,
) -> list[dict[str, Any]]:
    """Per-query metrics for dense, BM25 and hybrid RRF retrieval.

    Each returned row is shaped for the significance_tests harness:
    experiment, system, id and the per-query retrieval metrics. Only
    non-empty questions are evaluated; empty questions would confuse
    the retrievers and are dropped symmetrically across systems.
    """
    qp = QueryProcessor()

    encoder = ChunkEncoder(config.embedding_model)
    chunks = encoder.load_chunks(config.chunks_path)
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

    logger.info("Building dense index")
    embeddings = encoder.encode_chunks(chunks)
    builder = FAISSIndexBuilder()
    index = builder.build_index(embeddings, metadata)
    dense = FAISSRetriever(index, metadata, config)

    logger.info("Building BM25 index")
    bm25 = BM25Retriever(metadata, config)
    hybrid = HybridRetriever(dense=dense, sparse=bm25, k_rrf=config.rrf_k)

    strategies = {
        "dense_faiss": lambda q, q_emb: dense.retrieve(q_emb, top_k=config.top_k),
        "sparse_bm25": lambda q, q_emb: bm25.retrieve(q, top_k=config.top_k),
        "hybrid_rrf": lambda q, q_emb: hybrid.retrieve(q, q_emb, top_k=config.top_k),
    }

    out: list[dict[str, Any]] = []
    for strategy, retrieve in strategies.items():
        logger.info("Evaluating strategy=%s", strategy)
        for i, qa in enumerate(qa_pairs):
            question = qa["question"].strip()
            if not question:
                continue
            processed = qp.process(question)
            query_emb = encoder.encode_query(processed)
            retrieved = retrieve(processed, query_emb)
            retrieved_ids = [r.chunk_id for r in retrieved]
            relevant_ids = sections_to_chunk_ids(
                config.chunks_path, qa.get("relevant_sections", [])
            )
            metrics = evaluate_retrieval(retrieved_ids, relevant_ids)
            out.append(
                {
                    "experiment": "retrieval_comparison",
                    "system": strategy,
                    "id": _qa_id(qa, i),
                    **metrics,
                }
            )
    return out


def _per_query_reranking(
    qa_pairs: list[dict[str, Any]],
    config: RAGConfig,
) -> list[dict[str, Any]]:
    """Per-query metrics for dense_only, dense_rerank, bm25_rerank.

    The rerank paths reuse the production ``CrossEncoderReranker``
    so the numbers are identical to those the aggregate
    reranking_comparison script would produce.
    """
    qp = QueryProcessor()
    encoder = ChunkEncoder(config.embedding_model)
    chunks = encoder.load_chunks(config.chunks_path)
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

    logger.info("Building dense index (rerank run)")
    embeddings = encoder.encode_chunks(chunks)
    builder = FAISSIndexBuilder()
    index = builder.build_index(embeddings, metadata)
    dense = FAISSRetriever(index, metadata, config)
    bm25 = BM25Retriever(metadata, config)
    reranker = CrossEncoderReranker(config.reranker_model)

    fetch_k = config.rerank_fetch_k
    top_k = config.top_k

    def run_dense_only(processed: str, query_emb: Any) -> list[RetrievalResult]:
        return dense.retrieve(query_emb, top_k=top_k)

    def run_dense_rerank(processed: str, query_emb: Any) -> list[RetrievalResult]:
        candidates = dense.retrieve(query_emb, top_k=fetch_k)
        return reranker.rerank(processed, candidates, top_k=top_k)

    def run_bm25_rerank(processed: str, query_emb: Any) -> list[RetrievalResult]:
        candidates = bm25.retrieve(processed, top_k=fetch_k)
        return reranker.rerank(processed, candidates, top_k=top_k)

    strategies = {
        "dense_only": run_dense_only,
        "dense_rerank": run_dense_rerank,
        "bm25_rerank": run_bm25_rerank,
    }

    out: list[dict[str, Any]] = []
    for strategy, run in strategies.items():
        logger.info("Evaluating reranking strategy=%s", strategy)
        for i, qa in enumerate(qa_pairs):
            question = qa["question"].strip()
            if not question:
                continue
            processed = qp.process(question)
            query_emb = encoder.encode_query(processed)
            retrieved = run(processed, query_emb)
            retrieved_ids = [r.chunk_id for r in retrieved]
            relevant_ids = sections_to_chunk_ids(
                config.chunks_path, qa.get("relevant_sections", [])
            )
            metrics = evaluate_retrieval(retrieved_ids, relevant_ids)
            out.append(
                {
                    "experiment": "reranking_comparison",
                    "system": strategy,
                    "id": _qa_id(qa, i),
                    **metrics,
                }
            )
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-pairs",
        type=Path,
        default=Path("evaluation/test_set/qa_pairs.json"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_PER_QUERY_DIR
    )
    parser.add_argument(
        "--skip-retrieval", action="store_true",
        help="Skip the dense/sparse/hybrid retrieval experiment.",
    )
    parser.add_argument(
        "--skip-reranking", action="store_true",
        help="Skip the dense_only/dense_rerank/bm25_rerank reranking experiment.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    with args.qa_pairs.open() as f:
        qa_pairs = json.load(f)

    config = RAGConfig()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_retrieval:
        t0 = time.perf_counter()
        rows = _per_query_dense_vs_sparse_vs_hybrid(qa_pairs, config)
        out_path = args.out_dir / "retrieval_comparison.json"
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        logger.info(
            "Wrote %d rows to %s in %.1fs",
            len(rows), out_path, time.perf_counter() - t0,
        )

    if not args.skip_reranking:
        t0 = time.perf_counter()
        rows = _per_query_reranking(qa_pairs, config)
        out_path = args.out_dir / "reranking_comparison.json"
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        logger.info(
            "Wrote %d rows to %s in %.1fs",
            len(rows), out_path, time.perf_counter() - t0,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
