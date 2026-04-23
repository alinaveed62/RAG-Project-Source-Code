"""Cross-encoder reranking ablation experiment.

Compares three retrieval and rerank strategies on the full 50-pair
QA test set:

  dense_only:   bi-encoder retrieval, top-k = 5 (baseline).
  dense_rerank: bi-encoder retrieval top-20, cross-encoder rerank to
                top-5.
  bm25_rerank:  BM25 retrieval top-20, cross-encoder rerank to top-5
                (a sanity check that the rerank helps sparse retrieval
                too).

Writes evaluation/results/reranking_comparison.csv with the mean
metrics and the per-query metric vectors that feed the bootstrap
confidence interval computation downstream.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from evaluation.metrics.bootstrap import add_ci_columns
from evaluation.metrics.retrieval_metrics import (
    evaluate_retrieval,
    sections_to_chunk_ids,
)
from rag_pipeline.config import RAGConfig
from rag_pipeline.embeddings.encoder import ChunkEncoder
from rag_pipeline.embeddings.index_builder import FAISSIndexBuilder
from rag_pipeline.models import RetrievalResult
from rag_pipeline.retrieval.bm25_retriever import BM25Retriever
from rag_pipeline.retrieval.query_processor import QueryProcessor
from rag_pipeline.retrieval.reranker import CrossEncoderReranker
from rag_pipeline.retrieval.retriever import FAISSRetriever

logger = logging.getLogger(__name__)


def _retrieve_dense_only(
    dense: FAISSRetriever,
    query_emb: Any,
    top_k: int,
) -> list[RetrievalResult]:
    return dense.retrieve(query_emb, top_k=top_k)


def _retrieve_dense_rerank(
    dense: FAISSRetriever,
    reranker: CrossEncoderReranker,
    processed_query: str,
    query_emb: Any,
    fetch_k: int,
    top_k: int,
) -> tuple[list[RetrievalResult], float]:
    bi_results = dense.retrieve(query_emb, top_k=fetch_k)
    t0 = time.perf_counter()
    reranked = reranker.rerank(processed_query, bi_results, top_k=top_k)
    rerank_ms = (time.perf_counter() - t0) * 1000.0
    return reranked, rerank_ms


def _retrieve_bm25_rerank(
    bm25: BM25Retriever,
    reranker: CrossEncoderReranker,
    processed_query: str,
    fetch_k: int,
    top_k: int,
) -> tuple[list[RetrievalResult], float]:
    bm25_results = bm25.retrieve(processed_query, top_k=fetch_k)
    t0 = time.perf_counter()
    reranked = reranker.rerank(processed_query, bm25_results, top_k=top_k)
    rerank_ms = (time.perf_counter() - t0) * 1000.0
    return reranked, rerank_ms


def run_reranking_comparison(
    qa_pairs: list[dict[str, Any]],
    config: RAGConfig,
) -> pd.DataFrame:
    """Run the three-strategy reranking ablation.

    Args:
        qa_pairs: Evaluation QA pairs with question and
            relevant_sections keys.
        config: RAGConfig with chunks_path set. rerank_fetch_k is the
            bi-encoder over-fetch window for the rerank strategies,
            and top_k is the final top-k.

    Returns:
        DataFrame with one row per strategy. Columns include each
        metric's mean, bootstrap CI bounds, and the average per-query
        retrieval and rerank latency.
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

    logger.info("Building dense (FAISS) index ...")
    embeddings = encoder.encode_chunks(chunks)
    builder = FAISSIndexBuilder()
    index = builder.build_index(embeddings, metadata)
    dense = FAISSRetriever(index, metadata, config)

    logger.info("Building sparse (BM25) index ...")
    bm25 = BM25Retriever(metadata, config)

    logger.info("Loading cross-encoder reranker (lazy) ...")
    reranker = CrossEncoderReranker(config.reranker_model)

    fetch_k = config.rerank_fetch_k
    top_k = config.top_k

    rows: list[dict[str, Any]] = []

    for strategy_name in ("dense_only", "dense_rerank", "bm25_rerank"):
        logger.info("Evaluating strategy: %s", strategy_name)
        per_query_metrics: list[dict[str, float]] = []
        total_retrieval_s = 0.0
        total_rerank_ms = 0.0

        for qa in qa_pairs:
            question = qa["question"].strip()
            if not question:
                continue
            processed = qp.process(question)
            query_emb = encoder.encode_query(processed)

            t0 = time.perf_counter()
            if strategy_name == "dense_only":
                retrieved = _retrieve_dense_only(dense, query_emb, top_k)
                rerank_ms = 0.0
            elif strategy_name == "dense_rerank":
                retrieved, rerank_ms = _retrieve_dense_rerank(
                    dense, reranker, processed, query_emb, fetch_k, top_k
                )
            else:  # bm25_rerank, the last of the hardcoded strategies
                retrieved, rerank_ms = _retrieve_bm25_rerank(
                    bm25, reranker, processed, fetch_k, top_k
                )
            total_retrieval_s += time.perf_counter() - t0
            total_rerank_ms += rerank_ms

            retrieved_ids = [r.chunk_id for r in retrieved]
            relevant_sections = qa.get("relevant_sections", [])
            relevant_ids = sections_to_chunk_ids(config.chunks_path, relevant_sections)

            per_query_metrics.append(evaluate_retrieval(retrieved_ids, relevant_ids))

        if not per_query_metrics:
            continue

        n = len(per_query_metrics)
        avg = {key: sum(m[key] for m in per_query_metrics) / n for key in per_query_metrics[0]}
        row: dict[str, Any] = {
            "strategy": strategy_name,
            "n_queries": n,
            "avg_retrieval_ms": round((total_retrieval_s / n) * 1000.0, 2),
            "avg_rerank_ms": round(total_rerank_ms / n, 2),
            **avg,
        }
        # Attach a bootstrap CI to every numeric metric so the
        # LaTeX-table renderer can include the CI bounds.
        for metric_name in per_query_metrics[0]:
            add_ci_columns(
                metric_name,
                [m[metric_name] for m in per_query_metrics],
                row,
            )
        rows.append(row)

    return pd.DataFrame(rows)
