"""Compare dense (FAISS), sparse (BM25) and RRF-hybrid retrieval.

The RRF hybrid logic lives in the production HybridRetriever class,
and this experiment calls into it, so the experiment exercises the
same code path that would ship in a hybrid deployment.
"""

import logging
import time

import pandas as pd

from evaluation.metrics.bootstrap import add_ci_columns
from evaluation.metrics.retrieval_metrics import evaluate_retrieval, sections_to_chunk_ids
from rag_pipeline.config import RAGConfig
from rag_pipeline.embeddings.encoder import ChunkEncoder
from rag_pipeline.embeddings.index_builder import FAISSIndexBuilder
from rag_pipeline.retrieval.bm25_retriever import BM25Retriever
from rag_pipeline.retrieval.hybrid_retriever import HybridRetriever
from rag_pipeline.retrieval.query_processor import QueryProcessor
from rag_pipeline.retrieval.retriever import FAISSRetriever

logger = logging.getLogger(__name__)


def run_retrieval_comparison(
    qa_pairs: list[dict],
    config: RAGConfig,
) -> pd.DataFrame:
    """Compare dense, sparse and hybrid retrieval on the QA test set.

    Args:
        qa_pairs: QA pair dicts with question and relevant_sections.
        config: RAGConfig with chunks_path set.

    Returns:
        DataFrame with one row per strategy and its retrieval metrics.
    """
    qp = QueryProcessor()

    # Load the chunks.
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

    # Build the dense (FAISS) index.
    logger.info("Building dense (FAISS) index ...")
    embeddings = encoder.encode_chunks(chunks)
    builder = FAISSIndexBuilder()
    index = builder.build_index(embeddings, metadata)
    dense_retriever = FAISSRetriever(index, metadata, config)

    # Build the sparse (BM25) index.
    logger.info("Building sparse (BM25) index ...")
    bm25_retriever = BM25Retriever(metadata, config)

    hybrid_retriever = HybridRetriever(
        dense=dense_retriever, sparse=bm25_retriever, k_rrf=config.rrf_k
    )

    strategies = {
        "dense_faiss": lambda q, q_emb: dense_retriever.retrieve(q_emb, top_k=config.top_k),
        "sparse_bm25": lambda q, q_emb: bm25_retriever.retrieve(q, top_k=config.top_k),
        "hybrid_rrf": lambda q, q_emb: hybrid_retriever.retrieve(
            q, q_emb, top_k=config.top_k
        ),
    }

    results = []

    for strategy_name, retrieve_fn in strategies.items():
        logger.info("Evaluating retrieval strategy: %s", strategy_name)

        all_metrics = []
        total_time = 0.0

        for qa in qa_pairs:
            question = qa["question"].strip()
            if not question:
                continue

            processed = qp.process(question)
            query_emb = encoder.encode_query(processed)

            t0 = time.perf_counter()
            retrieved = retrieve_fn(processed, query_emb)
            total_time += time.perf_counter() - t0

            retrieved_ids = [r.chunk_id for r in retrieved]
            relevant_sections = qa.get("relevant_sections", [])
            relevant_ids = sections_to_chunk_ids(config.chunks_path, relevant_sections)

            metrics = evaluate_retrieval(retrieved_ids, relevant_ids)
            all_metrics.append(metrics)

        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0]:
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        n_queries = len(all_metrics) or 1
        row: dict[str, float | int | str] = {
            "strategy": strategy_name,
            "n_queries": len(all_metrics),
            "avg_retrieval_ms": round((total_time / n_queries) * 1000, 2),
            **avg_metrics,
        }
        if all_metrics:
            for metric_name in all_metrics[0]:
                add_ci_columns(
                    metric_name,
                    [m[metric_name] for m in all_metrics],
                    row,
                )
        results.append(row)

    return pd.DataFrame(results)
