"""Compare different embedding models on retrieval quality."""

import logging
import time

import pandas as pd

from evaluation.metrics.bootstrap import add_ci_columns
from evaluation.metrics.retrieval_metrics import evaluate_retrieval, sections_to_chunk_ids
from rag_pipeline.config import RAGConfig
from rag_pipeline.embeddings.encoder import ChunkEncoder
from rag_pipeline.embeddings.index_builder import FAISSIndexBuilder
from rag_pipeline.retrieval.query_processor import QueryProcessor
from rag_pipeline.retrieval.retriever import FAISSRetriever

logger = logging.getLogger(__name__)

EMBEDDING_MODELS = [
    {"name": "all-MiniLM-L6-v2", "dim": 384},
    {"name": "all-mpnet-base-v2", "dim": 768},
    {"name": "multi-qa-MiniLM-L6-cos-v1", "dim": 384},
]


def run_embedding_comparison(
    qa_pairs: list[dict],
    config: RAGConfig,
    models: list[dict] | None = None,
) -> pd.DataFrame:
    """Compare retrieval quality across embedding models.

    Args:
        qa_pairs: List of QA pair dicts with question, relevant_sections.
        config: RAGConfig with chunks_path set.
        models: List of model dicts with 'name' and 'dim' keys.

    Returns:
        DataFrame with columns: model, encoding_time_s, and retrieval metrics.
    """
    models = models or EMBEDDING_MODELS
    qp = QueryProcessor()
    results = []

    for model_info in models:
        model_name = model_info["name"]
        logger.info("Evaluating embedding model: %s", model_name)

        # Encode the chunks with this model.
        encoder = ChunkEncoder(model_name)
        chunks = encoder.load_chunks(config.chunks_path)

        t0 = time.perf_counter()
        embeddings = encoder.encode_chunks(chunks)
        encoding_time = time.perf_counter() - t0

        # Build the FAISS index from the encoded embeddings.
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
        builder = FAISSIndexBuilder()
        index = builder.build_index(embeddings, metadata)
        retriever = FAISSRetriever(index, metadata, config)

        # Evaluate retrieval on every QA pair.
        all_metrics = []
        for qa in qa_pairs:
            question = qa["question"].strip()
            if not question:
                continue

            processed = qp.process(question)
            query_emb = encoder.encode_query(processed)
            retrieved = retriever.retrieve(query_emb, top_k=config.top_k)

            retrieved_ids = [r.chunk_id for r in retrieved]
            relevant_sections = qa.get("relevant_sections", [])
            relevant_ids = sections_to_chunk_ids(chunks, relevant_sections)

            metrics = evaluate_retrieval(retrieved_ids, relevant_ids)
            all_metrics.append(metrics)

        # Average the metrics across queries and attach a bootstrap
        # confidence interval to each.
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0]:
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        row = {
            "model": model_name,
            "n_queries": len(all_metrics),
            "encoding_time_s": round(encoding_time, 2),
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
