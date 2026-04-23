"""Compare different chunk sizes on retrieval and answer quality."""

import json
import logging
from pathlib import Path

import pandas as pd

from evaluation.metrics.bootstrap import add_ci_columns
from evaluation.metrics.retrieval_metrics import evaluate_retrieval, sections_to_chunk_ids
from keats_scraper.config import ChunkConfig
from keats_scraper.models.document import Document
from keats_scraper.processors.chunker import Chunker
from rag_pipeline.config import RAGConfig
from rag_pipeline.embeddings.encoder import ChunkEncoder
from rag_pipeline.embeddings.index_builder import FAISSIndexBuilder
from rag_pipeline.retrieval.query_processor import QueryProcessor
from rag_pipeline.retrieval.retriever import FAISSRetriever

logger = logging.getLogger(__name__)

CHUNK_SIZES = [256, 512, 1024]


def run_chunk_size_comparison(
    qa_pairs: list[dict],
    documents_path: Path,
    config: RAGConfig,
    chunk_sizes: list[int] | None = None,
) -> pd.DataFrame:
    """Compare retrieval quality across different chunk sizes.

    Re-chunks the source documents at each target size using the
    production Chunker, then builds a fresh FAISS index and
    evaluates retrieval quality. The embedding model and retriever
    config stay fixed, so only the chunk size varies.

    Args:
        qa_pairs: QA pair dicts.
        documents_path: Path to documents.jsonl.
        config: RAGConfig.
        chunk_sizes: List of chunk sizes in tokens.

    Returns:
        DataFrame with chunk_size, num_chunks and retrieval metrics.
    """
    chunk_sizes = chunk_sizes or CHUNK_SIZES
    qp = QueryProcessor()

    results = []

    for size in chunk_sizes:
        logger.info("Evaluating chunk size: %d tokens", size)
        overlap = int(size * 0.1)  # use a 10% overlap ratio

        chunk_config = ChunkConfig(chunk_size=size, chunk_overlap=overlap)
        chunker = Chunker(config=chunk_config)

        documents = []
        with open(documents_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    documents.append(json.loads(line))

        all_chunks = []
        for doc_data in documents:
            doc = Document(**doc_data)
            chunks = chunker.chunk_document(doc)
            for chunk in chunks:
                all_chunks.append(
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "source": chunk.metadata.source_url,
                        "title": chunk.metadata.document_title,
                        "section": chunk.metadata.section,
                        "heading_path": list(
                            getattr(chunk.metadata, "heading_path", []) or []
                        ),
                    }
                )

        logger.info("Generated %d chunks at size=%d", len(all_chunks), size)

        # Encode the re-chunked corpus and build a fresh index.
        encoder = ChunkEncoder(config.embedding_model)
        embeddings = encoder.encode_chunks(all_chunks)

        builder = FAISSIndexBuilder()
        index = builder.build_index(embeddings, all_chunks)
        retriever = FAISSRetriever(index, all_chunks, config)

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
            relevant_ids = sections_to_chunk_ids(all_chunks, relevant_sections)

            metrics = evaluate_retrieval(retrieved_ids, relevant_ids)
            all_metrics.append(metrics)

        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0]:
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        row: dict[str, float | int] = {
            "chunk_size": size,
            "chunk_overlap": overlap,
            "num_chunks": len(all_chunks),
            "n_queries": len(all_metrics),
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
