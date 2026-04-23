"""Per-category metric breakdown for the QA test set.

Splits the 50-pair test set by its nine category labels (attendance,
assessment, office hours, representation, programmes, support,
administrative, general, edge cases) and reports per-category MRR,
Precision@5, ROUGE-1 F1, BERTScore F1, faithfulness, SGF, refusal
rate and mean latency.

Presenting disaggregated results alongside the overall averages
shows where the system is strong (usually categories with
tightly-scoped single-section answers) and where it struggles
(multi-section or out-of-scope queries).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import pandas as pd

from evaluation.metrics.answer_metrics import evaluate_answer_quality
from evaluation.metrics.bootstrap import add_ci_columns
from evaluation.metrics.retrieval_metrics import (
    evaluate_retrieval,
    sections_to_chunk_ids,
)
from evaluation.metrics.sgf import section_grounded_faithfulness
from rag_pipeline.pipeline import LOW_CONFIDENCE_ANSWER, RAGPipeline

logger = logging.getLogger(__name__)


def run_per_category_breakdown(
    pipeline: RAGPipeline,
    qa_pairs: list[dict[str, Any]],
    compute_bert: bool = True,
    with_sgf: bool = True,
) -> pd.DataFrame:
    """Compute per-category retrieval and answer-quality metrics.

    Args:
        pipeline: A fully set up RAGPipeline (reranker and citation
            injection enabled per the production config).
        qa_pairs: QA test set; each item must have a category key.
        compute_bert: Whether to compute BERTScore (slow).
        with_sgf: Whether to compute SGF per query.

    Returns:
        DataFrame with one row per category. Columns include each
        metric's mean with a bootstrap CI, plus n_queries,
        n_refusals, refusal_rate and mean_latency_ms.
    """
    # Collect per-query metrics grouped by category.
    by_cat: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_cat_refusals: dict[str, int] = defaultdict(int)
    by_cat_totals: dict[str, int] = defaultdict(int)
    by_cat_latency: dict[str, list[float]] = defaultdict(list)

    for i, qa in enumerate(qa_pairs):
        question = qa["question"].strip()
        if not question:
            continue
        category = qa.get("category", "unknown")
        by_cat_totals[category] += 1

        logger.info(
            "  [%s] %d/%d: %s", category, i + 1, len(qa_pairs), question[:50]
        )
        response = pipeline.answer(question)
        total_latency = (
            response.retrieval_time_ms
            + response.rerank_time_ms
            + response.generation_time_ms
        )
        by_cat_latency[category].append(total_latency)

        is_refusal = response.answer == LOW_CONFIDENCE_ANSWER
        if is_refusal:
            by_cat_refusals[category] += 1

        # Retrieval metrics are computed whether or not the pipeline
        # refused to generate an answer.
        retrieved_ids = [s.chunk_id for s in response.sources]
        relevant_sections = qa.get("relevant_sections", [])
        relevant_ids = sections_to_chunk_ids(
            pipeline.config.chunks_path, relevant_sections
        )
        r_metrics = evaluate_retrieval(retrieved_ids, relevant_ids)

        per_query: dict[str, float] = {**r_metrics}
        if not is_refusal:
            a_metrics = evaluate_answer_quality(
                predicted=response.answer,
                reference=qa.get("expected_answer", ""),
                context_texts=[s.text for s in response.sources],
                compute_bert=compute_bert,
            )
            per_query.update(a_metrics)
            if with_sgf:
                sgf = section_grounded_faithfulness(
                    answer=response.answer,
                    contexts=[s.text for s in response.sources],
                    retrieved_sections=[s.section for s in response.sources],
                    relevant_sections=relevant_sections,
                )
                per_query["sgf"] = sgf["sgf"]
                per_query["nli_faith"] = sgf["nli_faith"]
                per_query["section_match"] = sgf["section_match"]
        by_cat[category].append(per_query)

    rows: list[dict[str, Any]] = []
    for category, metrics_list in by_cat.items():
        keys: set[str] = set()
        for m in metrics_list:
            keys.update(m.keys())
        avg: dict[str, float] = {}
        for key in sorted(keys):
            vals = [m[key] for m in metrics_list if key in m]
            avg[key] = sum(vals) / len(vals)

        total = by_cat_totals[category]
        refusals = by_cat_refusals[category]
        latencies = by_cat_latency[category]
        row: dict[str, Any] = {
            "category": category,
            "n_queries": total,
            "n_refusals": refusals,
            "refusal_rate": refusals / total if total else 0.0,
            "mean_latency_ms": (
                round(sum(latencies) / len(latencies), 2) if latencies else 0.0
            ),
            **avg,
        }
        # Attach a bootstrap CI to each metric. Two or more values
        # are needed for the CI to be well-defined.
        for key in sorted(keys):
            vals = [m[key] for m in metrics_list if key in m]
            if len(vals) >= 2:
                add_ci_columns(key, vals, row)
        rows.append(row)

    # Sort rows alphabetically by category so the output is stable.
    rows.sort(key=lambda r: r["category"])
    return pd.DataFrame(rows)
