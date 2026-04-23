"""Compare different top-k values on retrieval and answer quality."""

from __future__ import annotations

import logging

import pandas as pd

from evaluation.metrics.answer_metrics import compute_rouge
from evaluation.metrics.bootstrap import add_ci_columns
from evaluation.metrics.retrieval_metrics import (
    evaluate_retrieval,
    sections_to_chunk_ids,
)
from rag_pipeline.pipeline import LOW_CONFIDENCE_ANSWER, RAGPipeline

logger = logging.getLogger(__name__)

TOPK_VALUES = [3, 5, 7, 10]


def _normalise_retrieval_metrics(metrics: dict[str, float], k: int) -> dict[str, float]:
    """Rename precision_at_{k}, recall_at_{k} and ndcg_at_{k} to
    k-free column names so every row of the summary CSV uses the
    same schema. Without this rename, each row produced with
    k_values=[k] would be keyed by the specific k, so the summary
    CSV would carry precision_at_3 only on the k=3 row and
    precision_at_5 only on the k=5 row, leaving NaN everywhere else.
    """
    return {
        "mrr": metrics["mrr"],
        "precision_at_k": metrics[f"precision_at_{k}"],
        "recall_at_k": metrics[f"recall_at_{k}"],
        "ndcg_at_k": metrics[f"ndcg_at_{k}"],
    }


def run_topk_comparison(
    pipeline: RAGPipeline,
    qa_pairs: list[dict],
    topk_values: list[int] | None = None,
    capture_per_query: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Compare retrieval and answer quality at different top-k values.

    Args:
        pipeline: A fully set up RAGPipeline.
        qa_pairs: QA pair dicts.
        topk_values: List of k values to test. Defaults to [3, 5, 7, 10].
        capture_per_query: If True, also return a per-query
            DataFrame with one row per (k, qa_id) pair for paired
            significance tests.

    Returns:
        When capture_per_query is False (default): a summary
        DataFrame with one row per k. When True: a tuple of
        (summary_df, per_query_df).
    """
    topk_values = topk_values or TOPK_VALUES
    summary_rows: list[dict] = []
    per_query_rows: list[dict] = []

    for k in topk_values:
        logger.info("Evaluating top_k=%d", k)
        pipeline.config.top_k = k

        all_retrieval_metrics: list[dict[str, float]] = []
        all_rouge_scores: list[dict[str, float]] = []
        n_refusals = 0

        for idx, qa in enumerate(qa_pairs):
            question = qa["question"].strip()
            if not question:
                continue

            response = pipeline.answer(question)

            retrieved_ids = [s.chunk_id for s in response.sources]
            relevant_sections = qa.get("relevant_sections", [])
            relevant_ids = sections_to_chunk_ids(
                pipeline.config.chunks_path, relevant_sections
            )
            raw_metrics = evaluate_retrieval(
                retrieved_ids, relevant_ids, k_values=[k]
            )
            r_metrics = _normalise_retrieval_metrics(raw_metrics, k)
            all_retrieval_metrics.append(r_metrics)

            is_refusal = response.answer == LOW_CONFIDENCE_ANSWER
            rouge: dict[str, float] = {}
            if is_refusal:
                n_refusals += 1
            else:
                rouge = compute_rouge(
                    response.answer, qa.get("expected_answer", "")
                )
                all_rouge_scores.append(rouge)

            if capture_per_query:
                per_query_rows.append(
                    {
                        "top_k": k,
                        "qa_id": qa.get("id") or f"qa_{idx}",
                        "is_refusal": is_refusal,
                        **r_metrics,
                        **rouge,
                    }
                )

        # Aggregate means and attach bootstrap CIs to each metric.
        avg_retrieval = _average_dicts(all_retrieval_metrics)
        avg_rouge = _average_dicts(all_rouge_scores)
        n_generated = len(all_rouge_scores)
        n_queries = len(all_retrieval_metrics)

        row: dict[str, float | int] = {
            "top_k": k,
            "n_queries": n_queries,
            "n_refusals": n_refusals,
            "n_generated": n_generated,
            **avg_retrieval,
            **avg_rouge,
        }
        if all_retrieval_metrics:
            for metric_name in all_retrieval_metrics[0]:
                add_ci_columns(
                    metric_name,
                    [m[metric_name] for m in all_retrieval_metrics],
                    row,
                )
        if all_rouge_scores:
            for metric_name in all_rouge_scores[0]:
                add_ci_columns(
                    metric_name,
                    [m[metric_name] for m in all_rouge_scores],
                    row,
                )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if capture_per_query:
        return summary_df, pd.DataFrame(per_query_rows)
    return summary_df


def _average_dicts(records: list[dict[str, float]]) -> dict[str, float]:
    """Average each numeric key across the given list of dicts.

    Returns an empty dict if the input is empty.
    """
    if not records:
        return {}
    keys: set[str] = set()
    for r in records:
        keys.update(r.keys())
    return {
        k: sum(r[k] for r in records if k in r) / sum(1 for r in records if k in r)
        for k in keys
    }
