"""Compare different LLM models on answer quality and resource demands.

Each metric in the summary CSV carries a bootstrap confidence
interval derived from its per-query vector, and the pipeline also
records P50, P95 and P99 latency plus a per-component breakdown so
the report can show the resource profile of each model.
"""

import logging

import numpy as np
import pandas as pd

from evaluation.metrics.answer_metrics import evaluate_answer_quality
from evaluation.metrics.bootstrap import add_ci_columns
from evaluation.metrics.sgf import section_grounded_faithfulness
from rag_pipeline.pipeline import LOW_CONFIDENCE_ANSWER, RAGPipeline

logger = logging.getLogger(__name__)

# Models available through Ollama, with approximate parameter
# counts and on-disk sizes.
LLM_MODELS = [
    {"name": "mistral", "params_b": 7.0, "size_gb": 4.1},
    {"name": "llama3.2", "params_b": 3.0, "size_gb": 2.0},
    {"name": "phi3:mini", "params_b": 3.8, "size_gb": 2.3},
    {"name": "gemma2:2b", "params_b": 2.0, "size_gb": 1.6},
]


def run_llm_comparison(
    pipeline: RAGPipeline,
    qa_pairs: list[dict],
    models: list[dict] | None = None,
    compute_bert: bool = True,
    with_sgf: bool = True,
) -> pd.DataFrame:
    """Compare answer quality, SGF and resource demands across LLMs.

    Args:
        pipeline: A fully set up RAGPipeline with the Ollama backend.
        qa_pairs: QA pair dicts with question and expected_answer.
        models: List of model dicts with name, params_b and size_gb.
        compute_bert: Whether to compute BERTScore (slow).
        with_sgf: If True, also compute the Section-Grounded
            Faithfulness metric.

    Returns:
        One row per model with model, params_b, size_gb, refusal
        count, per-component latency percentiles (retrieval, rerank,
        generation and total), averaged quality metrics, and
        bootstrap CIs on each averaged metric.
    """
    models = models or LLM_MODELS
    results = []

    for model_info in models:
        model_name = model_info["name"]
        logger.info(
            "Evaluating LLM: %s (%s B params)", model_name, model_info["params_b"]
        )
        pipeline.reload_generator(model_name)

        all_metrics: list[dict[str, float]] = []
        all_sgf: list[dict[str, float]] = []
        total_ms_samples: list[float] = []
        retrieval_ms_samples: list[float] = []
        rerank_ms_samples: list[float] = []
        generation_ms_samples: list[float] = []
        n_refusals = 0

        for i, qa in enumerate(qa_pairs):
            question = qa["question"].strip()
            if not question:
                continue

            logger.info(
                "  [%s] %d/%d: %s", model_name, i + 1, len(qa_pairs), question[:50]
            )

            response = pipeline.answer(question)
            if response.answer == LOW_CONFIDENCE_ANSWER:
                n_refusals += 1
                continue

            # Record the per-component latency for this query.
            retrieval_ms_samples.append(response.retrieval_time_ms)
            rerank_ms_samples.append(response.rerank_time_ms)
            generation_ms_samples.append(response.generation_time_ms)
            total_ms_samples.append(
                response.retrieval_time_ms
                + response.rerank_time_ms
                + response.generation_time_ms
            )

            metrics = evaluate_answer_quality(
                predicted=response.answer,
                reference=qa.get("expected_answer", ""),
                context_texts=[s.text for s in response.sources],
                compute_bert=compute_bert,
            )
            all_metrics.append(metrics)

            # Capture per-query SGF so the LLM comparison table can
            # carry an SGF column alongside ROUGE and BERTScore.
            if with_sgf:
                sgf = section_grounded_faithfulness(
                    answer=response.answer,
                    contexts=[s.text for s in response.sources],
                    retrieved_sections=[s.section for s in response.sources],
                    relevant_sections=qa.get("relevant_sections", []),
                )
                all_sgf.append(sgf)

        # Average quality metrics + bootstrap CI on each of them.
        avg_metrics: dict[str, float] = {}
        if all_metrics:
            all_keys: set[str] = set()
            for m in all_metrics:
                all_keys.update(m.keys())
            for key in all_keys:
                values = [m[key] for m in all_metrics if key in m]
                avg_metrics[f"avg_{key}"] = (
                    sum(values) / len(values) if values else 0.0
                )

        row: dict[str, float | int | str] = {
            "model": model_name,
            "params_b": model_info["params_b"],
            "size_gb": model_info["size_gb"],
            "n_refusals": n_refusals,
            "n_generated": len(all_metrics),
            **avg_metrics,
        }

        # Latency percentiles across the per-query samples.
        if total_ms_samples:
            row["p50_total_ms"] = float(np.percentile(total_ms_samples, 50))
            row["p95_total_ms"] = float(np.percentile(total_ms_samples, 95))
            row["p99_total_ms"] = float(np.percentile(total_ms_samples, 99))
            row["retrieval_p50_ms"] = float(np.percentile(retrieval_ms_samples, 50))
            row["rerank_p50_ms"] = float(np.percentile(rerank_ms_samples, 50))
            row["generation_p50_ms"] = float(np.percentile(generation_ms_samples, 50))
            # Mean kept for backwards compatibility with older tables.
            row["avg_generation_ms"] = round(
                sum(generation_ms_samples) / len(generation_ms_samples), 1
            )

        # Mean SGF plus a bootstrap CI. A CI needs at least two
        # samples to be well-defined; the same len(values) >= 2 guard
        # is applied to the answer-quality averages below, so a
        # single SGF sample yields the mean without a degenerate CI.
        if all_sgf:
            row["sgf"] = sum(s["sgf"] for s in all_sgf) / len(all_sgf)
            row["nli_faith"] = sum(s["nli_faith"] for s in all_sgf) / len(all_sgf)
            row["section_match"] = sum(s["section_match"] for s in all_sgf) / len(all_sgf)
            if len(all_sgf) >= 2:
                add_ci_columns("sgf", [s["sgf"] for s in all_sgf], row)

        # Bootstrap CIs on the answer-quality averages.
        if all_metrics:
            for key in all_keys:
                values = [m[key] for m in all_metrics if key in m]
                if len(values) >= 2:
                    add_ci_columns(f"avg_{key}", values, row)

        results.append(row)

    return pd.DataFrame(results)
