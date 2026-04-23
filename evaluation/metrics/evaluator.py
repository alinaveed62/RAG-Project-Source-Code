"""Combined evaluation runner for the RAG pipeline."""

import json
import logging
from pathlib import Path

from evaluation.metrics.answer_metrics import evaluate_answer_quality
from evaluation.metrics.retrieval_metrics import (
    evaluate_retrieval,
    sections_to_chunk_ids,
)
from evaluation.metrics.sgf import section_grounded_faithfulness
from rag_pipeline.pipeline import LOW_CONFIDENCE_ANSWER, RAGPipeline

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs evaluation on a test set using the RAG pipeline."""

    def __init__(self, pipeline: RAGPipeline, qa_pairs: list[dict]):
        self.pipeline = pipeline
        self.qa_pairs = qa_pairs

    def run(
        self,
        compute_bert: bool = True,
        k_values: list[int] | None = None,
        with_sgf: bool = False,
        sgf_alpha: float = 0.5,
    ) -> list[dict]:
        """Run evaluation on all QA pairs.

        Args:
            compute_bert: Whether to compute BERTScore (slow).
            k_values: k values for retrieval metrics.

        Returns:
            List of result dicts, one per QA pair.
        """
        k_values = k_values or [1, 3, 5, 10]
        results = []

        for i, qa in enumerate(self.qa_pairs):
            question = qa["question"]
            if not question.strip():
                logger.info("Skipping empty question (id=%s)", qa.get("id"))
                continue

            logger.info(
                "Evaluating %d/%d: %s", i + 1, len(self.qa_pairs), question[:50]
            )

            response = self.pipeline.answer(question)
            is_refusal = response.answer == LOW_CONFIDENCE_ANSWER

            # Retrieval metrics are computed on chunk IDs, not
            # section labels. The ground truth is the set of chunks
            # that belong to any section listed in relevant_sections.
            retrieved_ids = [s.chunk_id for s in response.sources]
            relevant_sections = qa.get("relevant_sections", [])
            relevant_ids = sections_to_chunk_ids(
                self.pipeline.config.chunks_path, relevant_sections
            )

            retrieval_scores = evaluate_retrieval(
                retrieved_ids, relevant_ids, k_values
            )

            # Answer quality metrics only make sense when the system
            # actually generated a response. For refusals we note the
            # refusal and skip ROUGE, BERTScore and faithfulness, so
            # the aggregate quality figures reflect generated answers
            # only.
            if is_refusal:
                answer_scores = {}
                sgf_scores = {}
            else:
                answer_scores = evaluate_answer_quality(
                    predicted=response.answer,
                    reference=qa.get("expected_answer", ""),
                    context_texts=[s.text for s in response.sources],
                    compute_bert=compute_bert,
                )
                if with_sgf:
                    sgf_scores = section_grounded_faithfulness(
                        answer=response.answer,
                        contexts=[s.text for s in response.sources],
                        retrieved_sections=[s.section for s in response.sources],
                        relevant_sections=relevant_sections,
                        alpha=sgf_alpha,
                    )
                else:
                    sgf_scores = {}

            result = {
                "id": qa.get("id", f"q{i}"),
                "question": question,
                "category": qa.get("category", ""),
                "difficulty": qa.get("difficulty", ""),
                "answer": response.answer,
                "expected_answer": qa.get("expected_answer", ""),
                "is_refusal": is_refusal,
                "num_sources": len(response.sources),
                "retrieval_time_ms": response.retrieval_time_ms,
                "generation_time_ms": response.generation_time_ms,
                **retrieval_scores,
                **answer_scores,
                **sgf_scores,
            }
            results.append(result)

        return results

    @staticmethod
    def print_summary(results: list[dict]) -> None:
        """Print a summary of evaluation results."""
        if not results:
            print("No results to summarize.")
            return

        # Gather numeric metric keys from every result, not just
        # results[0]: a refusal at position 0 leaves answer_scores
        # empty, so keys like rouge_1 only appear on later generated
        # answers. bool is excluded explicitly because in Python
        # isinstance(True, int) is True, and flags like is_refusal
        # would otherwise be averaged as if they were metrics.
        excluded = {"retrieval_time_ms", "generation_time_ms", "num_sources"}
        metric_keys_set: set[str] = set()
        for r in results:
            for k, v in r.items():
                if (
                    isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and k not in excluded
                ):
                    metric_keys_set.add(k)
        metric_keys = sorted(metric_keys_set)

        n_refusals = sum(1 for r in results if r.get("is_refusal"))
        n_generated = len(results) - n_refusals

        print(f"\nEvaluation Summary ({len(results)} questions)")
        print(
            f"  ({n_generated} generated, {n_refusals} FR9 refusals excluded "
            f"from generation metrics)"
        )
        print("=" * 50)

        for key in metric_keys:
            # metric_keys is the union of numeric keys across all
            # results, so every key appears in at least one row and
            # values is never empty by construction.
            values = [r[key] for r in results if key in r]
            avg = sum(values) / len(values)
            print(f"  {key:30s}: {avg:.4f}")

        # Retrieval runs for every query (including refusals), but
        # generation only runs for non-refusals.
        avg_retrieval = sum(r["retrieval_time_ms"] for r in results) / len(results)
        gen_values = [
            r["generation_time_ms"] for r in results if not r.get("is_refusal")
        ]
        avg_generation = sum(gen_values) / len(gen_values) if gen_values else 0.0
        print(f"\n  {'avg_retrieval_ms':30s}: {avg_retrieval:.1f}")
        print(f"  {'avg_generation_ms':30s}: {avg_generation:.1f}")

    @staticmethod
    def save_results(results: list[dict], output_path: Path) -> None:
        """Save results to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d results to %s", len(results), output_path)
