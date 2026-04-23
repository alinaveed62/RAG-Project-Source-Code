"""Tests for the LLM-comparison experiment.

These tests check that refusals are not averaged into per-model
generation latency or quality metrics, and that the output
DataFrame exposes an n_refusals column so the published table can
report how many queries were refused.
"""

from unittest.mock import MagicMock, patch

import pytest

from evaluation.experiments.llm_comparison import run_llm_comparison
from rag_pipeline.models import RAGResponse
from rag_pipeline.pipeline import LOW_CONFIDENCE_ANSWER


def _qa_pairs() -> list[dict]:
    return [
        {
            "id": "q1",
            "question": "What are the office hours?",
            "expected_answer": "Office hours are listed on the department page.",
        },
        {
            "id": "q2",
            "question": "Who won the World Cup?",
            "expected_answer": "Out of scope.",
        },
        {
            "id": "q3",
            "question": "How do I enrol?",
            "expected_answer": "Enrol through the online portal.",
        },
    ]


def _generated_response(generation_ms: float = 500.0) -> RAGResponse:
    return RAGResponse(
        question="Q?",
        answer="This is a generated answer.",
        sources=[],
        retrieval_time_ms=10.0,
        generation_time_ms=generation_ms,
    )


def _refusal_response() -> RAGResponse:
    return RAGResponse(
        question="Q?",
        answer=LOW_CONFIDENCE_ANSWER,
        sources=[],
        retrieval_time_ms=10.0,
        generation_time_ms=0.0,
    )


class TestLLMComparisonRefusalHandling:
    """Refusals must be excluded from the metric averages and
    reported separately in the n_refusals column.
    """

    def test_n_refusals_column_present(self):
        pipeline = MagicMock()
        pipeline.answer.return_value = _generated_response()

        with patch(
            "evaluation.experiments.llm_comparison.evaluate_answer_quality",
            return_value={"rouge_1": 0.5, "rouge_l": 0.4, "faithfulness": 0.8},
        ):
            df = run_llm_comparison(
                pipeline,
                _qa_pairs(),
                models=[{"name": "mistral", "params_b": 7.0, "size_gb": 4.1}],
                compute_bert=False,
            )

        assert "n_refusals" in df.columns
        assert "n_generated" in df.columns

    def test_refusal_counted_not_averaged(self):
        pipeline = MagicMock()
        pipeline.answer.side_effect = [
            _generated_response(generation_ms=400.0),
            _refusal_response(),
            _generated_response(generation_ms=600.0),
        ]

        with patch(
            "evaluation.experiments.llm_comparison.evaluate_answer_quality",
            return_value={"rouge_1": 0.5, "rouge_l": 0.4, "faithfulness": 0.8},
        ) as mock_quality:
            df = run_llm_comparison(
                pipeline,
                _qa_pairs(),
                models=[{"name": "mistral", "params_b": 7.0, "size_gb": 4.1}],
                compute_bert=False,
            )

        row = df.iloc[0]
        assert row["n_refusals"] == 1
        assert row["n_generated"] == 2
        # avg_generation_ms = (400 + 600) / 2 = 500; the 0.0 refusal must be
        # excluded from the numerator AND denominator.
        assert row["avg_generation_ms"] == pytest.approx(500.0)
        # evaluate_answer_quality called once per non-refusal question.
        assert mock_quality.call_count == 2

    def test_all_refusals_yields_zero_avg_generation_ms_not_division_error(self):
        pipeline = MagicMock()
        pipeline.answer.return_value = _refusal_response()

        with patch(
            "evaluation.experiments.llm_comparison.evaluate_answer_quality"
        ) as mock_quality:
            df = run_llm_comparison(
                pipeline,
                _qa_pairs(),
                models=[{"name": "mistral", "params_b": 7.0, "size_gb": 4.1}],
                compute_bert=False,
            )

        row = df.iloc[0]
        assert row["n_refusals"] == len(_qa_pairs())
        assert row["n_generated"] == 0
        # Latency columns (including avg_generation_ms) are intentionally
        # omitted when no non-refusal samples were collected: the row contract
        # is "no data means no key" rather than "0 == didn't measure".
        assert "avg_generation_ms" not in df.columns
        # No answer-quality computation happened.
        mock_quality.assert_not_called()

    def test_reload_generator_called_per_model(self):
        pipeline = MagicMock()
        pipeline.answer.return_value = _generated_response()

        with patch(
            "evaluation.experiments.llm_comparison.evaluate_answer_quality",
            return_value={"rouge_1": 0.5},
        ):
            run_llm_comparison(
                pipeline,
                _qa_pairs(),
                models=[
                    {"name": "mistral", "params_b": 7.0, "size_gb": 4.1},
                    {"name": "llama3.2", "params_b": 3.0, "size_gb": 2.0},
                ],
                compute_bert=False,
            )

        assert pipeline.reload_generator.call_count == 2
        assert pipeline.reload_generator.call_args_list[0].args == ("mistral",)
        assert pipeline.reload_generator.call_args_list[1].args == ("llama3.2",)

    def test_skips_empty_questions(self):
        pipeline = MagicMock()
        pipeline.answer.return_value = _generated_response()
        qa = _qa_pairs()
        qa.append({"id": "empty", "question": "   ", "expected_answer": ""})

        with patch(
            "evaluation.experiments.llm_comparison.evaluate_answer_quality",
            return_value={"rouge_1": 0.5},
        ):
            df = run_llm_comparison(
                pipeline,
                qa,
                models=[{"name": "mistral", "params_b": 7.0, "size_gb": 4.1}],
                compute_bert=False,
            )

        # Only the 3 non-empty questions should be answered.
        assert pipeline.answer.call_count == 3
        row = df.iloc[0]
        assert row["n_generated"] == 3

    def test_with_sgf_false_skips_sgf_columns(self):
        pipeline = MagicMock()
        pipeline.answer.return_value = _generated_response()
        with patch(
            "evaluation.experiments.llm_comparison.evaluate_answer_quality",
            return_value={"rouge_1": 0.5},
        ):
            df = run_llm_comparison(
                pipeline,
                _qa_pairs(),
                models=[{"name": "mistral", "params_b": 7.0, "size_gb": 4.1}],
                compute_bert=False,
                with_sgf=False,
            )
        assert "sgf" not in df.columns

    def test_single_generated_answer_skips_ci_columns(self):
        pipeline = MagicMock()
        pipeline.answer.side_effect = [
            _refusal_response(),
            _refusal_response(),
            _generated_response(),
        ]
        with patch(
            "evaluation.experiments.llm_comparison.evaluate_answer_quality",
            return_value={"rouge_1": 0.5},
        ):
            df = run_llm_comparison(
                pipeline,
                _qa_pairs(),
                models=[{"name": "mistral", "params_b": 7.0, "size_gb": 4.1}],
                compute_bert=False,
                with_sgf=False,
            )
        row = df.iloc[0]
        assert row["n_generated"] == 1
        # With only 1 generated answer, no bootstrap CI can be computed.
        assert "avg_rouge_1_ci_low" not in df.columns

    def test_single_sgf_sample_reports_mean_without_ci(self):
        # Symmetric guard to the answer-quality >= 2 gate: a single
        # non-refusal sample should emit the SGF mean column but skip the
        # bootstrap CI columns, because a 1-sample bootstrap is degenerate.
        pipeline = MagicMock()
        pipeline.answer.side_effect = [
            _refusal_response(),
            _refusal_response(),
            _generated_response(),
        ]
        with patch(
            "evaluation.experiments.llm_comparison.evaluate_answer_quality",
            return_value={"rouge_1": 0.5},
        ), patch(
            "evaluation.experiments.llm_comparison.section_grounded_faithfulness",
            return_value={"sgf": 0.6, "nli_faith": 0.5, "section_match": 0.7},
        ):
            df = run_llm_comparison(
                pipeline,
                _qa_pairs(),
                models=[{"name": "mistral", "params_b": 7.0, "size_gb": 4.1}],
                compute_bert=False,
                with_sgf=True,
            )
        row = df.iloc[0]
        assert row["n_generated"] == 1
        assert row["sgf"] == pytest.approx(0.6)
        # The CI columns must be absent, matching the other metrics'
        # "no data means no key" row-contract.
        assert "sgf_ci_low" not in df.columns
        assert "sgf_ci_high" not in df.columns
