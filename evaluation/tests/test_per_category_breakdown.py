"""Tests for the per-category breakdown experiment."""

from __future__ import annotations

from unittest.mock import patch

from evaluation.experiments.per_category_breakdown import run_per_category_breakdown
from rag_pipeline.pipeline import LOW_CONFIDENCE_ANSWER


def _qa_pairs() -> list[dict]:
    return [
        {
            "id": "att_01",
            "question": "Do I have to attend lectures?",
            "expected_answer": "Yes.",
            "category": "attendance",
            "relevant_sections": ["Teaching & Assessment"],
            "difficulty": "easy",
        },
        {
            "id": "att_02",
            "question": "Are tutorials mandatory?",
            "expected_answer": "Yes.",
            "category": "attendance",
            "relevant_sections": ["Teaching & Assessment"],
            "difficulty": "easy",
        },
        {
            "id": "ass_01",
            "question": "How do I submit?",
            "expected_answer": "Via KEATS.",
            "category": "assessment",
            "relevant_sections": ["Teaching & Assessment"],
            "difficulty": "easy",
        },
        {
            "id": "edge_01",
            "question": "Out of scope?",
            "expected_answer": "Refuse.",
            "category": "edge_cases",
            "relevant_sections": [],
            "difficulty": "hard",
        },
        {"id": "empty", "question": "   ", "category": "misc", "relevant_sections": []},
    ]


def _build_response(answer: str, make_rag_response, make_retrieval_result):
    return make_rag_response(
        answer=answer,
        sources=[
            make_retrieval_result(chunk_id="c1", section="Teaching & Assessment"),
        ],
        retrieval_time_ms=30.0,
        rerank_time_ms=0.0,
        generation_time_ms=100.0,
    )


class TestRunPerCategoryBreakdown:
    def test_happy_path_groups_categories(
        self, mock_pipeline, make_rag_response, make_retrieval_result
    ):
        mock_pipeline.answer.side_effect = [
            _build_response("Yes.", make_rag_response, make_retrieval_result),
            _build_response("Yes.", make_rag_response, make_retrieval_result),
            _build_response("Via KEATS.", make_rag_response, make_retrieval_result),
            _build_response(LOW_CONFIDENCE_ANSWER, make_rag_response, make_retrieval_result),
        ]
        with (
            patch(
                "evaluation.experiments.per_category_breakdown.evaluate_answer_quality",
                return_value={
                    "rouge_1": 0.5,
                    "rouge_l": 0.4,
                    "bert_score_f1": 0.8,
                    "faithfulness": 0.7,
                },
            ),
            patch(
                "evaluation.experiments.per_category_breakdown.section_grounded_faithfulness",
                return_value={"sgf": 0.6, "nli_faith": 0.5, "section_match": 0.7, "alpha": 0.5},
            ),
        ):
            df = run_per_category_breakdown(pipeline=mock_pipeline, qa_pairs=_qa_pairs())
        assert set(df["category"]) == {"attendance", "assessment", "edge_cases"}
        att = df[df["category"] == "attendance"].iloc[0]
        assert att["n_queries"] == 2
        assert att["n_refusals"] == 0
        assert "sgf" in df.columns
        edge = df[df["category"] == "edge_cases"].iloc[0]
        assert edge["n_refusals"] == 1
        assert edge["refusal_rate"] == 1.0

    def test_missing_category_key_falls_back_to_unknown(
        self, mock_pipeline, make_rag_response, make_retrieval_result
    ):
        qa = [
            {
                "id": "x",
                "question": "Q?",
                "expected_answer": "A",
                "relevant_sections": ["Teaching & Assessment"],
            }
        ]
        mock_pipeline.answer.return_value = _build_response(
            "A", make_rag_response, make_retrieval_result
        )
        with (
            patch(
                "evaluation.experiments.per_category_breakdown.evaluate_answer_quality",
                return_value={"rouge_1": 0.5},
            ),
            patch(
                "evaluation.experiments.per_category_breakdown.section_grounded_faithfulness",
                return_value={"sgf": 0.5, "nli_faith": 0.5, "section_match": 0.5, "alpha": 0.5},
            ),
        ):
            df = run_per_category_breakdown(pipeline=mock_pipeline, qa_pairs=qa)
        assert df.iloc[0]["category"] == "unknown"

    def test_rows_sorted_alphabetically(
        self, mock_pipeline, make_rag_response, make_retrieval_result
    ):
        qa = [
            {"id": "z", "question": "Q?", "expected_answer": "A", "category": "z_cat", "relevant_sections": []},
            {"id": "a", "question": "Q?", "expected_answer": "A", "category": "a_cat", "relevant_sections": []},
        ]
        mock_pipeline.answer.return_value = _build_response(
            "A", make_rag_response, make_retrieval_result
        )
        with (
            patch(
                "evaluation.experiments.per_category_breakdown.evaluate_answer_quality",
                return_value={"rouge_1": 0.5},
            ),
            patch(
                "evaluation.experiments.per_category_breakdown.section_grounded_faithfulness",
                return_value={"sgf": 0.5, "nli_faith": 0.5, "section_match": 0.5, "alpha": 0.5},
            ),
        ):
            df = run_per_category_breakdown(pipeline=mock_pipeline, qa_pairs=qa)
        assert list(df["category"]) == ["a_cat", "z_cat"]

    def test_with_sgf_false_skips_sgf_column(
        self, mock_pipeline, make_rag_response, make_retrieval_result
    ):
        qa = [
            {
                "id": "x",
                "question": "Q?",
                "expected_answer": "A",
                "category": "c",
                "relevant_sections": ["Teaching & Assessment"],
            }
        ]
        mock_pipeline.answer.return_value = _build_response(
            "A", make_rag_response, make_retrieval_result
        )
        with patch(
            "evaluation.experiments.per_category_breakdown.evaluate_answer_quality",
            return_value={"rouge_1": 0.5},
        ):
            df = run_per_category_breakdown(
                pipeline=mock_pipeline, qa_pairs=qa, with_sgf=False
            )
        assert "sgf" not in df.columns

    def test_compute_bert_false_passes_through(
        self, mock_pipeline, make_rag_response, make_retrieval_result
    ):
        qa = [
            {
                "id": "x",
                "question": "Q?",
                "expected_answer": "A",
                "category": "c",
                "relevant_sections": [],
            }
        ]
        mock_pipeline.answer.return_value = _build_response(
            "A", make_rag_response, make_retrieval_result
        )
        with (
            patch(
                "evaluation.experiments.per_category_breakdown.evaluate_answer_quality",
                return_value={"rouge_1": 0.5},
            ) as mock_eval,
            patch(
                "evaluation.experiments.per_category_breakdown.section_grounded_faithfulness",
                return_value={"sgf": 0.5, "nli_faith": 0.5, "section_match": 0.5, "alpha": 0.5},
            ),
        ):
            run_per_category_breakdown(
                pipeline=mock_pipeline, qa_pairs=qa, compute_bert=False
            )
        assert mock_eval.call_args.kwargs["compute_bert"] is False

    def test_single_query_category_no_ci_columns(
        self, mock_pipeline, make_rag_response, make_retrieval_result
    ):
        qa = [
            {
                "id": "a",
                "question": "Q?",
                "expected_answer": "A",
                "category": "solo",
                "relevant_sections": ["Teaching & Assessment"],
            }
        ]
        mock_pipeline.answer.return_value = _build_response(
            "A", make_rag_response, make_retrieval_result
        )
        with (
            patch(
                "evaluation.experiments.per_category_breakdown.evaluate_answer_quality",
                return_value={"rouge_1": 0.5},
            ),
            patch(
                "evaluation.experiments.per_category_breakdown.section_grounded_faithfulness",
                return_value={"sgf": 0.5, "nli_faith": 0.5, "section_match": 0.5, "alpha": 0.5},
            ),
        ):
            df = run_per_category_breakdown(pipeline=mock_pipeline, qa_pairs=qa)
        # 1 query in category → no bootstrap CI columns.
        assert "mrr_ci_low" not in df.columns
        assert df.iloc[0]["n_queries"] == 1

    def test_latency_sums_three_components(
        self, mock_pipeline, make_rag_response, make_retrieval_result
    ):
        mock_pipeline.answer.return_value = make_rag_response(
            answer="A",
            sources=[make_retrieval_result(chunk_id="c", section="S")],
            retrieval_time_ms=10.0,
            rerank_time_ms=5.0,
            generation_time_ms=100.0,
        )
        qa = [
            {
                "id": "a",
                "question": "Q?",
                "expected_answer": "A",
                "category": "solo",
                "relevant_sections": [],
            }
        ]
        with (
            patch(
                "evaluation.experiments.per_category_breakdown.evaluate_answer_quality",
                return_value={"rouge_1": 0.5},
            ),
            patch(
                "evaluation.experiments.per_category_breakdown.section_grounded_faithfulness",
                return_value={"sgf": 0.5, "nli_faith": 0.5, "section_match": 0.5, "alpha": 0.5},
            ),
        ):
            df = run_per_category_breakdown(pipeline=mock_pipeline, qa_pairs=qa)
        assert df.iloc[0]["mean_latency_ms"] == 115.0
