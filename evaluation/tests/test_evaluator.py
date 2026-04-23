"""Tests for the combined evaluation runner (Evaluator class)."""

import json
from unittest.mock import MagicMock, patch

from evaluation.metrics.evaluator import Evaluator
from rag_pipeline.models import RAGResponse, RetrievalResult
from rag_pipeline.pipeline import LOW_CONFIDENCE_ANSWER


def _make_response(question: str = "test?") -> RAGResponse:
    """Helper to build a RAGResponse with plausible defaults."""
    return RAGResponse(
        question=question,
        answer="The answer is 42.",
        sources=[
            RetrievalResult(
                chunk_id="c1",
                text="Chunk text one.",
                score=0.85,
                source="handbook.pdf",
                title="Attendance",
                section="Teaching & Assessment",
            ),
            RetrievalResult(
                chunk_id="c2",
                text="Chunk text two.",
                score=0.72,
                source="handbook.pdf",
                title="Exams",
                section="Programmes",
            ),
        ],
        retrieval_time_ms=12.5,
        generation_time_ms=340.0,
    )


def _make_qa_pairs(n: int = 3) -> list[dict]:
    """Helper to build a list of QA pairs."""
    return [
        {
            "id": f"q{i}",
            "question": f"Question number {i}?",
            "expected_answer": f"Expected answer {i}.",
            "relevant_sections": ["Teaching & Assessment"],
            "category": "attendance",
            "difficulty": "easy",
        }
        for i in range(1, n + 1)
    ]


# ---- Shared patch decorators ------------------------------------------------

_patch_retrieval = patch(
    "evaluation.metrics.evaluator.evaluate_retrieval",
    return_value={
        "mrr": 1.0,
        "precision_at_1": 1.0,
        "recall_at_1": 1.0,
        "ndcg_at_1": 1.0,
        "precision_at_5": 0.4,
        "recall_at_5": 1.0,
        "ndcg_at_5": 1.0,
    },
)

_patch_answer = patch(
    "evaluation.metrics.evaluator.evaluate_answer_quality",
    return_value={
        "rouge_1": 0.75,
        "rouge_2": 0.50,
        "rouge_l": 0.70,
        "faithfulness": 0.90,
    },
)


# ---- TestEvaluatorRun --------------------------------------------------------


@_patch_answer
@_patch_retrieval
class TestEvaluatorRun:

    def test_run_calls_pipeline_answer_for_each_pair(
        self, mock_retrieval, mock_answer
    ):
        pipeline = MagicMock()
        pipeline.answer.return_value = _make_response()
        qa_pairs = _make_qa_pairs(4)

        evaluator = Evaluator(pipeline, qa_pairs)
        evaluator.run(compute_bert=False)

        assert pipeline.answer.call_count == 4

    def test_run_skips_empty_questions(self, mock_retrieval, mock_answer):
        pipeline = MagicMock()
        pipeline.answer.return_value = _make_response()
        qa_pairs = _make_qa_pairs(2)
        qa_pairs.append({"id": "empty", "question": "   "})

        evaluator = Evaluator(pipeline, qa_pairs)
        results = evaluator.run(compute_bert=False)

        assert len(results) == 2
        assert pipeline.answer.call_count == 2

    def test_run_returns_list_of_result_dicts(self, mock_retrieval, mock_answer):
        pipeline = MagicMock()
        pipeline.answer.return_value = _make_response()

        evaluator = Evaluator(pipeline, _make_qa_pairs(2))
        results = evaluator.run(compute_bert=False)

        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_run_includes_retrieval_and_answer_scores(
        self, mock_retrieval, mock_answer
    ):
        pipeline = MagicMock()
        pipeline.answer.return_value = _make_response()

        evaluator = Evaluator(pipeline, _make_qa_pairs(1))
        results = evaluator.run(compute_bert=False)

        result = results[0]
        # Retrieval scores from mock
        assert result["mrr"] == 1.0
        assert result["precision_at_1"] == 1.0
        # Answer scores from mock
        assert result["rouge_1"] == 0.75
        assert result["faithfulness"] == 0.90

    def test_run_result_has_expected_keys(self, mock_retrieval, mock_answer):
        pipeline = MagicMock()
        pipeline.answer.return_value = _make_response()

        evaluator = Evaluator(pipeline, _make_qa_pairs(1))
        results = evaluator.run(compute_bert=False)

        result = results[0]
        expected_keys = {
            "id",
            "question",
            "category",
            "difficulty",
            "answer",
            "expected_answer",
            "num_sources",
            "retrieval_time_ms",
            "generation_time_ms",
        }
        assert expected_keys.issubset(result.keys())

    def test_run_custom_k_values(self, mock_retrieval, mock_answer):
        pipeline = MagicMock()
        pipeline.answer.return_value = _make_response()

        evaluator = Evaluator(pipeline, _make_qa_pairs(1))
        evaluator.run(compute_bert=False, k_values=[3, 7])

        # evaluate_retrieval should have been called with the custom k_values
        call_args = mock_retrieval.call_args
        assert call_args[0][2] == [3, 7]

    def test_run_without_bert_score(self, mock_retrieval, mock_answer):
        pipeline = MagicMock()
        pipeline.answer.return_value = _make_response()

        evaluator = Evaluator(pipeline, _make_qa_pairs(1))
        evaluator.run(compute_bert=False)

        call_kwargs = mock_answer.call_args[1]
        assert call_kwargs["compute_bert"] is False

    def test_run_records_timing(self, mock_retrieval, mock_answer):
        response = _make_response()
        pipeline = MagicMock()
        pipeline.answer.return_value = response

        evaluator = Evaluator(pipeline, _make_qa_pairs(1))
        results = evaluator.run(compute_bert=False)

        result = results[0]
        assert result["retrieval_time_ms"] == response.retrieval_time_ms
        assert result["generation_time_ms"] == response.generation_time_ms

    def test_run_marks_refusal_and_skips_answer_metrics(
        self, mock_retrieval, mock_answer
    ):
        """Refusals recorded by the evaluator have is_refusal=True,
        no ROUGE, BERTScore or faithfulness is computed, and the
        answer-quality keys stay absent so aggregate tables ignore
        refusals instead of averaging them in.
        """
        refusal_response = RAGResponse(
            question="What is the weather?",
            answer=LOW_CONFIDENCE_ANSWER,
            sources=[],
            retrieval_time_ms=8.0,
            generation_time_ms=0.0,
        )
        pipeline = MagicMock()
        pipeline.answer.return_value = refusal_response

        evaluator = Evaluator(pipeline, _make_qa_pairs(1))
        results = evaluator.run(compute_bert=False)

        assert len(results) == 1
        assert results[0]["is_refusal"] is True
        assert "rouge_1" not in results[0]
        assert "faithfulness" not in results[0]
        # evaluate_answer_quality must NOT have been called on the refusal.
        mock_answer.assert_not_called()


# ---- TestEvaluatorPrintSummary -----------------------------------------------


class TestEvaluatorPrintSummary:

    def test_print_summary_with_results(self, capsys):
        results = [
            {
                "id": "q1",
                "question": "Q1?",
                "mrr": 1.0,
                "rouge_1": 0.8,
                "retrieval_time_ms": 10.0,
                "generation_time_ms": 200.0,
                "num_sources": 2,
            },
            {
                "id": "q2",
                "question": "Q2?",
                "mrr": 0.5,
                "rouge_1": 0.6,
                "retrieval_time_ms": 20.0,
                "generation_time_ms": 300.0,
                "num_sources": 3,
            },
        ]

        Evaluator.print_summary(results)
        captured = capsys.readouterr()

        assert "Evaluation Summary (2 questions)" in captured.out
        assert "mrr" in captured.out
        assert "rouge_1" in captured.out
        assert "avg_retrieval_ms" in captured.out
        assert "avg_generation_ms" in captured.out

    def test_print_summary_empty_results(self, capsys):
        Evaluator.print_summary([])
        captured = capsys.readouterr()

        assert "No results to summarize." in captured.out

    def test_print_summary_excludes_refusals_from_generation_time(self, capsys):
        """Refusals have generation_time_ms == 0.0; averaging them
        in would deflate per-model latency, so the summary must
        exclude them.
        """
        results = [
            {
                "id": "q1",
                "question": "Q1?",
                "is_refusal": False,
                "mrr": 1.0,
                "retrieval_time_ms": 10.0,
                "generation_time_ms": 400.0,
                "num_sources": 2,
            },
            {
                "id": "q2",
                "question": "Q2?",
                "is_refusal": True,
                "mrr": 0.0,
                "retrieval_time_ms": 10.0,
                "generation_time_ms": 0.0,
                "num_sources": 0,
            },
        ]
        Evaluator.print_summary(results)
        captured = capsys.readouterr()

        # 1 refusal acknowledged in the header, generation latency averaged
        # over the 1 generated result only: 400.0 ms.
        assert "1 FR9 refusals excluded" in captured.out
        assert "avg_generation_ms" in captured.out
        assert "400.0" in captured.out


# ---- TestEvaluatorSaveResults ------------------------------------------------


class TestEvaluatorSaveResults:

    def test_save_results_creates_file(self, tmp_path):
        output = tmp_path / "results.json"
        Evaluator.save_results([{"id": "q1"}], output)

        assert output.exists()

    def test_save_results_creates_parent_dirs(self, tmp_path):
        output = tmp_path / "nested" / "deep" / "results.json"
        Evaluator.save_results([{"id": "q1"}], output)

        assert output.exists()

    def test_save_results_valid_json(self, tmp_path):
        output = tmp_path / "results.json"
        data = [{"id": "q1", "mrr": 0.9}, {"id": "q2", "mrr": 0.5}]

        Evaluator.save_results(data, output)

        loaded = json.loads(output.read_text(encoding="utf-8"))
        assert isinstance(loaded, list)

    def test_save_results_correct_count(self, tmp_path):
        output = tmp_path / "results.json"
        data = [{"id": f"q{i}"} for i in range(5)]

        Evaluator.save_results(data, output)

        loaded = json.loads(output.read_text(encoding="utf-8"))
        assert len(loaded) == 5


class TestEvaluatorWithSGF:
    """Cover the with_sgf=True branch inside Evaluator.run."""

    def test_sgf_scores_included_when_with_sgf_true(self, tmp_path):
        pipeline = MagicMock()
        pipeline.answer.return_value = _make_response()
        pipeline.config = MagicMock(chunks_path=tmp_path / "chunks.jsonl")
        (tmp_path / "chunks.jsonl").write_text(
            '{"id": "c1", "section": "Teaching & Assessment"}\n'
            '{"id": "c2", "section": "Programmes"}\n'
        )

        evaluator = Evaluator(pipeline, _make_qa_pairs(1))

        with (
            _patch_retrieval,
            _patch_answer,
            patch(
                "evaluation.metrics.evaluator.section_grounded_faithfulness",
                return_value={"sgf": 0.77, "nli_faith": 0.7, "section_match": 0.84, "alpha": 0.5},
            ) as mock_sgf,
        ):
            results = evaluator.run(compute_bert=False, with_sgf=True, sgf_alpha=0.6)

        assert mock_sgf.called
        assert mock_sgf.call_args.kwargs["alpha"] == 0.6
        assert results[0]["sgf"] == 0.77


class TestPrintSummaryEdgeCases:
    def test_empty_metric_values_skipped(self, capsys):
        """Metric key present in result dict with missing value still prints header."""
        results = [
            {"id": "q1", "mrr": 0.5, "retrieval_time_ms": 1.0, "generation_time_ms": 2.0},
        ]
        Evaluator.print_summary(results)
        captured = capsys.readouterr().out
        assert "Evaluation Summary" in captured
