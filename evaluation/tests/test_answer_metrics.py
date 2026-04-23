"""Tests for answer quality metrics."""

from unittest.mock import patch

import pytest

from evaluation.metrics.answer_metrics import (
    compute_bert_score,
    compute_faithfulness,
    compute_rouge,
    evaluate_answer_quality,
)


class _FakeTensor:
    def __init__(self, v: float):
        self._v = v

    def item(self) -> float:
        return self._v


class TestComputeRouge:

    def test_identical_texts(self):
        scores = compute_rouge("The cat sat on the mat", "The cat sat on the mat")
        assert scores["rouge_1"] == pytest.approx(1.0)
        assert scores["rouge_l"] == pytest.approx(1.0)

    def test_disjoint_texts(self):
        scores = compute_rouge("hello world", "foo bar baz")
        assert scores["rouge_1"] == pytest.approx(0.0)

    def test_partial_overlap(self):
        scores = compute_rouge("The cat sat on the mat", "The cat slept on the mat")
        assert 0 < scores["rouge_1"] < 1

    def test_returns_all_keys(self):
        scores = compute_rouge("a", "b")
        assert "rouge_1" in scores
        assert "rouge_2" in scores
        assert "rouge_l" in scores


class TestComputeFaithfulness:

    def test_full_overlap(self):
        answer = "Students must attend lectures."
        context = ["Students must attend all scheduled lectures and tutorials."]
        score = compute_faithfulness(answer, context)
        assert score == pytest.approx(1.0)

    def test_no_overlap(self):
        answer = "Quantum physics is fascinating."
        context = ["Students must attend lectures."]
        score = compute_faithfulness(answer, context)
        assert score == 0.0

    def test_empty_answer(self):
        assert compute_faithfulness("", ["some context"]) == 0.0

    def test_empty_context(self):
        score = compute_faithfulness("Some answer here.", [])
        assert score == 0.0

    def test_partial_overlap(self):
        answer = "Students must attend. The weather is nice."
        context = ["Students must attend all lectures."]
        score = compute_faithfulness(answer, context)
        assert 0 < score < 1


class TestEvaluateAnswerQuality:

    def test_without_bert(self):
        result = evaluate_answer_quality(
            predicted="test answer",
            reference="test answer",
            compute_bert=False,
        )
        assert "rouge_1" in result
        assert "bert_score_f1" not in result

    def test_with_context_adds_faithfulness(self):
        result = evaluate_answer_quality(
            predicted="students attend",
            reference="students attend",
            context_texts=["students must attend lectures"],
            compute_bert=False,
        )
        assert "faithfulness" in result

    def test_without_context_no_faithfulness(self):
        result = evaluate_answer_quality(
            predicted="test",
            reference="test",
            context_texts=None,
            compute_bert=False,
        )
        assert "faithfulness" not in result


class TestComputeBertScore:

    def test_returns_p_r_f1_from_mock_bert_score(self):
        with patch(
            "bert_score.score",
            return_value=([_FakeTensor(0.91)], [_FakeTensor(0.82)], [_FakeTensor(0.865)]),
        ) as mock_bert:
            scores = compute_bert_score("pred", "ref")
        assert scores["bert_score_precision"] == pytest.approx(0.91)
        assert scores["bert_score_recall"] == pytest.approx(0.82)
        assert scores["bert_score_f1"] == pytest.approx(0.865)
        mock_bert.assert_called_once()


class TestEvaluateAnswerQualityEdgeCases:

    def test_with_bert_delegates_to_bert_score(self):
        with patch(
            "bert_score.score",
            return_value=([_FakeTensor(0.7)], [_FakeTensor(0.6)], [_FakeTensor(0.65)]),
        ):
            result = evaluate_answer_quality(
                predicted="p",
                reference="r",
                context_texts=None,
                compute_bert=True,
            )
        assert result["bert_score_f1"] == pytest.approx(0.65)

    def test_whitespace_only_answer_has_zero_faithfulness(self):
        score = compute_faithfulness("   \n\t", ["ctx"])
        assert score == 0.0

    def test_answer_sentence_with_only_stopwords_counts_as_faithful(self):
        # "the a and or" is one sentence whose content_words set is empty after
        # stopword filtering (every token is a stopword); the function counts
        # it as trivially faithful (if not content_words: faithful_count +=1).
        score = compute_faithfulness("the a and or", ["unrelated context"])
        assert score == pytest.approx(1.0)
