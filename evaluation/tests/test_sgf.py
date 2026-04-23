"""Tests for Section-Grounded Faithfulness."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evaluation.metrics import sgf as sgf_module
from evaluation.metrics.sgf import (
    _reset_default_nli_model,
    _section_match,
    _split_sentences,
    section_grounded_faithfulness,
)


@pytest.fixture(autouse=True)
def _clear_nli_cache():
    _reset_default_nli_model()
    yield
    _reset_default_nli_model()


def _mock_nli_model(entailment_probs: list[float]) -> MagicMock:
    """Build a mock CrossEncoder whose predict returns the given entailment
    probabilities (one per answer sentence). Contradiction and neutral are
    filled in to sum to 1."""
    model = MagicMock()

    def _predict(pairs, apply_softmax=True, batch_size=16):
        assert len(pairs) == len(entailment_probs)
        return [
            [0.5 * (1.0 - e), e, 0.5 * (1.0 - e)] for e in entailment_probs
        ]

    model.predict.side_effect = _predict
    return model


class TestSentenceSplitter:
    def test_splits_on_period(self):
        assert _split_sentences("One. Two. Three.") == ["One.", "Two.", "Three."]

    def test_splits_on_question_mark(self):
        assert _split_sentences("Hi? Bye.") == ["Hi?", "Bye."]

    def test_drops_empty(self):
        assert _split_sentences("") == []


class TestDefaultNLIModel:
    def test_loads_lazily_and_caches(self, monkeypatch):
        fake_cls_calls = []

        class _FakeCrossEncoder:
            def __init__(self, model_name):
                fake_cls_calls.append(model_name)

            def predict(self, pairs, apply_softmax=True, batch_size=16):
                return [[0.1, 0.8, 0.1] for _ in pairs]

        import sentence_transformers

        monkeypatch.setattr(sentence_transformers, "CrossEncoder", _FakeCrossEncoder)
        first = sgf_module._get_default_nli_model()
        second = sgf_module._get_default_nli_model()
        assert first is second
        assert len(fake_cls_calls) == 1
        assert fake_cls_calls[0] == sgf_module._DEFAULT_NLI_MODEL


class TestNLIFaithfulnessPrivate:
    def test_empty_sentences_returns_zero(self):
        assert sgf_module._nli_faithfulness([], "context", MagicMock(), batch_size=1) == 0.0

    def test_drops_whitespace_only(self):
        assert _split_sentences("   ") == []


class TestSectionMatch:
    def test_exact_match_is_one(self):
        assert _section_match(["A", "B"], ["A", "B"]) == 1.0

    def test_disjoint_is_zero(self):
        assert _section_match(["A"], ["B"]) == 0.0

    def test_partial_overlap_is_jaccard(self):
        # |{A,B} ∩ {B,C}| / |{A,B,C}| = 1 / 3
        assert _section_match(["A", "B"], ["B", "C"]) == pytest.approx(1 / 3)

    def test_both_empty_is_one(self):
        assert _section_match([], []) == 1.0

    def test_one_empty_is_zero(self):
        assert _section_match(["A"], []) == 0.0
        assert _section_match([], ["A"]) == 0.0

    def test_duplicates_collapsed(self):
        assert _section_match(["A", "A", "A"], ["A"]) == 1.0


class TestSGFAlpha:
    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            section_grounded_faithfulness(
                "answer.", ["ctx"], ["A"], ["A"], alpha=-0.1
            )
        with pytest.raises(ValueError, match="alpha"):
            section_grounded_faithfulness(
                "answer.", ["ctx"], ["A"], ["A"], alpha=1.5
            )

    def test_alpha_one_is_pure_nli(self):
        model = _mock_nli_model([0.9])
        out = section_grounded_faithfulness(
            "answer.",
            contexts=["ctx"],
            retrieved_sections=["A"],
            relevant_sections=["B"],  # zero section match
            alpha=1.0,
            nli_model=model,
        )
        assert out["nli_faith"] == pytest.approx(0.9)
        assert out["section_match"] == 0.0
        assert out["sgf"] == pytest.approx(0.9)

    def test_alpha_zero_is_pure_section_match(self):
        model = _mock_nli_model([0.0])  # nli_faith=0
        out = section_grounded_faithfulness(
            "answer.",
            contexts=["ctx"],
            retrieved_sections=["A"],
            relevant_sections=["A"],
            alpha=0.0,
            nli_model=model,
        )
        assert out["section_match"] == 1.0
        assert out["sgf"] == pytest.approx(1.0)


class TestSGFWhole:
    def test_perfect_match_and_entailment(self):
        model = _mock_nli_model([1.0, 1.0, 1.0])
        out = section_grounded_faithfulness(
            "One. Two. Three.",
            contexts=["full context"],
            retrieved_sections=["Teaching & Assessment"],
            relevant_sections=["Teaching & Assessment"],
            alpha=0.5,
            nli_model=model,
        )
        assert out["section_match"] == 1.0
        assert out["nli_faith"] == pytest.approx(1.0)
        assert out["sgf"] == pytest.approx(1.0)

    def test_zero_entailment_zero_match(self):
        model = _mock_nli_model([0.0, 0.0])
        out = section_grounded_faithfulness(
            "One. Two.",
            contexts=["ctx"],
            retrieved_sections=["A"],
            relevant_sections=["B"],
            alpha=0.5,
            nli_model=model,
        )
        assert out["sgf"] == 0.0

    def test_mixed_components_weighted_correctly(self):
        model = _mock_nli_model([0.8])
        out = section_grounded_faithfulness(
            "answer.",
            contexts=["ctx"],
            retrieved_sections=["A", "B"],
            relevant_sections=["B", "C"],  # Jaccard = 1/3
            alpha=0.5,
            nli_model=model,
        )
        expected_section = 1 / 3
        expected_sgf = 0.5 * 0.8 + 0.5 * expected_section
        assert out["nli_faith"] == pytest.approx(0.8)
        assert out["section_match"] == pytest.approx(expected_section)
        assert out["sgf"] == pytest.approx(expected_sgf)

    def test_empty_answer_yields_zero_nli(self):
        out = section_grounded_faithfulness(
            answer="",
            contexts=["ctx"],
            retrieved_sections=["A"],
            relevant_sections=["A"],
            alpha=0.5,
            nli_model=_mock_nli_model([]),
        )
        assert out["nli_faith"] == 0.0
        # Section match is still 1.0 so sgf = 0.5 * 0 + 0.5 * 1 = 0.5
        assert out["sgf"] == pytest.approx(0.5)

    def test_empty_contexts_yields_zero_nli(self):
        out = section_grounded_faithfulness(
            answer="An answer.",
            contexts=[],
            retrieved_sections=["A"],
            relevant_sections=["A"],
            alpha=0.5,
            nli_model=_mock_nli_model([]),
        )
        assert out["nli_faith"] == 0.0

    def test_single_sentence_answer(self):
        model = _mock_nli_model([0.7])
        out = section_grounded_faithfulness(
            answer="The only sentence.",
            contexts=["ctx"],
            retrieved_sections=["A"],
            relevant_sections=["A"],
            alpha=0.5,
            nli_model=model,
        )
        assert out["nli_faith"] == pytest.approx(0.7)
        assert out["section_match"] == 1.0
        assert out["sgf"] == pytest.approx(0.85)

    def test_result_keys_shape(self):
        out = section_grounded_faithfulness(
            "a.",
            ["c"],
            ["A"],
            ["A"],
            alpha=0.3,
            nli_model=_mock_nli_model([0.5]),
        )
        assert set(out.keys()) == {"sgf", "nli_faith", "section_match", "alpha"}
        assert out["alpha"] == 0.3


class TestDefaultModelLazyLoading:
    def test_default_model_loaded_only_when_needed(self, monkeypatch):
        """Calls that skip the NLI branch (empty answer or empty contexts)
        must not load the 400 MB model."""
        fake_loader = MagicMock()
        monkeypatch.setattr(sgf_module, "_get_default_nli_model", fake_loader)

        section_grounded_faithfulness(
            answer="",
            contexts=["ctx"],
            retrieved_sections=["A"],
            relevant_sections=["A"],
            alpha=0.5,
        )
        section_grounded_faithfulness(
            answer="something.",
            contexts=[],
            retrieved_sections=["A"],
            relevant_sections=["A"],
            alpha=0.5,
        )

        fake_loader.assert_not_called()

    def test_default_model_loaded_once(self, monkeypatch):
        mock_model = _mock_nli_model([0.5])
        fake_loader = MagicMock(return_value=mock_model)
        monkeypatch.setattr(sgf_module, "_get_default_nli_model", fake_loader)

        section_grounded_faithfulness(
            "answer.",
            ["ctx"],
            ["A"],
            ["A"],
            alpha=0.5,
        )
        fake_loader.assert_called_once()
