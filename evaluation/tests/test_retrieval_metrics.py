"""Tests for retrieval quality metrics."""

import json

import pytest

from evaluation.metrics.retrieval_metrics import (
    _reset_section_cache,
    evaluate_retrieval,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    sections_to_chunk_ids,
)


class TestPrecisionAtK:

    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["a", "b", "c"], ["x", "y"], k=3) == 0.0

    def test_partial_relevant(self):
        assert precision_at_k(["a", "b", "c"], ["a", "c"], k=3) == pytest.approx(2 / 3)

    def test_k_zero(self):
        assert precision_at_k(["a"], ["a"], k=0) == 0.0

    def test_k_larger_than_retrieved(self):
        assert precision_at_k(["a"], ["a"], k=5) == pytest.approx(1 / 5)


class TestRecallAtK:

    def test_full_recall(self):
        assert recall_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_no_recall(self):
        assert recall_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x"], ["a", "b"], k=2) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["a"], [], k=1) == 0.0


class TestMRR:

    def test_first_position(self):
        assert mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_position(self):
        assert mrr(["x", "a", "c"], ["a"]) == 0.5

    def test_no_relevant(self):
        assert mrr(["x", "y"], ["a"]) == 0.0

    def test_multiple_relevant_returns_first(self):
        assert mrr(["x", "a", "b"], ["a", "b"]) == 0.5


class TestNDCGAtK:

    def test_perfect_ranking(self):
        # All relevant items at top positions
        assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_no_relevant(self):
        assert ndcg_at_k(["x", "y"], ["a"], k=2) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], [], k=2) == 0.0

    def test_k_zero(self):
        assert ndcg_at_k(["a"], ["a"], k=0) == 0.0

    def test_imperfect_ranking(self):
        # Relevant item at position 2, not position 1
        result = ndcg_at_k(["x", "a"], ["a"], k=2)
        assert 0 < result < 1


class TestEvaluateRetrieval:

    def test_returns_all_metric_keys(self):
        result = evaluate_retrieval(["a", "b"], ["a"], k_values=[1, 5])
        assert "mrr" in result
        assert "precision_at_1" in result
        assert "recall_at_5" in result
        assert "ndcg_at_1" in result

    def test_default_k_values(self):
        result = evaluate_retrieval(["a"], ["a"])
        assert "precision_at_1" in result
        assert "precision_at_3" in result
        assert "precision_at_5" in result
        assert "precision_at_10" in result


class TestSectionsToChunkIds:
    """Retrieval metrics must be computed on chunk IDs, not section
    labels. If section labels were used in place of chunk IDs, recall
    and nDCG could exceed 1.0 whenever two retrieved chunks shared a
    section. The helper expands sections into the correct chunk IDs.
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _reset_section_cache()
        yield
        _reset_section_cache()

    def _fixture_chunks(self):
        return [
            {"id": "a1", "section": "A", "text": "foo"},
            {"id": "a2", "section": "A", "text": "bar"},
            {"id": "b1", "section": "B", "text": "baz"},
        ]

    def test_iterable_source_expands_sections(self):
        ids = sections_to_chunk_ids(self._fixture_chunks(), ["A"])
        assert ids == ["a1", "a2"]

    def test_multiple_sections_concatenate(self):
        ids = sections_to_chunk_ids(self._fixture_chunks(), ["A", "B"])
        assert ids == ["a1", "a2", "b1"]

    def test_missing_section_yields_nothing(self):
        assert sections_to_chunk_ids(self._fixture_chunks(), ["Z"]) == []

    def test_duplicate_section_deduplicates_ids(self):
        ids = sections_to_chunk_ids(self._fixture_chunks(), ["A", "A"])
        assert ids == ["a1", "a2"]

    def test_path_source_reads_jsonl(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        chunks_file.write_text(
            "\n".join(json.dumps(c) for c in self._fixture_chunks()) + "\n",
            encoding="utf-8",
        )
        assert sections_to_chunk_ids(chunks_file, ["B"]) == ["b1"]

    def test_path_source_cached(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        chunks_file.write_text(
            "\n".join(json.dumps(c) for c in self._fixture_chunks()) + "\n",
            encoding="utf-8",
        )
        sections_to_chunk_ids(chunks_file, ["A"])
        # Mutate file on disk; cached result should not change.
        chunks_file.write_text("", encoding="utf-8")
        assert sections_to_chunk_ids(chunks_file, ["A"]) == ["a1", "a2"]

    def test_metrics_bounded_in_unit_interval(self):
        """When two retrieved chunks share a section, retrieval
        metrics must stay within [0, 1], because they operate on
        chunk IDs rather than on section labels.
        """
        chunks = self._fixture_chunks()
        retrieved_ids = ["a1", "a2", "b1"]
        relevant_ids = sections_to_chunk_ids(chunks, ["A"])

        result = evaluate_retrieval(retrieved_ids, relevant_ids, k_values=[1, 3, 5])

        for key, value in result.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} not in [0,1]"
        # Precise expected values: 2 of 3 retrieved are relevant (A-section).
        assert result["precision_at_3"] == pytest.approx(2 / 3)
        assert result["recall_at_3"] == 1.0
        assert 0.0 < result["ndcg_at_3"] <= 1.0

    def test_chunk_without_id_skipped_during_index_build(self):
        chunks = [
            {"id": "a1", "section": "A", "text": "foo"},
            {"section": "A", "text": "no-id chunk"},  # no id key
            {"id": "a2", "section": "A", "text": "bar"},
        ]
        ids = sections_to_chunk_ids(chunks, ["A"])
        assert ids == ["a1", "a2"]

    def test_jsonl_blank_line_skipped(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        chunks_file.write_text(
            json.dumps({"id": "a1", "section": "A"}) + "\n\n"
            + json.dumps({"id": "a2", "section": "A"}) + "\n",
            encoding="utf-8",
        )
        assert sections_to_chunk_ids(chunks_file, ["A"]) == ["a1", "a2"]


class TestNDCGEdgeCases:
    def test_idcg_zero_returns_zero(self):
        # No retrieved IDs match any relevant IDs, so DCG is 0; IDCG is positive
        # (there's a relevant item so IDCG > 0). The result is DCG/IDCG = 0.
        assert ndcg_at_k(["x", "y"], ["a", "b"], k=2) == 0.0
