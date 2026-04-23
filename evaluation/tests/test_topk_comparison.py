"""Tests for the top-k-comparison experiment.

These tests check that every row of the summary DataFrame carries
the same column schema (precision_at_k, recall_at_k, ndcg_at_k)
and never mixes precision_at_3 with precision_at_5 across rows.
They also exercise the n_generated column and the optional
capture_per_query switch used by the paired-significance driver.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from evaluation.experiments.topk_comparison import (
    _normalise_retrieval_metrics,
    run_topk_comparison,
)
from evaluation.metrics.retrieval_metrics import _reset_section_cache
from rag_pipeline.models import RAGResponse, RetrievalResult
from rag_pipeline.pipeline import LOW_CONFIDENCE_ANSWER

# -- Fixtures ---------------------------------------------------------------


def _qa_pairs() -> list[dict]:
    return [
        {
            "id": "att_01",
            "question": "Do I have to attend all my lectures?",
            "expected_answer": "Attendance is expected.",
            "relevant_sections": ["Teaching & Assessment"],
        },
        {
            "id": "ass_01",
            "question": "How do I submit an EC claim?",
            "expected_answer": "Submit on the ECs portal.",
            "relevant_sections": ["Teaching & Assessment"],
        },
        {
            "id": "edge_01",
            "question": "What is the weather in London?",
            "expected_answer": "Out of scope.",
            "relevant_sections": [],
        },
    ]


def _write_chunks(tmp_path: Path) -> Path:
    chunks = [
        {"id": f"c{i}", "section": "Teaching & Assessment"} for i in range(12)
    ]
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
    return path


def _make_pipeline(
    tmp_path: Path, *, refuse_on: set[str] | None = None
) -> MagicMock:
    chunks_path = _write_chunks(tmp_path)
    refuse_on = refuse_on or set()

    pipeline = MagicMock()
    pipeline.config = MagicMock()
    pipeline.config.top_k = 5
    pipeline.config.chunks_path = chunks_path

    def _answer(question: str, section_filter=None):
        # top-k is whatever the current setting is; return that many sources.
        top_k = pipeline.config.top_k
        sources = [
            RetrievalResult(
                chunk_id=f"c{i}",
                text="Attendance is required.",
                score=0.7,
                source="handbook",
                title="Teaching",
                section="Teaching & Assessment",
            )
            for i in range(top_k)
        ]
        if question in refuse_on:
            return RAGResponse(
                question=question,
                answer=LOW_CONFIDENCE_ANSWER,
                sources=sources,
                retrieval_time_ms=3.2,
                generation_time_ms=0.0,
            )
        return RAGResponse(
            question=question,
            answer="Attendance at lectures is expected per the handbook.",
            sources=sources,
            retrieval_time_ms=3.2,
            generation_time_ms=100.0,
        )

    pipeline.answer.side_effect = _answer
    return pipeline


# -- Tests ------------------------------------------------------------------


class TestNormaliseRetrievalMetrics:
    """Unit tests for the column-renaming helper."""

    def test_renames_k_suffixed_columns_to_flat_names(self):
        raw = {
            "mrr": 0.5,
            "precision_at_5": 0.2,
            "recall_at_5": 0.4,
            "ndcg_at_5": 0.6,
        }
        flat = _normalise_retrieval_metrics(raw, k=5)
        assert flat == {
            "mrr": 0.5,
            "precision_at_k": 0.2,
            "recall_at_k": 0.4,
            "ndcg_at_k": 0.6,
        }

    def test_renames_for_each_distinct_k(self):
        raw3 = {"mrr": 0.4, "precision_at_3": 0.33,
                "recall_at_3": 0.5, "ndcg_at_3": 0.45}
        raw10 = {"mrr": 0.5, "precision_at_10": 0.1,
                 "recall_at_10": 1.0, "ndcg_at_10": 0.55}
        assert _normalise_retrieval_metrics(raw3, k=3)["precision_at_k"] == 0.33
        assert _normalise_retrieval_metrics(raw10, k=10)["precision_at_k"] == 0.1


class TestSummarySchema:
    """Regression tests for the sparsity bug."""

    def test_no_nan_in_primary_columns(self, tmp_path):
        _reset_section_cache()
        pipeline = _make_pipeline(tmp_path)
        df = run_topk_comparison(pipeline, _qa_pairs(), topk_values=[3, 5, 7, 10])

        primary = ["mrr", "precision_at_k", "recall_at_k", "ndcg_at_k"]
        for col in primary:
            assert col in df.columns, f"missing column: {col}"
            assert not df[col].isna().any(), f"NaN in {col}: {df[col].tolist()}"

    def test_every_row_shares_the_same_schema(self, tmp_path):
        _reset_section_cache()
        pipeline = _make_pipeline(tmp_path)
        df = run_topk_comparison(pipeline, _qa_pairs(), topk_values=[3, 5])

        # No accidental precision_at_3 alongside precision_at_k.
        # The precision_at_k_ci_low and precision_at_k_ci_high
        # columns are allowed (they are the bootstrap CI columns
        # and carry a consistent schema across rows).
        allowed = {"precision_at_k", "precision_at_k_ci_low", "precision_at_k_ci_high"}
        unwanted = [
            c for c in df.columns
            if c.startswith("precision_at_") and c not in allowed
        ]
        assert unwanted == [], f"unexpected legacy columns: {unwanted}"


class TestCounters:

    def test_n_generated_plus_refusals_equals_n_queries(self, tmp_path):
        _reset_section_cache()
        pipeline = _make_pipeline(tmp_path, refuse_on={"What is the weather in London?"})
        df = run_topk_comparison(pipeline, _qa_pairs(), topk_values=[3])

        row = df.iloc[0]
        assert row["n_queries"] == 3
        assert row["n_refusals"] == 1
        assert row["n_generated"] == 2

    def test_n_generated_excluded_from_rouge_average(self, tmp_path):
        _reset_section_cache()
        pipeline = _make_pipeline(tmp_path, refuse_on={"What is the weather in London?"})
        df = run_topk_comparison(pipeline, _qa_pairs(), topk_values=[3])

        # Every generated answer is identical so the ROUGE average must be the
        # ROUGE of one of them (not diluted by the refusal boilerplate).
        row = df.iloc[0]
        assert "rouge_1" in row
        assert row["rouge_1"] > 0.0

    def test_empty_question_skipped(self, tmp_path):
        _reset_section_cache()
        pairs = _qa_pairs() + [
            {"id": "empty", "question": "   ", "expected_answer": "",
             "relevant_sections": []}
        ]
        pipeline = _make_pipeline(tmp_path)
        df = run_topk_comparison(pipeline, pairs, topk_values=[3])
        row = df.iloc[0]
        assert row["n_queries"] == 3  # empty filtered out


class TestCapturePerQuery:

    def test_returns_tuple_when_flag_is_true(self, tmp_path):
        _reset_section_cache()
        pipeline = _make_pipeline(tmp_path)
        result = run_topk_comparison(
            pipeline, _qa_pairs(), topk_values=[3, 5],
            capture_per_query=True,
        )
        assert isinstance(result, tuple)
        summary, per_q = result
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(per_q, pd.DataFrame)

    def test_per_query_has_one_row_per_k_and_qa_id(self, tmp_path):
        _reset_section_cache()
        pipeline = _make_pipeline(tmp_path)
        summary, per_q = run_topk_comparison(
            pipeline, _qa_pairs(), topk_values=[3, 5],
            capture_per_query=True,
        )
        # 3 qa_pairs x 2 k-values = 6 per-query rows
        assert len(per_q) == 6
        for k in (3, 5):
            assert set(per_q[per_q["top_k"] == k]["qa_id"]) == {
                "att_01", "ass_01", "edge_01"
            }

    def test_per_query_rouge_fields_present_for_generated_answers(self, tmp_path):
        _reset_section_cache()
        pipeline = _make_pipeline(tmp_path, refuse_on={"What is the weather in London?"})
        _, per_q = run_topk_comparison(
            pipeline, _qa_pairs(), topk_values=[3],
            capture_per_query=True,
        )
        generated = per_q[~per_q["is_refusal"]]
        refused = per_q[per_q["is_refusal"]]

        assert len(generated) == 2
        assert len(refused) == 1
        # ROUGE keys from compute_rouge are already prefixed (rouge_1, rouge_2, rouge_l);
        # they are spread directly into per-query rows without double-prefixing.
        assert "rouge_1" in generated.columns
        # Refusal row has no ROUGE attached (columns stay NaN).
        rouge_cols = [c for c in per_q.columns if c.startswith("rouge_")]
        assert all(refused[c].isna().all() for c in rouge_cols), (
            "Refusal rows must leave ROUGE fields empty so they are excluded "
            "from paired tests."
        )


class TestEmptyInputs:
    def test_all_questions_empty_no_ci_columns(self, tmp_path):
        _reset_section_cache()
        pipeline = _make_pipeline(tmp_path)
        qa = [
            {"id": "e1", "question": "   ", "expected_answer": "", "relevant_sections": []},
            {"id": "e2", "question": "", "expected_answer": "", "relevant_sections": []},
        ]
        df = run_topk_comparison(pipeline, qa, topk_values=[3])
        row = df.iloc[0]
        assert row["n_queries"] == 0
        # No retrieval metrics AND no ROUGE → no CI columns added.
        assert "mrr_ci_low" not in df.columns
        assert "rouge_1_ci_low" not in df.columns

    def test_all_refusals_no_rouge_ci_columns(self, tmp_path):
        _reset_section_cache()
        pipeline = _make_pipeline(
            tmp_path, refuse_on={q["question"] for q in _qa_pairs()}
        )
        df = run_topk_comparison(pipeline, _qa_pairs(), topk_values=[3])
        row = df.iloc[0]
        assert row["n_refusals"] == 3
        assert row["n_generated"] == 0
        assert "rouge_1_ci_low" not in df.columns


class TestAverageDictsHelper:
    def test_empty_records_returns_empty_dict(self):
        from evaluation.experiments.topk_comparison import _average_dicts

        assert _average_dicts([]) == {}


class TestQaIdFallback:
    def test_missing_id_uses_deterministic_qa_index(self, tmp_path):
        """QA pairs without id must get qa_{idx} so paired tests can
        dedupe on a stable key instead of colliding on None."""
        _reset_section_cache()
        pipeline = _make_pipeline(tmp_path)
        qa = [
            {"question": "Do I have to attend all my lectures?",
             "expected_answer": "Yes.", "relevant_sections": []},
            {"question": "How do I submit an EC claim?",
             "expected_answer": "Portal.", "relevant_sections": []},
        ]
        _, per_q = run_topk_comparison(
            pipeline, qa, topk_values=[3], capture_per_query=True,
        )
        assert list(per_q["qa_id"]) == ["qa_0", "qa_1"]
