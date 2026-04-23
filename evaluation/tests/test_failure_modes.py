"""Tests for the failure-mode tagger and reporter."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.experiments.failure_modes import (
    FAILURE_MODES,
    classify_row,
    main,
    plot_stacked_bar,
    run_failure_mode_analysis,
    summarise_by_category,
    tag_rows,
)


class TestClassifyRow:
    def test_correct_refusal_when_mrr_zero(self):
        assert classify_row({"is_refusal": True, "mrr": 0.0}) == "correct_refusal"

    def test_over_refusal_when_relevant_chunk_present(self):
        assert (
            classify_row({"is_refusal": True, "mrr": 0.5})
            == "over_refusal"
        )

    def test_wrong_retrieval_when_answered_but_nothing_relevant(self):
        row = {"is_refusal": False, "mrr": 0.0, "rouge_l": 0.6}
        assert classify_row(row) == "wrong_retrieval"

    def test_low_quality_answer_below_threshold(self):
        row = {"is_refusal": False, "mrr": 0.5, "rouge_l": 0.1}
        assert classify_row(row) == "low_quality_answer"

    def test_partial_answer_mid_threshold(self):
        row = {"is_refusal": False, "mrr": 0.5, "rouge_l": 0.3}
        assert classify_row(row) == "partial_answer"

    def test_ok_above_threshold(self):
        row = {"is_refusal": False, "mrr": 0.5, "rouge_l": 0.5}
        assert classify_row(row) == "ok"

    def test_missing_rouge_treated_as_ok_when_retrieval_good(self):
        row = {"is_refusal": False, "mrr": 0.5}
        # Retrieval hit, but answer-quality metrics are absent; the
        # tagger refuses to pretend and labels the row on the retrieval
        # signal alone.
        assert classify_row(row) == "ok"

    def test_mrr_none_treated_as_zero(self):
        row = {"is_refusal": True, "mrr": None}
        assert classify_row(row) == "correct_refusal"

    def test_rouge_l_edge_at_0_2_is_partial(self):
        # 0.2 is the _ROUGE_L_LOW cutoff; the label transitions from
        # low_quality to partial at that point.
        row = {"is_refusal": False, "mrr": 0.5, "rouge_l": 0.2}
        assert classify_row(row) == "partial_answer"

    def test_rouge_l_edge_at_0_4_is_ok(self):
        row = {"is_refusal": False, "mrr": 0.5, "rouge_l": 0.4}
        assert classify_row(row) == "ok"


class TestTagRows:
    def test_returns_copied_rows(self):
        rows = [{"is_refusal": False, "mrr": 0.5, "rouge_l": 0.5, "id": "q1"}]
        tagged = tag_rows(rows)
        assert tagged[0]["failure_mode"] == "ok"
        # Originals are untouched.
        assert "failure_mode" not in rows[0]

    def test_empty_input_returns_empty(self):
        assert tag_rows([]) == []


class TestSummariseByCategory:
    def test_tallies_counts_per_category(self):
        tagged = [
            {"category": "attendance", "failure_mode": "ok"},
            {"category": "attendance", "failure_mode": "ok"},
            {"category": "attendance", "failure_mode": "correct_refusal"},
            {"category": "assessment", "failure_mode": "wrong_retrieval"},
        ]
        summary = summarise_by_category(tagged)
        assert set(summary.columns) >= {"category", "n_queries", *FAILURE_MODES}
        att = summary[summary["category"] == "attendance"].iloc[0]
        assert att["ok"] == 2
        assert att["correct_refusal"] == 1
        assert att["n_queries"] == 3
        asn = summary[summary["category"] == "assessment"].iloc[0]
        assert asn["wrong_retrieval"] == 1

    def test_missing_category_bucketed_as_unknown(self):
        tagged = [
            {"failure_mode": "ok"},
            {"category": None, "failure_mode": "ok"},
        ]
        summary = summarise_by_category(tagged)
        assert summary.iloc[0]["category"] == "unknown"
        assert summary.iloc[0]["n_queries"] == 2

    def test_empty_returns_empty_frame(self):
        summary = summarise_by_category([])
        assert summary.empty


class TestPlotStackedBar:
    def test_writes_png_for_populated_summary(self, tmp_path: Path):
        tagged = [
            {"category": "attendance", "failure_mode": "ok"},
            {"category": "attendance", "failure_mode": "correct_refusal"},
            {"category": "assessment", "failure_mode": "partial_answer"},
        ]
        summary = summarise_by_category(tagged)
        out = tmp_path / "fig.pdf"
        plot_stacked_bar(summary, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_writes_placeholder_for_empty_summary(self, tmp_path: Path):
        out = tmp_path / "fig.pdf"
        plot_stacked_bar(pd.DataFrame(), out)
        assert out.exists()
        assert out.stat().st_size > 0


class TestRunFailureModeAnalysis:
    def test_missing_baseline_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            run_failure_mode_analysis(
                baseline_json=tmp_path / "does_not_exist.json"
            )

    def test_non_list_raises(self, tmp_path: Path):
        baseline = tmp_path / "baseline.json"
        baseline.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a JSON list"):
            run_failure_mode_analysis(
                baseline_json=baseline,
                out_csv=tmp_path / "out.csv",
                summary_csv=tmp_path / "sum.csv",
                out_fig=tmp_path / "fig.pdf",
            )

    def test_happy_path_writes_all_artefacts(self, tmp_path: Path):
        baseline = tmp_path / "baseline.json"
        rows = [
            {
                "id": "att_01",
                "category": "attendance",
                "is_refusal": False,
                "mrr": 0.5,
                "rouge_l": 0.5,
            },
            {
                "id": "att_02",
                "category": "attendance",
                "is_refusal": True,
                "mrr": 0.0,
            },
            {
                "id": "ass_01",
                "category": "assessment",
                "is_refusal": False,
                "mrr": 0.0,
                "rouge_l": 0.9,
            },
            {
                "id": "ass_02",
                "category": "assessment",
                "is_refusal": False,
                "mrr": 0.5,
                "rouge_l": 0.1,
            },
        ]
        baseline.write_text(json.dumps(rows), encoding="utf-8")
        per_query_df, summary_df = run_failure_mode_analysis(
            baseline_json=baseline,
            out_csv=tmp_path / "fm.csv",
            summary_csv=tmp_path / "sum.csv",
            out_fig=tmp_path / "fig.pdf",
        )
        assert (tmp_path / "fm.csv").exists()
        assert (tmp_path / "sum.csv").exists()
        assert (tmp_path / "fig.pdf").exists()
        assert set(per_query_df["failure_mode"]) == {
            "ok",
            "correct_refusal",
            "wrong_retrieval",
            "low_quality_answer",
        }
        att_row = summary_df[summary_df["category"] == "attendance"].iloc[0]
        assert att_row["n_queries"] == 2

    def test_main_returns_zero(self, tmp_path: Path):
        baseline = tmp_path / "baseline.json"
        baseline.write_text(
            json.dumps(
                [
                    {
                        "id": "q1",
                        "category": "attendance",
                        "is_refusal": False,
                        "mrr": 0.5,
                        "rouge_l": 0.5,
                    }
                ]
            ),
            encoding="utf-8",
        )
        rc = main(
            [
                "--baseline",
                str(baseline),
                "--out-csv",
                str(tmp_path / "fm.csv"),
                "--summary-csv",
                str(tmp_path / "sum.csv"),
                "--out-fig",
                str(tmp_path / "fig.pdf"),
            ]
        )
        assert rc == 0


def test_failure_modes_constant_order():
    # The chart's stacking order depends on the declared tuple order, so
    # a silent reorder would change every report figure on the next
    # re-run.
    assert FAILURE_MODES[0] == "correct_refusal"
    assert FAILURE_MODES[-1] == "ok"
