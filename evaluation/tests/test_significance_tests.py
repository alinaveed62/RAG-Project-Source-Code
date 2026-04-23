"""Tests for the cross-experiment significance harness."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.experiments.significance_tests import (
    DEFAULT_METRICS,
    holm_bonferroni,
    load_per_query_rows,
    main,
    pairwise_significance,
    run_all_significance_tests,
    write_outputs,
)


def _rows_two_systems_clear_win(metric: str = "rouge_l") -> list[dict]:
    """Construct per-query rows where system A beats B with varied gaps.

    The gap is perturbed per query so the paired-difference vector has
    non-zero variance; a perfectly constant gap would collapse Cohen's d
    to 0 even though Wilcoxon would still flag significance.
    """
    return (
        [
            {
                "experiment": "llm_comparison",
                "system": "winner",
                "id": f"q{i:02d}",
                metric: 0.9 - 0.005 * i,
            }
            for i in range(30)
        ]
        + [
            {
                "experiment": "llm_comparison",
                "system": "loser",
                "id": f"q{i:02d}",
                metric: 0.3 + 0.01 * (i % 5),
            }
            for i in range(30)
        ]
    )


class TestHolmBonferroni:
    def test_empty_returns_empty(self):
        assert holm_bonferroni([]) == []

    def test_single_p_unchanged(self):
        # n = 1 Holm equals the identity: only one hypothesis in the family.
        assert holm_bonferroni([0.03]) == [0.03]

    def test_step_down_matches_textbook_example(self):
        # Known textbook example: p = [0.01, 0.04, 0.03]
        # sorted asc: 0.01 at idx 0, 0.03 at idx 2, 0.04 at idx 1
        # step-down multipliers: 3, 2, 1
        # adjusted sorted: [0.03, 0.06, 0.06] (monotone clamped)
        # clamped to [0, 1]; remapped to original order:
        #   idx 0 -> 0.03, idx 1 -> 0.06, idx 2 -> 0.06
        result = holm_bonferroni([0.01, 0.04, 0.03])
        assert result[0] == pytest.approx(0.03)
        assert result[2] == pytest.approx(0.06)
        assert result[1] == pytest.approx(0.06)

    def test_clamps_to_one(self):
        # Every multiplier pushes over 1.0; all should cap at 1.0.
        result = holm_bonferroni([0.5, 0.6, 0.7])
        assert all(p == 1.0 for p in result)

    def test_monotone_in_sorted_order(self):
        p_in = [0.001, 0.002, 0.04, 0.05]
        p_out = holm_bonferroni(p_in)
        sorted_indices = sorted(range(len(p_in)), key=lambda i: p_in[i])
        sorted_adjusted = [p_out[i] for i in sorted_indices]
        # Step-down enforces non-decreasing adjusted values along the
        # sorted ascending order of raw p-values.
        for lo, hi in zip(sorted_adjusted, sorted_adjusted[1:]):
            assert lo <= hi


class TestLoadPerQueryRows:
    def test_loads_valid_list(self, tmp_path: Path):
        path = tmp_path / "llm.json"
        path.write_text(
            json.dumps(
                [
                    {"experiment": "x", "system": "a", "id": "q1", "rouge_l": 0.1}
                ]
            ),
            encoding="utf-8",
        )
        rows = load_per_query_rows(path)
        assert len(rows) == 1
        assert rows[0]["system"] == "a"

    def test_non_list_raises(self, tmp_path: Path):
        path = tmp_path / "llm.json"
        path.write_text("{\"not\": \"a list\"}", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a JSON list"):
            load_per_query_rows(path)

    def test_missing_keys_raises(self, tmp_path: Path):
        path = tmp_path / "llm.json"
        path.write_text(
            json.dumps([{"system": "a", "id": "q1"}]), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="missing required keys"):
            load_per_query_rows(path)

    def test_non_dict_row_raises(self, tmp_path: Path):
        # A row that is a list / scalar / null must be rejected with a
        # clear message before the set(row) call would emit a confusing
        # "unsupported operand type" error.
        path = tmp_path / "llm.json"
        path.write_text(
            json.dumps([["not", "an", "object"]]), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="expected a JSON object"):
            load_per_query_rows(path)


class TestPairwiseSignificance:
    def test_empty_returns_empty_frame(self):
        assert pairwise_significance([]).empty

    def test_mixed_experiments_raises(self):
        rows = [
            {"experiment": "a", "system": "x", "id": "1", "rouge_l": 0.1},
            {"experiment": "b", "system": "y", "id": "1", "rouge_l": 0.2},
        ]
        with pytest.raises(ValueError, match="single experiment"):
            pairwise_significance(rows)

    def test_clear_win_flags_significant(self):
        rows = _rows_two_systems_clear_win()
        df = pairwise_significance(rows, metrics=("rouge_l",))
        assert not df.empty
        assert len(df) == 1  # one pair: winner vs loser
        row = df.iloc[0]
        assert row["system_a"] == "loser"
        assert row["system_b"] == "winner"
        assert row["n_pairs"] == 30
        assert row["p_holm"] < 0.01
        # DataFrame returns numpy.bool_, so compare by value not identity.
        assert bool(row["significant_holm_05"]) is True
        # winner's per-query rouge > loser's, so cohens_d(loser, winner) < 0.
        assert row["cohens_d"] < 0
        assert row["effect_label"] == "large"

    def test_metric_absent_is_skipped(self):
        # No row carries "faithfulness" so the harness should skip it and
        # not add a NaN placeholder that would break downstream CSV
        # consumers.
        rows = _rows_two_systems_clear_win(metric="rouge_l")
        df = pairwise_significance(
            rows, metrics=("rouge_l", "faithfulness")
        )
        assert set(df["metric"]) == {"rouge_l"}

    def test_missing_per_query_values_align(self):
        # System A has q01..q05; system B has q03..q07. Only q03..q05
        # form paired vectors, so n_pairs must be 3 not 5.
        rows = [
            {"experiment": "e", "system": "A", "id": f"q{i:02d}", "rouge_l": 0.5}
            for i in range(1, 6)
        ] + [
            {"experiment": "e", "system": "B", "id": f"q{i:02d}", "rouge_l": 0.2}
            for i in range(3, 8)
        ]
        df = pairwise_significance(rows, metrics=("rouge_l",))
        assert len(df) == 1
        assert df.iloc[0]["n_pairs"] == 3

    def test_none_value_drops_pair_symmetrically(self):
        rows = [
            {"experiment": "e", "system": "A", "id": "q1", "rouge_l": 0.5},
            {"experiment": "e", "system": "A", "id": "q2", "rouge_l": None},
            {"experiment": "e", "system": "B", "id": "q1", "rouge_l": 0.2},
            {"experiment": "e", "system": "B", "id": "q2", "rouge_l": 0.3},
        ]
        df = pairwise_significance(rows, metrics=("rouge_l",))
        assert df.iloc[0]["n_pairs"] == 1

    def test_three_systems_produce_three_pairs(self):
        rows = []
        for system, base in [("A", 0.1), ("B", 0.5), ("C", 0.9)]:
            for i in range(20):
                rows.append(
                    {
                        "experiment": "e",
                        "system": system,
                        "id": f"q{i:02d}",
                        "rouge_l": base + 0.003 * (i % 7),
                    }
                )
        df = pairwise_significance(rows, metrics=("rouge_l",))
        assert len(df) == 3
        # Holm correction applied across the 3 tests in the family.
        assert (df["p_holm"] >= df["p_raw"]).all()

    def test_multiple_metrics_outer_loop_iterates(self):
        # Exercises the outer for metric in metrics loop executing
        # more than once so both iteration paths are covered.
        rows = []
        for system, r_base, f_base in [("A", 0.2, 0.3), ("B", 0.7, 0.8)]:
            for i in range(15):
                rows.append(
                    {
                        "experiment": "e",
                        "system": system,
                        "id": f"q{i:02d}",
                        "rouge_l": r_base + 0.004 * (i % 5),
                        "faithfulness": f_base + 0.004 * (i % 5),
                    }
                )
        df = pairwise_significance(
            rows, metrics=("rouge_l", "faithfulness")
        )
        assert set(df["metric"]) == {"rouge_l", "faithfulness"}
        assert len(df) == 2


class TestRunAllSignificanceTests:
    def test_missing_dir_returns_empty(self, tmp_path: Path):
        df = run_all_significance_tests(per_query_dir=tmp_path / "missing")
        assert df.empty

    def test_empty_dir_returns_empty(self, tmp_path: Path):
        (tmp_path / "per_query").mkdir()
        df = run_all_significance_tests(per_query_dir=tmp_path / "per_query")
        assert df.empty

    def test_loads_and_concatenates(self, tmp_path: Path):
        d = tmp_path / "per_query"
        d.mkdir()
        rows = _rows_two_systems_clear_win()
        (d / "llm.json").write_text(json.dumps(rows), encoding="utf-8")
        # Second file with a different experiment tag.
        rows2 = [dict(r, experiment="retrieval_comparison") for r in rows]
        (d / "retrieval.json").write_text(json.dumps(rows2), encoding="utf-8")
        df = run_all_significance_tests(per_query_dir=d, metrics=("rouge_l",))
        assert set(df["experiment"]) == {
            "llm_comparison",
            "retrieval_comparison",
        }

    def test_empty_json_is_skipped_with_warning(self, tmp_path: Path):
        d = tmp_path / "per_query"
        d.mkdir()
        (d / "empty.json").write_text("[]", encoding="utf-8")
        df = run_all_significance_tests(per_query_dir=d)
        assert df.empty

    def test_file_with_no_requested_metrics_yields_empty_frame(
        self, tmp_path: Path
    ):
        # When the per-query JSON has rows but none of the rows carry any
        # metric in the requested set, pairwise_significance() returns an
        # empty frame, and run_all_significance_tests must skip appending
        # it so the final concat does not see an empty DataFrame.
        d = tmp_path / "per_query"
        d.mkdir()
        rows = [
            {
                "experiment": "e",
                "system": "A",
                "id": f"q{i}",
                "some_other_metric": 0.5,
            }
            for i in range(5)
        ] + [
            {
                "experiment": "e",
                "system": "B",
                "id": f"q{i}",
                "some_other_metric": 0.3,
            }
            for i in range(5)
        ]
        (d / "one.json").write_text(json.dumps(rows), encoding="utf-8")
        df = run_all_significance_tests(
            per_query_dir=d, metrics=("rouge_l",)
        )
        assert df.empty


class TestWriteOutputs:
    def test_writes_csv_and_tex(self, tmp_path: Path):
        df = pairwise_significance(
            _rows_two_systems_clear_win(), metrics=("rouge_l",)
        )
        csv_path = tmp_path / "out.csv"
        tex_path = tmp_path / "out.tex"
        write_outputs(df, out_csv=csv_path, out_tex=tex_path)
        assert csv_path.exists()
        assert tex_path.exists()
        # LaTeX contains the compact columns; raw p and n_nonzero_pairs
        # only appear in the CSV.
        tex_text = tex_path.read_text(encoding="utf-8")
        assert "cohens\\_d" in tex_text or "cohens_d" in tex_text
        assert "p\\_holm" in tex_text or "p_holm" in tex_text
        assert "n_nonzero_pairs" not in tex_text
        csv_text = csv_path.read_text(encoding="utf-8")
        assert "n_nonzero_pairs" in csv_text

    def test_empty_frame_writes_placeholder_tex(self, tmp_path: Path):
        csv_path = tmp_path / "out.csv"
        tex_path = tmp_path / "out.tex"
        write_outputs(pd.DataFrame(), out_csv=csv_path, out_tex=tex_path)
        assert csv_path.exists()
        # Placeholder mentions the reason explicitly so the chapter
        # reader understands why the table is missing numbers.
        assert "no per-query data" in tex_path.read_text(encoding="utf-8")


class TestMain:
    def test_main_with_missing_dir_writes_placeholder(self, tmp_path: Path):
        csv_path = tmp_path / "sig.csv"
        tex_path = tmp_path / "sig.tex"
        rc = main(
            [
                "--per-query-dir",
                str(tmp_path / "missing"),
                "--out-csv",
                str(csv_path),
                "--out-tex",
                str(tex_path),
            ]
        )
        assert rc == 0
        assert csv_path.exists()
        assert tex_path.exists()

    def test_main_with_populated_dir_writes_table(self, tmp_path: Path):
        d = tmp_path / "per_query"
        d.mkdir()
        (d / "llm.json").write_text(
            json.dumps(_rows_two_systems_clear_win()), encoding="utf-8"
        )
        csv_path = tmp_path / "sig.csv"
        tex_path = tmp_path / "sig.tex"
        rc = main(
            [
                "--per-query-dir",
                str(d),
                "--out-csv",
                str(csv_path),
                "--out-tex",
                str(tex_path),
                "--metric",
                "rouge_l",
            ]
        )
        assert rc == 0
        csv = pd.read_csv(csv_path)
        assert len(csv) == 1
        # Reading the bool back from CSV gives either a Python bool, a
        # numpy bool, or the string "True"; normalise and compare.
        assert str(csv.iloc[0]["significant_holm_05"]).lower() == "true"


def test_default_metrics_contains_expected():
    # Guard against silent edits that drop a metric from the
    # significance family without a conscious decision.
    assert "rouge_l" in DEFAULT_METRICS
    assert "faithfulness" in DEFAULT_METRICS
    assert "sgf" in DEFAULT_METRICS
