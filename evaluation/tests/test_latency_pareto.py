"""Tests for the latency-quality Pareto frontier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluation.experiments.latency_pareto import (
    PARETO_INPUTS,
    ParetoInput,
    compute_frontier,
    is_on_frontier,
    load_all_frontiers,
    main,
    plot_frontier,
    run_latency_pareto,
)


class TestIsOnFrontier:
    def test_single_point_is_on_frontier(self):
        assert is_on_frontier([(10.0, 0.5)], 0) is True

    def test_dominated_point_removed(self):
        # Point B is dominated by A: A is faster AND higher quality.
        points = [(10.0, 0.9), (20.0, 0.5)]
        assert is_on_frontier(points, 0) is True
        assert is_on_frontier(points, 1) is False

    def test_trade_off_points_both_on_frontier(self):
        # Point A: fast but low quality; point B: slow but high quality.
        # Neither dominates the other.
        points = [(10.0, 0.5), (50.0, 0.9)]
        assert is_on_frontier(points, 0) is True
        assert is_on_frontier(points, 1) is True

    def test_duplicate_points_both_on_frontier(self):
        # Two points with identical coordinates are tied and neither
        # dominates; both are on the frontier by convention.
        points = [(10.0, 0.5), (10.0, 0.5)]
        assert is_on_frontier(points, 0) is True
        assert is_on_frontier(points, 1) is True

    def test_strictly_better_on_both_axes_dominates(self):
        points = [(10.0, 0.9), (20.0, 0.5), (15.0, 0.7)]
        # (10, 0.9) dominates both others.
        assert is_on_frontier(points, 0) is True
        assert is_on_frontier(points, 1) is False
        assert is_on_frontier(points, 2) is False

    def test_equal_latency_worse_quality_is_dominated(self):
        points = [(10.0, 0.9), (10.0, 0.5)]
        # Same latency; (10, 0.9) is strictly better in quality.
        assert is_on_frontier(points, 0) is True
        assert is_on_frontier(points, 1) is False


class TestComputeFrontier:
    def _spec(self, **kwargs) -> ParetoInput:
        defaults = dict(
            experiment="test_experiment",
            csv_name="test.csv",
            latency_column="lat",
            quality_column="q",
            label_column="name",
        )
        defaults.update(kwargs)
        return ParetoInput(**defaults)

    def test_missing_columns_raise(self):
        df = pd.DataFrame({"lat": [10], "q": [0.5]})  # no name
        with pytest.raises(ValueError, match="missing required columns"):
            compute_frontier(df, self._spec())

    def test_tags_frontier_and_dominated(self):
        df = pd.DataFrame(
            {
                "lat": [10, 20, 15],
                "q": [0.9, 0.5, 0.7],
                "name": ["A", "B", "C"],
            }
        )
        out = compute_frontier(df, self._spec())
        frontier = dict(zip(out["label"], out["is_on_frontier"]))
        # out["is_on_frontier"] is a pandas Series of numpy.bool_.
        # numpy.bool_(True) is True is False under recent numpy, so
        # prefer truthiness checks over identity comparisons.
        assert frontier["A"]
        assert not frontier["B"]
        assert not frontier["C"]

    def test_dropped_rows_warn_and_keep_rest(self, caplog):
        df = pd.DataFrame(
            {
                "lat": [10, None, 15],
                "q": [0.9, 0.5, 0.7],
                "name": ["A", "B", "C"],
            }
        )
        with caplog.at_level("WARNING"):
            out = compute_frontier(df, self._spec())
        assert len(out) == 2
        assert "dropping" in caplog.text.lower()

    def test_all_missing_returns_empty(self):
        df = pd.DataFrame(
            {
                "lat": [None, None],
                "q": [None, None],
                "name": ["A", "B"],
            }
        )
        out = compute_frontier(df, self._spec())
        assert out.empty
        # Column schema is preserved even when empty so downstream
        # concatenation does not error.
        assert set(out.columns) >= {
            "experiment",
            "label",
            "latency_ms",
            "quality",
            "is_on_frontier",
        }


class TestLoadAllFrontiers:
    def test_missing_dir_yields_empty_frame(self, tmp_path: Path):
        out = load_all_frontiers(tmp_path / "missing")
        assert out.empty

    def test_single_csv_loaded(self, tmp_path: Path):
        results = tmp_path / "results"
        results.mkdir()
        pd.DataFrame(
            {
                "model": ["gemma2:2b", "mistral"],
                "p50_total_ms": [100.0, 500.0],
                "avg_rouge_l": [0.5, 0.4],
            }
        ).to_csv(results / "llm_comparison.csv", index=False)
        out = load_all_frontiers(
            results, inputs=(PARETO_INPUTS[0],)
        )
        assert len(out) == 2
        assert out[out["label"] == "gemma2:2b"]["is_on_frontier"].iloc[0]

    def test_empty_csv_is_skipped(self, tmp_path: Path):
        results = tmp_path / "results"
        results.mkdir()
        pd.DataFrame(
            columns=["model", "p50_total_ms", "avg_rouge_l"]
        ).to_csv(results / "llm_comparison.csv", index=False)
        out = load_all_frontiers(results, inputs=(PARETO_INPUTS[0],))
        assert out.empty

    def test_missing_values_everywhere_returns_empty(self, tmp_path: Path):
        results = tmp_path / "results"
        results.mkdir()
        pd.DataFrame(
            {
                "model": ["a"],
                "p50_total_ms": [None],
                "avg_rouge_l": [None],
            }
        ).to_csv(results / "llm_comparison.csv", index=False)
        out = load_all_frontiers(results, inputs=(PARETO_INPUTS[0],))
        assert out.empty


class TestPlotFrontier:
    def test_writes_populated_figure(self, tmp_path: Path):
        frontier = pd.DataFrame(
            {
                "experiment": ["llm", "llm"],
                "label": ["gemma2:2b", "mistral"],
                "latency_ms": [100.0, 500.0],
                "quality": [0.5, 0.4],
                "is_on_frontier": [True, False],
            }
        )
        out = tmp_path / "fig.pdf"
        plot_frontier(frontier, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_writes_placeholder_for_empty_frontier(self, tmp_path: Path):
        out = tmp_path / "fig.pdf"
        plot_frontier(pd.DataFrame(), out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_renders_multiple_experiments(self, tmp_path: Path):
        frontier = pd.DataFrame(
            {
                "experiment": ["llm", "llm", "retrieval", "retrieval"],
                "label": ["gemma", "mistral", "dense", "bm25"],
                "latency_ms": [100.0, 500.0, 50.0, 30.0],
                "quality": [0.5, 0.4, 0.7, 0.5],
                "is_on_frontier": [True, False, True, True],
            }
        )
        out = tmp_path / "fig.pdf"
        plot_frontier(frontier, out)
        assert out.exists()

    def test_renders_experiment_without_frontier_points(self, tmp_path: Path):
        # Artificial case: every row is marked non-frontier. Ensures the
        # empty-frontier subset branch is covered.
        frontier = pd.DataFrame(
            {
                "experiment": ["llm", "llm"],
                "label": ["a", "b"],
                "latency_ms": [1.0, 2.0],
                "quality": [0.5, 0.5],
                "is_on_frontier": [False, False],
            }
        )
        out = tmp_path / "fig.pdf"
        plot_frontier(frontier, out)
        assert out.exists()


class TestRunLatencyPareto:
    def test_full_pipeline(self, tmp_path: Path):
        results = tmp_path / "results"
        results.mkdir()
        pd.DataFrame(
            {
                "model": ["gemma2:2b", "mistral"],
                "p50_total_ms": [100.0, 500.0],
                "avg_rouge_l": [0.5, 0.4],
            }
        ).to_csv(results / "llm_comparison.csv", index=False)
        csv_out = tmp_path / "pareto.csv"
        fig_out = tmp_path / "pareto.pdf"
        out = run_latency_pareto(
            results_dir=results,
            out_csv=csv_out,
            out_fig=fig_out,
            inputs=(PARETO_INPUTS[0],),
        )
        assert csv_out.exists()
        assert fig_out.exists()
        assert not out.empty


class TestMain:
    def test_main_missing_dir(self, tmp_path: Path):
        rc = main(
            [
                "--results-dir",
                str(tmp_path / "nope"),
                "--out-csv",
                str(tmp_path / "pareto.csv"),
                "--out-fig",
                str(tmp_path / "pareto.pdf"),
            ]
        )
        assert rc == 0
        assert (tmp_path / "pareto.csv").exists()
