"""Tests for the chunking-strategy ablation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.experiments.chunking_strategy_comparison import (
    AGGREGATED_METRICS,
    StrategyInput,
    aggregate_results,
    describe_chunks,
    load_chunks,
    parse_args,
    plot_comparison,
    run_chunking_strategy_comparison,
    run_comparison,
)


class TestLoadChunks:
    def test_loads_jsonl(self, tmp_path: Path):
        path = tmp_path / "chunks.jsonl"
        path.write_text(
            json.dumps({"id": "c1", "text": "one two"}) + "\n"
            + json.dumps({"id": "c2", "text": "three four five"}) + "\n",
            encoding="utf-8",
        )
        rows = load_chunks(path)
        assert [r["id"] for r in rows] == ["c1", "c2"]

    def test_skips_blank_lines(self, tmp_path: Path):
        path = tmp_path / "chunks.jsonl"
        path.write_text(
            "\n"
            + json.dumps({"id": "c1", "text": "one"}) + "\n"
            + "\n"
            + json.dumps({"id": "c2", "text": "two"}) + "\n",
            encoding="utf-8",
        )
        rows = load_chunks(path)
        assert len(rows) == 2

    def test_invalid_line_raises(self, tmp_path: Path):
        path = tmp_path / "chunks.jsonl"
        path.write_text(
            json.dumps({"id": "c1"}) + "\nnot json at all\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="invalid JSON line"):
            load_chunks(path)

    def test_empty_file_returns_empty(self, tmp_path: Path):
        path = tmp_path / "chunks.jsonl"
        path.write_text("", encoding="utf-8")
        assert load_chunks(path) == []


class TestDescribeChunks:
    def test_empty_returns_zeros(self):
        assert describe_chunks([]) == {
            "n_chunks": 0,
            "mean_tokens": 0.0,
            "median_tokens": 0.0,
        }

    def test_basic_descriptors(self):
        chunks = [
            {"text": "one two three"},
            {"text": "four five"},
            {"text": "six seven eight nine"},
        ]
        out = describe_chunks(chunks)
        assert out["n_chunks"] == 3
        assert out["mean_tokens"] == pytest.approx(3.0)
        assert out["median_tokens"] == 3.0

    def test_handles_missing_text(self):
        out = describe_chunks([{"id": "c1"}])
        # .get('text', '') falls back to an empty string -> zero
        # tokens; the helper must not raise.
        assert out["n_chunks"] == 1
        assert out["mean_tokens"] == 0.0


class TestAggregateResults:
    def test_handles_empty(self):
        out = aggregate_results([])
        assert out["n_results"] == 0
        assert out["n_refusals"] == 0

    def test_averages_across_rows(self):
        results = [
            {
                "rouge_l": 0.5,
                "mrr": 1.0,
                "precision_at_3": 0.3333,
                "is_refusal": False,
            },
            {
                "rouge_l": 0.3,
                "mrr": 0.5,
                "precision_at_3": 0.6667,
                "is_refusal": False,
            },
        ]
        out = aggregate_results(results)
        assert out["rouge_l"] == pytest.approx(0.4)
        assert out["n_results"] == 2
        assert out["n_refusals"] == 0

    def test_skips_missing_metrics(self):
        # A refusal row drops BERTScore / ROUGE but retains retrieval
        # metrics; the aggregator must average only over rows where each
        # metric is present.
        results = [
            {"mrr": 0.0, "is_refusal": True},
            {"mrr": 1.0, "rouge_l": 0.6, "is_refusal": False},
        ]
        out = aggregate_results(results)
        assert out["mrr"] == pytest.approx(0.5)
        assert out["rouge_l"] == pytest.approx(0.6)
        assert out["n_refusals"] == 1

    def test_ignores_non_numeric(self):
        results = [
            {"mrr": "oops", "rouge_l": 0.5, "is_refusal": False},
            {"mrr": 0.7, "rouge_l": 0.3, "is_refusal": False},
        ]
        out = aggregate_results(results)
        assert out["mrr"] == pytest.approx(0.7)


class TestRunComparison:
    def test_invokes_evaluate_fn_per_strategy(self, tmp_path: Path):
        recursive = tmp_path / "recursive.jsonl"
        semantic = tmp_path / "semantic.jsonl"
        recursive.write_text(
            json.dumps({"id": "c1", "text": "alpha"}) + "\n",
            encoding="utf-8",
        )
        semantic.write_text(
            json.dumps({"id": "c1", "text": "beta"}) + "\n",
            encoding="utf-8",
        )
        strategies = (
            StrategyInput("recursive", recursive),
            StrategyInput("semantic", semantic),
        )
        seen: list[str] = []

        def evaluate_fn(spec, chunks):
            seen.append(spec.strategy)
            # Return two rows, one refusal one success, so refusal_rate
            # shows a non-trivial value.
            return [
                {"rouge_l": 0.5, "mrr": 1.0, "is_refusal": False},
                {"mrr": 0.0, "is_refusal": True},
            ]

        summary = run_comparison(strategies, evaluate_fn)
        assert seen == ["recursive", "semantic"]
        assert set(summary["strategy"]) == {"recursive", "semantic"}
        # Both strategies saw 1 refusal of 2 rows, so rate = 0.5.
        assert (summary["refusal_rate"] == 0.5).all()

    def test_zero_results_gives_zero_refusal_rate(self, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        strategies = (StrategyInput("recursive", path),)
        summary = run_comparison(strategies, lambda spec, chunks: [])
        assert summary.iloc[0]["refusal_rate"] == 0.0


class TestPlotComparison:
    def test_writes_populated_figure(self, tmp_path: Path):
        summary = pd.DataFrame(
            {
                "strategy": ["recursive", "semantic"],
                "mrr": [0.5, 0.6],
                "rouge_l": [0.4, 0.45],
            }
        )
        out = tmp_path / "fig.pdf"
        plot_comparison(summary, out)
        assert out.exists()

    def test_placeholder_when_empty(self, tmp_path: Path):
        out = tmp_path / "fig.pdf"
        plot_comparison(pd.DataFrame(), out)
        assert out.exists()

    def test_placeholder_when_no_metric_columns(self, tmp_path: Path):
        # The plot path guards against a DataFrame that has rows but no
        # aggregated-metric columns (e.g. every row refused).
        summary = pd.DataFrame({"strategy": ["recursive"], "n_results": [0]})
        out = tmp_path / "fig.pdf"
        plot_comparison(summary, out)
        assert out.exists()


class TestRunChunkingStrategyComparison:
    def test_end_to_end_with_injected_evaluator(self, tmp_path: Path):
        recursive = tmp_path / "recursive.jsonl"
        semantic = tmp_path / "semantic.jsonl"
        recursive.write_text(
            json.dumps({"id": "c1", "text": "alpha"}) + "\n",
            encoding="utf-8",
        )
        semantic.write_text(
            json.dumps({"id": "c1", "text": "beta"}) + "\n",
            encoding="utf-8",
        )

        def evaluate_fn(spec, chunks):
            return [{"rouge_l": 0.5, "mrr": 1.0, "is_refusal": False}]

        out_csv = tmp_path / "chunking.csv"
        out_fig = tmp_path / "chunking.pdf"
        summary = run_chunking_strategy_comparison(
            recursive_chunks=recursive,
            semantic_chunks=semantic,
            out_csv=out_csv,
            out_fig=out_fig,
            evaluate_fn=evaluate_fn,
        )
        assert out_csv.exists()
        assert out_fig.exists()
        assert set(summary["strategy"]) == {"recursive", "semantic"}


class TestParseArgs:
    def test_defaults(self):
        ns = parse_args([])
        assert ns.recursive_chunks.name.endswith(".jsonl")
        assert ns.semantic_chunks.name.endswith(".jsonl")

    def test_overrides(self):
        ns = parse_args(
            [
                "--recursive-chunks",
                "/tmp/a.jsonl",
                "--semantic-chunks",
                "/tmp/b.jsonl",
                "--out-csv",
                "/tmp/c.csv",
                "--out-fig",
                "/tmp/d.pdf",
            ]
        )
        assert str(ns.recursive_chunks) == "/tmp/a.jsonl"
        assert str(ns.out_csv) == "/tmp/c.csv"


def test_aggregated_metrics_contains_headline_metrics():
    assert "rouge_l" in AGGREGATED_METRICS
    assert "mrr" in AGGREGATED_METRICS
    assert "faithfulness" in AGGREGATED_METRICS
