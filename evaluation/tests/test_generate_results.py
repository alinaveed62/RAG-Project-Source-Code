"""Tests for generate_results.py: charts and LaTeX from the experiment CSVs."""

from __future__ import annotations

import json
import math

import pandas as pd

from evaluation.experiments.generate_results import (
    _baseline_summary,
    _bold_best,
    _fmt_float,
    _fmt_int,
    _fmt_latency_ms,
    _require_columns,
    _write_rows_tex,
    generate_all_results,
    generate_bar_chart,
    generate_latex_table,
    generate_report_tables,
)


class TestGenerateBarChart:
    def test_writes_file(self, tmp_path):
        df = pd.DataFrame({"model": ["a", "b"], "mrr": [0.5, 0.6]})
        out = tmp_path / "bar.png"
        generate_bar_chart(df, x_col="model", y_cols=["mrr"], title="T", output_path=out)
        assert out.exists() and out.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path):
        df = pd.DataFrame({"model": ["a"], "mrr": [0.5]})
        out = tmp_path / "nested" / "deep" / "bar.png"
        generate_bar_chart(df, x_col="model", y_cols=["mrr"], title="T", output_path=out)
        assert out.exists()

    def test_multiple_y_cols_grouped(self, tmp_path):
        df = pd.DataFrame({"model": ["a", "b"], "p1": [0.5, 0.6], "p2": [0.7, 0.8]})
        out = tmp_path / "bar.png"
        generate_bar_chart(
            df, x_col="model", y_cols=["p1", "p2"], title="T", output_path=out
        )
        assert out.exists()

    def test_ylabel_override(self, tmp_path):
        df = pd.DataFrame({"model": ["a"], "ms": [50.0]})
        out = tmp_path / "bar.png"
        generate_bar_chart(
            df,
            x_col="model",
            y_cols=["ms"],
            title="T",
            output_path=out,
            ylabel="Milliseconds",
        )
        assert out.exists()


class TestGenerateLatexTable:
    def test_writes_tex_tabular(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [0.12345, 0.54321]})
        out = tmp_path / "t.tex"
        generate_latex_table(df, output_path=out)
        text = out.read_text(encoding="utf-8")
        assert "\\begin{tabular}" in text
        assert "0.1235" in text  # 4dp float formatter
        assert "0.5432" in text

    def test_caption_and_label_wrap_in_table_env(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        out = tmp_path / "t.tex"
        generate_latex_table(
            df, output_path=out, caption="Example caption", label="tab:example"
        )
        text = out.read_text(encoding="utf-8")
        assert "\\begin{table}[htbp]" in text
        assert "\\centering" in text
        assert "\\caption{Example caption}" in text
        assert "\\label{tab:example}" in text
        assert "\\end{table}" in text

    def test_no_caption_no_table_env(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        out = tmp_path / "t.tex"
        generate_latex_table(df, output_path=out)
        text = out.read_text(encoding="utf-8")
        assert "\\begin{table}" not in text
        assert "\\begin{tabular}" in text

    def test_caption_only_wraps_table(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        out = tmp_path / "t.tex"
        generate_latex_table(df, output_path=out, caption="Only caption")
        text = out.read_text(encoding="utf-8")
        assert "\\begin{table}[htbp]" in text
        assert "\\caption{Only caption}" in text
        assert "\\label" not in text

    def test_label_only_wraps_table(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        out = tmp_path / "t.tex"
        generate_latex_table(df, output_path=out, label="tab:example")
        text = out.read_text(encoding="utf-8")
        assert "\\begin{table}[htbp]" in text
        assert "\\label{tab:example}" in text
        assert "\\caption" not in text

    def test_integer_column_not_4dp_formatted(self, tmp_path):
        df = pd.DataFrame({"n_queries": [49, 50]})
        out = tmp_path / "t.tex"
        generate_latex_table(df, output_path=out)
        text = out.read_text(encoding="utf-8")
        assert "49" in text
        assert "49.0000" not in text

    def test_creates_parent_dirs(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        out = tmp_path / "nested" / "deep" / "t.tex"
        generate_latex_table(df, output_path=out)
        assert out.exists()


def _write_csv(results_dir, filename: str, df: pd.DataFrame) -> None:
    path = results_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class TestGenerateAllResults:
    def test_skip_when_no_csvs(self, tmp_path):
        generate_all_results(tmp_path / "results")
        # No figures/tables directories created because every branch is a
        # csv.exists() guard that also creates parents inside.
        assert not (tmp_path / "results" / "figures").exists()

    def test_embedding_comparison_produces_chart_and_table(self, tmp_path):
        results = tmp_path / "results"
        _write_csv(
            results,
            "embedding_comparison.csv",
            pd.DataFrame(
                {
                    "model": ["a", "b"],
                    "mrr": [0.5, 0.6],
                    "precision_at_5": [0.4, 0.5],
                    "recall_at_5": [0.3, 0.4],
                    "ndcg_at_5": [0.5, 0.55],
                }
            ),
        )
        generate_all_results(results)
        assert (results / "figures" / "embedding_comparison.png").exists()
        assert (results / "tables" / "embedding_comparison.tex").exists()

    def test_topk_comparison_produces_chart_and_table(self, tmp_path):
        results = tmp_path / "results"
        _write_csv(
            results,
            "topk_comparison.csv",
            pd.DataFrame(
                {
                    "top_k": [3, 5, 7, 10],
                    "rouge_1": [0.5, 0.55, 0.6, 0.62],
                    "rouge_2": [0.3, 0.32, 0.34, 0.36],
                    "rouge_l": [0.45, 0.46, 0.48, 0.5],
                }
            ),
        )
        generate_all_results(results)
        assert (results / "figures" / "topk_answer_quality.png").exists()
        assert (results / "tables" / "topk_comparison.tex").exists()

    def test_chunk_size_comparison_produces_chart_and_table(self, tmp_path):
        results = tmp_path / "results"
        _write_csv(
            results,
            "chunk_size_comparison.csv",
            pd.DataFrame(
                {
                    "chunk_size": [256, 512, 1024],
                    "mrr": [0.4, 0.5, 0.55],
                    "precision_at_5": [0.3, 0.35, 0.4],
                    "recall_at_5": [0.2, 0.3, 0.4],
                }
            ),
        )
        generate_all_results(results)
        assert (results / "figures" / "chunk_size_comparison.png").exists()
        assert (results / "tables" / "chunk_size_comparison.tex").exists()

    def test_llm_comparison_with_quality_and_latency(self, tmp_path):
        results = tmp_path / "results"
        _write_csv(
            results,
            "llm_comparison.csv",
            pd.DataFrame(
                {
                    "model": ["mistral", "gemma2:2b"],
                    "avg_rouge_1": [0.5, 0.6],
                    "avg_rouge_l": [0.4, 0.5],
                    "avg_faithfulness": [0.8, 0.85],
                    "avg_generation_ms": [4200, 1800],
                }
            ),
        )
        generate_all_results(results)
        assert (results / "figures" / "llm_answer_quality.png").exists()
        assert (results / "figures" / "llm_latency.png").exists()
        assert (results / "tables" / "llm_comparison.tex").exists()

    def test_llm_comparison_without_quality_cols(self, tmp_path):
        results = tmp_path / "results"
        _write_csv(
            results,
            "llm_comparison.csv",
            pd.DataFrame({"model": ["mistral"], "avg_generation_ms": [4200]}),
        )
        generate_all_results(results)
        assert not (results / "figures" / "llm_answer_quality.png").exists()
        assert (results / "figures" / "llm_latency.png").exists()
        assert (results / "tables" / "llm_comparison.tex").exists()

    def test_llm_comparison_without_latency_col(self, tmp_path):
        results = tmp_path / "results"
        _write_csv(
            results,
            "llm_comparison.csv",
            pd.DataFrame({"model": ["mistral"], "avg_rouge_1": [0.5]}),
        )
        generate_all_results(results)
        assert (results / "figures" / "llm_answer_quality.png").exists()
        assert not (results / "figures" / "llm_latency.png").exists()

    def test_retrieval_comparison_with_metric_cols(self, tmp_path):
        results = tmp_path / "results"
        _write_csv(
            results,
            "retrieval_comparison.csv",
            pd.DataFrame(
                {
                    "strategy": ["dense", "bm25", "hybrid"],
                    "mrr": [0.5, 0.45, 0.55],
                    "precision_at_3": [0.4, 0.38, 0.42],
                    "recall_at_3": [0.3, 0.28, 0.33],
                    "ndcg_at_3": [0.45, 0.42, 0.48],
                }
            ),
        )
        generate_all_results(results)
        assert (results / "figures" / "retrieval_comparison.png").exists()
        assert (results / "tables" / "retrieval_comparison.tex").exists()

    def test_retrieval_comparison_without_metric_cols(self, tmp_path):
        results = tmp_path / "results"
        _write_csv(
            results,
            "retrieval_comparison.csv",
            pd.DataFrame({"strategy": ["dense"], "n_queries": [49]}),
        )
        generate_all_results(results)
        assert not (results / "figures" / "retrieval_comparison.png").exists()
        assert (results / "tables" / "retrieval_comparison.tex").exists()

    def test_reranking_comparison_with_metric_subset(self, tmp_path):
        results = tmp_path / "results"
        _write_csv(
            results,
            "reranking_comparison.csv",
            pd.DataFrame(
                {
                    "strategy": ["dense_only", "dense_rerank", "bm25_rerank"],
                    "mrr": [0.5, 0.55, 0.48],
                    "precision_at_1": [0.4, 0.45, 0.38],
                    "precision_at_5": [0.35, 0.4, 0.33],
                    "ndcg_at_5": [0.45, 0.48, 0.42],
                }
            ),
        )
        generate_all_results(results)
        assert (results / "figures" / "reranking_comparison.png").exists()
        assert (results / "tables" / "reranking_comparison.tex").exists()

    def test_reranking_comparison_without_metric_cols(self, tmp_path):
        results = tmp_path / "results"
        _write_csv(
            results,
            "reranking_comparison.csv",
            pd.DataFrame({"strategy": ["dense_only"], "avg_retrieval_ms": [30.0]}),
        )
        generate_all_results(results)
        assert not (results / "figures" / "reranking_comparison.png").exists()
        assert (results / "tables" / "reranking_comparison.tex").exists()

    def test_per_category_breakdown_with_metric_cols(self, tmp_path):
        results = tmp_path / "results"
        _write_csv(
            results,
            "per_category_breakdown.csv",
            pd.DataFrame(
                {
                    "category": ["attendance", "assessment"],
                    "mrr": [0.6, 0.5],
                    "precision_at_5": [0.4, 0.35],
                    "rouge_1": [0.5, 0.48],
                    "bertscore_f1": [0.75, 0.72],
                    "faithfulness": [0.8, 0.78],
                    "sgf": [0.7, 0.65],
                }
            ),
        )
        generate_all_results(results)
        assert (results / "figures" / "per_category_breakdown.png").exists()
        assert (results / "tables" / "per_category_breakdown.tex").exists()

    def test_per_category_breakdown_without_metric_cols(self, tmp_path):
        results = tmp_path / "results"
        _write_csv(
            results,
            "per_category_breakdown.csv",
            pd.DataFrame({"category": ["a"], "n_queries": [5]}),
        )
        generate_all_results(results)
        assert not (results / "figures" / "per_category_breakdown.png").exists()
        assert (results / "tables" / "per_category_breakdown.tex").exists()

    def test_resource_profile_with_percentile_cols(self, tmp_path):
        results = tmp_path / "results"
        _write_csv(
            results,
            "resource_profile.csv",
            pd.DataFrame(
                {
                    "model": ["mistral", "gemma2:2b"],
                    "p50_total_ms": [2100, 900],
                    "p95_total_ms": [4500, 1600],
                    "p99_total_ms": [6200, 2100],
                }
            ),
        )
        generate_all_results(results)
        assert (results / "figures" / "resource_percentiles.png").exists()
        assert (results / "tables" / "resource_profile.tex").exists()

    def test_resource_profile_without_latency_cols(self, tmp_path):
        results = tmp_path / "results"
        _write_csv(
            results,
            "resource_profile.csv",
            pd.DataFrame({"model": ["mistral"], "throughput": [0.5]}),
        )
        generate_all_results(results)
        assert not (results / "figures" / "resource_percentiles.png").exists()
        assert (results / "tables" / "resource_profile.tex").exists()


class TestFmtFloat:
    def test_normal(self):
        assert _fmt_float(0.12345, 3) == "0.123"

    def test_digits_respected(self):
        assert _fmt_float(0.12345, 4) == "0.1235"

    def test_integer_input(self):
        assert _fmt_float(1, 2) == "1.00"

    def test_none(self):
        assert _fmt_float(None) == "--"

    def test_nan(self):
        assert _fmt_float(float("nan")) == "--"


class TestFmtInt:
    def test_rounds_float(self):
        assert _fmt_int(1.6) == "2"

    def test_plain_int(self):
        assert _fmt_int(42) == "42"

    def test_none(self):
        assert _fmt_int(None) == "--"

    def test_nan(self):
        assert _fmt_int(float("nan")) == "--"


class TestFmtLatencyMs:
    def test_sub_hundred_two_decimals(self):
        assert _fmt_latency_ms(0.08) == "0.08"

    def test_sub_hundred_boundary(self):
        assert _fmt_latency_ms(99.9) == "99.90"

    def test_three_digit_integer(self):
        assert _fmt_latency_ms(855.4) == "855"

    def test_thousands_separator(self):
        assert _fmt_latency_ms(3961.2) == "3\\,961"

    def test_five_digit(self):
        assert _fmt_latency_ms(27841.54) == "27\\,842"

    def test_none(self):
        assert _fmt_latency_ms(None) == "--"

    def test_nan(self):
        assert _fmt_latency_ms(float("nan")) == "--"


class TestBoldBest:
    def test_max_bolds_highest(self):
        out = _bold_best([0.1, 0.2, 0.15], ["0.100", "0.200", "0.150"], "max")
        assert out == ["0.100", "\\textbf{0.200}", "0.150"]

    def test_min_bolds_lowest(self):
        out = _bold_best([5.0, 1.5, 3.0], ["5", "1.5", "3"], "min")
        assert out == ["5", "\\textbf{1.5}", "3"]

    def test_ties_not_bolded(self):
        # All-equal case: nothing wins.
        out = _bold_best([0.5, 0.5, 0.5], ["a", "b", "c"], "max")
        assert out == ["a", "b", "c"]

    def test_shared_max_both_bolded(self):
        out = _bold_best([0.5, 0.6, 0.6], ["a", "b", "c"], "max")
        assert out == ["a", "\\textbf{b}", "\\textbf{c}"]

    def test_single_value_not_bolded(self):
        out = _bold_best([0.3], ["x"], "max")
        assert out == ["x"]

    def test_nan_skipped(self):
        out = _bold_best([float("nan"), 0.5, 0.3], ["nan", "0.5", "0.3"], "max")
        assert out == ["nan", "\\textbf{0.5}", "0.3"]

    def test_none_skipped(self):
        out = _bold_best([None, 0.2, 0.4], ["none", "0.2", "0.4"], "max")
        assert out == ["none", "0.2", "\\textbf{0.4}"]

    def test_all_missing_returns_copy(self):
        out = _bold_best([None, float("nan")], ["a", "b"], "max")
        assert out == ["a", "b"]

    def test_display_precision_tie_bolds_both(self):
        # Two raw values that round identically at the chosen precision
        # should both be bolded, even though the underlying floats differ.
        out = _bold_best(
            [0.1198, 0.1203, 0.0677],
            ["0.120", "0.120", "0.068"],
            "max",
        )
        assert out == ["\\textbf{0.120}", "\\textbf{0.120}", "0.068"]


class TestWriteRowsTex:
    def test_writes_and_creates_parent(self, tmp_path):
        out = tmp_path / "nested" / "rows.tex"
        _write_rows_tex([["a", "b"], ["c", "d"]], out)
        text = out.read_text(encoding="utf-8")
        assert text == "a & b \\\\\nc & d \\\\\n"


class TestRequireColumns:
    def test_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert _require_columns(df, ("a", "b"), "x.csv") is True

    def test_missing_logged(self, caplog):
        df = pd.DataFrame({"a": [1]})
        import logging
        with caplog.at_level(logging.WARNING):
            assert _require_columns(df, ("a", "b"), "x.csv") is False
        assert "missing required columns" in caplog.text
        assert "['b']" in caplog.text


class TestBaselineSummary:
    def _write_baseline(self, tmp_path, rows):
        path = tmp_path / "baseline_results.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def test_retrieval_metrics_include_refusals(self, tmp_path):
        rows = [
            {"is_refusal": False, "mrr": 0.6, "precision_at_5": 0.5,
             "retrieval_time_ms": 30, "generation_time_ms": 8000,
             "rouge_1": 0.2, "bert_score_f1": 0.85},
            {"is_refusal": True, "mrr": 1.0, "precision_at_5": 0.4,
             "retrieval_time_ms": 20, "generation_time_ms": 0},
        ]
        path = self._write_baseline(tmp_path, rows)
        summary = _baseline_summary(path)
        assert summary["n_total"] == 2
        assert summary["n_refusals"] == 1
        assert summary["n_generated"] == 1
        # MRR averages over both rows because retrieval runs for
        # every query, including refusals.
        assert math.isclose(summary["mrr"], (0.6 + 1.0) / 2)
        # Answer-quality keys present only from the generated row.
        assert math.isclose(summary["rouge_1"], 0.2)
        assert math.isclose(summary["bert_score_f1"], 0.85)
        # Retrieval latency averages over both; generation latency only over generated.
        assert math.isclose(summary["retrieval_time_ms"], 25.0)
        assert math.isclose(summary["generation_time_ms"], 8000.0)

    def test_missing_metric_skipped(self, tmp_path):
        rows = [{"is_refusal": False, "mrr": 0.5}]
        path = self._write_baseline(tmp_path, rows)
        summary = _baseline_summary(path)
        assert "mrr" in summary
        assert "rouge_1" not in summary
        assert "generation_time_ms" not in summary

    def test_empty_list_returns_zero_counts(self, tmp_path):
        path = self._write_baseline(tmp_path, [])
        summary = _baseline_summary(path)
        assert summary["n_total"] == 0
        assert summary["n_refusals"] == 0
        assert summary["n_generated"] == 0


class TestGenerateReportTables:
    """End-to-end tests for the report-ready row-file generation."""

    def _baseline_json(self, results_dir, rows=None):
        results_dir.mkdir(parents=True, exist_ok=True)
        default_row = {
            "is_refusal": False,
            "mrr": 0.6,
            "precision_at_1": 0.5,
            "precision_at_5": 0.3,
            "recall_at_5": 0.08,
            "ndcg_at_5": 0.45,
            "ndcg_at_10": 0.35,
            "rouge_1": 0.15,
            "rouge_2": 0.07,
            "rouge_l": 0.13,
            "bert_score_precision": 0.8,
            "bert_score_recall": 0.88,
            "bert_score_f1": 0.84,
            "faithfulness": 0.76,
            "retrieval_time_ms": 30.0,
            "generation_time_ms": 8000.0,
        }
        payload = rows if rows is not None else [default_row]
        (results_dir / "baseline_results.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_writes_baseline_rows(self, tmp_path):
        results = tmp_path / "results"
        self._baseline_json(results)
        out = tmp_path / "Tables"
        generate_report_tables(results, out_dir=out)
        assert (out / "baseline_retrieval_rows.tex").exists()
        assert (out / "baseline_answer_rows.tex").exists()
        retrieval = (out / "baseline_retrieval_rows.tex").read_text(encoding="utf-8")
        assert "MRR & 0.600 \\\\" in retrieval
        answer = (out / "baseline_answer_rows.tex").read_text(encoding="utf-8")
        assert "Average generation latency & 8.0\\,s \\\\" in answer
        assert "Average retrieval latency & 30.0\\,ms \\\\" in answer

    def test_embedding_rows(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        pd.DataFrame({
            "model": ["a", "b"],
            "encoding_time_s": [1.5, 2.5],
            "mrr": [0.5, 0.6],
            "precision_at_5": [0.3, 0.35],
            "ndcg_at_5": [0.4, 0.45],
        }).to_csv(results / "embedding_comparison.csv", index=False)
        out = tmp_path / "Tables"
        generate_report_tables(results, out_dir=out)
        text = (out / "embedding_comparison_rows.tex").read_text(encoding="utf-8")
        # model b wins every metric → bolded values for b only.
        assert "a & 1.50 & 0.500 & 0.300 & 0.400 \\\\" in text
        assert "b & 2.50 & \\textbf{0.600} & \\textbf{0.350} & \\textbf{0.450} \\\\" in text

    def test_retrieval_rows_latency_is_min_bolded(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        pd.DataFrame({
            "strategy": ["dense_faiss", "sparse_bm25"],
            "avg_retrieval_ms": [0.08, 0.55],
            "mrr": [0.6, 0.5],
            "precision_at_5": [0.3, 0.2],
            "ndcg_at_5": [0.4, 0.3],
        }).to_csv(results / "retrieval_comparison.csv", index=False)
        out = tmp_path / "Tables"
        generate_report_tables(results, out_dir=out)
        text = (out / "retrieval_comparison_rows.tex").read_text(encoding="utf-8")
        # Dense wins min-latency and max-quality.
        assert "Dense (FAISS) & \\textbf{0.08} & \\textbf{0.600}" in text
        # Sparse alias renamed in display mapping.
        assert "Sparse (BM25)" in text

    def test_topk_rows(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        pd.DataFrame({
            "top_k": [3, 5],
            "n_refusals": [12, 16],
            "mrr": [0.55, 0.6],
            "precision_at_k": [0.5, 0.45],
            "rouge_1": [0.25, 0.22],
            "rouge_2": [0.1, 0.09],
            "rouge_l": [0.2, 0.17],
        }).to_csv(results / "topk_comparison.csv", index=False)
        out = tmp_path / "Tables"
        generate_report_tables(results, out_dir=out)
        text = (out / "topk_comparison_rows.tex").read_text(encoding="utf-8")
        assert "3 & 12" in text
        assert "\\textbf{0.250}" in text  # rouge_1 winner at k=3

    def test_llm_rows_display_names_and_bolding(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        pd.DataFrame({
            "model": ["mistral", "gemma2:2b"],
            "params_b": [7.0, 2.0],
            "size_gb": [4.1, 1.6],
            "n_refusals": [17, 17],
            "p50_total_ms": [27841.5, 8377.6],
            "avg_rouge_1": [0.14, 0.15],
            "avg_rouge_l": [0.12, 0.14],
            "avg_bert_score_f1": [0.85, 0.84],
            "avg_faithfulness": [0.77, 0.75],
        }).to_csv(results / "llm_comparison.csv", index=False)
        out = tmp_path / "Tables"
        generate_report_tables(results, out_dir=out)
        text = (out / "llm_comparison_rows.tex").read_text(encoding="utf-8")
        assert "Mistral 7B" in text
        assert "Gemma 2 2B & 2.0 & 1.6 & 17 & \\textbf{8\\,378}" in text

    def test_chunk_size_rows(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        pd.DataFrame({
            "chunk_size": [256, 512],
            "num_chunks": [659, 379],
            "mrr": [0.58, 0.60],
            "precision_at_5": [0.29, 0.30],
            "ndcg_at_5": [0.35, 0.37],
        }).to_csv(results / "chunk_size_comparison.csv", index=False)
        out = tmp_path / "Tables"
        generate_report_tables(results, out_dir=out)
        text = (out / "chunk_size_comparison_rows.tex").read_text(encoding="utf-8")
        assert "256 & 659" in text
        assert "512 & 379" in text

    def test_skips_missing_baseline(self, tmp_path, caplog):
        results = tmp_path / "results"
        results.mkdir()
        out = tmp_path / "Tables"
        import logging
        with caplog.at_level(logging.WARNING):
            generate_report_tables(results, out_dir=out)
        assert "baseline_results.json missing" in caplog.text
        assert not (out / "baseline_retrieval_rows.tex").exists()

    def test_skips_missing_csv(self, tmp_path, caplog):
        results = tmp_path / "results"
        self._baseline_json(results)
        out = tmp_path / "Tables"
        import logging
        with caplog.at_level(logging.WARNING):
            generate_report_tables(results, out_dir=out)
        # No CSVs provided; baseline rows ARE written but embedding etc. are not.
        assert (out / "baseline_retrieval_rows.tex").exists()
        assert not (out / "embedding_comparison_rows.tex").exists()
        assert "embedding_comparison.csv missing" in caplog.text

    def test_skips_csv_with_missing_columns(self, tmp_path, caplog):
        results = tmp_path / "results"
        results.mkdir()
        # Skinny CSV without precision_at_5.
        pd.DataFrame({"model": ["a"], "encoding_time_s": [1.0], "mrr": [0.5]}).to_csv(
            results / "embedding_comparison.csv", index=False
        )
        out = tmp_path / "Tables"
        import logging
        with caplog.at_level(logging.WARNING):
            generate_report_tables(results, out_dir=out)
        assert "missing required columns" in caplog.text
        assert not (out / "embedding_comparison_rows.tex").exists()

    def test_out_dir_defaults_to_report_tables_dir(self, tmp_path, monkeypatch):
        # Monkey-patch the constant so we do not pollute the real report dir.
        fake_report_dir = tmp_path / "fake_report_tables"
        monkeypatch.setattr(
            "evaluation.experiments.generate_results.REPORT_TABLES_DIR",
            fake_report_dir,
        )
        results = tmp_path / "results"
        self._baseline_json(results)
        generate_report_tables(results)
        assert (fake_report_dir / "baseline_retrieval_rows.tex").exists()
