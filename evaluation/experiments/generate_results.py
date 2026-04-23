"""Generate CSV tables, matplotlib charts, and LaTeX tables from experiment results."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from evaluation.config import REPORT_TABLES_DIR

logger = logging.getLogger(__name__)


def generate_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    output_path: Path,
    ylabel: str = "Score",
) -> None:
    """Generate a grouped bar chart.

    Args:
        df: DataFrame with data.
        x_col: Column for x-axis categories.
        y_cols: Columns to plot as bars.
        title: Chart title.
        output_path: Path to save the figure.
        ylabel: Y-axis label.
    """
    _fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df))
    width = 0.8 / len(y_cols)

    for i, col in enumerate(y_cols):
        offset = (i - len(y_cols) / 2 + 0.5) * width
        ax.bar(
            [xi + offset for xi in x],
            df[col],
            width=width,
            label=col,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col], rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved chart to %s", output_path)


def generate_latex_table(
    df: pd.DataFrame, output_path: Path, caption: str = "", label: str = ""
) -> None:
    """Generate a LaTeX table from a DataFrame.

    Args:
        df: DataFrame to convert.
        output_path: Path to save the .tex file.
        caption: LaTeX table caption.
        label: LaTeX table label.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Format float columns to 4 decimal places
    formatters = {}
    for col in df.columns:
        if df[col].dtype in ("float64", "float32"):
            formatters[col] = lambda x: f"{x:.4f}"

    latex = df.to_latex(index=False, formatters=formatters, escape=True)

    if caption or label:
        # Insert caption and label after \begin{tabular}
        header = [
            "\\begin{table}[htbp]",
            "\\centering",
        ]
        if caption:
            header.append(f"\\caption{{{caption}}}")
        if label:
            header.append(f"\\label{{{label}}}")
        latex = "\n".join(header) + "\n" + latex + "\n\\end{table}"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)
    logger.info("Saved LaTeX table to %s", output_path)


_BASELINE_RETRIEVAL_ROWS = (
    ("mrr", "MRR", 3),
    ("precision_at_1", "Precision@1", 3),
    ("precision_at_5", "Precision@5", 3),
    ("recall_at_5", "Recall@5", 3),
    ("ndcg_at_5", "nDCG@5", 3),
    ("ndcg_at_10", "nDCG@10", 3),
)

_BASELINE_ANSWER_ROWS = (
    ("rouge_1", "ROUGE-1 F1", 3),
    ("rouge_2", "ROUGE-2 F1", 3),
    ("rouge_l", "ROUGE-L F1", 3),
    ("bert_score_precision", "BERTScore Precision", 3),
    ("bert_score_recall", "BERTScore Recall", 3),
    ("bert_score_f1", "BERTScore F1", 3),
    ("faithfulness", "Faithfulness", 3),
)

_RETRIEVAL_STRATEGY_DISPLAY = {
    "dense_faiss": "Dense (FAISS)",
    "sparse_bm25": "Sparse (BM25)",
    "hybrid_rrf": "Hybrid (RRF)",
}

_LLM_MODEL_DISPLAY = {
    "mistral": "Mistral 7B",
    "llama3.2": "Llama 3.2",
    "phi3:mini": "Phi-3 Mini",
    "gemma2:2b": "Gemma 2 2B",
}


def _is_nan(x: object) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _fmt_float(x: float | int | None, digits: int = 3) -> str:
    """Format a numeric value as a decimal string, NaN-safe."""
    if x is None or _is_nan(x):
        return "--"
    return f"{float(x):.{digits}f}"


def _fmt_int(x: float | int | None) -> str:
    if x is None or _is_nan(x):
        return "--"
    return str(round(float(x)))


def _fmt_latency_ms(x: float | int | None) -> str:
    """Format a latency in milliseconds with sensible precision.

    Below 100 ms: two decimal places (for example 0.08).
    Between 100 and 999 ms: rounded to an integer (for example 855).
    1000 ms or more: grouped into thousands with a LaTeX thin space
    (for example 3\\,961).
    """
    if x is None or _is_nan(x):
        return "--"
    value = float(x)
    if value < 100:
        return f"{value:.2f}"
    rounded = round(value)
    if rounded < 1000:
        return str(rounded)
    thousands, rest = divmod(rounded, 1000)
    return f"{thousands}\\,{rest:03d}"


def _bold_best(
    raw_values: list[float | int | None],
    formatted: list[str],
    direction: str = "max",
) -> list[str]:
    """Wrap the winning cell (or cells) in \\textbf{}.

    Ties are judged against the formatted display string rather than
    the raw float, so two rows that round to the same displayed
    precision are both shown in bold. This avoids the situation where
    one 0.120 is bolded and another 0.120 is not because their
    underlying float representations differ past the display
    precision. If every numeric value is identical, or fewer than
    two rows have values, the cells are returned unbolded.
    """
    numeric: list[tuple[int, float]] = []
    for idx, v in enumerate(raw_values):
        if v is None or _is_nan(v):
            continue
        numeric.append((idx, float(v)))
    if len(numeric) < 2:
        return list(formatted)
    values = [v for _, v in numeric]
    if max(values) == min(values):
        return list(formatted)
    target = max(values) if direction == "max" else min(values)
    # Bold every row whose numeric value equals the extremum, or
    # whose formatted string matches the extremum's formatted
    # string. This way real ties and display-precision ties both
    # appear consistent.
    target_idx = next(i for i, v in numeric if v == target)
    target_fmt = formatted[target_idx]
    out = list(formatted)
    for idx, v in numeric:
        if v == target or formatted[idx] == target_fmt:
            out[idx] = f"\\textbf{{{out[idx]}}}"
    return out


def _write_rows_tex(rows: list[list[str]], out_path: Path) -> None:
    """Write row bodies only: cells joined with & and ending with \\\\."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(" & ".join(cells) + " \\\\" for cells in rows) + "\n"
    out_path.write_text(body, encoding="utf-8")
    logger.info("Wrote %d rows to %s", len(rows), out_path)


def _require_columns(df: pd.DataFrame, cols: tuple[str, ...], csv_name: str) -> bool:
    """Return True if every column in cols is present on df.

    Logs a warning listing any missing columns, so a partially
    populated CSV (for example a fixture that omits latency) does
    not crash the whole report-table regeneration.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning(
            "%s missing required columns %s; skipping rows file.",
            csv_name,
            sorted(missing),
        )
        return False
    return True


def _baseline_summary(baseline_json_path: Path) -> dict[str, float]:
    """Aggregate per-query baseline_results.json into flat averages.

    Retrieval metrics average over every scored row (refusals
    included, because retrieval runs before the refusal gate fires).
    Answer-quality metrics average only over rows where is_refusal is
    False, so the boilerplate refusal text does not pollute ROUGE,
    BERTScore or faithfulness. Generation latency also skips refusals
    (they have zero generation time by construction). Retrieval
    latency averages over every row, because retrieval runs for every
    query.
    """
    with baseline_json_path.open(encoding="utf-8") as f:
        rows = json.load(f)

    retrieval_keys = (
        "mrr",
        "precision_at_1",
        "precision_at_3",
        "precision_at_5",
        "precision_at_10",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "ndcg_at_1",
        "ndcg_at_3",
        "ndcg_at_5",
        "ndcg_at_10",
    )
    answer_keys = (
        "rouge_1",
        "rouge_2",
        "rouge_l",
        "bert_score_precision",
        "bert_score_recall",
        "bert_score_f1",
        "faithfulness",
        "sgf",
        "nli_faith",
        "section_match",
    )

    summary: dict[str, float] = {}
    summary["n_total"] = float(len(rows))
    summary["n_refusals"] = float(sum(1 for r in rows if r.get("is_refusal")))
    summary["n_generated"] = summary["n_total"] - summary["n_refusals"]

    for key in retrieval_keys:
        values = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        if values:
            summary[key] = sum(values) / len(values)

    for key in answer_keys:
        values = [
            float(r[key])
            for r in rows
            if not r.get("is_refusal") and isinstance(r.get(key), (int, float))
        ]
        if values:
            summary[key] = sum(values) / len(values)

    retrieval_lat = [
        float(r["retrieval_time_ms"])
        for r in rows
        if isinstance(r.get("retrieval_time_ms"), (int, float))
    ]
    if retrieval_lat:
        summary["retrieval_time_ms"] = sum(retrieval_lat) / len(retrieval_lat)
    generation_lat = [
        float(r["generation_time_ms"])
        for r in rows
        if not r.get("is_refusal")
        and isinstance(r.get("generation_time_ms"), (int, float))
    ]
    if generation_lat:
        summary["generation_time_ms"] = sum(generation_lat) / len(generation_lat)

    return summary


def _write_baseline_retrieval_rows(summary: dict[str, float], out_path: Path) -> None:
    rows = [[label, _fmt_float(summary.get(key), digits)] for key, label, digits in _BASELINE_RETRIEVAL_ROWS]
    _write_rows_tex(rows, out_path)


def _write_baseline_answer_rows(summary: dict[str, float], out_path: Path) -> None:
    rows: list[list[str]] = [
        [label, _fmt_float(summary.get(key), digits)]
        for key, label, digits in _BASELINE_ANSWER_ROWS
    ]
    gen_ms = summary.get("generation_time_ms")
    if gen_ms is not None:
        rows.append(["Average generation latency", f"{gen_ms / 1000:.1f}\\,s"])
    ret_ms = summary.get("retrieval_time_ms")
    if ret_ms is not None:
        rows.append(["Average retrieval latency", f"{ret_ms:.1f}\\,ms"])
    _write_rows_tex(rows, out_path)


def _write_embedding_rows(df: pd.DataFrame, out_path: Path) -> None:
    required = ("model", "encoding_time_s", "mrr", "precision_at_5", "ndcg_at_5")
    if not _require_columns(df, required, "embedding_comparison.csv"):
        return
    models = df["model"].tolist()
    encoding = df["encoding_time_s"].tolist()
    mrr = df["mrr"].tolist()
    p5 = df["precision_at_5"].tolist()
    ndcg5 = df["ndcg_at_5"].tolist()

    enc_fmt = [f"{float(v):.2f}" for v in encoding]
    mrr_fmt = _bold_best(mrr, [_fmt_float(v, 3) for v in mrr], "max")
    p5_fmt = _bold_best(p5, [_fmt_float(v, 3) for v in p5], "max")
    ndcg5_fmt = _bold_best(ndcg5, [_fmt_float(v, 3) for v in ndcg5], "max")

    rows = [list(cells) for cells in zip(models, enc_fmt, mrr_fmt, p5_fmt, ndcg5_fmt)]
    _write_rows_tex(rows, out_path)


def _write_chunk_size_rows(df: pd.DataFrame, out_path: Path) -> None:
    required = ("chunk_size", "num_chunks", "mrr", "precision_at_5", "ndcg_at_5")
    if not _require_columns(df, required, "chunk_size_comparison.csv"):
        return
    sizes = [_fmt_int(v) for v in df["chunk_size"].tolist()]
    chunks = [_fmt_int(v) for v in df["num_chunks"].tolist()]
    mrr = df["mrr"].tolist()
    p5 = df["precision_at_5"].tolist()
    ndcg5 = df["ndcg_at_5"].tolist()

    mrr_fmt = _bold_best(mrr, [_fmt_float(v, 3) for v in mrr], "max")
    p5_fmt = _bold_best(p5, [_fmt_float(v, 3) for v in p5], "max")
    ndcg5_fmt = _bold_best(ndcg5, [_fmt_float(v, 3) for v in ndcg5], "max")

    rows = [list(cells) for cells in zip(sizes, chunks, mrr_fmt, p5_fmt, ndcg5_fmt)]
    _write_rows_tex(rows, out_path)


def _write_topk_rows(df: pd.DataFrame, out_path: Path) -> None:
    required = (
        "top_k",
        "n_refusals",
        "mrr",
        "precision_at_k",
        "rouge_1",
        "rouge_2",
        "rouge_l",
    )
    if not _require_columns(df, required, "topk_comparison.csv"):
        return
    ks = [str(int(v)) for v in df["top_k"].tolist()]
    refusals = [str(int(v)) for v in df["n_refusals"].tolist()]
    mrr = df["mrr"].tolist()
    p_at_k = df["precision_at_k"].tolist()
    r1 = df["rouge_1"].tolist()
    r2 = df["rouge_2"].tolist()
    rl = df["rouge_l"].tolist()

    mrr_fmt = _bold_best(mrr, [_fmt_float(v, 3) for v in mrr], "max")
    p_fmt = _bold_best(p_at_k, [_fmt_float(v, 3) for v in p_at_k], "max")
    r1_fmt = _bold_best(r1, [_fmt_float(v, 3) for v in r1], "max")
    r2_fmt = _bold_best(r2, [_fmt_float(v, 3) for v in r2], "max")
    rl_fmt = _bold_best(rl, [_fmt_float(v, 3) for v in rl], "max")

    rows = [
        list(cells)
        for cells in zip(ks, refusals, mrr_fmt, p_fmt, r1_fmt, r2_fmt, rl_fmt)
    ]
    _write_rows_tex(rows, out_path)


def _write_retrieval_rows(df: pd.DataFrame, out_path: Path) -> None:
    required = (
        "strategy",
        "avg_retrieval_ms",
        "mrr",
        "precision_at_5",
        "ndcg_at_5",
    )
    if not _require_columns(df, required, "retrieval_comparison.csv"):
        return
    strategies = [
        _RETRIEVAL_STRATEGY_DISPLAY.get(str(s), str(s)) for s in df["strategy"].tolist()
    ]
    latency = df["avg_retrieval_ms"].tolist()
    mrr = df["mrr"].tolist()
    p5 = df["precision_at_5"].tolist()
    ndcg5 = df["ndcg_at_5"].tolist()

    lat_fmt = _bold_best(latency, [f"{float(v):.2f}" for v in latency], "min")
    mrr_fmt = _bold_best(mrr, [_fmt_float(v, 3) for v in mrr], "max")
    p5_fmt = _bold_best(p5, [_fmt_float(v, 3) for v in p5], "max")
    ndcg5_fmt = _bold_best(ndcg5, [_fmt_float(v, 3) for v in ndcg5], "max")

    rows = [list(cells) for cells in zip(strategies, lat_fmt, mrr_fmt, p5_fmt, ndcg5_fmt)]
    _write_rows_tex(rows, out_path)


def _write_llm_rows(df: pd.DataFrame, out_path: Path) -> None:
    required = (
        "model",
        "params_b",
        "size_gb",
        "n_refusals",
        "p50_total_ms",
        "avg_rouge_1",
        "avg_rouge_l",
        "avg_bert_score_f1",
        "avg_faithfulness",
    )
    if not _require_columns(df, required, "llm_comparison.csv"):
        return
    models = [_LLM_MODEL_DISPLAY.get(str(m), str(m)) for m in df["model"].tolist()]
    params = [f"{float(v):.1f}" for v in df["params_b"].tolist()]
    size = [f"{float(v):.1f}" for v in df["size_gb"].tolist()]
    refusals = [str(int(v)) for v in df["n_refusals"].tolist()]
    latency = df["p50_total_ms"].tolist()
    r1 = df["avg_rouge_1"].tolist()
    rl = df["avg_rouge_l"].tolist()
    bert_f1 = df["avg_bert_score_f1"].tolist()
    faith = df["avg_faithfulness"].tolist()

    lat_fmt = _bold_best(latency, [_fmt_latency_ms(v) for v in latency], "min")
    r1_fmt = _bold_best(r1, [_fmt_float(v, 3) for v in r1], "max")
    rl_fmt = _bold_best(rl, [_fmt_float(v, 3) for v in rl], "max")
    bert_fmt = _bold_best(bert_f1, [_fmt_float(v, 3) for v in bert_f1], "max")
    faith_fmt = _bold_best(faith, [_fmt_float(v, 3) for v in faith], "max")

    rows = [
        list(cells)
        for cells in zip(models, params, size, refusals, lat_fmt, r1_fmt, rl_fmt, bert_fmt, faith_fmt)
    ]
    _write_rows_tex(rows, out_path)


def generate_report_tables(
    results_dir: Path,
    out_dir: Path | None = None,
) -> None:
    """Write rows-only .tex files that Evaluation.tex \\inputs.

    Each file contains only the row bodies (cells & cells & \\\\)
    for one table. The chapter keeps the outer table, tabular,
    header, \\midrule and \\bottomrule wrapper, and the auto-
    generated rows slot in between via \\input{}. This keeps
    captions, labels and column choices under editorial control
    while the numeric content tracks the CSVs produced by the
    experiment scripts.

    Args:
        results_dir: Directory with the experiment CSVs and
            baseline_results.json (typically evaluation/results).
        out_dir: Directory to write the *_rows.tex files to.
            Defaults to the TeX project's tables/ directory via
            evaluation.config.REPORT_TABLES_DIR.
    """
    effective_out_dir: Path = out_dir if out_dir is not None else REPORT_TABLES_DIR

    baseline_json = results_dir / "baseline_results.json"
    if baseline_json.exists():
        summary = _baseline_summary(baseline_json)
        _write_baseline_retrieval_rows(
            summary, effective_out_dir / "baseline_retrieval_rows.tex"
        )
        _write_baseline_answer_rows(
            summary, effective_out_dir / "baseline_answer_rows.tex"
        )
    else:
        logger.warning(
            "baseline_results.json missing at %s; skipping baseline report rows.",
            baseline_json,
        )

    csv_writers = (
        ("embedding_comparison.csv", _write_embedding_rows, "embedding_comparison_rows.tex"),
        ("chunk_size_comparison.csv", _write_chunk_size_rows, "chunk_size_comparison_rows.tex"),
        ("topk_comparison.csv", _write_topk_rows, "topk_comparison_rows.tex"),
        ("retrieval_comparison.csv", _write_retrieval_rows, "retrieval_comparison_rows.tex"),
        ("llm_comparison.csv", _write_llm_rows, "llm_comparison_rows.tex"),
    )
    for csv_name, writer, out_name in csv_writers:
        csv_path = results_dir / csv_name
        if not csv_path.exists():
            logger.warning(
                "%s missing; skipping %s report rows.", csv_name, out_name
            )
            continue
        writer(pd.read_csv(csv_path), effective_out_dir / out_name)


def generate_all_results(results_dir: Path) -> None:
    """Generate charts and tables from all available result CSVs.

    Args:
        results_dir: Directory containing experiment result CSVs.
    """
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"

    # Embedding comparison
    emb_csv = results_dir / "embedding_comparison.csv"
    if emb_csv.exists():
        df = pd.read_csv(emb_csv)
        generate_bar_chart(
            df,
            x_col="model",
            y_cols=["mrr", "precision_at_5", "recall_at_5", "ndcg_at_5"],
            title="Retrieval Metrics by Embedding Model",
            output_path=figures_dir / "embedding_comparison.png",
        )
        generate_latex_table(
            df,
            output_path=tables_dir / "embedding_comparison.tex",
            caption="Retrieval metrics by embedding model",
            label="tab:embedding_comparison",
        )

    # Top-k comparison
    topk_csv = results_dir / "topk_comparison.csv"
    if topk_csv.exists():
        df = pd.read_csv(topk_csv)
        generate_bar_chart(
            df,
            x_col="top_k",
            y_cols=["rouge_1", "rouge_2", "rouge_l"],
            title="Answer Quality by Top-K",
            output_path=figures_dir / "topk_answer_quality.png",
        )
        generate_latex_table(
            df,
            output_path=tables_dir / "topk_comparison.tex",
            caption="Answer quality metrics by top-k value",
            label="tab:topk_comparison",
        )

    # Chunk size comparison
    chunk_csv = results_dir / "chunk_size_comparison.csv"
    if chunk_csv.exists():
        df = pd.read_csv(chunk_csv)
        generate_bar_chart(
            df,
            x_col="chunk_size",
            y_cols=["mrr", "precision_at_5", "recall_at_5"],
            title="Retrieval Metrics by Chunk Size",
            output_path=figures_dir / "chunk_size_comparison.png",
        )
        generate_latex_table(
            df,
            output_path=tables_dir / "chunk_size_comparison.tex",
            caption="Retrieval metrics by chunk size",
            label="tab:chunk_size_comparison",
        )

    # LLM model comparison
    llm_csv = results_dir / "llm_comparison.csv"
    if llm_csv.exists():
        df = pd.read_csv(llm_csv)
        # Answer quality chart
        quality_cols = [c for c in df.columns if c.startswith("avg_rouge") or c == "avg_faithfulness"]
        if quality_cols:
            generate_bar_chart(
                df,
                x_col="model",
                y_cols=quality_cols,
                title="Answer Quality by LLM Model",
                output_path=figures_dir / "llm_answer_quality.png",
            )
        # Generation time chart
        if "avg_generation_ms" in df.columns:
            generate_bar_chart(
                df,
                x_col="model",
                y_cols=["avg_generation_ms"],
                title="Average Generation Latency by LLM Model",
                output_path=figures_dir / "llm_latency.png",
                ylabel="Milliseconds",
            )
        generate_latex_table(
            df,
            output_path=tables_dir / "llm_comparison.tex",
            caption="Answer quality and resource demands by LLM model",
            label="tab:llm_comparison",
        )

    # Retrieval strategy comparison (dense vs BM25 vs hybrid)
    ret_csv = results_dir / "retrieval_comparison.csv"
    if ret_csv.exists():
        df = pd.read_csv(ret_csv)
        metric_cols = [c for c in df.columns if c.startswith(("mrr", "precision", "recall", "ndcg"))]
        if metric_cols:
            generate_bar_chart(
                df,
                x_col="strategy",
                y_cols=metric_cols[:4],  # Top 4 metrics
                title="Retrieval Metrics by Strategy (Dense vs Sparse vs Hybrid)",
                output_path=figures_dir / "retrieval_comparison.png",
            )
        generate_latex_table(
            df,
            output_path=tables_dir / "retrieval_comparison.tex",
            caption="Retrieval metrics by strategy (dense vs BM25 vs hybrid)",
            label="tab:retrieval_comparison",
        )

    # Cross-encoder reranking ablation chart and table.
    rerank_csv = results_dir / "reranking_comparison.csv"
    if rerank_csv.exists():
        df = pd.read_csv(rerank_csv)
        metric_cols = [c for c in df.columns if c in {"mrr", "precision_at_1", "precision_at_5", "ndcg_at_5"}]
        if metric_cols:
            generate_bar_chart(
                df,
                x_col="strategy",
                y_cols=metric_cols,
                title="Effect of Cross-Encoder Reranking on Retrieval Quality",
                output_path=figures_dir / "reranking_comparison.png",
            )
        generate_latex_table(
            df,
            output_path=tables_dir / "reranking_comparison.tex",
            caption="Effect of cross-encoder reranking on retrieval metrics (50 QA pairs, 95\\% bootstrap CIs)",
            label="tab:reranking_comparison",
        )

    # Per-category breakdown chart and table.
    cat_csv = results_dir / "per_category_breakdown.csv"
    if cat_csv.exists():
        df = pd.read_csv(cat_csv)
        metric_cols = [c for c in df.columns if c in {"mrr", "precision_at_5", "rouge_1", "bertscore_f1", "faithfulness", "sgf"}]
        if metric_cols:
            generate_bar_chart(
                df,
                x_col="category",
                y_cols=metric_cols,
                title="Per-Category Metric Breakdown",
                output_path=figures_dir / "per_category_breakdown.png",
            )
        generate_latex_table(
            df,
            output_path=tables_dir / "per_category_breakdown.tex",
            caption="Per-category retrieval and answer-quality metrics",
            label="tab:per_category_breakdown",
        )

    # Resource profile chart and table: P50, P95 and P99 latency
    # together with memory footprint.
    res_csv = results_dir / "resource_profile.csv"
    if res_csv.exists():
        df = pd.read_csv(res_csv)
        latency_cols = [c for c in df.columns if c in {"p50_total_ms", "p95_total_ms", "p99_total_ms"}]
        if latency_cols:
            generate_bar_chart(
                df,
                x_col="model",
                y_cols=latency_cols,
                title="LLM Latency Percentiles by Model (M2 Pro CPU)",
                output_path=figures_dir / "resource_percentiles.png",
                ylabel="Milliseconds",
            )
        generate_latex_table(
            df,
            output_path=tables_dir / "resource_profile.tex",
            caption="Resource profile by LLM model: latency percentiles, memory, throughput",
            label="tab:resource_profile",
        )

    generate_report_tables(results_dir)

    logger.info("Result generation complete")
