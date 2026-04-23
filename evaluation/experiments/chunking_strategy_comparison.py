"""Ablation: recursive chunker vs semantic-boundary chunker.

The production corpus is built with the recursive chunker in
keats_scraper/processors/chunker.py; the semantic-boundary chunker
in keats_scraper/processors/semantic_chunker.py is an opt-in
alternative. This script runs the QA evaluation against each chunk
set and writes two artefacts: a CSV with one row per strategy
(carrying the headline metrics plus chunk-count and mean-token
descriptors) and a side-by-side bar chart of the quality metrics.

The pipeline used by the evaluation is supplied through an injected
evaluate_fn callable, so the orchestration can be unit tested
without loading Ollama or FAISS. The CLI entry point constructs
the real pipeline in the same way run_all.py does.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

# Force the non-interactive Agg backend before matplotlib.pyplot
# is imported; otherwise the backend is locked in by the pyplot
# import and matplotlib.use("Agg") becomes a silent no-op.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from evaluation.config import REPORT_FIGURES_DIR

logger = logging.getLogger(__name__)

DEFAULT_RECURSIVE_CHUNKS = Path(
    "keats_scraper/data/chunks/chunks_for_embedding.jsonl"
)
DEFAULT_SEMANTIC_CHUNKS = Path(
    "keats_scraper/data/chunks/chunks_for_embedding_semantic.jsonl"
)

DEFAULT_OUT_CSV = Path("evaluation/results/chunking_strategy_comparison.csv")
DEFAULT_OUT_FIG = REPORT_FIGURES_DIR / "chunking_strategy_comparison.pdf"

# Metrics summarised into the comparison CSV. These are the same keys the
# baseline evaluator already emits per query, so the aggregation below
# treats any missing metric as "not recorded" rather than a zero.
AGGREGATED_METRICS = (
    "mrr",
    "precision_at_3",
    "ndcg_at_5",
    "rouge_l",
    "bert_score_f1",
    "faithfulness",
)


@dataclass(frozen=True)
class StrategyInput:
    """Maps a named strategy to the chunks JSONL that produced it."""

    strategy: str
    chunks_path: Path


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Load a chunks-for-embedding JSONL file.

    Each line is a JSON object with at least id and text keys.
    Blank lines are skipped; lines that fail to parse as JSON raise
    immediately, with the line number in the message, so silent
    corpus corruption does not sneak into the comparison.
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON line: {exc}"
                ) from exc
    return rows


def describe_chunks(chunks: list[dict[str, Any]]) -> dict[str, float]:
    """Compute corpus-level descriptors for a chunk set.

    Returns n_chunks, mean_tokens (a whitespace word count used as a
    cheap proxy for tokens), and median_tokens, so the report can
    compare one strategy to another in concrete numbers.
    """
    if not chunks:
        return {"n_chunks": 0, "mean_tokens": 0.0, "median_tokens": 0.0}
    token_counts = [len(c.get("text", "").split()) for c in chunks]
    return {
        "n_chunks": len(chunks),
        "mean_tokens": float(statistics.fmean(token_counts)),
        "median_tokens": float(statistics.median(token_counts)),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, float]:
    """Average the headline metrics over a per-query result list.

    Refusals drop out of answer-quality averages the same way
    Evaluator.print_summary treats them: a metric that is absent
    from a row contributes nothing to its own average, so an
    evaluator that skips BERTScore on refusals does not pull the
    mean toward zero.
    """
    aggregated: dict[str, float] = {}
    for key in AGGREGATED_METRICS:
        values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        if values:
            aggregated[key] = float(statistics.fmean(values))
    aggregated["n_results"] = len(results)
    aggregated["n_refusals"] = sum(1 for r in results if r.get("is_refusal"))
    return aggregated


def run_comparison(
    strategies: tuple[StrategyInput, ...],
    evaluate_fn: Callable[[StrategyInput, list[dict[str, Any]]], list[dict[str, Any]]],
) -> pd.DataFrame:
    """Run the QA evaluation once per strategy and return a summary.

    Args:
        strategies: Ordered tuple of StrategyInput. Each entry names
            the strategy and the JSONL of chunks produced under it.
        evaluate_fn: Callable that receives the strategy spec and
            the loaded chunks, and returns the list of per-query
            result dicts (same shape as baseline_results.json). It
            is injected so tests can substitute a fake evaluator.

    Returns:
        DataFrame with one row per strategy, carrying the chunk
        descriptors, refusal rate and averaged quality metrics.
    """
    rows: list[dict[str, Any]] = []
    for spec in strategies:
        logger.info("Loading chunks for %s from %s", spec.strategy, spec.chunks_path)
        chunks = load_chunks(spec.chunks_path)
        descriptors = describe_chunks(chunks)
        logger.info(
            "Evaluating %s (%d chunks, mean %.1f tokens per chunk)",
            spec.strategy,
            descriptors["n_chunks"],
            descriptors["mean_tokens"],
        )
        results = evaluate_fn(spec, chunks)
        aggregated = aggregate_results(results)
        row: dict[str, Any] = {"strategy": spec.strategy, **descriptors, **aggregated}
        if row["n_results"]:
            row["refusal_rate"] = row["n_refusals"] / row["n_results"]
        else:
            row["refusal_rate"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    """Render a grouped bar chart of the aggregated quality metrics."""
    metric_cols = [m for m in AGGREGATED_METRICS if m in summary.columns]
    if summary.empty or not metric_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No chunking-strategy data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    n_strategies = len(summary)
    width = 0.8 / max(n_strategies, 1)
    x_positions = range(len(metric_cols))
    for i, (_, row) in enumerate(summary.iterrows()):
        offsets = [x + (i - n_strategies / 2 + 0.5) * width for x in x_positions]
        heights = [float(row.get(m, 0.0)) for m in metric_cols]
        ax.bar(offsets, heights, width=width, label=str(row["strategy"]))
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(metric_cols, rotation=30, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Chunking strategy comparison: recursive vs semantic")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _real_evaluate_fn(
    qa_pairs_path: Path, index_root: Path
) -> Callable[[StrategyInput, list[dict[str, Any]]], list[dict[str, Any]]]:  # pragma: no cover - wires the real pipeline
    """Build the real evaluator used by the CLI entry point.

    Split out as a factory so the unit tests do not need to import
    RAGPipeline or load embedders; the CLI main function calls this
    to produce the evaluate_fn passed into run_comparison.
    """
    from evaluation.metrics.evaluator import Evaluator
    from rag_pipeline.config import RAGConfig
    from rag_pipeline.pipeline import RAGPipeline

    with qa_pairs_path.open(encoding="utf-8") as f:
        qa_pairs = json.load(f)

    def evaluate(spec: StrategyInput, _chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        config = RAGConfig(
            chunks_path=spec.chunks_path,
            index_dir=index_root / spec.strategy,
        )
        pipeline = RAGPipeline(config)
        if not (config.index_dir / "index.faiss").exists():
            pipeline.build_index()
        pipeline.setup()
        evaluator = Evaluator(pipeline=pipeline, qa_pairs=qa_pairs)
        return evaluator.run(compute_bert=True, with_sgf=True)

    return evaluate


def run_chunking_strategy_comparison(
    recursive_chunks: Path = DEFAULT_RECURSIVE_CHUNKS,
    semantic_chunks: Path = DEFAULT_SEMANTIC_CHUNKS,
    out_csv: Path = DEFAULT_OUT_CSV,
    out_fig: Path = DEFAULT_OUT_FIG,
    evaluate_fn: Callable[
        [StrategyInput, list[dict[str, Any]]], list[dict[str, Any]]
    ]
    | None = None,
) -> pd.DataFrame:
    """Top-level orchestrator used by both the CLI and the tests."""
    if evaluate_fn is None:  # pragma: no cover - default wires real pipeline
        evaluate_fn = _real_evaluate_fn(
            qa_pairs_path=Path("evaluation/test_set/qa_pairs.json"),
            index_root=Path("rag_pipeline/data"),
        )
    strategies = (
        StrategyInput("recursive", recursive_chunks),
        StrategyInput("semantic", semantic_chunks),
    )
    summary = run_comparison(strategies, evaluate_fn)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    logger.info("Wrote %d rows to %s", len(summary), out_csv)
    plot_comparison(summary, out_fig)
    logger.info("Wrote comparison figure to %s", out_fig)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recursive-chunks", type=Path, default=DEFAULT_RECURSIVE_CHUNKS
    )
    parser.add_argument(
        "--semantic-chunks", type=Path, default=DEFAULT_SEMANTIC_CHUNKS
    )
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-fig", type=Path, default=DEFAULT_OUT_FIG)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    run_chunking_strategy_comparison(
        recursive_chunks=args.recursive_chunks,
        semantic_chunks=args.semantic_chunks,
        out_csv=args.out_csv,
        out_fig=args.out_fig,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
