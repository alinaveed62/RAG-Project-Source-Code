"""Driver that runs every evaluation experiment.

The eleven experiment modules in this package each expose a run_*
function. This driver is the single entry point that wires them
together, loads the QA test set, builds the production pipeline
once, and dispatches each experiment in turn.

The eleven split into two classes. Seven primary ablations need a
live pipeline or the QA test set (embedding_comparison,
chunk_size_comparison, retrieval_comparison, reranking_comparison,
topk_comparison, llm_comparison, per_category_breakdown). Four
post-hoc analyses reuse the CSVs or baseline JSON emitted by the
primary ablations (chunking_strategy_comparison, failure_modes,
latency_pareto, significance_tests). The post-hoc analyses short-
circuit gracefully when their upstream inputs are missing.

Usage:

    python -m evaluation.experiments.run_all
        Skip any CSV that already exists.
    python -m evaluation.experiments.run_all --force
        Rerun everything.
    python -m evaluation.experiments.run_all --only llm_comparison topk_comparison
    python -m evaluation.experiments.run_all --skip llm_comparison

Each experiment is wrapped in a try/except so one failure does not
halt the rest of the run. Progress and exceptions go to
evaluation/results/run_all.log and to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.config import REPORT_FIGURES_DIR
from evaluation.experiments.chunk_size_comparison import run_chunk_size_comparison
from evaluation.experiments.chunking_strategy_comparison import (
    run_chunking_strategy_comparison,
)
from evaluation.experiments.embedding_comparison import run_embedding_comparison
from evaluation.experiments.failure_modes import run_failure_mode_analysis
from evaluation.experiments.generate_results import generate_all_results
from evaluation.experiments.latency_pareto import run_latency_pareto
from evaluation.experiments.llm_comparison import run_llm_comparison
from evaluation.experiments.per_category_breakdown import run_per_category_breakdown
from evaluation.experiments.reranking_comparison import run_reranking_comparison
from evaluation.experiments.retrieval_comparison import run_retrieval_comparison
from evaluation.experiments.significance_tests import (
    run_all_significance_tests,
)
from evaluation.experiments.significance_tests import (
    write_outputs as write_significance_outputs,
)
from evaluation.experiments.topk_comparison import run_topk_comparison
from evaluation.metrics.evaluator import Evaluator
from rag_pipeline.config import RAGConfig
from rag_pipeline.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class ExperimentSpec:
    """Describes a single experiment for the driver loop."""

    name: str
    csv_name: str
    needs_pipeline: bool
    func: Callable[..., pd.DataFrame]


EXPERIMENTS: list[ExperimentSpec] = [
    ExperimentSpec(
        "embedding_comparison",
        "embedding_comparison.csv",
        needs_pipeline=False,
        func=run_embedding_comparison,
    ),
    ExperimentSpec(
        "chunk_size_comparison",
        "chunk_size_comparison.csv",
        needs_pipeline=False,
        func=run_chunk_size_comparison,
    ),
    ExperimentSpec(
        "retrieval_comparison",
        "retrieval_comparison.csv",
        needs_pipeline=False,
        func=run_retrieval_comparison,
    ),
    ExperimentSpec(
        "reranking_comparison",
        "reranking_comparison.csv",
        needs_pipeline=False,
        func=run_reranking_comparison,
    ),
    ExperimentSpec(
        "topk_comparison",
        "topk_comparison.csv",
        needs_pipeline=True,
        func=run_topk_comparison,
    ),
    ExperimentSpec(
        "llm_comparison",
        "llm_comparison.csv",
        needs_pipeline=True,
        func=run_llm_comparison,
    ),
    ExperimentSpec(
        "per_category_breakdown",
        "per_category_breakdown.csv",
        needs_pipeline=True,
        func=run_per_category_breakdown,
    ),
    # Post-hoc analyses below. They do not need a live pipeline; they
    # read the CSVs / baseline JSON that the primary ablations emitted.
    ExperimentSpec(
        "chunking_strategy_comparison",
        "chunking_strategy_comparison.csv",
        needs_pipeline=False,
        func=run_chunking_strategy_comparison,
    ),
    ExperimentSpec(
        "failure_modes",
        "failure_modes.csv",
        needs_pipeline=False,
        func=run_failure_mode_analysis,
    ),
    ExperimentSpec(
        "latency_pareto",
        "pareto_frontier.csv",
        needs_pipeline=False,
        func=run_latency_pareto,
    ),
    ExperimentSpec(
        "significance_tests",
        "significance_tests.csv",
        needs_pipeline=False,
        func=run_all_significance_tests,
    ),
]


def setup_logging(log_path: Path) -> None:
    """Root-logger setup so every experiment's logger writes to both
    stderr and a log file for later inspection."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(stream=sys.stderr),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def load_qa_pairs(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def run_one(
    spec: ExperimentSpec,
    csv_path: Path,
    force: bool,
    qa_pairs: list[dict[str, Any]],
    config: RAGConfig,
    documents_path: Path,
    pipeline: RAGPipeline | None,
    results_dir: Path | None = None,
) -> bool:
    """Run a single experiment and write its summary CSV.

    Returns True if the experiment produced output, or was skipped
    because the CSV already exists. Returns False if it raised.
    """
    if csv_path.exists() and not force:
        logger.info("[skip] %s (exists: %s)", spec.name, csv_path)
        return True

    effective_results_dir = results_dir if results_dir is not None else csv_path.parent

    logger.info("[run ] %s", spec.name)
    start = time.perf_counter()
    try:
        if spec.name == "chunk_size_comparison":
            df = spec.func(qa_pairs=qa_pairs, documents_path=documents_path, config=config)
        elif spec.name == "failure_modes":
            baseline_json = effective_results_dir / "baseline_results.json"
            if not baseline_json.exists():
                logger.warning(
                    "[skip] failure_modes: %s missing; run baseline first.",
                    baseline_json,
                )
                return True
            per_query_df, _summary_df = spec.func(
                baseline_json=baseline_json,
                out_csv=csv_path,
                summary_csv=effective_results_dir / "failure_mode_summary.csv",
                out_fig=REPORT_FIGURES_DIR / "failure_modes.pdf",
            )
            df = per_query_df
        elif spec.name == "latency_pareto":
            df = spec.func(
                results_dir=effective_results_dir,
                out_csv=csv_path,
                out_fig=REPORT_FIGURES_DIR / "latency_pareto.pdf",
            )
        elif spec.name == "significance_tests":
            df = spec.func(per_query_dir=effective_results_dir / "per_query")
            # significance_tests also writes a compact LaTeX
            # companion file; keeping the CSV and the LaTeX file in
            # lockstep means the report can \\input either. Any write
            # failure is propagated so the driver records it as a
            # normal experiment failure (the outer try/except wraps
            # this path).
            write_significance_outputs(df, out_csv=csv_path)
            elapsed = time.perf_counter() - start
            logger.info(
                "[done] %s %d rows in %.1fs -> %s",
                spec.name,
                len(df),
                elapsed,
                csv_path,
            )
            return True
        elif spec.name == "chunking_strategy_comparison":
            # Semantic chunk set is opt-in; its absence is a signal to skip
            # rather than a hard failure, since the recursive corpus is
            # always available and the ablation only makes sense against
            # both.
            semantic_chunks = Path(
                "keats_scraper/data/chunks/chunks_for_embedding_semantic.jsonl"
            )
            recursive_chunks = config.chunks_path
            if not semantic_chunks.exists():
                logger.warning(
                    "[skip] chunking_strategy_comparison: semantic chunks "
                    "missing at %s; re-chunk with the SemanticChunker to "
                    "populate.",
                    semantic_chunks,
                )
                return True
            df = spec.func(
                recursive_chunks=recursive_chunks,
                semantic_chunks=semantic_chunks,
                out_csv=csv_path,
                out_fig=REPORT_FIGURES_DIR / "chunking_strategy_comparison.pdf",
            )
        elif spec.needs_pipeline:
            if pipeline is None:
                logger.error("[fail] %s: pipeline required but unavailable", spec.name)
                return False
            df = spec.func(pipeline=pipeline, qa_pairs=qa_pairs)
        else:
            df = spec.func(qa_pairs=qa_pairs, config=config)
    except Exception:
        elapsed = time.perf_counter() - start
        logger.error("[fail] %s after %.1fs", spec.name, elapsed)
        logger.error(traceback.format_exc())
        return False

    if not isinstance(df, pd.DataFrame):
        logger.error("[fail] %s: expected DataFrame, got %s", spec.name, type(df))
        return False

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    elapsed = time.perf_counter() - start
    logger.info("[done] %s %d rows in %.1fs -> %s", spec.name, len(df), elapsed, csv_path)
    return True


def run_baseline(
    pipeline: RAGPipeline,
    qa_pairs: list[dict[str, Any]],
    output_path: Path,
    force: bool,
) -> bool:
    """Regenerate baseline_results.json under the current defaults."""
    if output_path.exists() and not force:
        logger.info("[skip] baseline (exists: %s)", output_path)
        return True

    logger.info(
        "[run ] baseline (%d QA pairs, BERTScore, SGF)", len(qa_pairs)
    )
    start = time.perf_counter()
    try:
        evaluator = Evaluator(pipeline=pipeline, qa_pairs=qa_pairs)
        results = evaluator.run(compute_bert=True, with_sgf=True)
    except Exception:
        logger.error("[fail] baseline after %.1fs", time.perf_counter() - start)
        logger.error(traceback.format_exc())
        return False

    Evaluator.save_results(results, output_path)
    logger.info(
        "[done] baseline %d records in %.1fs", len(results), time.perf_counter() - start
    )
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Where experiment CSVs, figures, and tables are written.",
    )
    parser.add_argument(
        "--qa-pairs",
        type=Path,
        default=Path("evaluation/test_set/qa_pairs.json"),
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("keats_scraper/data/processed/documents.jsonl"),
        help="Required by chunk_size_comparison (re-chunks the corpus).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="If set, run only these experiment names (plus 'baseline' to "
        "regenerate baseline_results.json).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=None,
        help="Skip these experiments (space-separated names).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even when the CSV already exists.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the baseline_results.json regeneration step.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip generate_all_results() at the end.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    results_dir: Path = args.results_dir
    log_path = results_dir / "run_all.log"
    setup_logging(log_path)

    logger.info("=== run_all start ===")
    logger.info("results_dir=%s force=%s only=%s skip=%s", results_dir, args.force, args.only, args.skip)

    qa_pairs = load_qa_pairs(args.qa_pairs)
    logger.info("Loaded %d QA pairs from %s", len(qa_pairs), args.qa_pairs)

    config = RAGConfig()
    logger.info(
        "RAGConfig: embedding=%s, ollama_model=%s, top_k=%d, reranking=%s, mode=%s",
        config.embedding_model,
        config.ollama_model,
        config.top_k,
        config.enable_reranking,
        config.retrieval_mode,
    )

    selected: list[ExperimentSpec] = EXPERIMENTS
    if args.only:
        only_set = set(args.only)
        selected = [s for s in selected if s.name in only_set]
    if args.skip:
        skip_set = set(args.skip)
        selected = [s for s in selected if s.name not in skip_set]

    needs_pipeline = (
        any(s.needs_pipeline for s in selected)
        or (not args.no_baseline and (args.only is None or "baseline" in (args.only or [])))
    )

    pipeline: RAGPipeline | None = None
    if needs_pipeline:
        logger.info("Building RAGPipeline (index + encoder + reranker + Ollama)")
        pipeline = RAGPipeline(config)
        if not (config.index_dir / "index.faiss").exists():
            logger.info("Index missing; running build_index()")
            pipeline.build_index()
        pipeline.setup()
        logger.info("Pipeline ready.")

    failures: list[str] = []
    for spec in selected:
        csv_path = results_dir / spec.csv_name
        ok = run_one(
            spec,
            csv_path=csv_path,
            force=args.force,
            qa_pairs=qa_pairs,
            config=config,
            documents_path=args.documents,
            pipeline=pipeline,
            results_dir=results_dir,
        )
        if not ok:
            failures.append(spec.name)

    if not args.no_baseline and (args.only is None or "baseline" in (args.only or [])):
        # The needs_pipeline condition above guarantees the pipeline
        # is not None on this branch, because the same condition that
        # triggers the baseline also forces the pipeline to be built.
        # The assert is defensive belt-and-braces.
        assert pipeline is not None  # noqa: S101 - type narrowing; enforced by needs_pipeline gate above
        baseline_path = results_dir / "baseline_results.json"
        if not run_baseline(pipeline, qa_pairs, baseline_path, force=args.force):
            failures.append("baseline")

    if not args.no_figures:
        logger.info("[run ] generate_all_results -> %s", results_dir)
        try:
            generate_all_results(results_dir)
        except Exception:
            logger.error("[fail] generate_all_results")
            logger.error(traceback.format_exc())
            failures.append("generate_all_results")

    if failures:
        logger.error("Run completed with failures: %s", failures)
        return 1
    logger.info("=== run_all done, 0 failures ===")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
