"""Latency-quality Pareto frontier across comparison experiments.

The evaluation chapter argues that Gemma 2 2B is Pareto-dominant on
this corpus. This script produces the figure and CSV that back the
claim. For each supported experiment CSV, it reads the latency and
quality columns, computes the Pareto frontier (the set of points
where no other point is both faster and higher-quality at the same
time), and draws a scatter plot with the frontier highlighted.

Different experiments report latency in different columns (the LLM
comparison uses p50_total_ms; the retrieval comparison uses
avg_retrieval_ms), so each CSV is mapped to its own
(latency, quality, label) triple via the PARETO_INPUTS registry.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from evaluation.config import REPORT_FIGURES_DIR

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("evaluation/results")
DEFAULT_OUT_CSV = Path("evaluation/results/pareto_frontier.csv")
DEFAULT_OUT_FIG = REPORT_FIGURES_DIR / "latency_pareto.pdf"


@dataclass(frozen=True)
class ParetoInput:
    """Maps a comparison CSV to the columns used for Pareto analysis."""

    experiment: str
    csv_name: str
    latency_column: str
    quality_column: str
    label_column: str


# Only the experiments whose CSV already carries a per-configuration
# latency column participate here. topk_comparison,
# chunk_size_comparison and embedding_comparison report latency in a
# different shape (encoding_time_s, or none at all) and are better
# analysed within their own section of the evaluation chapter.
PARETO_INPUTS: tuple[ParetoInput, ...] = (
    ParetoInput(
        experiment="llm_comparison",
        csv_name="llm_comparison.csv",
        latency_column="p50_total_ms",
        quality_column="avg_rouge_l",
        label_column="model",
    ),
    ParetoInput(
        experiment="retrieval_comparison",
        csv_name="retrieval_comparison.csv",
        latency_column="avg_retrieval_ms",
        quality_column="mrr",
        label_column="strategy",
    ),
    ParetoInput(
        experiment="reranking_comparison",
        csv_name="reranking_comparison.csv",
        latency_column="avg_retrieval_ms",
        quality_column="mrr",
        label_column="strategy",
    ),
)


def is_on_frontier(
    points: list[tuple[float, float]],
    target_index: int,
) -> bool:
    """Return True if points[target_index] is on the Pareto frontier.

    A 2-D point (latency, quality) is on the Pareto frontier when no
    other point has both lower-or-equal latency and higher-or-equal
    quality, with strict inequality on at least one axis. Ties are
    treated as non-dominated, so a configuration that shares its
    position with another still shows up as a frontier point.
    """
    lat, qual = points[target_index]
    for i, (other_lat, other_qual) in enumerate(points):
        if i == target_index:
            continue
        # other dominates target when it is at least as good on
        # both axes and strictly better on at least one.
        dominates = (
            other_lat <= lat
            and other_qual >= qual
            and (other_lat < lat or other_qual > qual)
        )
        if dominates:
            return False
    return True


def compute_frontier(df: pd.DataFrame, spec: ParetoInput) -> pd.DataFrame:
    """Attach an is_on_frontier column to a comparison DataFrame.

    The returned frame carries only the columns needed for plotting
    and downstream CSV writing: experiment, label, latency_ms,
    quality and is_on_frontier. Rows with missing latency or
    quality values are dropped with a warning, so NaN does not
    corrupt the dominance comparisons.
    """
    required = {spec.latency_column, spec.quality_column, spec.label_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{spec.experiment}: CSV is missing required columns "
            f"{sorted(missing)}"
        )
    mask = df[spec.latency_column].notna() & df[spec.quality_column].notna()
    if (~mask).any():
        dropped = int((~mask).sum())
        logger.warning(
            "%s: dropping %d rows with missing latency/quality.",
            spec.experiment,
            dropped,
        )
    filtered = df[mask].copy()
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "experiment",
                "label",
                "latency_ms",
                "quality",
                "is_on_frontier",
            ]
        )
    points = list(
        zip(
            filtered[spec.latency_column].astype(float).tolist(),
            filtered[spec.quality_column].astype(float).tolist(),
        )
    )
    flags = [is_on_frontier(points, i) for i in range(len(points))]

    return pd.DataFrame(
        {
            "experiment": spec.experiment,
            "label": filtered[spec.label_column].astype(str).tolist(),
            "latency_ms": [p[0] for p in points],
            "quality": [p[1] for p in points],
            "is_on_frontier": flags,
        }
    )


def load_all_frontiers(
    results_dir: Path,
    inputs: tuple[ParetoInput, ...] = PARETO_INPUTS,
) -> pd.DataFrame:
    """Load each supported CSV and concatenate its frontier frame.

    CSVs that are not present in results_dir are skipped with a
    warning, so the script still works on a partially populated
    checkout.
    """
    frames: list[pd.DataFrame] = []
    for spec in inputs:
        path = results_dir / spec.csv_name
        if not path.exists():
            logger.warning(
                "Skipping %s: CSV not found at %s", spec.experiment, path
            )
            continue
        df = pd.read_csv(path)
        if df.empty:
            logger.warning(
                "Skipping %s: CSV is empty at %s", spec.experiment, path
            )
            continue
        frame = compute_frontier(df, spec)
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "experiment",
                "label",
                "latency_ms",
                "quality",
                "is_on_frontier",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def plot_frontier(frontier: pd.DataFrame, out_path: Path) -> None:
    """Render the scatter + frontier overlay as a single figure.

    Each experiment gets its own marker shape so the reader can tell the
    LLM comparison points from the retrieval comparison points at a
    glance. Frontier points are drawn on top with a bold outline and are
    connected with a step-down line within each experiment.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if frontier.empty:
        ax.text(
            0.5,
            0.5,
            "No frontier data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    markers = ["o", "s", "D", "^", "v", "P", "*"]
    experiments = sorted(frontier["experiment"].unique())
    for idx, experiment in enumerate(experiments):
        sub = frontier[frontier["experiment"] == experiment].sort_values(
            "latency_ms"
        )
        marker = markers[idx % len(markers)]
        ax.scatter(
            sub["latency_ms"],
            sub["quality"],
            marker=marker,
            s=80,
            alpha=0.5,
            label=f"{experiment} (all)",
        )
        frontier_sub = sub[sub["is_on_frontier"]]
        if not frontier_sub.empty:
            ax.scatter(
                frontier_sub["latency_ms"],
                frontier_sub["quality"],
                marker=marker,
                s=180,
                edgecolors="black",
                linewidths=1.4,
                facecolors="none",
                label=f"{experiment} (frontier)",
            )
            for _, row in frontier_sub.iterrows():
                ax.annotate(
                    str(row["label"]),
                    (row["latency_ms"], row["quality"]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                )
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Quality (ROUGE-L or MRR, depending on experiment)")
    ax.set_title(
        "Latency-quality Pareto frontier across comparison experiments"
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_latency_pareto(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    out_csv: Path = DEFAULT_OUT_CSV,
    out_fig: Path = DEFAULT_OUT_FIG,
    inputs: tuple[ParetoInput, ...] = PARETO_INPUTS,
) -> pd.DataFrame:
    """Run the full pipeline: load CSVs, compute frontier, write outputs."""
    frontier = load_all_frontiers(results_dir, inputs=inputs)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frontier.to_csv(out_csv, index=False)
    logger.info("Wrote %d frontier rows to %s", len(frontier), out_csv)
    plot_frontier(frontier, out_fig)
    logger.info("Wrote frontier figure to %s", out_fig)
    return frontier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR
    )
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-fig", type=Path, default=DEFAULT_OUT_FIG)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    run_latency_pareto(
        results_dir=args.results_dir,
        out_csv=args.out_csv,
        out_fig=args.out_fig,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
