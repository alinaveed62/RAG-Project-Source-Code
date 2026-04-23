"""Quantitative failure-mode taxonomy for the baseline evaluation run.

The qualitative error analysis in the evaluation chapter lists
failure patterns (verbose generation, refusals on edge cases,
cross-section queries, abbreviation handling) but does not quantify
how many queries fall into each bucket or how the buckets interact
with the nine test-set categories. This script closes that gap by
tagging every query in baseline_results.json with one failure-mode
label, then writing a stacked-bar figure and a category by
failure-mode contingency table.

Labels are assigned by pure-Python heuristics on the per-query
metrics, so the tagger is deterministic, cheap and auditable and
does not call an LLM judge. The thresholds sit at the top of this
file with a short rationale for each cutoff.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
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

DEFAULT_BASELINE_JSON = Path("evaluation/results/baseline_results.json")
DEFAULT_OUT_CSV = Path("evaluation/results/failure_modes.csv")
DEFAULT_SUMMARY_CSV = Path("evaluation/results/failure_mode_summary.csv")
DEFAULT_OUT_FIG = REPORT_FIGURES_DIR / "failure_modes.pdf"

# ROUGE-L thresholds used to bucket generated answers. They match
# the thresholds the evaluation chapter discusses in its error
# analysis section ("answers with ROUGE-L below 0.2 tend to be
# off-topic; 0.2 to 0.4 are partial"), so the labels and the prose
# stay consistent.
_ROUGE_L_LOW = 0.2
_ROUGE_L_PARTIAL = 0.4

FAILURE_MODES = (
    "correct_refusal",
    "over_refusal",
    "wrong_retrieval",
    "low_quality_answer",
    "partial_answer",
    "ok",
)


def classify_row(row: dict[str, Any]) -> str:
    """Assign one of six failure-mode labels to a per-query result row.

    The classification uses three signals from the per-query record:
    is_refusal (did the pipeline refuse?), mrr (was any relevant
    chunk retrieved? MRR is zero only when none of the retrieved
    chunks belongs to a relevant section), and rouge_l (how well
    does the generated answer lexically match the reference?).

    The retrieval-related labels correct_refusal, over_refusal and
    wrong_retrieval short-circuit before ROUGE is inspected: ROUGE
    is undefined on refusals and uninformative when retrieval has
    already missed the mark.
    """
    is_refusal = bool(row.get("is_refusal", False))
    mrr = float(row.get("mrr", 0.0) or 0.0)
    rouge_l = row.get("rouge_l")

    if is_refusal:
        # The pipeline refused. That is the right call only if no
        # relevant chunk could be retrieved in the first place. If a
        # relevant chunk was available but the system refused anyway,
        # the call is counted as an over-refusal.
        return "correct_refusal" if mrr == 0.0 else "over_refusal"

    if mrr == 0.0:
        # The pipeline generated an answer even though no relevant
        # chunk was in the top-k, so the answer is not grounded in
        # the handbook and the answer-quality metrics are meaningless.
        return "wrong_retrieval"

    # Retrieval hit something relevant; classify by answer quality.
    if rouge_l is None:
        # is_refusal is False and mrr > 0, but answer-quality scores
        # are missing (for example, BERTScore was skipped on this
        # row). Label as ok on the retrieval signal alone rather
        # than silently dropping the row from the tally.
        return "ok"

    rouge_l = float(rouge_l)
    if rouge_l < _ROUGE_L_LOW:
        return "low_quality_answer"
    if rouge_l < _ROUGE_L_PARTIAL:
        return "partial_answer"
    return "ok"


def tag_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach a failure_mode label to each input row.

    The original row is shallow-copied so the caller's list stays
    untouched, and the CSV writer sees a stable schema even when the
    input JSON has different keys from row to row.
    """
    tagged = []
    for row in rows:
        new_row = dict(row)
        new_row["failure_mode"] = classify_row(row)
        tagged.append(new_row)
    return tagged


def summarise_by_category(tagged_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a category by failure-mode contingency table.

    Rows without a category field fall into the "unknown" bucket so
    none are dropped silently. Columns follow the stable
    FAILURE_MODES order so charts rendered from this frame stack in
    the same order across runs.
    """
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {mode: 0 for mode in FAILURE_MODES}
    )
    totals: dict[str, int] = defaultdict(int)
    for row in tagged_rows:
        category = row.get("category") or "unknown"
        mode = row["failure_mode"]
        counts[category][mode] += 1
        totals[category] += 1

    records = []
    for category in sorted(counts):
        record: dict[str, Any] = {
            "category": category,
            "n_queries": totals[category],
        }
        for mode in FAILURE_MODES:
            record[mode] = counts[category][mode]
        records.append(record)
    return pd.DataFrame(records)


def plot_stacked_bar(summary: pd.DataFrame, out_path: Path) -> None:
    """Render the per-category failure-mode stacked bar chart.

    The chart is saved with matplotlib using the Agg backend, so it
    runs headless in CI or a Colab runtime. Colours come from the
    default palette; the legend labels follow FAILURE_MODES in
    display order so the stack reads bottom-up.
    """
    if summary.empty:
        # Still emit a placeholder figure so downstream
        # \\includegraphics does not fail at LaTeX compile time.
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No per-query data available",
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
    bottoms = [0.0] * len(summary)
    x = range(len(summary))
    for mode in FAILURE_MODES:
        heights = summary[mode].astype(float).tolist()
        ax.bar(x, heights, bottom=bottoms, label=mode.replace("_", " "))
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["category"], rotation=30, ha="right")
    ax.set_ylabel("Number of queries")
    ax.set_title("Failure-mode distribution per test-set category")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_failure_mode_analysis(
    baseline_json: Path = DEFAULT_BASELINE_JSON,
    out_csv: Path = DEFAULT_OUT_CSV,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    out_fig: Path = DEFAULT_OUT_FIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full failure-mode pipeline and write every artefact.

    Returns the per-query DataFrame and the category-level summary,
    in that order, so tests can assert on both without re-reading
    the CSVs.
    """
    if not baseline_json.exists():
        logger.error("baseline_results.json not found at %s", baseline_json)
        raise FileNotFoundError(baseline_json)
    with baseline_json.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(
            f"{baseline_json}: expected a JSON list; "
            f"got {type(rows).__name__}"
        )

    tagged = tag_rows(rows)
    per_query_df = pd.DataFrame(tagged)
    summary_df = summarise_by_category(tagged)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    per_query_df.to_csv(out_csv, index=False)
    logger.info("Wrote %d per-query rows to %s", len(per_query_df), out_csv)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Wrote %d category rows to %s", len(summary_df), summary_csv)

    plot_stacked_bar(summary_df, out_fig)
    logger.info("Wrote stacked-bar figure to %s", out_fig)

    return per_query_df, summary_df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=DEFAULT_BASELINE_JSON,
    )
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument(
        "--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV
    )
    parser.add_argument("--out-fig", type=Path, default=DEFAULT_OUT_FIG)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    run_failure_mode_analysis(
        baseline_json=args.baseline,
        out_csv=args.out_csv,
        summary_csv=args.summary_csv,
        out_fig=args.out_fig,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
