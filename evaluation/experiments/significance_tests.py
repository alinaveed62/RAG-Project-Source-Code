"""Cross-experiment pairwise significance testing.

The existing experiment CSVs report point estimates with bootstrap
CIs, but they do not answer "is system A significantly better than
system B on this metric?". That claim needs a paired test on the
per-query metric vectors, an effect-size estimate for magnitude, and
a multiple-comparisons correction because each experiment runs many
pairwise tests at once.

This module loads per-query JSON files emitted by the comparison
experiments (evaluation/results/per_query/<experiment>.json) and
computes, for every pair of systems within an experiment, the
paired Wilcoxon signed-rank p-value (ties excluded), Cohen's d on
the paired-difference vector, and Holm-Bonferroni corrected
p-values within each (experiment, metric) family.

The tidy CSV it writes is the evidence the statistical analysis
section cites; the LaTeX table it writes is included into the
report.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.config import REPORT_TABLES_DIR
from evaluation.metrics.bootstrap import paired_wilcoxon
from evaluation.metrics.effect_size import cohens_d, effect_size_label

logger = logging.getLogger(__name__)

DEFAULT_PER_QUERY_DIR = Path("evaluation/results/per_query")
DEFAULT_OUT_CSV = Path("evaluation/results/significance_tests.csv")
DEFAULT_OUT_TEX = REPORT_TABLES_DIR / "significance_tests.tex"

# Metrics eligible for pairwise testing. Everything else in the per-query
# JSON (question text, refusal flag, latency, chunk id) is either
# non-numeric, non-continuous, or not appropriate for a Wilcoxon test.
DEFAULT_METRICS = (
    "rouge_1",
    "rouge_l",
    "bert_score_f1",
    "faithfulness",
    "sgf",
    "mrr",
    "precision_at_3",
    "ndcg_at_5",
)


def load_per_query_rows(path: Path) -> list[dict[str, Any]]:
    """Load a per-query JSON file and validate the minimum schema.

    Each row must carry experiment, system and id keys so the
    pairwise tests can group by (experiment, system) and align
    vectors query-for-query across systems. Missing keys are a
    programming error rather than a data condition, so a ValueError
    is raised at load time instead of silently producing malformed
    results.
    """
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list; got {type(rows).__name__}")
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"{path} row {i}: expected a JSON object; got "
                f"{type(row).__name__}"
            )
        missing = {"experiment", "system", "id"} - set(row)
        if missing:
            raise ValueError(
                f"{path} row {i}: missing required keys {sorted(missing)}"
            )
    return rows


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction on a family of p-values.

    The Holm procedure is uniformly more powerful than plain
    Bonferroni and controls the family-wise error rate at the same
    level, which is what the evaluation chapter needs when it
    quotes "significant after correction". Ties and empty inputs are
    handled explicitly so callers do not need to pre-filter.
    """
    n = len(p_values)
    if n == 0:
        return []
    # Sort p-values ascending, remembering each one's original index.
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        candidate = (n - rank) * p
        # Step-down: the adjusted p-value is monotone non-decreasing
        # along the sorted order, so each correction is clamped by the
        # running max of earlier (smaller-p) corrections.
        candidate = max(running_max, candidate)
        # Adjusted p-values cap at 1.0.
        candidate = min(candidate, 1.0)
        adjusted[orig_idx] = candidate
        running_max = candidate
    return adjusted


def pairwise_significance(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Run paired Wilcoxon, Cohen's d and Holm for one experiment.

    Args:
        rows: Per-query dicts with experiment, system and id keys,
            plus numeric metric columns. All rows must share the
            same experiment value.
        metrics: Which metric columns to test. Metrics absent from
            every row are silently skipped, so a caller can pass
            the superset of everything any experiment emits.

    Returns:
        A long-format DataFrame with one row per
        (metric, system_a, system_b) pair. Holm correction is
        applied within each (experiment, metric) family.
    """
    if not rows:
        return pd.DataFrame()
    experiments = {r["experiment"] for r in rows}
    if len(experiments) != 1:
        raise ValueError(
            f"rows must belong to a single experiment; got {sorted(experiments)}"
        )
    experiment = next(iter(experiments))

    # Group (system -> {qa_id -> row}) so each pairwise comparison can
    # align by question id. Keeping the full dict lets us quote specific
    # metrics without a second pass over the JSON.
    by_system: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_system.setdefault(row["system"], {})[row["id"]] = row
    systems = sorted(by_system)

    records: list[dict[str, Any]] = []
    for metric in metrics:
        # Records for this metric get Holm-corrected together as one
        # family; collect raw rows first, correct in a second pass.
        per_metric: list[dict[str, Any]] = []
        for sys_a, sys_b in itertools.combinations(systems, 2):
            rows_a = by_system[sys_a]
            rows_b = by_system[sys_b]
            common_ids = sorted(set(rows_a) & set(rows_b))
            values_a: list[float] = []
            values_b: list[float] = []
            for qa_id in common_ids:
                a_val = rows_a[qa_id].get(metric)
                b_val = rows_b[qa_id].get(metric)
                # Skip question-for-question when either side is missing
                # the metric (e.g. ROUGE is not recorded for a refusal).
                # The Wilcoxon test needs paired vectors so the drop is
                # symmetric on both sides.
                if a_val is None or b_val is None:
                    continue
                values_a.append(float(a_val))
                values_b.append(float(b_val))
            if not values_a:
                # Whole metric absent from this experiment's rows; skip
                # silently so callers can pass a liberal DEFAULT_METRICS
                # list.
                continue
            wilcoxon = paired_wilcoxon(values_a, values_b)
            d = cohens_d(values_a, values_b)
            per_metric.append(
                {
                    "experiment": experiment,
                    "metric": metric,
                    "system_a": sys_a,
                    "system_b": sys_b,
                    "n_pairs": wilcoxon["n_pairs"],
                    "n_nonzero_pairs": wilcoxon["n_nonzero_pairs"],
                    "statistic": wilcoxon["statistic"],
                    "p_raw": wilcoxon["pvalue"],
                    "cohens_d": d,
                    "effect_label": effect_size_label(d),
                }
            )
        # Holm correction within this metric family.
        p_raw = [rec["p_raw"] for rec in per_metric]
        p_holm = holm_bonferroni(p_raw)
        for rec, p in zip(per_metric, p_holm, strict=True):
            rec["p_holm"] = p
            rec["significant_holm_05"] = bool(p < 0.05)
            records.append(rec)

    return pd.DataFrame(records)


def run_all_significance_tests(
    per_query_dir: Path = DEFAULT_PER_QUERY_DIR,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Run pairwise significance tests on every per-query JSON.

    Args:
        per_query_dir: Directory with <experiment>.json files
            emitted by the comparison experiments.
        metrics: Superset of metrics to test. Metrics that do not
            appear in a given experiment's rows are skipped silently.

    Returns:
        A concatenated long-format DataFrame across all experiments.
        When the directory is missing or empty, an empty DataFrame
        is returned along with a log message so the caller can
        decide whether to error or proceed.
    """
    if not per_query_dir.exists():
        logger.warning(
            "per-query directory %s does not exist yet; re-run the "
            "comparison experiments with --per-query-dir to populate it.",
            per_query_dir,
        )
        return pd.DataFrame()
    json_files = sorted(per_query_dir.glob("*.json"))
    if not json_files:
        logger.warning(
            "per-query directory %s exists but is empty.", per_query_dir
        )
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in json_files:
        logger.info("Loading per-query rows from %s", path)
        rows = load_per_query_rows(path)
        if not rows:
            logger.warning("%s is empty; skipping.", path)
            continue
        frame = pairwise_significance(rows, metrics=metrics)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_outputs(
    df: pd.DataFrame,
    out_csv: Path = DEFAULT_OUT_CSV,
    out_tex: Path = DEFAULT_OUT_TEX,
) -> None:
    """Persist the significance-test DataFrame as CSV + LaTeX.

    The CSV carries every column (including the raw p-values and the
    n_nonzero_pairs diagnostic) for audit. The LaTeX table is a compact
    subset suitable for a single-column chapter table: experiment,
    metric, pair, Cohen's d, Holm p-value, and the significance flag.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_csv)

    if df.empty:
        # Still emit a placeholder LaTeX file so \\input does not error at
        # compile time; the chapter reads this file and conveys an
        # explicit "no per-query data yet" status.
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        out_tex.write_text(
            "% significance_tests: no per-query data found; re-run the "
            "comparison experiments with per-query emission to populate "
            "this table.\n",
            encoding="utf-8",
        )
        logger.info("Wrote placeholder %s", out_tex)
        return

    compact = df[
        [
            "experiment",
            "metric",
            "system_a",
            "system_b",
            "cohens_d",
            "effect_label",
            "p_holm",
            "significant_holm_05",
        ]
    ].copy()
    compact["cohens_d"] = compact["cohens_d"].map(lambda v: f"{v:.2f}")
    compact["p_holm"] = compact["p_holm"].map(lambda v: f"{v:.3f}")
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(
        compact.to_latex(
            index=False,
            longtable=True,
            escape=True,
            caption=(
                "Pairwise significance tests per experiment. Wilcoxon "
                "signed-rank with Holm-Bonferroni correction within each "
                "(experiment, metric) family; Cohen's d on the paired "
                "difference vector."
            ),
            label="tab:significance_tests",
        ),
        encoding="utf-8",
    )
    logger.info("Wrote LaTeX table to %s", out_tex)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-query-dir",
        type=Path,
        default=DEFAULT_PER_QUERY_DIR,
        help="Directory of per-query JSON files (one per experiment).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=DEFAULT_OUT_TEX,
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help=(
            "Metric to test; may be repeated. Defaults to the "
            "DEFAULT_METRICS list."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    metrics = tuple(args.metric) if args.metric else DEFAULT_METRICS
    df = run_all_significance_tests(
        per_query_dir=args.per_query_dir, metrics=metrics
    )
    write_outputs(df, out_csv=args.out_csv, out_tex=args.out_tex)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
