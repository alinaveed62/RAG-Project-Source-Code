"""Bootstrap confidence intervals and paired hypothesis tests.

Small test sets (n = 50) produce averages whose reliability is easy
to overstate. This module wraps scipy.stats so every reported metric
in the evaluation can carry a 95% bootstrap CI, every model-vs-model
claim can be tested with a paired Wilcoxon signed-rank test, and
every binary hit-rate comparison with McNemar.

All helpers use fixed default seeds so the reported numbers are
reproducible.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_N_RESAMPLES = 1000
_DEFAULT_RNG_SEED = 42


def bootstrap_ci(
    values: list[float],
    confidence_level: float = 0.95,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    seed: int = _DEFAULT_RNG_SEED,
) -> tuple[float, float, float]:
    """Return (mean, CI-low, CI-high) for a one-sample metric vector.

    Uses the BCa method (the scipy.stats.bootstrap default), which
    behaves well on skewed distributions. Returns (mean, mean, mean)
    when fewer than two values are supplied, since bootstrapping a
    single point is not meaningful.

    Args:
        values: Per-query metric values (for example MRR for each of
            50 questions).
        confidence_level: Defaults to 0.95.
        n_resamples: Number of bootstrap resamples. Defaults to 1000.
        seed: RNG seed for reproducibility.

    Returns:
        (mean, ci_low, ci_high).
    """
    import numpy as np
    from scipy.stats import bootstrap

    if not values:
        return 0.0, 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0]), float(values[0])

    data = (np.asarray(values, dtype=float),)
    rng = np.random.default_rng(seed)
    result = bootstrap(
        data,
        statistic=np.mean,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method="BCa",
        random_state=rng,
    )
    mean = float(np.mean(values))
    ci_low = float(result.confidence_interval.low)
    ci_high = float(result.confidence_interval.high)
    return mean, ci_low, ci_high


def paired_wilcoxon(
    values_a: list[float],
    values_b: list[float],
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test on per-query metric vectors.

    Args:
        values_a: Per-query metrics for system A.
        values_b: Per-query metrics for system B, same length as
            values_a and paired query-for-query.
        alternative: "two-sided", "greater" (A > B) or "less" (A < B).

    Returns:
        Dict with statistic, pvalue, n_pairs and n_nonzero_pairs.
        If every pair ties, the p-value is returned as 1.0 (no
        difference); scipy would otherwise error out on that input.
    """
    from scipy.stats import wilcoxon

    if len(values_a) != len(values_b):
        raise ValueError(
            f"paired samples must have equal length; got {len(values_a)} vs "
            f"{len(values_b)}"
        )
    if not values_a:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "n_pairs": 0,
            "n_nonzero_pairs": 0,
        }

    diffs = [a - b for a, b in zip(values_a, values_b)]
    n_nonzero = sum(1 for d in diffs if d != 0.0)
    if n_nonzero == 0:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "n_pairs": len(values_a),
            "n_nonzero_pairs": 0,
        }

    res = wilcoxon(values_a, values_b, alternative=alternative, zero_method="wilcox")
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "n_pairs": len(values_a),
        "n_nonzero_pairs": n_nonzero,
    }


def mcnemar_hit_at_k(
    hits_a: list[bool],
    hits_b: list[bool],
    exact: bool = True,
) -> dict[str, float]:
    """McNemar test on paired binary hit@k vectors.

    Args:
        hits_a: True/False for whether system A hit for each query.
        hits_b: True/False for whether system B hit for each query.
        exact: If True, use the exact binomial test (suitable for
            small n).

    Returns:
        Dict with statistic, pvalue, b (A hit, B miss) and c (A miss,
        B hit).
    """
    from statsmodels.stats.contingency_tables import mcnemar

    if len(hits_a) != len(hits_b):
        raise ValueError(
            f"hit vectors must have equal length; got {len(hits_a)} vs "
            f"{len(hits_b)}"
        )

    # 2x2 contingency table: [[both hit, A-only], [B-only, neither]].
    a_b = sum(1 for a, b in zip(hits_a, hits_b) if a and b)
    a_only = sum(1 for a, b in zip(hits_a, hits_b) if a and not b)
    b_only = sum(1 for a, b in zip(hits_a, hits_b) if not a and b)
    neither = sum(1 for a, b in zip(hits_a, hits_b) if not a and not b)

    table = [[a_b, a_only], [b_only, neither]]
    if a_only + b_only == 0:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "b": a_only,
            "c": b_only,
        }
    result = mcnemar(table, exact=exact)
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "b": a_only,
        "c": b_only,
    }


def add_ci_columns(
    metric_name: str,
    per_query_values: list[float],
    row: dict[str, Any],
    confidence_level: float = 0.95,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    seed: int = _DEFAULT_RNG_SEED,
) -> None:
    """Add <metric>_ci_low and <metric>_ci_high keys to a row dict.

    Convenience helper for the experiment scripts. They already emit
    one row per configuration with the mean; this function mutates
    the row in place to add the bootstrap CI bounds so that LaTeX
    tables can render mean [low, high].
    """
    _, ci_low, ci_high = bootstrap_ci(
        per_query_values,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
    )
    row[f"{metric_name}_ci_low"] = ci_low
    row[f"{metric_name}_ci_high"] = ci_high
