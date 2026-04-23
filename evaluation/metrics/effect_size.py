"""Paired Cohen's d effect size for evaluation comparisons.

The paired Wilcoxon signed-rank test in bootstrap.py answers "is
system A reliably different from system B on the 50-query test
set?" with a p-value. It does not answer "by how much?". With
n = 50, a p-value can slip into the significance band on a
difference too small to matter in practice. Pairing Cohen's d on
the per-query difference vector with the Wilcoxon test gives the
magnitude alongside the significance, which is the pattern the
evaluation chapter relies on when defending the size of a change.

The standardised mean of paired differences (often called Cohen's
d_z) is used rather than the independent-samples formula, because
every experiment in this project is paired question-for-question.

Thresholds follow Cohen 1988: |d| < 0.2 negligible, [0.2, 0.5]
small, [0.5, 0.8] medium, >= 0.8 large.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

_NEGLIGIBLE = 0.2
_SMALL = 0.5
_MEDIUM = 0.8


def cohens_d(values_a: list[float], values_b: list[float]) -> float:
    """Paired-sample Cohen's d on the per-query difference vector.

    Args:
        values_a: Per-query metric values for system A.
        values_b: Per-query metric values for system B, paired
            query-for-query with values_a.

    Returns:
        The effect size, defined as mean(diffs) / stddev(diffs,
        ddof=1). Returns 0.0 when fewer than two pairs are supplied
        (no variance to normalise against) or when every difference
        is identical (zero standard deviation; the Wilcoxon test
        reports p = 1.0 in the same case, so the pair is self
        consistent).

    Raises:
        ValueError: If the two vectors have different lengths. The
            paired design requires query-for-query alignment; a
            silent mismatch would produce a meaningless magnitude.
    """
    if len(values_a) != len(values_b):
        raise ValueError(
            f"paired samples must have equal length; got {len(values_a)} vs "
            f"{len(values_b)}"
        )
    n = len(values_a)
    if n < 2:
        return 0.0

    diffs = [a - b for a, b in zip(values_a, values_b)]
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    # A constant shift across all pairs has zero variance in exact
    # arithmetic but usually a tiny positive variance in IEEE-754
    # because of the a - b subtraction. A variance floor of 1e-30
    # (standard deviation well below 1e-15) catches that noise
    # without swallowing real signal: answer-quality metrics live in
    # [0, 1], so any genuine paired-diff variance is far above this
    # floor.
    if variance < 1e-30:
        return 0.0
    return mean / math.sqrt(variance)


def effect_size_label(d: float) -> str:
    """Map a Cohen's d magnitude to Cohen's 1988 interpretation band.

    The label goes into the significance-test CSV and LaTeX table so
    the reader can see the magnitude band next to the numeric value.
    The sign is ignored: a large negative effect and a large positive
    effect both return "large".
    """
    magnitude = abs(d)
    if magnitude < _NEGLIGIBLE:
        return "negligible"
    if magnitude < _SMALL:
        return "small"
    if magnitude < _MEDIUM:
        return "medium"
    return "large"
