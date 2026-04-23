"""Tests for the paired Cohen's d effect-size helper."""

from __future__ import annotations

import math

import pytest

from evaluation.metrics.effect_size import cohens_d, effect_size_label


class TestCohensD:
    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            cohens_d([0.1, 0.2], [0.1])

    def test_empty_returns_zero(self):
        assert cohens_d([], []) == 0.0

    def test_single_pair_returns_zero(self):
        # n=1 means no variance is defined; helper returns 0 rather than
        # dividing by zero or returning NaN.
        assert cohens_d([0.5], [0.3]) == 0.0

    def test_identical_vectors_return_zero(self):
        same = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert cohens_d(same, same) == 0.0

    def test_all_pairs_differ_by_constant_returns_zero(self):
        # Constant shift has zero variance in the diff vector, so Cohen's d
        # is 0.0 by convention (the Wilcoxon helper reports p = 1.0 in the
        # same case).
        a = [0.2, 0.4, 0.6, 0.8]
        b = [0.1, 0.3, 0.5, 0.7]
        assert cohens_d(a, b) == 0.0

    def test_known_value(self):
        # Hand-computed: diffs = [0.1, 0.2, 0.3], mean = 0.2,
        # variance (ddof=1) = ((-0.1)**2 + 0**2 + 0.1**2) / 2 = 0.01,
        # std = 0.1, d = 0.2 / 0.1 = 2.0.
        a = [0.3, 0.5, 0.7]
        b = [0.2, 0.3, 0.4]
        assert math.isclose(cohens_d(a, b), 2.0, abs_tol=1e-9)

    def test_sign_preserved_when_a_greater(self):
        a = [0.9, 0.9, 0.9, 0.8]
        b = [0.1, 0.1, 0.1, 0.2]
        d = cohens_d(a, b)
        assert d > 0.0

    def test_sign_flips_when_b_greater(self):
        a = [0.1, 0.1, 0.1, 0.2]
        b = [0.9, 0.9, 0.9, 0.8]
        d = cohens_d(a, b)
        assert d < 0.0


class TestEffectSizeLabel:
    def test_negligible_lower_edge(self):
        assert effect_size_label(0.0) == "negligible"

    def test_negligible_upper_edge_exclusive(self):
        # |d| < 0.2 is negligible. 0.199 counts.
        assert effect_size_label(0.199) == "negligible"

    def test_small_lower_edge_inclusive(self):
        # |d| >= 0.2 and < 0.5 is small.
        assert effect_size_label(0.2) == "small"

    def test_small_upper_edge_exclusive(self):
        assert effect_size_label(0.499) == "small"

    def test_medium_lower_edge_inclusive(self):
        assert effect_size_label(0.5) == "medium"

    def test_medium_upper_edge_exclusive(self):
        assert effect_size_label(0.799) == "medium"

    def test_large_lower_edge_inclusive(self):
        assert effect_size_label(0.8) == "large"

    def test_large_unbounded(self):
        assert effect_size_label(5.0) == "large"

    def test_negative_magnitude_treated_as_positive(self):
        # A negative d of large magnitude maps to "large" just like a
        # positive one. Sign carries direction; the label carries size.
        assert effect_size_label(-1.2) == "large"
        assert effect_size_label(-0.3) == "small"
