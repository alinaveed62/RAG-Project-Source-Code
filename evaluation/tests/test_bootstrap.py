"""Tests for the bootstrap and paired-hypothesis helpers."""

from __future__ import annotations

import random

import pytest

from evaluation.metrics.bootstrap import (
    add_ci_columns,
    bootstrap_ci,
    mcnemar_hit_at_k,
    paired_wilcoxon,
)


class TestBootstrapCI:
    def test_empty_returns_zeros(self):
        mean, lo, hi = bootstrap_ci([])
        assert (mean, lo, hi) == (0.0, 0.0, 0.0)

    def test_single_value_collapses(self):
        mean, lo, hi = bootstrap_ci([0.5])
        assert mean == 0.5
        assert lo == 0.5
        assert hi == 0.5

    def test_ci_brackets_mean(self):
        random.seed(7)
        values = [random.uniform(0.3, 0.6) for _ in range(50)]
        mean, lo, hi = bootstrap_ci(values, n_resamples=500)
        assert lo <= mean <= hi

    def test_tight_distribution_gives_tight_ci(self):
        tight = [0.5] * 50 + [0.51, 0.49]
        _, lo, hi = bootstrap_ci(tight, n_resamples=500)
        assert hi - lo < 0.05

    def test_deterministic_with_seed(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        out1 = bootstrap_ci(values, n_resamples=200, seed=99)
        out2 = bootstrap_ci(values, n_resamples=200, seed=99)
        assert out1 == out2


class TestPairedWilcoxon:
    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            paired_wilcoxon([0.1, 0.2], [0.1])

    def test_all_identical_returns_pvalue_one(self):
        same = [0.5] * 10
        result = paired_wilcoxon(same, same)
        assert result["pvalue"] == 1.0
        assert result["n_nonzero_pairs"] == 0

    def test_clear_difference_yields_small_pvalue(self):
        a = [0.6 + 0.01 * i for i in range(20)]
        b = [0.3 + 0.01 * i for i in range(20)]
        result = paired_wilcoxon(a, b, alternative="greater")
        assert result["pvalue"] < 0.01
        assert result["n_pairs"] == 20
        assert result["n_nonzero_pairs"] == 20

    def test_empty_returns_sensible_default(self):
        result = paired_wilcoxon([], [])
        assert result["pvalue"] == 1.0
        assert result["n_pairs"] == 0


class TestMcNemar:
    def test_no_disagreement_returns_pvalue_one(self):
        hits = [True, False, True, True, False]
        result = mcnemar_hit_at_k(hits, hits)
        assert result["pvalue"] == 1.0
        assert result["b"] == 0
        assert result["c"] == 0

    def test_clear_asymmetry_yields_small_pvalue(self):
        # System A hits every query; B misses every query.
        a = [True] * 20
        b = [False] * 20
        result = mcnemar_hit_at_k(a, b)
        assert result["pvalue"] < 0.01
        assert result["b"] == 20
        assert result["c"] == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            mcnemar_hit_at_k([True, False], [True])


class TestAddCiColumns:
    def test_appends_keys_in_place(self):
        row = {"mrr": 0.5}
        add_ci_columns("mrr", [0.4, 0.5, 0.6, 0.55, 0.45], row, n_resamples=200)
        assert "mrr_ci_low" in row
        assert "mrr_ci_high" in row
        assert row["mrr_ci_low"] <= 0.5 <= row["mrr_ci_high"]

    def test_ci_values_in_unit_interval(self):
        row = {}
        add_ci_columns(
            "precision_at_5",
            [0.2] * 30 + [0.3] * 20,
            row,
            n_resamples=200,
        )
        assert 0.0 <= row["precision_at_5_ci_low"] <= 1.0
        assert 0.0 <= row["precision_at_5_ci_high"] <= 1.0
