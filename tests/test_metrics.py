"""Tests for rq_probe.metrics — hand-computed cases and property checks."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from rq_probe.metrics import (
    auroc,
    auroc_hanley_mcneil,
    spearman,
    spearman_bootstrap,
    top_bottom_jaccard,
)


# ---------- AUROC ----------

def test_auroc_sklearn_example():
    # sklearn documentation example, hand-checkable: 3 of 4 (pos, neg) pairs ordered
    y = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    assert auroc(y, s) == pytest.approx(0.75)
    assert auroc(y, s) == roc_auc_score(y, s)


def test_auroc_perfect_and_reversed():
    y = [0, 0, 1, 1]
    assert auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auroc_scale_invariant():
    # AUROC is rank-based: any positive affine transform leaves it unchanged
    y = [0, 0, 1, 1, 0, 1]
    s = [0.3, -1.2, 2.0, 0.5, 0.1, 3.3]
    assert auroc(y, s) == auroc(y, np.asarray(s) * 2 + 5)


def test_auroc_single_class_raises():
    with pytest.raises(ValueError):
        auroc([1, 1, 1], [0.1, 0.2, 0.3])


# ---------- Hanley–McNeil CI ----------

def test_hm_perfect_separation_zero_variance():
    # auc = 1 -> Q1 = Q2 = 1 -> var = 0 -> CI collapses to [1, 1]
    lo_hi = auroc_hanley_mcneil([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert lo_hi == (1.0, 1.0, 1.0)


def test_hm_hand_computed():
    # y = [0,0,0,1,1], scores [0.1,0.2,0.3,0.2,0.9]:
    # pos 0.2 ties a 0.2 neg (1 win + 0.5 tie + 1 loss) and 0.9 wins all -> auc = 4.5/6 = 0.75
    # Q1 = 0.6, Q2 = 0.642857..., var = 0.385714.../6 = 0.0642857, se = 0.25355
    # lo = 0.75 - 1.96*se = 0.25305, hi clamps to 1.0
    auc, lo, hi = auroc_hanley_mcneil([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.2, 0.9])
    assert auc == pytest.approx(0.75)
    assert lo == pytest.approx(0.25305, rel=1e-4)
    assert hi == 1.0


def test_hm_ci_contains_estimate_and_shrinks_with_n():
    rng = np.random.default_rng(7)
    for n in (50, 5000):
        y = rng.integers(0, 2, size=n)
        s = rng.normal(size=n) + y * 0.5  # mildly separable
        auc, lo, hi = auroc_hanley_mcneil(y, s)
        assert lo <= auc <= hi
        if n == 5000:
            assert hi - lo < 0.1


# ---------- Spearman ----------

def test_spearman_monotonic():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_hand_computed():
    # ranks of s2 = [1, 3, 2, 5, 4]; d = [0,-1,1,-1,-1]; sum d^2 = 4
    # rho = 1 - 6*4/(5*(25-1)) = 1 - 0.2 = 0.8
    assert spearman([1, 2, 3, 4, 5], [1, 3, 2, 5, 4]) == pytest.approx(0.8)


def test_spearman_bootstrap_deterministic_and_sane():
    rng = np.random.default_rng(3)
    s1 = rng.normal(size=200)
    s2 = s1 * 0.7 + rng.normal(size=200, scale=0.5)
    r1 = spearman_bootstrap(s1, s2)
    r2 = spearman_bootstrap(s1, s2)  # same seed -> identical
    assert r1 == r2
    rho, lo, hi = r1
    assert lo <= rho <= hi
    assert rho == pytest.approx(spearman(s1, s2))


def test_spearman_bootstrap_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        spearman_bootstrap([1, 2, 3], [1, 2])


# ---------- Jaccard ----------

def test_jaccard_hand_computed():
    # n = 10, frac 0.2 -> k = 2
    # top: a keeps {0,1}, b keeps {0,2} -> overlap 1, union 3 -> 1/3
    # bottom: a keeps {8,9}, b keeps {6,9} -> overlap 1, union 3 -> 1/3
    a = [0.95, 0.90, 0.85, 0.60, 0.55, 0.50, 0.45, 0.40, 0.10, 0.05]
    b = [0.93, 0.70, 0.88, 0.62, 0.58, 0.08, 0.04, 0.42, 0.12, 0.06]
    top, bottom = top_bottom_jaccard(a, b, frac=0.2)
    assert top == pytest.approx(1 / 3)
    assert bottom == pytest.approx(1 / 3)


def test_jaccard_identical_scores():
    s = list(range(10))
    top, bottom = top_bottom_jaccard(s, s, frac=0.2)
    assert (top, bottom) == (1.0, 1.0)


def test_jaccard_k_size_floor():
    # n = 4, frac 0.2 -> round(0.8) = 1 -> k floored at 1, sets of size 1
    a = [0.4, 0.3, 0.2, 0.1]
    b = [0.1, 0.2, 0.3, 0.4]
    top, bottom = top_bottom_jaccard(a, b, frac=0.2)
    assert top == 0.0  # a's top {0}, b's top {3}
    assert bottom == 0.0  # a's bottom {3}, b's bottom {0}


def test_jaccard_default_frac_is_20_percent():
    # protocol decision: paper's 0.2, not upstream's 0.1 (EXPERIMENT_PROTOCOL §6)
    import inspect

    sig = inspect.signature(top_bottom_jaccard)
    assert sig.parameters["frac"].default == 0.2


def test_jaccard_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        top_bottom_jaccard([1, 2, 3], [1, 2, 3, 4])
