"""Evaluation metrics (Phase 2) — CPU-only, ports of the upstream analysis code.

Deviations resolved in favor of the paper (docs/EXPERIMENT_PROTOCOL.md §6):
Jaccard frac defaults to 0.2 (paper Figs. 2–3 captions), not upstream's 0.1.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

Z_95 = 1.96


def auroc(y_true, scores) -> float:
    """AUROC of `scores` against binary labels.

    Raises ValueError if only one class is present (sklearn >= 1.9 returns nan
    with a warning instead; the pipeline wants a hard error).
    """
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        raise ValueError(f"auroc is undefined with a single class present: {np.unique(y).tolist()}")
    return float(roc_auc_score(y, scores))


def auroc_hanley_mcneil(y_true, scores, z: float = Z_95) -> tuple[float, float, float]:
    """AUROC with analytic Hanley–McNeil 95% CI, clamped to [0, 1].

    Exact port of upstream `auc_confint_hanley_mcneil` (diffmean_analysis.ipynb).
    """
    auc = auroc(y_true, scores)
    y = np.asarray(y_true)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc**2)
        + (n_neg - 1) * (q2 - auc**2)
    ) / (n_pos * n_neg)
    se = float(np.sqrt(var))
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


def spearman(s1, s2) -> float:
    """Spearman rank correlation between two score vectors (paper Eq. 3)."""
    return float(spearmanr(np.asarray(s1).ravel(), np.asarray(s2).ravel()).statistic)


def spearman_bootstrap(
    s1, s2, n_boot: int = 2000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Spearman correlation with bootstrap CI.

    Port of upstream `spearman_ci`: resampling unit = examples (rows),
    n_boot=2000, seed=0.
    """
    a = np.asarray(s1).ravel()
    b = np.asarray(s2).ravel()
    if a.shape != b.shape:
        raise ValueError(f"score vectors must have equal length, got {a.shape} vs {b.shape}")
    n = a.shape[0]
    rho = spearman(a, b)
    rng = np.random.default_rng(seed)
    rhos = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rhos[i] = spearmanr(a[idx], b[idx]).statistic
    alpha = (1.0 - ci) / 2.0
    return rho, float(np.quantile(rhos, alpha)), float(np.quantile(rhos, 1.0 - alpha))


def top_bottom_jaccard(scores_a, scores_b, frac: float = 0.2) -> tuple[float, float]:
    """Jaccard overlap of the top-frac and bottom-frac ranked sets (paper Eq. 4).

    Port of upstream `top_bottom_jaccard`: k = max(1, round(frac * n)),
    argpartition selection, ranking by raw scores (larger = more rhetorical).
    Returns (top_jaccard, bottom_jaccard).
    """
    a = np.asarray(scores_a).ravel()
    b = np.asarray(scores_b).ravel()
    if a.shape != b.shape:
        raise ValueError(f"score vectors must have equal length, got {a.shape} vs {b.shape}")
    n = a.shape[0]
    k = max(1, int(round(frac * n)))

    top_a = set(np.argpartition(a, -k)[-k:].tolist())
    top_b = set(np.argpartition(b, -k)[-k:].tolist())
    bot_a = set(np.argpartition(a, k)[:k].tolist())
    bot_b = set(np.argpartition(b, k)[:k].tolist())

    def jaccard(x: set[int], y: set[int]) -> float:
        union = x | y
        return len(x & y) / len(union) if union else 0.0

    return jaccard(top_a, top_b), jaccard(bot_a, bot_b)
