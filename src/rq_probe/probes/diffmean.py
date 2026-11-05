"""diffMean probe (Phase 5): w = µ⁺ − µ⁻ from train-split class means (paper Eq. 2).

Port of upstream `diffmean_analysis.ipynb` (docs/EXPERIMENT_PROTOCOL.md §5):
train split only, unit-normalized before projection (AUROC-invariant, kept for
fidelity with the upstream analysis and later cosine comparisons).
"""

from __future__ import annotations

import torch


class DiffMeanError(RuntimeError):
    pass


def diffmean_direction(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Unit-normalized µ⁺ − µ⁻ from the given rows.

    Callers must pass **train-split rows only** (leakage check PROTOCOL §7.2);
    run_train enforces this by masking before the call.
    """
    if X.dim() != 2:
        raise DiffMeanError(f"expected [N, D] rows, got shape {tuple(X.shape)}")
    y = y.long()
    plus, minus = X[y == 1], X[y == 0]
    if plus.shape[0] == 0 or minus.shape[0] == 0:
        raise DiffMeanError(
            f"diffMean needs both classes; got {plus.shape[0]} rhetorical / {minus.shape[0]} informational"
        )
    w = plus.mean(dim=0) - minus.mean(dim=0)
    norm = w.norm()
    if not torch.isfinite(norm) or norm == 0:
        raise DiffMeanError("degenerate diffMean direction (zero or non-finite norm)")
    return (w / norm).float()
