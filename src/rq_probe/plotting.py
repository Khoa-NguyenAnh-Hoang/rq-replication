"""Figures (Phase 6+): Fig-1-style layer-wise AUROC plot (paper §3.4 / Fig. 1)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: figures are written to disk, never shown
import matplotlib.pyplot as plt

PROBE_STYLES = {
    "logistic": {"color": "tab:blue", "label": "logistic"},
    "hinge": {"color": "tab:orange", "label": "hinge"},
    "diffmean": {"color": "tab:green", "label": "diffMean"},
}

PAIR_STYLES = {
    "logistic+hinge": {"color": "tab:blue"},
    "logistic+diffmean": {"color": "tab:orange"},
    "hinge+diffmean": {"color": "tab:green"},
}


def read_auroc_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fig1_auroc(jsonl_path: str | Path, out_path: str | Path, title: str | None = None) -> Path:
    """Layer-wise test AUROC per probe with Hanley–McNeil 95% CI bands."""
    records = [r for r in read_auroc_jsonl(jsonl_path) if "auroc_test" in r]
    probes = list(dict.fromkeys(r["probe"] for r in records))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for probe in probes:
        rows = sorted((r for r in records if r["probe"] == probe), key=lambda r: r["layer"])
        style = PROBE_STYLES.get(probe, {"color": None, "label": probe})
        layers = [r["layer"] for r in rows]
        auc = [r["auroc_test"] for r in rows]
        lo = [r.get("ci95_lo", r["auroc_test"]) for r in rows]
        hi = [r.get("ci95_hi", r["auroc_test"]) for r in rows]
        ax.plot(layers, auc, marker=".", ms=4, color=style["color"], label=style["label"])
        ax.fill_between(layers, lo, hi, alpha=0.15, color=style["color"])
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("layer (0 = embedding output, 36 = last)")
    ax.set_ylabel("test AUROC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title(title or "Layer-wise rhetorical-question decodability (RQ, Qwen3-4B, PCA-64)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def fig2_alignment(jsonl_path: str | Path, out_path: str | Path, title: str | None = None) -> Path:
    """Fig-2-style panel: per-layer direction cosine between probe pairs."""
    records = [r for r in read_auroc_jsonl(jsonl_path) if "cosine" in r]
    pairs = list(dict.fromkeys(r["pair"] for r in records))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pair in pairs:
        rows = sorted((r for r in records if r["pair"] == pair), key=lambda r: r["layer"])
        style = PAIR_STYLES.get(pair, {"color": None})
        ax.plot([r["layer"] for r in rows], [r["cosine"] for r in rows],
                marker=".", ms=4, color=style["color"], label=pair.replace("+", " + ").replace("diffmean", "diffMean"))
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("layer (0 = embedding output, 36 = last)")
    ax.set_ylabel("direction cosine")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title or "Probe direction alignment across layers (RQ, Qwen3-4B, PCA-64)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
