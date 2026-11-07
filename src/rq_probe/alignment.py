"""Alignment metrics (Phase 7a, Tier 2): cosine / Spearman / Jaccard between
probe pairs + PCA map-back sanity (paper §3.4 Figs. 2; App. B closed form).

Per layer and probe pair (a, b):
- cosine of the unit-normalized directions W_a, W_b (upstream
  within_dataset_cosine.ipynb);
- Spearman ρ between the two probes' test-split raw projection scores, with
  bootstrap CI (n=2000, seed=0, unit=examples — port of upstream spearman_ci);
- Jaccard overlap of the top-20% / bottom-20% test-score sets (paper Fig. 2
  caption frac=0.2, D5).

Map-back (App. B): a PCA-space probe w_z scores original-space rows exactly —
w_x = Cᵀ w_z, b_x = −µ·w_x, since z = (x−µ)Cᵀ. Recorded per (probe, layer) as
auroc_pca vs auroc_original on the test split; agreement validates the closed
form (resolves REPOSITORY_DESIGN §7's Tier-2 open point). Test-split scores
are re-read here as the sanctioned analysis stage (PROTOCOL §6 metric table),
never fed back into training.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .config import Config
from .evaluate import resolve_pca, split_masks
from .metrics import auroc, spearman_bootstrap, top_bottom_jaccard
from .plotting import fig2_alignment
from .probes.trainable import probe_artifact_path

PROBE_PAIRS = (("logistic", "hinge"), ("logistic", "diffmean"), ("hinge", "diffmean"))


class AlignmentError(RuntimeError):
    pass


def cosine_directions(w1: torch.Tensor, w2: torch.Tensor) -> float:
    """Cosine between two direction vectors (unit-normalized inside)."""
    n1, n2 = w1.norm(), w2.norm()
    if n1 == 0 or n2 == 0:
        raise AlignmentError("zero-norm direction — cannot take cosine")
    return float((w1 / n1) @ (w2 / n2))


def map_back(w_z: torch.Tensor, components: torch.Tensor, mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Original-space linear probe from a PCA-space one (App. B closed form).

    w_z: [K] PCA-space direction; components: [K, D] (rows = PCs); mean: [D].
    Returns (w_x [D], b_x scalar): score(x) = x·w_x + b_x ≡ z·w_z.
    """
    w_x = components.T @ w_z
    b_x = -(mean @ w_x)
    return w_x, b_x


def run_analyze(
    cfg: Config,
    *,
    pca_path: str | Path | None = None,
    fit_split: str | None = None,
    pairs: list[tuple[str, str]] | None = None,
    log=print,
) -> Path:
    """Compute alignment metrics for the probe artifacts of one PCA stem.

    Returns the alignment JSONL path.
    """
    from .config import PROBE_NAMES
    from .evaluate import _write_jsonl

    pca_path = resolve_pca(cfg, pca_path, fit_split)
    obj = torch.load(pca_path, map_location="cpu", weights_only=True)
    X = obj["embeddings"].float()
    N, L, K = X.shape
    if obj.get("ids") != list(range(N)):
        raise AlignmentError("source artifact ids violate the row-alignment contract (ids must be 0..N-1)")
    y = torch.tensor(obj["labels"], dtype=torch.long)
    splits = list(obj["splits"])
    masks = split_masks(splits)

    src_meta = obj.get("meta", {})
    meta_common = {
        "source_artifact": str(pca_path),
        "fit_split": src_meta.get("fit_split"),
        "seed": cfg.experiment.seed,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    probes = [p for p in cfg.probes if probe_artifact_path(pca_path, p).is_file()]
    if len(probes) < 2:
        raise AlignmentError(
            f"need probe artifacts for at least 2 probes under {pca_path.stem}; found {probes} "
            f"(run `rq-probe train` first)"
        )
    Ws = {p: torch.load(probe_artifact_path(pca_path, p), map_location="cpu", weights_only=True)["W"].float() for p in probes}

    if pairs is None:
        pairs = [(a, b) for a, b in PROBE_PAIRS if a in probes and b in probes]
    for a, b in pairs:
        unknown = {a, b} - set(PROBE_NAMES)
        if unknown or a == b:
            raise AlignmentError(f"invalid probe pair {(a, b)!r} (expected two distinct of {PROBE_NAMES})")

    # map-back needs the PCA basis + the original embedding artifact
    basis_path = pca_path.parent / f"{pca_path.stem}_basis.pt"  # pca_step naming: same stem + _basis
    if not basis_path.is_file():
        raise AlignmentError(f"PCA basis not found: {basis_path} (run `rq-probe pca` first)")
    basis = torch.load(basis_path, map_location="cpu", weights_only=True)
    components, means = basis["components"].float(), basis["means"].float()

    orig_path = Path(src_meta["source_artifact"]) if src_meta.get("source_artifact") else None
    X_orig = None
    if orig_path is not None and orig_path.is_file():
        X_orig = torch.load(orig_path, map_location="cpu", weights_only=True)["embeddings"].float()
    else:
        log(f"original embedding artifact {orig_path} not found — skipping map-back sanity")

    records = []
    t0 = time.time()
    for layer in range(L):
        scores_test = {p: (X[:, layer, :] @ Ws[p][layer])[masks["test"]] for p in probes}
        for a, b in pairs:
            rho, ci_lo, ci_hi = spearman_bootstrap(scores_test[a].numpy(), scores_test[b].numpy())
            j_top, j_bot = top_bottom_jaccard(scores_test[a].numpy(), scores_test[b].numpy())
            records.append(
                {
                    "layer": layer,
                    "pair": f"{a}+{b}",
                    "cosine": cosine_directions(Ws[a][layer], Ws[b][layer]),
                    "spearman": rho,
                    "spearman_ci_lo": ci_lo,
                    "spearman_ci_hi": ci_hi,
                    "jaccard_top": j_top,
                    "jaccard_bottom": j_bot,
                    "meta": meta_common,
                }
            )
        if X_orig is not None:
            for p in probes:
                w_x, b_x = map_back(Ws[p][layer], components[layer], means[layer])
                s_orig = (X_orig[:, layer, :] @ w_x + b_x)[masks["test"]]
                records.append(
                    {
                        "layer": layer,
                        "probe": p,
                        "auroc_pca": auroc(y[masks["test"]].numpy(), scores_test[p].numpy()),
                        "auroc_original": auroc(y[masks["test"]].numpy(), s_orig.numpy()),
                        "meta": meta_common,
                    }
                )
        if (layer + 1) % 10 == 0 or layer == L - 1:
            log(f"analyze: layer {layer + 1}/{L} done ({time.time() - t0:.0f}s)")

    jsonl_path = Path("outputs/metrics") / f"alignment_{pca_path.stem}.jsonl"
    _write_jsonl(jsonl_path, records)
    _write_alignment_table(jsonl_path, pairs)
    fig = fig2_alignment(jsonl_path, Path("outputs/figures") / f"fig2_alignment_{pca_path.stem}.png")

    pair_recs = [r for r in records if "pair" in r]
    for a, b in pairs:
        cos = [r["cosine"] for r in pair_recs if r["pair"] == f"{a}+{b}"]
        jt = [r["jaccard_top"] for r in pair_recs if r["pair"] == f"{a}+{b}"]
        jb = [r["jaccard_bottom"] for r in pair_recs if r["pair"] == f"{a}+{b}"]
        log(f"{a}+{b}: cosine min/mean/max = {min(cos):.3f}/{sum(cos) / len(cos):.3f}/{max(cos):.3f}; "
            f"jaccard top mean {sum(jt) / len(jt):.3f} vs bottom mean {sum(jb) / len(jb):.3f}")
    map_recs = [r for r in records if "probe" in r]
    if map_recs:
        worst = max(map_recs, key=lambda r: abs(r["auroc_pca"] - r["auroc_original"]))
        log(
            f"map-back sanity: max |auroc_pca - auroc_original| = "
            f"{abs(worst['auroc_pca'] - worst['auroc_original']):.5f} "
            f"({worst['probe']} layer {worst['layer']}) — 0 expected"
        )
    log(f"saved {len(records)} records to {jsonl_path}")
    log(f"saved figure to {fig}")
    return jsonl_path


def _write_alignment_table(jsonl_path: Path, pairs: list[tuple[str, str]]) -> Path:
    from .evaluate import read_records

    records = read_records(jsonl_path)
    pair_recs = [r for r in records if "pair" in r]
    map_recs = {(r["probe"], r["layer"]): r for r in records if "probe" in r}
    by_layer: dict[int, dict[str, dict]] = {}
    for r in pair_recs:
        by_layer.setdefault(r["layer"], {})[r["pair"]] = r

    header = ["layer"]
    for a, b in pairs:
        header += [f"cos {a}+{b}", f"ρ {a}+{b} [95% CI]", f"Jtop {a}+{b}", f"Jbot {a}+{b}"]
    map_probes = sorted({probe for probe, _ in map_recs})
    for p in map_probes:
        header += [f"AUROC {p} pca", f"AUROC {p} orig"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for layer in sorted(by_layer):
        row = [str(layer)]
        for a, b in pairs:
            r = by_layer[layer].get(f"{a}+{b}")
            row += (
                [f"{r['cosine']:.3f}", f"{r['spearman']:.3f} [{r['spearman_ci_lo']:.3f}, {r['spearman_ci_hi']:.3f}]",
                 f"{r['jaccard_top']:.3f}", f"{r['jaccard_bottom']:.3f}"]
                if r else ["—"] * 4
            )
        for p in map_probes:
            r = map_recs.get((p, layer))
            row += [f"{r['auroc_pca']:.3f}", f"{r['auroc_original']:.3f}"] if r else ["—", "—"]
        lines.append("| " + " | ".join(row) + " |")
    out = jsonl_path.parent / f"alignment_table_{jsonl_path.stem.removeprefix('alignment_')}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
