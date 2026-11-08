"""Robustness & error analysis (Phase 8, M8): errors, length proxy, seed stability.

- Error analysis at one operating layer (default: argmax of logistic test AUROC
  from the existing `outputs/metrics/auroc_{stem}.jsonl`): confusion counts at
  the natural threshold 0 (bias-free probes), pairwise probe agreement, and a
  per-test-example CSV (id, label, scores, lengths, text joined by id via
  data.load_dataset — read-only; the CSV lands in gitignored outputs/).
- Length-proxy check (paper §7 Table 2 analog): word-count-only AUROC baseline,
  Spearman ρ(score, length) per (probe, layer), length-tercile AUROC at the
  operating layer.
- Seed stability (--seed-stability): retrains logistic/hinge per layer under
  seeds {cfg.seed, 1, 2} via train_layer and reports the across-seed std of
  test AUROC next to the Hanley–McNeil CI width. Weights are not persisted
  (same throwaway pattern as the Phase 6 negative control). Test-split
  discipline: one evaluation-stage read per (seed, probe, layer); variance is
  reported, no seed is selected on test (PROTOCOL §7.4 intent).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import load_dataset
from .evaluate import read_records, resolve_pca, split_masks
from .metrics import auroc, auroc_hanley_mcneil, spearman
from .probes.trainable import probe_artifact_path, train_layer

METRICS_DIR = Path("outputs/metrics")


class RobustnessError(RuntimeError):
    pass


def _confusion(y_true: torch.Tensor, scores: torch.Tensor, threshold: float = 0.0) -> dict[str, int]:
    """TP/FP/FN/TN at the given threshold (probes are bias-free: 0 is natural)."""
    pred = (scores > threshold).long()
    y = y_true.long()
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
    }


def _agreement(pred_a: torch.Tensor, pred_b: torch.Tensor) -> float:
    """Fraction of identical binary predictions."""
    return float((pred_a.long() == pred_b.long()).float().mean())


def _best_layer(jsonl_path: Path, probe: str = "logistic") -> int:
    recs = [r for r in read_records(jsonl_path) if r.get("probe") == probe]
    if not recs:
        raise RobustnessError(f"no {probe} records in {jsonl_path}")
    return max(recs, key=lambda r: r["auroc_test"])["layer"]


def run_robustness(
    cfg: Config,
    *,
    pca_path: str | Path | None = None,
    fit_split: str | None = None,
    layer: int | None = None,
    seed_stability: bool = False,
    log=print,
) -> dict:
    """Error analysis + length proxy (+ optional seed stability). Returns a
    summary dict with the output paths and headline numbers."""
    from .probes.trainable import TRAINABLE_LOSSES

    pca_path = resolve_pca(cfg, pca_path, fit_split)
    obj = torch.load(pca_path, map_location="cpu", weights_only=True)
    X = obj["embeddings"].float()
    N, L, _ = X.shape
    if obj.get("ids") != list(range(N)):
        raise RobustnessError("source artifact ids violate the row-alignment contract (ids must be 0..N-1)")
    y = torch.tensor(obj["labels"], dtype=torch.long)
    splits = list(obj["splits"])
    masks = split_masks(splits)

    dataset = load_dataset(cfg.dataset.file)
    if dataset.n_rows != N:
        raise RobustnessError(f"CSV rows ({dataset.n_rows}) != artifact rows ({N})")
    texts = dataset.texts
    n_words = torch.tensor([len(t.split()) for t in texts], dtype=torch.float32)
    n_chars = torch.tensor([len(t) for t in texts], dtype=torch.float32)

    probes = [p for p in cfg.probes if probe_artifact_path(pca_path, p).is_file()]
    if not probes:
        raise RobustnessError(f"no probe artifacts under {pca_path.stem} (run `rq-probe train` first)")
    Ws = {p: torch.load(probe_artifact_path(pca_path, p), map_location="cpu", weights_only=True)["W"].float() for p in probes}

    if layer is None:
        jsonl = METRICS_DIR / f"auroc_{pca_path.stem}.jsonl"
        if not jsonl.is_file():
            raise RobustnessError(f"{jsonl} not found — run `rq-probe evaluate` first or pass --layer")
        layer = _best_layer(jsonl)
    if not 0 <= layer < L:
        raise RobustnessError(f"layer must be within 0..{L - 1}, got {layer}")

    summary: dict = {"stem": pca_path.stem, "layer": layer, "probes": probes}

    # ---------- error analysis at the operating layer ----------
    y_test = y[masks["test"]]
    words_test = n_words[masks["test"]]
    scores_test = {p: (X[:, layer, :] @ Ws[p][layer])[masks["test"]] for p in probes}
    preds_test = {p: (s > 0).long() for p, s in scores_test.items()}

    confusion = {p: _confusion(y_test, scores_test[p]) for p in probes}
    agreement = {
        f"{a}+{b}": _agreement(preds_test[a], preds_test[b])
        for i, a in enumerate(probes) for b in probes[i + 1:]
    }
    log(f"layer {layer} confusion (threshold 0, test n={int(masks['test'].sum())}):")
    for p in probes:
        c = confusion[p]
        log(f"  {p}: tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']} (acc {(c['tp'] + c['tn']) / sum(c.values()):.3f})")
    for pair, ag in agreement.items():
        log(f"  agreement {pair}: {ag:.3f}")

    csv_path = METRICS_DIR / f"errors_{pca_path.stem}_L{layer}.csv"
    _write_error_csv(csv_path, masks, y, splits, n_words, n_chars, scores_test, preds_test, probes, texts)
    log(f"saved per-example error CSV to {csv_path}")

    for p in probes:
        c = confusion[p]
        fn_words = words_test[(y_test == 1) & (preds_test[p] == 0)]
        fp_words = words_test[(y_test == 0) & (preds_test[p] == 1)]
        log(
            f"  {p} length (words): FN mean {float(fn_words.mean()) if fn_words.numel() else float('nan'):.1f} | "
            f"FP mean {float(fp_words.mean()) if fp_words.numel() else float('nan'):.1f} | "
            f"rhetorical mean {float(words_test[y_test == 1].mean()):.1f} | "
            f"informational mean {float(words_test[y_test == 0].mean()):.1f}"
        )

    # ---------- length proxy (all layers) ----------
    t0 = time.time()
    lp_records, lp_summary = _length_proxy(X, Ws, y, masks, n_words, probes, layer)
    lp_path = METRICS_DIR / f"lengthproxy_{pca_path.stem}.jsonl"
    _write_jsonl(lp_path, lp_records)
    summary["length_baseline_auroc"] = lp_summary["baseline_auroc"]
    summary["max_abs_spearman_length"] = lp_summary["max_abs_rho"]
    log(
        f"length proxy: word-count-only test AUROC {lp_summary['baseline_auroc']:.3f}; "
        f"|rho(score, length)| max {lp_summary['max_abs_rho']:.3f} "
        f"({lp_summary['max_rho_where']}); bucketed AUROC at L{layer} in {lp_path} ({time.time() - t0:.0f}s)"
    )

    summary["confusion"] = confusion
    summary["agreement"] = agreement
    summary["errors_csv"] = str(csv_path)
    summary["lengthproxy_jsonl"] = str(lp_path)

    # ---------- optional seed stability ----------
    if seed_stability:
        ss_path, ss_summary = _seed_stability(
            cfg, pca_path.stem, X, y, splits, masks, [p for p in probes if p in TRAINABLE_LOSSES], log
        )
        summary["seed_stability"] = ss_summary
        summary["seedstability_jsonl"] = str(ss_path)
        log(
            f"seed stability (seeds {ss_summary['seeds']}): across-seed std of test AUROC "
            f"max {ss_summary['max_std']:.4f} / mean {ss_summary['mean_std']:.4f} vs "
            f"mean Hanley–McNeil CI width {ss_summary['mean_hm_width']:.4f} (canonical seed)"
        )

    return summary


def _write_error_csv(path, masks, y, splits, n_words, n_chars, scores_test, preds_test, probes, texts) -> None:
    test_idx = torch.nonzero(masks["test"]).flatten().tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["id", "split", "label", "n_words", "n_chars"]
            + [f"{p}_score" for p in probes] + [f"{p}_pred" for p in probes] + [f"{p}_correct" for p in probes]
            + ["n_wrong", "probes_disagree", "text"]
        )
        for k, i in enumerate(test_idx):
            row = [i, splits[i], int(y[i]), int(n_words[i]), int(n_chars[i])]
            row += [f"{float(scores_test[p][k]):.6f}" for p in probes]
            row += [int(preds_test[p][k]) for p in probes]
            row += [int(preds_test[p][k] == y[i]) for p in probes]
            n_wrong = sum(int(preds_test[p][k] != y[i]) for p in probes)
            disagree = len({int(preds_test[p][k]) for p in probes}) > 1
            row += [n_wrong, int(disagree), texts[i]]
            w.writerow(row)


def _length_proxy(X, Ws, y, masks, n_words, probes, layer) -> tuple[list[dict], dict]:
    y_test = y[masks["test"]]
    words_test = n_words[masks["test"]]
    records = [
        {"probe": "word_count", "auroc_test": auroc(y_test.numpy(), words_test.numpy()), "meta": {"kind": "length_baseline"}}
    ]
    baseline = records[0]["auroc_test"]
    max_abs_rho, max_where = 0.0, ""
    for p in probes:
        for layer_l in range(X.shape[1]):
            s_test = (X[:, layer_l, :] @ Ws[p][layer_l])[masks["test"]]
            rho = spearman(s_test.numpy(), words_test.numpy())
            records.append({"layer": layer_l, "probe": p, "spearman_length": rho})
            if abs(rho) > max_abs_rho:
                max_abs_rho, max_where = abs(rho), f"{p} L{layer_l}"

    # length-tercile AUROC at the operating layer
    qs = np.quantile(words_test.numpy(), [1 / 3, 2 / 3])
    buckets = np.digitize(words_test.numpy(), qs)  # 0, 1, 2
    for p in probes:
        s_test = scores_at(X, Ws[p], layer, masks)
        for b in (0, 1, 2):
            sel = torch.tensor(buckets == b)
            rec = {"layer": layer, "probe": p, "bucket": int(b), "n": int(sel.sum())}
            if rec["n"] == 0:
                rec["word_range"] = None
                rec["auroc_test"] = None
            else:
                rec["word_range"] = [int(words_test[sel].min()), int(words_test[sel].max())]
                yb = y_test[sel]
                rec["auroc_test"] = auroc(yb.numpy(), s_test[sel].numpy()) if len(np.unique(yb.numpy())) > 1 else None
            records.append(rec)
    summary = {"baseline_auroc": baseline, "max_abs_rho": max_abs_rho, "max_rho_where": max_where}
    return records, summary


def scores_at(X, W, layer, masks):
    return (X[:, layer, :] @ W[layer])[masks["test"]]


def _seed_stability(cfg, stem, X, y, splits, masks, probes, log) -> tuple[Path, dict]:
    seeds = [cfg.experiment.seed, 1, 2]
    hp = cfg.probe_hparams
    records = []
    t0 = time.time()
    for p in probes:
        for layer in range(X.shape[1]):
            aucs, hm_widths = [], []
            for s in seeds:
                res = train_layer(
                    X[:, layer, :], y, splits,
                    loss_type=p, seed=s,
                    lr=hp.lr, weight_decay=hp.weight_decay, batch_size=hp.batch_size,
                    max_epochs=hp.max_epochs, patience=hp.patience,
                    early_stopping_metric=hp.early_stopping_metric, log=lambda *_: None,
                )
                s_test = (X[:, layer, :] @ res["W"])[masks["test"]]
                auc, lo, hi = auroc_hanley_mcneil(y[masks["test"]].numpy(), s_test.numpy())
                aucs.append(auc)
                hm_widths.append(hi - lo)
            records.append(
                {"layer": layer, "probe": p, "seeds": seeds, "auroc_per_seed": aucs,
                 "std": float(np.std(aucs, ddof=1)), "hm_ci_width_canonical": hm_widths[0]}
            )
        log(f"seed stability: {p} done ({time.time() - t0:.0f}s)")
    stds = [r["std"] for r in records]
    widths = [r["hm_ci_width_canonical"] for r in records]
    out = METRICS_DIR / f"seedstability_{stem}.jsonl"
    _write_jsonl(out, records)
    summary = {"seeds": seeds, "max_std": max(stds), "mean_std": float(np.mean(stds)), "mean_hm_width": float(np.mean(widths))}
    return out, summary


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
