"""Layer-wise evaluation (Phase 6, Tier 1): test AUROC + Hanley–McNeil CIs.

Scores each probe artifact against its PCA artifact. Protocol (docs/
EXPERIMENT_PROTOCOL.md §6, §7.4): the test split is touched exactly once per
(probe, layer) here and never fed back; train/dev AUROCs are logged in the
JSONL for diagnosis only, clearly labeled. The shuffled-label negative control
(`negative_control=True`) permutes TRAIN labels only (seeded), retrains the
logistic probe per layer, and scores the untouched real test labels — chance
performance is the expected outcome.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .extraction import artifact_path
from .metrics import auroc, auroc_hanley_mcneil
from .plotting import fig1_auroc
from .probes.trainable import probe_artifact_path, train_layer

METRICS_DIR = Path("outputs/metrics")


class EvaluateError(RuntimeError):
    pass


def resolve_pca(cfg: Config, pca_path: str | Path | None, fit_split: str | None) -> Path:
    if pca_path is None:
        fs = fit_split or cfg.pca.fit_split
        from .pca_step import pca_artifact_paths

        pca_path = pca_artifact_paths(artifact_path(cfg), cfg.pca.n_components, fs)[0]
    pca_path = Path(pca_path)
    if not pca_path.is_file():
        raise EvaluateError(f"source PCA artifact not found: {pca_path} (run `rq-probe pca` first)")
    return pca_path


def split_masks(splits: list[str]) -> dict[str, torch.Tensor]:
    return {
        name: torch.tensor([s == name for s in splits])
        for name in ("train", "validation", "test")
    }


def _scores_for_layer(X: torch.Tensor, layer: int, W: torch.Tensor) -> torch.Tensor:
    return X[:, layer, :] @ W[layer]


def _record(layer, probe, y, scores, masks, meta) -> dict:
    y_test = y[masks["test"]]
    s_test = scores[masks["test"]]
    auc, lo, hi = auroc_hanley_mcneil(y_test.numpy(), s_test.numpy())
    return {
        "layer": layer,
        "probe": probe,
        "auroc_test": auc,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "auroc_dev": auroc(y[masks["validation"]].numpy(), scores[masks["validation"]].numpy()),
        "auroc_train": auroc(y[masks["train"]].numpy(), scores[masks["train"]].numpy()),
        "n_pos_test": int((y_test == 1).sum()),
        "n_neg_test": int((y_test == 0).sum()),
        "meta": meta,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _fmt(r: dict) -> str:
    return f"{r['auroc_test']:.3f} [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]"


def _write_table(path: Path, records: list[dict], probes: list[str]) -> None:
    by_layer: dict[int, dict[str, dict]] = {}
    for r in records:
        by_layer.setdefault(r["layer"], {})[r["probe"]] = r
    lines = ["| layer | " + " | ".join(probes) + " |", "|---|" + "---|" * len(probes)]
    for layer in sorted(by_layer):
        cells = [_fmt(by_layer[layer][p]) if p in by_layer[layer] else "—" for p in probes]
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_fit_comparison(pca_path: Path, log) -> None:
    """Side-by-side fit-all vs fit-train table, written when both JSONLs exist."""
    other_stem = pca_path.stem.replace("_fit-all", "_fit-train") if "_fit-all" in pca_path.stem else pca_path.stem.replace("_fit-train", "_fit-all")
    if other_stem == pca_path.stem:
        return
    paths = {stem: METRICS_DIR / f"auroc_{stem}.jsonl" for stem in (pca_path.stem, other_stem)}
    if not all(p.is_file() for p in paths.values()):
        return
    fits = {}
    for stem, p in paths.items():
        for r in read_records(p):
            fits[(r["probe"], r["layer"], stem.split("_fit-")[1])] = r["auroc_test"]
    probes = list(dict.fromkeys(pr for pr, _, _ in fits))
    layers = sorted({l for _, l, _ in fits})
    header = ["layer"]
    for probe in probes:
        header += [f"{probe} fit-all", f"{probe} fit-train", f"{probe} Δ"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for layer in layers:
        row = [str(layer)]
        for probe in probes:
            a_all = fits.get((probe, layer, "all"))
            a_tr = fits.get((probe, layer, "train"))
            row += [
                f"{a_all:.3f}" if a_all is not None else "—",
                f"{a_tr:.3f}" if a_tr is not None else "—",
                f"{a_all - a_tr:+.3f}" if a_all is not None and a_tr is not None else "—",
            ]
        lines.append("| " + " | ".join(row) + " |")
    out = METRICS_DIR / "auroc_fit-comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"saved fit-all vs fit-train comparison to {out}")


def read_records(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_evaluate(
    cfg: Config,
    *,
    pca_path: str | Path | None = None,
    fit_split: str | None = None,
    probes: list[str] | None = None,
    negative_control: bool = False,
    log=print,
) -> Path:
    """Evaluate probe artifacts on the test split; returns the JSONL path."""
    probes = list(probes) if probes is not None else list(cfg.probes)
    pca_path = resolve_pca(cfg, pca_path, fit_split)

    obj = torch.load(pca_path, map_location="cpu", weights_only=True)
    X = obj["embeddings"].float()
    N, L, _ = X.shape
    if obj.get("ids") != list(range(N)):
        raise EvaluateError("source artifact ids violate the row-alignment contract (ids must be 0..N-1)")
    y = torch.tensor(obj["labels"], dtype=torch.long)
    splits = list(obj["splits"])
    masks = split_masks(splits)
    if masks["test"].sum() == 0:
        raise EvaluateError("test split is empty — nothing to evaluate")

    meta_common = {
        "source_artifact": str(pca_path),
        "fit_split": obj.get("meta", {}).get("fit_split"),
        "seed": cfg.experiment.seed,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    records: list[dict] = []
    for probe in probes:
        probe_file = probe_artifact_path(pca_path, probe)
        if not probe_file.is_file():
            raise EvaluateError(f"probe artifact not found: {probe_file} (run `rq-probe train` first)")
        art = torch.load(probe_file, map_location="cpu", weights_only=True)
        if art["meta"]["source_artifact"] != str(pca_path):
            raise EvaluateError(f"{probe_file} was trained on {art['meta']['source_artifact']}, not {pca_path}")
        W = art["W"].float()
        t0 = time.time()
        for layer in range(L):
            records.append(_record(layer, probe, y, _scores_for_layer(X, layer, W), masks, meta_common))
        log(f"evaluated {probe}: {L} layers on test n={int(masks['test'].sum())} ({time.time() - t0:.1f}s)")

    jsonl_path = METRICS_DIR / f"auroc_{pca_path.stem}.jsonl"
    _write_jsonl(jsonl_path, records)
    _write_table(METRICS_DIR / f"auroc_table_{pca_path.stem}.md", records, probes)
    fig = fig1_auroc(jsonl_path, Path("outputs/figures") / f"fig1_auroc_{pca_path.stem}.png")
    log(f"saved {len(records)} records to {jsonl_path}")
    log(f"saved AUROC table to {METRICS_DIR / f'auroc_table_{pca_path.stem}.md'}")
    log(f"saved figure to {fig}")

    if negative_control:
        _run_negative_control(cfg, pca_path, X, y, splits, masks, meta_common, log)

    _write_fit_comparison(pca_path, log)
    return jsonl_path


def _run_negative_control(cfg, pca_path, X, y, splits, masks, meta_common, log) -> None:
    """Shuffle TRAIN labels (seeded), retrain logistic per layer, score the
    untouched real test labels; ≈0.5 is the expected outcome."""
    train_idx = torch.tensor([i for i, s in enumerate(splits) if s == "train"], dtype=torch.long)
    rng = np.random.default_rng(cfg.experiment.seed)
    y_shuffled = y.clone()
    y_shuffled[train_idx] = y[train_idx][torch.from_numpy(rng.permutation(train_idx.numel()))]

    hp = cfg.probe_hparams
    records = []
    t0 = time.time()
    for layer in range(X.shape[1]):
        res = train_layer(
            X[:, layer, :], y_shuffled, splits,
            loss_type="logistic", seed=cfg.experiment.seed,
            lr=hp.lr, weight_decay=hp.weight_decay, batch_size=hp.batch_size,
            max_epochs=hp.max_epochs, patience=hp.patience,
            early_stopping_metric=hp.early_stopping_metric, log=lambda *_: None,
        )
        scores = X[:, layer, :] @ res["W"]  # res["W"] is this layer's [K] vector
        records.append(
            {
                "layer": layer,
                "probe": "logistic_shuffled",
                "auroc_test": auroc(y[masks["test"]].numpy(), scores[masks["test"]].numpy()),
                "control": "shuffled_train_labels",
                "meta": meta_common,
            }
        )
    aucs = [r["auroc_test"] for r in records]
    out = METRICS_DIR / f"auroc_{pca_path.stem}_shuffled.jsonl"
    _write_jsonl(out, records)
    log(
        f"negative control (shuffled train labels, real test labels): AUROC "
        f"min/mean/max = {min(aucs):.3f}/{sum(aucs) / len(aucs):.3f}/{max(aucs):.3f} -> {out} "
        f"({time.time() - t0:.0f}s)"
    )
