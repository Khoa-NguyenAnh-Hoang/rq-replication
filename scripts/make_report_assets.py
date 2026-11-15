"""Generate report assets (Phase 9 scaffolding): tables, figures, headline numbers.

Reads the metrics JSONLs / tables / figures under outputs/ and the dataset CSV,
then writes everything the report needs into report/assets/ (gitignored —
regenerate any time with this script):

- numbers.json            machine-readable headline numbers (traceability: every
                          report number should come from here)
- dataset_stats.md        split/class table (recomputed from the CSV)
- probe_comparison.md     per stem: mid-layer mean + peak per probe (RQ2)
- robustness_summary.md   negative control, length proxy, seed stability, errors
- copies of the generated AUROC/alignment/comparison tables and figures

Usage: uv run python scripts/make_report_assets.py [--metrics-dir ... --csv ... --out-dir ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from rq_probe.data import load_dataset

MID_LO, MID_HI = 13, 25  # "mid layers" band used throughout the docs
TRAINABLE = ("logistic", "hinge", "diffmean")


def _records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _mid_peak(records: list[dict], probe: str) -> dict | None:
    rows = [r for r in records if r.get("probe") == probe and "auroc_test" in r]
    if not rows:
        return None
    mid = [r["auroc_test"] for r in rows if MID_LO <= r["layer"] <= MID_HI]
    peak = max(rows, key=lambda r: r["auroc_test"])
    return {
        "mid_layer_mean": round(sum(mid) / len(mid), 4) if mid else None,
        "peak_layer": peak["layer"],
        "peak_auroc": round(peak["auroc_test"], 4),
        "peak_ci": [round(peak.get("ci95_lo", peak["auroc_test"]), 4), round(peak.get("ci95_hi", peak["auroc_test"]), 4)],
    }


def _split_pooling_fit(stem: str) -> tuple[str, str]:
    pooling = "mean_token" if "mean_token" in stem else "last_token"
    fit = "train" if stem.endswith("_fit-train") else "all"
    return pooling, fit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-dir", default="outputs/metrics")
    ap.add_argument("--figures-dir", default="outputs/figures")
    ap.add_argument("--csv", default="data/raw/RQ.csv")
    ap.add_argument("--out-dir", default="report/assets")
    args = ap.parse_args(argv)

    metrics, figures, out = Path(args.metrics_dir), Path(args.figures_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numbers: dict = {}

    # ---------- dataset ----------
    ds = load_dataset(args.csv)
    numbers["dataset"] = {"sha256": ds.sha256, "n_rows": ds.n_rows, "counts": ds.counts()}
    lines = ["| split | total | rhetorical | informational |", "|---|---|---|---|"]
    for split, c in ds.counts().items():
        lines.append(f"| {split} | {c['total']} | {c['rhetorical']} | {c['informational']} |")
    lines.append(f"\nCanonical sha256: `{ds.sha256}`")
    _write(out / "dataset_stats.md", "\n".join(lines) + "\n")

    # ---------- AUROC per stem ----------
    auroc_paths = sorted(p for p in metrics.glob("auroc_*.jsonl") if not p.stem.endswith("_shuffled"))
    numbers["auroc"] = {}
    comparison_rows = []
    for p in auroc_paths:
        stem = p.stem.removeprefix("auroc_")
        records = _records(p)
        per_probe = {probe: _mid_peak(records, probe) for probe in TRAINABLE}
        numbers["auroc"][stem] = per_probe
        table = Path(metrics) / f"auroc_table_{stem}.md"
        if table.is_file():
            shutil.copy(table, out / table.name)
        pooling, fit = _split_pooling_fit(stem)
        for probe in TRAINABLE:
            s = per_probe[probe]
            if s:
                comparison_rows.append((pooling, fit, probe, s["mid_layer_mean"], s["peak_layer"], s["peak_auroc"]))

        shuffled = metrics / f"auroc_{stem}_shuffled.jsonl"
        if shuffled.is_file():
            aucs = [r["auroc_test"] for r in _records(shuffled) if "auroc_test" in r]
            numbers.setdefault("negative_control", {})[stem] = {
                "min": round(min(aucs), 4), "mean": round(sum(aucs) / len(aucs), 4), "max": round(max(aucs), 4)
            }

    # ---------- probe comparison (RQ2) ----------
    lines = ["| pooling | fit | probe | mid-layer mean | peak layer | peak AUROC |", "|---|---|---|---|---|---|"]
    for pooling, fit, probe, mid, peak_l, peak in comparison_rows:
        lines.append(f"| {pooling} | {fit} | {probe} | {'—' if mid is None else f'{mid:.3f}'} | {peak_l} | {peak:.3f} |")
    _write(out / "probe_comparison.md", "\n".join(lines) + "\n")

    # ---------- fit / pooling comparisons ----------
    by_pooling: dict[str, dict[str, dict]] = defaultdict(dict)
    for stem, per_probe in numbers["auroc"].items():
        pooling, fit = _split_pooling_fit(stem)
        by_pooling[pooling][fit] = per_probe
    for pooling, fits in by_pooling.items():
        if "all" in fits and "train" in fits:
            deltas = {}
            for probe in TRAINABLE:
                a, b = fits["all"].get(probe), fits["train"].get(probe)
                if a and b and a["peak_auroc"] is not None and b["peak_auroc"] is not None:
                    deltas[probe] = round(abs(a["mid_layer_mean"] - b["mid_layer_mean"]), 4)
            numbers.setdefault("fit_split_max_abs_delta", {})[pooling] = deltas
    if "last_token" in by_pooling and "mean_token" in by_pooling:
        deltas = {}
        for probe in TRAINABLE:
            a = by_pooling["last_token"]["all"].get(probe)
            b = by_pooling["mean_token"]["all"].get(probe)
            if a and b:
                deltas[probe] = round(b["mid_layer_mean"] - a["mid_layer_mean"], 4)
        numbers["mean_vs_last_mid_layer_delta"] = deltas

    # ---------- alignment ----------
    numbers["alignment"] = {}
    for p in sorted(metrics.glob("alignment_*.jsonl")):
        stem = p.stem.removeprefix("alignment_")
        records = _records(p)
        pairs: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        mapback = []
        for r in records:
            if "cosine" in r:
                pairs[r["pair"]]["cosine"].append(r["cosine"])
                pairs[r["pair"]]["jtop"].append(r["jaccard_top"])
                pairs[r["pair"]]["jbot"].append(r["jaccard_bottom"])
            elif "auroc_pca" in r:
                mapback.append(abs(r["auroc_pca"] - r["auroc_original"]))
        entry = {
            pair: {
                "cosine_mean": round(sum(v["cosine"]) / len(v["cosine"]), 4),
                "cosine_min": round(min(v["cosine"]), 4),
                "jaccard_top_mean": round(sum(v["jtop"]) / len(v["jtop"]), 4),
                "jaccard_bottom_mean": round(sum(v["jbot"]) / len(v["jbot"]), 4),
            }
            for pair, v in pairs.items()
        }
        if mapback:
            entry["mapback_max_abs_auroc_delta"] = round(max(mapback), 5)
        numbers["alignment"][stem] = entry
        table = metrics / f"alignment_table_{stem}.md"
        if table.is_file():
            shutil.copy(table, out / table.name)

    # ---------- robustness ----------
    numbers["robustness"] = {}
    for p in sorted(metrics.glob("errors_*.csv")):
        import csv as _csv

        rows = list(_csv.DictReader(open(p, encoding="utf-8")))
        probes = [c.removesuffix("_pred") for c in rows[0] if c.endswith("_pred")]
        entry = {}
        for probe in probes:
            tp = sum(1 for r in rows if r["label"] == "1" and r[f"{probe}_pred"] == "1")
            fp = sum(1 for r in rows if r["label"] == "0" and r[f"{probe}_pred"] == "1")
            fn = sum(1 for r in rows if r["label"] == "1" and r[f"{probe}_pred"] == "0")
            tn = sum(1 for r in rows if r["label"] == "0" and r[f"{probe}_pred"] == "0")
            entry[probe] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        if len(probes) >= 2:
            entry["agreement"] = {
                f"{probes[i]}+{probes[j]}": round(
                    sum(1 for r in rows if r[f"{probes[i]}_pred"] == r[f"{probes[j]}_pred"]) / len(rows), 4
                )
                for i in range(len(probes)) for j in range(i + 1, len(probes))
            }
        numbers["robustness"][p.stem.removeprefix("errors_")] = entry
    for p in sorted(metrics.glob("lengthproxy_*.jsonl")):
        recs = _records(p)
        baseline = next((r for r in recs if r.get("probe") == "word_count"), None)
        buckets = [r for r in recs if "bucket" in r]
        rhos = [abs(r["spearman_length"]) for r in recs if "spearman_length" in r]
        numbers.setdefault("length_proxy", {})[p.stem.removeprefix("lengthproxy_")] = {
            "word_count_only_auroc": None if baseline is None else round(baseline["auroc_test"], 4),
            "bucketed_auroc": [
                {"bucket": r["bucket"], "n": r["n"], "word_range": r["word_range"],
                 "auroc": None if r["auroc_test"] is None else round(r["auroc_test"], 4)}
                for r in buckets
            ],
            "max_abs_spearman_length": round(max(rhos), 4) if rhos else None,
        }
    for p in sorted(metrics.glob("seedstability_*.jsonl")):
        recs = _records(p)
        numbers.setdefault("seed_stability", {})[p.stem.removeprefix("seedstability_")] = {
            "seeds": recs[0]["seeds"],
            "max_std": round(max(r["std"] for r in recs), 4),
            "mean_std": round(sum(r["std"] for r in recs) / len(recs), 4),
            "mean_hm_ci_width": round(sum(r["hm_ci_width_canonical"] for r in recs) / len(recs), 4),
        }

    # ---------- robustness_summary.md ----------
    lines = ["# Robustness summary (generated)\n"]
    for stem, nc in numbers.get("negative_control", {}).items():
        lines.append(f"- Negative control ({stem}): shuffled-train-labels AUROC min/mean/max = {nc['min']}/{nc['mean']}/{nc['max']}")
    for stem, lp in numbers.get("length_proxy", {}).items():
        flat = ", ".join(f"{b['auroc']:.3f}" for b in lp["bucketed_auroc"] if b["auroc"] is not None)
        lines.append(f"- Length proxy ({stem}): word-count-only AUROC {lp['word_count_only_auroc']}; tercile AUROC {flat}; max |rho| {lp['max_abs_spearman_length']}")
    for stem, ss in numbers.get("seed_stability", {}).items():
        lines.append(f"- Seed stability ({stem}): seeds {ss['seeds']}, across-seed std max {ss['max_std']} / mean {ss['mean_std']} vs mean HM CI width {ss['mean_hm_ci_width']}")
    for stem, entry in numbers.get("robustness", {}).items():
        for probe in TRAINABLE:
            if probe in entry and isinstance(entry[probe], dict):
                c = entry[probe]
                lines.append(f"- Errors {stem} {probe}: tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']}")
        if "agreement" in entry:
            lines.append(f"- Errors {stem} agreement: {entry['agreement']}")
    _write(out / "robustness_summary.md", "\n".join(lines) + "\n")

    # ---------- comparisons already generated as md ----------
    for name in ("auroc_fit-comparison.md", "auroc_pooling-comparison.md"):
        if (metrics / name).is_file():
            shutil.copy(metrics / name, out / name)

    # ---------- figures ----------
    if figures.is_dir():
        for png in sorted(figures.glob("*.png")):
            shutil.copy(png, out / png.name)

    _write(out / "numbers.json", json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    print(f"report assets written to {out} ({len(list(out.iterdir()))} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
