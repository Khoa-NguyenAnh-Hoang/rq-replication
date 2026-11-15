"""Tests for scripts/make_report_assets.py — synthetic metrics fixtures."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from make_report_assets import main  # noqa: E402

N = 12
SPLITS = ["train"] * 8 + ["validation"] * 2 + ["test"] * 2


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics = Path("outputs/metrics")
    metrics.mkdir(parents=True)

    with open("data_raw.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["context", "question", "label", "dataset", "id", "question_with_context", "binary_label"])
        for i in range(N):
            lab = i % 2
            w.writerow([f"c{i}", f"q{i}?", str(lab), SPLITS[i], str(i), f"text {i}", str(lab)])

    def auroc_line(layer, probe):
        return json.dumps({"layer": layer, "probe": probe, "auroc_test": 0.5 + 0.1 * layer,
                           "ci95_lo": 0.4, "ci95_hi": 0.6 + 0.1 * layer})

    with open(metrics / "auroc_synth_pca64_fit-all.jsonl", "w", encoding="utf-8") as f:
        for l in range(3):
            for p in ("logistic", "hinge", "diffmean"):
                f.write(auroc_line(l, p) + "\n")
    (metrics / "auroc_table_synth_pca64_fit-all.md").write_text("| layer | logistic |\n", encoding="utf-8")
    with open(metrics / "auroc_synth_pca64_fit-all_shuffled.jsonl", "w", encoding="utf-8") as f:
        for l in range(3):
            f.write(json.dumps({"layer": l, "probe": "logistic_shuffled", "auroc_test": 0.5 + 0.01 * l}) + "\n")
    with open(metrics / "alignment_synth_pca64_fit-all.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"layer": 0, "pair": "logistic+hinge", "cosine": 0.9,
                            "jaccard_top": 0.5, "jaccard_bottom": 0.7}) + "\n")
        f.write(json.dumps({"layer": 0, "probe": "logistic", "auroc_pca": 0.75, "auroc_original": 0.75}) + "\n")
    with open(metrics / "lengthproxy_synth_pca64_fit-all.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"probe": "word_count", "auroc_test": 0.55}) + "\n")
        f.write(json.dumps({"layer": 0, "probe": "logistic", "spearman_length": 0.2}) + "\n")
        f.write(json.dumps({"layer": 0, "probe": "logistic", "bucket": 0, "n": 2, "word_range": [3, 4], "auroc_test": 0.8}) + "\n")
    with open(metrics / "seedstability_synth_pca64_fit-all.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"layer": 0, "probe": "logistic", "seeds": [42, 1, 2],
                            "auroc_per_seed": [0.8, 0.7, 0.6], "std": 0.1, "hm_ci_width_canonical": 0.06}) + "\n")
    with open(metrics / "errors_synth_pca64_fit-all_L0.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "label", "n_words", "n_chars", "logistic_score", "logistic_pred",
                    "logistic_correct", "hinge_score", "hinge_pred", "hinge_correct", "n_wrong", "probes_disagree", "text"])
        for i in range(2):
            lab = i % 2
            w.writerow([8 + i, "test", lab, 3, 10, 0.5 * (i + 1), str(lab), "1", -0.5, str(1 - lab), "0", 1 - lab, lab, f"text {8 + i}"])

    Path("outputs/figures").mkdir(parents=True)
    (Path("outputs/figures") / "fig1_auroc_synth_pca64_fit-all.png").write_bytes(b"png" * 100)
    return metrics


def test_main_generates_assets(workspace):
    rc = main(["--csv", "data_raw.csv"])
    assert rc == 0
    out = Path("report/assets")

    numbers = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    assert numbers["dataset"]["n_rows"] == N
    assert numbers["dataset"]["counts"]["train"]["total"] == 8
    a = numbers["auroc"]["synth_pca64_fit-all"]
    assert a["logistic"]["peak_layer"] == 2 and a["logistic"]["peak_auroc"] == pytest.approx(0.7, rel=1e-3)
    assert a["logistic"]["mid_layer_mean"] is None  # 3-layer fixture has no layer in the 13-25 band
    assert numbers["negative_control"]["synth_pca64_fit-all"]["mean"] == pytest.approx(0.51, rel=1e-3)
    assert numbers["alignment"]["synth_pca64_fit-all"]["logistic+hinge"]["cosine_mean"] == 0.9
    assert numbers["alignment"]["synth_pca64_fit-all"]["mapback_max_abs_auroc_delta"] == 0.0
    assert numbers["length_proxy"]["synth_pca64_fit-all"]["word_count_only_auroc"] == 0.55
    assert numbers["seed_stability"]["synth_pca64_fit-all"]["max_std"] == pytest.approx(0.1, rel=1e-3)
    errors = numbers["robustness"]["synth_pca64_fit-all_L0"]
    assert errors["logistic"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}
    assert errors["agreement"]["logistic+hinge"] == 0.0  # fixture preds are always opposite

    for name in ("dataset_stats.md", "probe_comparison.md", "robustness_summary.md",
                 "auroc_table_synth_pca64_fit-all.md", "fig1_auroc_synth_pca64_fit-all.png"):
        assert (out / name).is_file(), name
    summary = (out / "robustness_summary.md").read_text(encoding="utf-8")
    assert "Negative control" in summary and "Seed stability" in summary and "word-count-only AUROC 0.55" in summary
