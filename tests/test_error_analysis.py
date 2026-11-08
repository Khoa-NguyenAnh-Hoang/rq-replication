"""Tests for rq_probe.error_analysis — synthetic mini-pipeline fixtures, CPU only."""

import csv
import json
from pathlib import Path

import pytest
import torch
from sklearn.metrics import roc_auc_score

from rq_probe.config import Config
from rq_probe.error_analysis import RobustnessError, _agreement, _confusion, run_robustness
from rq_probe.evaluate import read_records, run_evaluate
from rq_probe.probes.trainable import run_train

N, L, K = 60, 3, 4
SPLITS = ["train"] * 40 + ["validation"] * 10 + ["test"] * 10
LOG_QUIET = lambda *_: None  # noqa: E731


def make_cfg() -> Config:
    cfg = Config()
    cfg.probe_hparams.max_epochs = 100
    return cfg


def _write_fixture_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["context", "question", "label", "dataset", "id", "question_with_context", "binary_label"])
        for i in range(N):
            lab = i % 2
            w.writerow([f"c{i}", f"q{i}?", str(lab), SPLITS[i], str(i),
                        f"question number {i} {'x ' * (i % 5)}end", str(lab)])


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_fixture_csv(Path("data/raw/RQ.csv"))
    y = torch.tensor([i % 2 for i in range(N)])
    g = torch.Generator().manual_seed(0)
    X = torch.randn(N, L, K, generator=g)
    X[y == 1, :, 0] += 3.0
    pca_path = Path("outputs/pca/synth_pca64_fit-all.pt")
    pca_path.parent.mkdir(parents=True)
    torch.save(
        {
            "embeddings": X, "ids": list(range(N)), "labels": y.tolist(), "splits": SPLITS,
            "meta": {"n_components": K, "fit_split": "all", "source_meta": {"model_checkpoint": "fake/model"}},
        },
        pca_path,
    )
    run_train(make_cfg(), pca_path=pca_path, log=LOG_QUIET)
    run_evaluate(make_cfg(), pca_path=pca_path, log=LOG_QUIET)  # auroc JSONL for best-layer lookup
    return pca_path


# ---------- pure helpers ----------

def test_confusion_hand_computed():
    y = torch.tensor([1, 1, 0, 0, 1])
    s = torch.tensor([2.0, -1.0, 3.0, -2.0, 0.5])  # thr 0: preds [1,0,1,0,1]
    assert _confusion(y, s) == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}


def test_agreement():
    assert _agreement(torch.tensor([1, 0, 1]), torch.tensor([1, 0, 0])) == pytest.approx(2 / 3)


# ---------- end to end ----------

def test_robustness_end_to_end(workspace):
    summary = run_robustness(make_cfg(), pca_path=workspace, log=LOG_QUIET)
    assert summary["layer"] >= 0 and set(summary["confusion"]) == {"diffmean", "logistic", "hinge"}
    for c in summary["confusion"].values():
        assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == 10  # test rows
    assert all(0.0 <= a <= 1.0 for a in summary["agreement"].values())

    errors_csv = Path(summary["errors_csv"])
    rows = list(csv.DictReader(open(errors_csv, encoding="utf-8")))
    assert len(rows) == 10
    assert rows[0]["text"] == f"question number {rows[0]['id']} {'x ' * (int(rows[0]['id']) % 5)}end"  # join-by-id
    assert {"logistic_score", "logistic_pred", "n_wrong", "probes_disagree"} <= set(rows[0])

    lp = read_records(Path(summary["lengthproxy_jsonl"]))
    baseline = next(r for r in lp if r["probe"] == "word_count")
    y_test = torch.tensor([i % 2 for i in range(N)][40:])
    words = torch.tensor([len(f"question number {i} {'x ' * (i % 5)}end".split()) for i in range(N)][40:])
    assert baseline["auroc_test"] == pytest.approx(roc_auc_score(y_test.numpy(), words.numpy()))
    rhos = [r for r in lp if "spearman_length" in r]
    assert len(rhos) == 3 * L and all(-1.0 <= r["spearman_length"] <= 1.0 for r in rhos)
    buckets = [r for r in lp if "bucket" in r]
    assert len(buckets) == 3 * 3  # probes x buckets at the operating layer


def test_seed_stability_deterministic_and_shaped(workspace):
    kw = dict(pca_path=workspace, layer=1, seed_stability=True, log=LOG_QUIET)
    s1 = run_robustness(make_cfg(), **kw)
    s2 = run_robustness(make_cfg(), **kw)
    assert s1["seed_stability"]["seeds"] == [42, 1, 2]
    assert s1["seed_stability"] == s2["seed_stability"]  # same seeds -> identical reruns
    recs = read_records(Path(s1["seedstability_jsonl"]))
    assert len(recs) == 2 * L  # logistic + hinge x layers
    assert all(len(r["auroc_per_seed"]) == 3 and r["std"] >= 0 for r in recs)


def test_layer_required_when_no_jsonl(workspace, tmp_path):
    for f in Path("outputs/metrics").glob("auroc_synth_pca64_fit-all.jsonl"):
        f.unlink()
    with pytest.raises(RobustnessError, match="evaluate"):
        run_robustness(make_cfg(), pca_path=workspace, log=LOG_QUIET)


def test_csv_row_count_mismatch_rejected(workspace):
    cfg = make_cfg()
    cfg.dataset.file = "data/raw/nope.csv"
    with pytest.raises(Exception, match="not found"):
        run_robustness(cfg, pca_path=workspace, layer=0, log=LOG_QUIET)
