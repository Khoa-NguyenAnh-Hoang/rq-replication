"""Tests for rq_probe.evaluate + plotting — synthetic fixtures, CPU only."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from sklearn.metrics import roc_auc_score

from rq_probe.config import Config
from rq_probe.evaluate import EvaluateError, read_records, run_evaluate
from rq_probe.probes.trainable import probe_artifact_path, run_train

N, L, K = 60, 3, 4
SPLITS = ["train"] * 40 + ["validation"] * 10 + ["test"] * 10
LOG_QUIET = lambda *_: None  # noqa: E731


def make_cfg(max_epochs: int = 100) -> Config:
    cfg = Config()
    cfg.probe_hparams.max_epochs = max_epochs
    return cfg


def _pca_payload(fit_split: str = "all", seed: int = 0):
    y = torch.tensor([i % 2 for i in range(N)])
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N, L, K, generator=g)
    X[y == 1, :, 0] += 3.0  # separable along dim 0 in every layer
    return {
        "embeddings": X,
        "ids": list(range(N)),
        "labels": y.tolist(),
        "splits": SPLITS,
        "meta": {
            "n_components": K,
            "fit_split": fit_split,
            "source_meta": {"model_checkpoint": "fake/model", "model_revision": None,
                             "pooling": "last_token", "max_length": 512, "dataset_sha256": "deadbeef"},
        },
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("outputs/pca/synth_pca64_fit-all.pt")
    path.parent.mkdir(parents=True)
    torch.save(_pca_payload("all"), path)
    run_train(make_cfg(), pca_path=path, log=LOG_QUIET)  # all three probes
    return path


def _test_scores(workspace, probe, layer):
    X = torch.load(workspace, weights_only=True)["embeddings"]
    y = torch.tensor([i % 2 for i in range(N)])
    W = torch.load(probe_artifact_path(workspace, probe), weights_only=True)["W"]
    scores = X[:, layer, :] @ W[layer]
    test = torch.tensor([s == "test" for s in SPLITS])
    return y[test].numpy(), scores[test].numpy()


def test_evaluate_jsonl_matches_sklearn(workspace):
    out = run_evaluate(make_cfg(), pca_path=workspace, probes=["logistic"], log=LOG_QUIET)
    records = read_records(out)
    assert len(records) == L
    r = next(r for r in records if r["layer"] == 0)
    y_test, s_test = _test_scores(workspace, "logistic", 0)
    assert r["auroc_test"] == pytest.approx(roc_auc_score(y_test, s_test))
    assert r["ci95_lo"] <= r["auroc_test"] <= r["ci95_hi"]
    assert r["n_pos_test"] == 5 and r["n_neg_test"] == 5
    assert r["auroc_dev"] > 0.9 and r["auroc_train"] > 0.9  # separable fixture
    assert r["meta"]["fit_split"] == "all"


def test_outputs_written(workspace):
    out = run_evaluate(make_cfg(), pca_path=workspace, log=LOG_QUIET)
    records = read_records(out)
    assert {r["probe"] for r in records} == {"diffmean", "logistic", "hinge"}
    assert len(records) == 3 * L
    table = Path("outputs/metrics") / f"auroc_table_{workspace.stem}.md"
    assert table.is_file() and "logistic" in table.read_text(encoding="utf-8")
    fig = Path("outputs/figures") / f"fig1_auroc_{workspace.stem}.png"
    assert fig.is_file() and fig.stat().st_size > 1000


def test_negative_control_is_chance(workspace):
    run_evaluate(make_cfg(), pca_path=workspace, probes=["logistic"], log=LOG_QUIET)
    out = run_evaluate(make_cfg(), pca_path=workspace, probes=["logistic"],
                       negative_control=True, log=LOG_QUIET)
    control = read_records(out.parent / f"auroc_{workspace.stem}_shuffled.jsonl")
    assert len(control) == L
    assert all(r["control"] == "shuffled_train_labels" for r in control)
    assert all(0.2 < r["auroc_test"] < 0.8 for r in control)
    real = {r["layer"]: r["auroc_test"] for r in read_records(out)}
    assert max(r["auroc_test"] for r in control) < min(real.values())  # signal beats shuffled labels


def test_fit_comparison_table(workspace):
    other = workspace.parent / "synth_pca64_fit-train.pt"
    torch.save(_pca_payload("train"), other)
    run_train(make_cfg(), pca_path=other, log=LOG_QUIET)
    run_evaluate(make_cfg(), pca_path=workspace, log=LOG_QUIET)
    run_evaluate(make_cfg(), pca_path=other, log=LOG_QUIET)
    comp = Path("outputs/metrics/auroc_fit-comparison.md")
    text = comp.read_text(encoding="utf-8")
    assert "fit-all" in text and "fit-train" in text and "Δ" in text


def test_wrong_pca_probe_pair_rejected(workspace, tmp_path):
    other = workspace.parent / "synth_pca64_fit-train.pt"
    torch.save(_pca_payload("train"), other)
    # plant a logistic artifact claiming it was trained on the fit-all PCA space
    shutil.copyfile(probe_artifact_path(workspace, "logistic"), probe_artifact_path(other, "logistic"))
    with pytest.raises(EvaluateError, match="trained on"):
        run_evaluate(make_cfg(), pca_path=other, probes=["logistic"], log=LOG_QUIET)


def test_missing_probe_artifact(workspace):
    probe_artifact_path(workspace, "hinge").unlink()
    with pytest.raises(EvaluateError, match="probe artifact not found"):
        run_evaluate(make_cfg(), pca_path=workspace, probes=["hinge"], log=LOG_QUIET)
