"""Tests for rq_probe.probes — synthetic fixtures, CPU only."""

from pathlib import Path

import pytest
import torch

from rq_probe.config import Config
from rq_probe.probes.diffmean import DiffMeanError, diffmean_direction
from rq_probe.probes.trainable import (
    ProbeError,
    hinge_loss,
    logistic_loss,
    probe_artifact_path,
    run_train,
    train_layer,
)

N, L, K = 60, 3, 4
SPLITS = ["train"] * 40 + ["validation"] * 10 + ["test"] * 10
LOG_QUIET = lambda *_: None  # noqa: E731


def make_cfg(max_epochs: int = 100) -> Config:
    # separable fixture converges in tens of epochs; cap epochs to keep the suite fast
    cfg = Config()
    cfg.probe_hparams.max_epochs = max_epochs
    return cfg


def _make_pca_artifact(path: Path, seed: int = 0):
    # labels alternate 0/1 so every contiguous split holds both classes
    y = torch.tensor([i % 2 for i in range(N)])
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N, L, K, generator=g)
    X[y == 1, :, 0] += 3.0  # separable along dim 0 in every layer
    torch.save(
        {
            "embeddings": X,
            "ids": list(range(N)),
            "labels": y.tolist(),
            "splits": SPLITS,
            "meta": {
                "n_components": K,
                "fit_split": "all",
                "source_meta": {
                    "model_checkpoint": "fake/model",
                    "model_revision": None,
                    "pooling": "last_token",
                    "max_length": 512,
                    "dataset_sha256": "deadbeef",
                },
            },
        },
        path,
    )
    return X


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("outputs/pca/synth_pca64_fit-all.pt")
    path.parent.mkdir(parents=True)
    _make_pca_artifact(path)
    return path


# ---------- vendored losses ----------

def test_hinge_converts_0_1_to_pm1():
    logits = torch.tensor([2.0, -1.0, 0.5])
    y01 = torch.tensor([1, 0, 1])
    expected = torch.clamp(1.0 - torch.tensor([1.0, -1.0, 1.0]) * logits, min=0).mean()
    assert torch.allclose(hinge_loss(logits, y01), expected)


def test_logistic_loss_is_bce():
    logits = torch.tensor([0.3, -2.0, 1.1])
    y = torch.tensor([0, 1, 1])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float())
    assert torch.allclose(logistic_loss(logits, y), expected)


# ---------- diffMean ----------

def test_diffmean_direction_matches_manual():
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [10.0, 5.0], [10.0, 4.0]])
    y = torch.tensor([0, 0, 1, 1])
    w = diffmean_direction(X, y)
    mu_p, mu_m = X[y == 1].mean(0), X[y == 0].mean(0)
    assert torch.allclose(w, (mu_p - mu_m) / (mu_p - mu_m).norm(), atol=1e-6)


def test_diffmean_unit_norm_and_single_class_raises():
    X = torch.randn(20, 5)
    y = torch.tensor([i % 2 for i in range(20)])
    assert diffmean_direction(X, y).norm() == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(DiffMeanError, match="both classes"):
        diffmean_direction(X, torch.ones(20, dtype=torch.long))


def test_diffmean_ignores_non_train_rows(workspace):
    # leakage check PROTOCOL §7.2: extreme validation/test rows must not move w
    cfg = Config()
    run_train(cfg, pca_path=workspace, probes=["diffmean"], log=LOG_QUIET)
    clean = torch.load(probe_artifact_path(workspace, "diffmean"), weights_only=True)["W"].clone()

    X = torch.load(workspace, weights_only=True)["embeddings"]
    X[50:, :, :] += 1000.0  # blow up all validation + test rows
    torch.save({**torch.load(workspace, weights_only=True), "embeddings": X}, workspace)
    run_train(cfg, pca_path=workspace, probes=["diffmean"], log=LOG_QUIET)
    dirty = torch.load(probe_artifact_path(workspace, "diffmean"), weights_only=True)["W"]
    assert torch.equal(clean, dirty)


# ---------- trainable probes ----------

def test_logistic_converges_on_separable(workspace):
    out = run_train(make_cfg(), pca_path=workspace, probes=["logistic"], log=LOG_QUIET)[0]
    art = torch.load(out, weights_only=True)
    assert art["dev_auroc"].min() > 0.95  # fixture is separable in every layer
    assert (art["b"] == 0).all()  # bias-free upstream quirk
    assert art["W"].shape == (L, K) and art["W"].dtype == torch.float32


def test_training_deterministic_under_fixed_seed(workspace):
    X = torch.load(workspace, weights_only=True)["embeddings"]
    y = torch.tensor([i % 2 for i in range(N)])
    kwargs = dict(
        loss_type="logistic", lr=0.1, weight_decay=0.0, batch_size=256,
        max_epochs=100, patience=20, early_stopping_metric="dev_loss", log=LOG_QUIET,
    )
    r1 = train_layer(X[:, 1, :], y, SPLITS, seed=42, **kwargs)
    r2 = train_layer(X[:, 1, :], y, SPLITS, seed=42, **kwargs)
    assert torch.equal(r1["W"], r2["W"])
    assert r1["history"] == r2["history"]


def test_layer_subset_matches_full_run(workspace):
    # per-layer reseeding (fix 3): training layer 1 alone == training it after layer 0
    cfg = make_cfg()
    full = run_train(cfg, pca_path=workspace, probes=["logistic"], log=LOG_QUIET)[0]
    w_full = torch.load(full, weights_only=True)["W"].clone()
    sub = run_train(cfg, pca_path=workspace, probes=["logistic"], layers=[1], log=LOG_QUIET)[0]
    art = torch.load(sub, weights_only=True)
    assert art["meta"]["layers"] == [1]
    assert torch.equal(art["W"][1], w_full[1])
    assert torch.isnan(art["W"][0]).all() and torch.isnan(art["W"][2]).all()  # untrained = nan


def test_artifact_records_pca_fit_and_hparams(workspace):
    out = run_train(make_cfg(), pca_path=workspace, probes=["logistic"], lr=0.01, log=LOG_QUIET)[0]
    assert out.name == "synth_pca64_fit-all_logistic.pt"
    art = torch.load(out, weights_only=True)
    meta = art["meta"]
    assert meta["probe"] == "logistic"
    assert meta["pca_fit_split"] == "all" and meta["pca_n_components"] == K
    assert meta["source_artifact"] == str(workspace)
    assert meta["seed"] == 42 and meta["device"] == "cpu"
    assert meta["hparams"]["kind"] == "logistic" and meta["hparams"]["lr"] == 0.01
    assert meta["layers"] == [0, 1, 2]
    assert meta["source_meta"]["dataset_sha256"] == "deadbeef"


def test_diffmean_artifact_values(workspace):
    out = run_train(make_cfg(), pca_path=workspace, probes=["diffmean"], log=LOG_QUIET)[0]
    art = torch.load(out, weights_only=True)
    X = torch.load(workspace, weights_only=True)["embeddings"]
    y = torch.tensor([i % 2 for i in range(N)])
    n_train = SPLITS.index("validation")  # first 40 rows are train
    for l in range(L):
        mu_p, mu_m = X[:n_train][y[:n_train] == 1, l, :].mean(0), X[:n_train][y[:n_train] == 0, l, :].mean(0)
        assert torch.allclose(art["W"][l], (mu_p - mu_m) / (mu_p - mu_m).norm(), atol=1e-6)
    assert art["meta"]["hparams"] == {"kind": "diffmean"}


# ---------- input validation ----------

def test_empty_validation_rejected(workspace):
    obj = torch.load(workspace, weights_only=True)
    obj["splits"] = ["train"] * 40 + ["test"] * 20
    torch.save(obj, workspace)
    with pytest.raises(ProbeError, match="validation"):
        run_train(make_cfg(), pca_path=workspace, probes=["diffmean"], log=LOG_QUIET)


def test_unknown_probe_rejected(workspace):
    with pytest.raises(ProbeError, match="unknown probes"):
        run_train(make_cfg(), pca_path=workspace, probes=["svm"], log=LOG_QUIET)


def test_missing_pca_artifact(workspace):
    with pytest.raises(ProbeError, match="not found"):
        run_train(make_cfg(), pca_path="outputs/pca/nope.pt", log=LOG_QUIET)


def test_layer_out_of_range_rejected(workspace):
    with pytest.raises(ProbeError, match="within 0.."):
        run_train(make_cfg(), pca_path=workspace, probes=["logistic"], layers=[99], log=LOG_QUIET)
