"""Tests for rq_probe.pca_step — synthetic embedding artifacts, CPU only."""

from pathlib import Path

import pytest
import torch
from sklearn.decomposition import PCA

from rq_probe.config import Config, PCAConfig
from rq_probe.pca_step import PCAError, pca_artifact_paths, run_pca

N, L, D, K = 12, 3, 10, 4
SPLITS = ["train"] * 8 + ["validation"] * 2 + ["test"] * 2


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    # distinct per-layer structure so layer isolation is observable
    X = torch.randn(N, L, D)
    X[:, 0, :] += 5.0  # layer 0 offset
    X[:, 2, 0] *= 3.0  # layer 2 outlier-heavy first dim
    path = Path("outputs/embeddings/fake_src.pt")
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "embeddings": X,
            "ids": list(range(N)),
            "labels": [1, 0] * (N // 2),
            "splits": SPLITS,
            "meta": {"model_checkpoint": "fake/model", "pooling": "last_token"},
        },
        path,
    )
    return tmp_path, path


def make_cfg() -> Config:
    cfg = Config()
    cfg.pca = PCAConfig(n_components=K, fit_split="all")
    return cfg


def test_output_structure(workspace):
    _, src = workspace
    out = run_pca(make_cfg(), embedding_path=src, log=lambda *_: None)
    basis = pca_artifact_paths(src, K, "all")[1]
    assert out.name == f"fake_src_pca{K}_fit-all.pt"
    assert basis.name == f"fake_src_pca{K}_fit-all_basis.pt"
    art = torch.load(out, weights_only=False)
    bs = torch.load(basis, weights_only=False)
    assert art["embeddings"].shape == (N, L, K)
    assert art["embeddings"].dtype == torch.float32
    assert art["ids"] == list(range(N)) and len(art["labels"]) == N and len(art["splits"]) == N
    assert bs["components"].shape == (L, K, D)
    assert bs["means"].shape == (L, D)
    assert bs["explained_variance_ratio"].shape == (L, K)
    assert bs["meta"]["n_components"] == K and bs["meta"]["fit_split"] == "all"


def test_transform_matches_closed_form(workspace):
    # z = (x - mean) @ components.T  (paper App. B Eq. 5), per layer
    _, src = workspace
    out = run_pca(make_cfg(), embedding_path=src, log=lambda *_: None)
    basis = pca_artifact_paths(src, K, "all")[1]
    X = torch.load(src, weights_only=False)["embeddings"].float()
    bs = torch.load(basis, weights_only=False)
    art = torch.load(out, weights_only=False)
    for layer in range(L):
        z = (X[:, layer, :] - bs["means"][layer]) @ bs["components"][layer].T
        assert torch.allclose(art["embeddings"][:, layer, :], z, atol=1e-4), f"layer {layer}"


def test_layer_isolation(workspace):
    # layer l's output equals an independent PCA fitted on layer l's own matrix
    _, src = workspace
    obj = torch.load(src, weights_only=False)
    out = run_pca(make_cfg(), embedding_path=src, log=lambda *_: None)
    art = torch.load(out, weights_only=False)
    for layer in range(L):
        pca = PCA(n_components=K, svd_solver="full").fit(obj["embeddings"][:, layer, :].numpy())
        ref = pca.transform(obj["embeddings"][:, layer, :].numpy())
        assert torch.allclose(art["embeddings"][:, layer, :], torch.from_numpy(ref), atol=1e-4)


def test_fit_all_vs_train_differ(workspace):
    # leakage variant must be a genuinely different fit (different mean/basis)
    _, src = workspace
    out_all = run_pca(make_cfg(), embedding_path=src, fit_split="all", log=lambda *_: None)
    out_train = run_pca(make_cfg(), embedding_path=src, fit_split="train", log=lambda *_: None)
    basis_train = pca_artifact_paths(src, K, "train")[1]
    assert out_all.name.endswith("_fit-all.pt") and out_train.name.endswith("_fit-train.pt")
    bs_all = torch.load(pca_artifact_paths(src, K, "all")[1], weights_only=False)
    bs_train = torch.load(basis_train, weights_only=False)
    assert not torch.allclose(bs_all["means"], bs_train["means"], atol=1e-6)


def test_deterministic_reruns(workspace):
    _, src = workspace
    out1 = run_pca(make_cfg(), embedding_path=src, log=lambda *_: None)
    first = torch.load(out1, weights_only=False)["embeddings"].clone()
    out2 = run_pca(make_cfg(), embedding_path=src, log=lambda *_: None)
    second = torch.load(out2, weights_only=False)["embeddings"]
    assert torch.equal(first, second)  # exact full SVD -> bit-identical reruns


def test_train_fit_ignores_test_outliers(workspace):
    # an extreme test-only outlier must shift the train-fitted mean far less
    # than the all-fitted one
    _, src = workspace
    obj = torch.load(src, weights_only=False)
    obj["embeddings"][N - 1, :, :] += 1000.0  # corrupt one test row
    torch.save(obj, src)
    run_pca(make_cfg(), embedding_path=src, fit_split="all", log=lambda *_: None)
    run_pca(make_cfg(), embedding_path=src, fit_split="train", log=lambda *_: None)
    all_means = torch.load(pca_artifact_paths(src, K, "all")[1], weights_only=False)["means"]
    train_means = torch.load(pca_artifact_paths(src, K, "train")[1], weights_only=False)["means"]
    clean = obj["embeddings"][:-1, 0, :].mean(dim=0)  # reference excludes the corrupted row
    assert (all_means[0] - clean).norm() > (train_means[0] - clean).norm()


def test_source_validation(workspace):
    _, src = workspace
    obj = torch.load(src, weights_only=False)
    del obj["splits"]
    torch.save(obj, src)
    with pytest.raises(PCAError, match="splits"):
        run_pca(make_cfg(), embedding_path=src, log=lambda *_: None)

    obj["splits"] = SPLITS
    obj["ids"] = [i + 100 for i in range(N)]
    torch.save(obj, src)
    with pytest.raises(PCAError, match="row-alignment"):
        run_pca(make_cfg(), embedding_path=src, log=lambda *_: None)


def test_missing_source_artifact(workspace):
    with pytest.raises(PCAError, match="not found"):
        run_pca(make_cfg(), embedding_path="outputs/embeddings/nope.pt", log=lambda *_: None)
