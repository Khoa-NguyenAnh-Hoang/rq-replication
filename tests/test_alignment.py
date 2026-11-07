"""Tests for rq_probe.alignment — synthetic mini-pipeline fixtures, CPU only."""

from pathlib import Path

import pytest
import torch
from sklearn.decomposition import PCA

from rq_probe.config import Config, PCAConfig
from rq_probe.alignment import AlignmentError, cosine_directions, map_back, run_analyze
from rq_probe.evaluate import read_records
from rq_probe.pca_step import run_pca
from rq_probe.probes.trainable import run_train

N, L, D, K = 60, 3, 10, 4
SPLITS = ["train"] * 40 + ["validation"] * 10 + ["test"] * 10
LOG_QUIET = lambda *_: None  # noqa: E731


def make_cfg() -> Config:
    cfg = Config()
    cfg.pca = PCAConfig(n_components=K, fit_split="all")
    cfg.probe_hparams.max_epochs = 100
    return cfg


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """embeddings artifact -> PCA (basis) -> probes -> ready for analyze."""
    monkeypatch.chdir(tmp_path)
    y = torch.tensor([i % 2 for i in range(N)])
    g = torch.Generator().manual_seed(0)
    X = torch.randn(N, L, D, generator=g)
    X[y == 1, :, 0] += 3.0
    emb = Path("outputs/embeddings/synth_last_token.pt")
    emb.parent.mkdir(parents=True)
    torch.save(
        {
            "embeddings": X,
            "ids": list(range(N)),
            "labels": y.tolist(),
            "splits": SPLITS,
            "meta": {"model_checkpoint": "fake/model", "model_revision": None,
                     "pooling": "last_token", "max_length": 512, "dataset_sha256": "deadbeef"},
        },
        emb,
    )
    pca_out = run_pca(make_cfg(), embedding_path=emb, log=LOG_QUIET)
    run_train(make_cfg(), pca_path=pca_out, log=LOG_QUIET)
    return pca_out


def test_map_back_closed_form():
    # App. B: z·w_z == x·w_x + b_x exactly, up to float error
    g = torch.Generator().manual_seed(1)
    X = torch.randn(40, D, generator=g)
    pca = PCA(n_components=K, svd_solver="full").fit(X.numpy())
    C = torch.from_numpy(pca.components_.copy())
    mean = torch.from_numpy(pca.mean_.copy())
    w_z = torch.randn(K, generator=g)
    z = (X - mean) @ C.T
    w_x, b_x = map_back(w_z, C, mean)
    assert torch.allclose(z @ w_z, X @ w_x + b_x, atol=1e-4)


def test_cosine_known_vectors():
    assert cosine_directions(torch.tensor([2.0, 0.0]), torch.tensor([3.0, 0.0])) == pytest.approx(1.0)
    assert cosine_directions(torch.tensor([2.0, 0.0]), torch.tensor([-3.0, 0.0])) == pytest.approx(-1.0)
    assert cosine_directions(torch.tensor([2.0, 0.0]), torch.tensor([0.0, 5.0])) == pytest.approx(0.0, abs=1e-12)


def test_analyze_end_to_end(workspace):
    out = run_analyze(make_cfg(), pca_path=workspace, log=LOG_QUIET)
    records = read_records(out)
    pairs = {r["pair"] for r in records if "pair" in r}
    assert pairs == {"logistic+hinge", "logistic+diffmean", "hinge+diffmean"}
    assert sum(1 for r in records if "pair" in r) == 3 * L
    assert sum(1 for r in records if "probe" in r) == 3 * L  # map-back sanity records
    for r in records:
        if "pair" in r:
            assert -1.0 <= r["cosine"] <= 1.0
            assert r["spearman_ci_lo"] <= r["spearman"] <= r["spearman_ci_hi"]
            assert 0.0 <= r["jaccard_top"] <= 1.0 and 0.0 <= r["jaccard_bottom"] <= 1.0
        else:
            # App. B closed form: PCA-space and original-space AUROC identical
            assert abs(r["auroc_pca"] - r["auroc_original"]) < 1e-5
    table = Path("outputs/metrics") / f"alignment_table_{workspace.stem}.md"
    fig = Path("outputs/figures") / f"fig2_alignment_{workspace.stem}.png"
    assert table.is_file() and "cos logistic+hinge" in table.read_text(encoding="utf-8")
    assert fig.is_file() and fig.stat().st_size > 1000


def test_analyze_needs_two_probes(workspace):
    from rq_probe.probes.trainable import probe_artifact_path

    for p in ("hinge", "diffmean"):
        probe_artifact_path(workspace, p).unlink()
    with pytest.raises(AlignmentError, match="at least 2"):
        run_analyze(make_cfg(), pca_path=workspace, log=LOG_QUIET)


def test_analyze_invalid_pair(workspace):
    with pytest.raises(AlignmentError, match="invalid probe pair"):
        run_analyze(make_cfg(), pca_path=workspace, pairs=[("logistic", "logistic")], log=LOG_QUIET)
