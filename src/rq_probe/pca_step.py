"""Per-layer PCA preprocessing (Phase 4), CPU-only.

Port of upstream `pca_transform.py` with the paper's protocol question made
explicit (docs/EXPERIMENT_PROTOCOL.md §4): the released code fits PCA on ALL
rows (train+dev+test) while paper App. B says the PCA mean is computed on the
training split. Both variants run and are reported side by side:

- fit_split="all"   -> published protocol (primary results)
- fit_split="train" -> leakage-controlled variant

Determinism deviation: sklearn's default solver auto-selects *randomized* SVD
at this scale and would be nondeterministic without a fixed seed; we force the
exact full SVD, which is deterministic and seed-free (numerically equivalent
up to SVD precision).

Outputs (row i <-> id i preserved):
- outputs/pca/{source}_pca{K}_fit-{split}.pt        embeddings [N, L, K] + ids/labels/splits + meta
- outputs/pca/{source}_pca{K}_fit-{split}_basis.pt  components [L, K, D], means [L, D], evr [L, K]
"""

from __future__ import annotations

import time
from pathlib import Path

import sklearn
import torch
from sklearn.decomposition import PCA

from .config import Config
from .extraction import artifact_path

SOURCE_META_KEYS = (
    "model_checkpoint",
    "model_revision",
    "pooling",
    "max_length",
    "dataset_sha256",
)


class PCAError(RuntimeError):
    pass


def pca_artifact_paths(embedding_path: str | Path, n_components: int, fit_split: str) -> tuple[Path, Path]:
    stem = Path(embedding_path).stem
    base = Path("outputs/pca") / f"{stem}_pca{n_components}_fit-{fit_split}"
    return base.with_suffix(".pt"), base.parent / f"{stem}_pca{n_components}_fit-{fit_split}_basis.pt"


def run_pca(
    cfg: Config,
    *,
    embedding_path: str | Path | None = None,
    fit_split: str | None = None,
    n_components: int | None = None,
    log=print,
) -> Path:
    """Fit per-layer PCA and transform the source embedding artifact.

    Returns the compressed-artifact path; the basis file path is derivable via
    pca_artifact_paths.
    """
    fit_split = fit_split or cfg.pca.fit_split
    if fit_split not in ("all", "train"):
        raise PCAError(f"fit_split must be 'all' or 'train', got {fit_split!r}")
    k = n_components or cfg.pca.n_components

    emb_path = Path(embedding_path) if embedding_path is not None else artifact_path(cfg)
    if not emb_path.is_file():
        raise PCAError(f"source embedding artifact not found: {emb_path} (run `rq-probe extract` first)")
    obj = torch.load(emb_path, map_location="cpu", weights_only=False)
    X = obj["embeddings"]
    if X.dim() != 3:
        raise PCAError(f"expected [N, L, D] embeddings, got shape {tuple(X.shape)}")
    X = X.float()
    N, L, D = X.shape
    ids, labels, splits = obj["ids"], obj["labels"], obj.get("splits")
    if ids != list(range(N)):
        raise PCAError("source artifact ids violate the row-alignment contract (ids must be 0..N-1)")
    if splits is None:
        raise PCAError("source artifact lacks 'splits' — re-extract with the current pipeline")
    if fit_split == "train":
        n_train = sum(1 for s in splits if s == "train")
        if n_train < k:
            raise PCAError(f"train split ({n_train} rows) smaller than n_components={k}")
        fit_rows = torch.tensor([i for i, s in enumerate(splits) if s == "train"])

    out_path, basis_path = pca_artifact_paths(emb_path, k, fit_split)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    emb_pca = torch.empty(N, L, k, dtype=torch.float32)
    components = torch.empty(L, k, D, dtype=torch.float32)
    means = torch.empty(L, D, dtype=torch.float32)
    evr = torch.empty(L, k, dtype=torch.float32)

    t0 = time.time()
    for layer in range(L):
        X_layer = X[:, layer, :].numpy()
        fit_matrix = X_layer if fit_split == "all" else X_layer[fit_rows.numpy()]
        pca = PCA(n_components=k, svd_solver="full")
        pca.fit(fit_matrix)
        emb_pca[:, layer, :] = torch.from_numpy(pca.transform(X_layer))
        components[layer] = torch.from_numpy(pca.components_)
        means[layer] = torch.from_numpy(pca.mean_)
        evr[layer] = torch.from_numpy(pca.explained_variance_ratio_)
        if (layer + 1) % 10 == 0 or layer == L - 1:
            log(f"pca: layer {layer + 1}/{L} done ({time.time() - t0:.0f}s)")

    source_meta = obj.get("meta", {})
    meta = {
        "source_artifact": str(emb_path),
        "source_meta": {key: source_meta.get(key) for key in SOURCE_META_KEYS},
        "n_components": k,
        "fit_split": fit_split,
        "svd_solver": "full",
        "seed": cfg.experiment.seed,
        "n_rows": N,
        "num_layers": L,
        "orig_dim": D,
        "explained_variance_last_component_max": float(evr[:, -1].max()),
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    torch.save(
        {"embeddings": emb_pca, "ids": ids, "labels": labels, "splits": splits, "meta": meta},
        out_path,
    )
    torch.save(
        {
            "components": components,
            "means": means,
            "explained_variance_ratio": evr,
            "meta": meta,
        },
        basis_path,
    )

    last_max = float(evr[:, -1].max())
    log(f"saved {tuple(emb_pca.shape)} PCA embeddings to {out_path}")
    log(f"saved PCA basis {L}x{k}x{D} to {basis_path}")
    note = "  WARNING: >= 1% contradicts the paper App. A expectation" if last_max >= 0.01 else ""
    log(f"last-component explained variance: max {last_max:.4%} across layers{note}")
    return out_path
