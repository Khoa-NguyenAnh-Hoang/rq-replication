"""Trainable probes (Phase 5): logistic / hinge, vendored from the upstream repo.

Vendored from ruyi101/rq-representation-probing `scripts/linear_model_training/`
(dataset.py, model.py, trainer.py, utils.py; entry-point logic from train.py).
No LICENSE file upstream — used for academic reproduction with citation
(docs/REPOSITORY_DESIGN.md §1). Vendored pieces are marked below and are
mathematically unchanged. Applied fixes (docs/MASTER_PLAN.md §4.1,
REPOSITORY_DESIGN.md §3):

1. split aliasing — split membership comes from the validated PCA artifact
   (data.py already normalised dev→validation), never a re-read of the raw CSV;
2. one load — the artifact is loaded once for all layers (upstream reloaded
   the .pt per layer);
3. seeds — torch/random/numpy are reseeded before every layer (upstream seeded
   once at module import: layer k's init depended on layers 0..k-1's RNG draws;
   per-layer seeding also makes `--layers` subsets match the full run exactly);
4. test discipline — the trainer never sees test rows (upstream computed
   test_loss/test_acc inside training); evaluation is `rq-probe evaluate`;
5. validation AUROC added to the per-layer output (DESIGN §3).

Everything runs on CPU (protocol §3: probes = CPU); the vendored
`use_cuda_if_available` parameter is kept but called with False.
"""

from __future__ import annotations

import platform
import random
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import PROBE_NAMES, Config
from ..extraction import artifact_path
from ..metrics import auroc
from ..pca_step import pca_artifact_paths
from .diffmean import diffmean_direction

TRAINABLE_LOSSES = ("logistic", "hinge")

# The existing Phase-4 artifacts embed torch.__version__ as a TorchVersion
# object (a str subclass, no code); allow exactly that global so artifacts can
# be loaded with weights_only=True. New artifacts store a plain str instead.
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])


class ProbeError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# vendored: linear_model_training/utils.py
# ---------------------------------------------------------------------------

def logistic_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary logistic loss (BCE with logits); targets in {0, 1}."""
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def hinge_loss(logits: torch.Tensor, targets: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Hinge loss (SVM-style); targets in {0, 1} are mapped to {-1, +1}."""
    unique = torch.unique(targets)
    vals = set(unique.tolist())
    if vals == {0, 1}:
        y = targets * 2 - 1  # 0 -> -1, 1 -> +1
    else:
        y = targets

    y = y.float()
    loss = torch.clamp(margin - y * logits, min=0.0)
    return loss.mean()


@torch.no_grad()
def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0) -> float:
    """Accuracy of sign(logits) against targets in {0, 1}."""
    preds = (logits > threshold).long()
    targets = targets.long()
    return (preds == targets).float().mean().item()


# ---------------------------------------------------------------------------
# vendored: linear_model_training/model.py
# ---------------------------------------------------------------------------

class LinearModel(torch.nn.Module):
    """f(x) = w^T x — linear, no bias (upstream quirk kept deliberately: AUROC
    is invariant to the affine shift, and the App. B map-back reduces to
    w_x = W^T w_z; docs/EXPERIMENT_PROTOCOL.md §5)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, 1, bias=False)
        # upstream: small Gaussian init instead of the torch default
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


# ---------------------------------------------------------------------------
# vendored: linear_model_training/dataset.py — EmbeddingDataset verbatim; the
# file-based loaders are replaced by make_dataloaders (fixes 1 + 2)
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """Simple dataset wrapping (X, y): X [N, D] float, y [N] long in {0, 1}."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert X.shape[0] == y.shape[0], (
            f"X and y must have same length, got {X.shape[0]} vs {y.shape[0]}"
        )
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_dataloaders(
    X: torch.Tensor,
    y: torch.Tensor,
    splits: list[str],
    batch_size: int = 256,
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    """Build train/validation loaders from one in-memory layer matrix.

    Fixes 1 + 2: splits come from the validated artifact (dev→validation
    already normalised by data.py) instead of a CSV re-read, and X is passed in
    already loaded instead of re-loading the .pt per layer. Test rows are not
    returned: they are only touched by `rq-probe evaluate` (fix 4).
    """
    train_idx = torch.tensor([i for i, s in enumerate(splits) if s == "train"], dtype=torch.long)
    val_idx = torch.tensor([i for i, s in enumerate(splits) if s == "validation"], dtype=torch.long)
    if train_idx.numel() == 0:
        raise ProbeError("train split is empty — cannot train")
    if val_idx.numel() == 0:
        raise ProbeError(
            "validation split is empty — cannot early-stop (dev→validation mapping broken?)"
        )

    return {
        "train": DataLoader(
            EmbeddingDataset(X[train_idx], y[train_idx]), batch_size=batch_size, shuffle=shuffle_train
        ),
        "validation": DataLoader(
            EmbeddingDataset(X[val_idx], y[val_idx]), batch_size=batch_size, shuffle=False
        ),
    }


# ---------------------------------------------------------------------------
# vendored: linear_model_training/trainer.py — test evaluation removed (fix 4),
# print → injected log, validation AUROC added (fix 5)
# ---------------------------------------------------------------------------

def _eval_loop(model, loader: DataLoader, loss_type: str, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    if len(loader.dataset) == 0:
        return float("nan"), float("nan")

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            if loss_type == "logistic":
                loss = logistic_loss(logits, y)
            elif loss_type == "hinge":
                loss = hinge_loss(logits, y)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            total_correct += int(compute_accuracy(logits, y) * batch_size)

    return total_loss / total_count, total_correct / total_count


def train_with_early_stopping(
    loaders: dict[str, DataLoader],
    loss_type: str = "logistic",
    lr: float = 0.1,
    weight_decay: float = 0.0,
    max_epochs: int = 1000,
    patience: int = 20,
    use_cuda_if_available: bool = True,
    early_stopping_metric: str = "dev_loss",
    log=print,
):
    """Train a LinearModel on train, monitor on validation, early-stop, restore
    the best state. Vendored; deviations: no test evaluation (fix 4), log
    injection, validation AUROC in the returned metrics (fix 5)."""
    device = torch.device("cuda" if (use_cuda_if_available and torch.cuda.is_available()) else "cpu")

    train_loader = loaders["train"]
    dev_loader = loaders["validation"]

    if len(train_loader.dataset) == 0:
        raise ValueError("Train split is empty — cannot train.")

    if len(dev_loader.dataset) == 0:
        raise ValueError("Dev split is empty — cannot early-stop without dev set.")

    log(f"Training samples: {len(train_loader.dataset)} | Dev samples: {len(dev_loader.dataset)}")

    # Infer input dim from one batch
    X_batch, _ = next(iter(train_loader))
    in_dim = X_batch.shape[1]

    model = LinearModel(in_dim=in_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "train_acc": [],
        "dev_loss": [],
        "dev_acc": [],
    }

    # Early-stopping state
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    if early_stopping_metric == "dev_loss":
        best_score = float("inf")
        mode = "min"
    elif early_stopping_metric == "dev_acc":
        best_score = -float("inf")
        mode = "max"
    else:
        raise ValueError(f"Unknown early_stopping_metric: {early_stopping_metric}")

    for epoch in range(1, max_epochs + 1):
        # ---------- training ----------
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)

            if loss_type == "logistic":
                loss = logistic_loss(logits, y)
            elif loss_type == "hinge":
                loss = hinge_loss(logits, y)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            total_correct += int(compute_accuracy(logits, y) * batch_size)

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count

        # ---------- dev evaluation ----------
        dev_loss, dev_acc = _eval_loop(model, dev_loader, loss_type, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["dev_loss"].append(dev_loss)
        history["dev_acc"].append(dev_acc)

        # ---------- early stopping check ----------
        score = dev_loss if early_stopping_metric == "dev_loss" else dev_acc
        improved = (mode == "min" and score < best_score) or (mode == "max" and score > best_score)

        if improved:
            best_score = score
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            log(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
            break

    # ---------- restore best model ----------
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    else:
        log("Warning: no improvement recorded; using last-epoch model.")

    # ---------- final dev evaluation (+ validation AUROC, fix 5) ----------
    dev_loss, dev_acc = _eval_loop(model, dev_loader, loss_type, device)

    X_val = torch.cat([X for X, _ in dev_loader])
    y_val = torch.cat([y for _, y in dev_loader])
    model.eval()
    with torch.no_grad():
        val_scores = model(X_val.to(device)).cpu()

    final_metrics = {
        "dev_loss": dev_loss,
        "dev_acc": dev_acc,
        "dev_auroc": auroc(y_val.numpy(), val_scores.numpy()),
        "best_epoch": best_epoch,
        "best_score": best_score,
    }
    return model, history, final_metrics


# ---------------------------------------------------------------------------
# our code — seeding (fix 3), per-layer entry points, artifact orchestration
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Upstream train.py's seed block, applied per layer (fix 3) instead of
    once at module import."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_layer(
    X: torch.Tensor,
    y: torch.Tensor,
    splits: list[str],
    *,
    loss_type: str,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    early_stopping_metric: str,
    seed: int,
    log=print,
) -> dict:
    """Train one probe on one layer's matrix [N, D]; reseeds first (fix 3)."""
    if loss_type not in TRAINABLE_LOSSES:
        raise ProbeError(f"unknown loss_type {loss_type!r} (expected one of {TRAINABLE_LOSSES})")
    seed_everything(seed)
    loaders = make_dataloaders(X, y, splits, batch_size=batch_size)
    model, history, final = train_with_early_stopping(
        loaders,
        loss_type=loss_type,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        use_cuda_if_available=False,  # protocol §3: probes are CPU — deterministic
        early_stopping_metric=early_stopping_metric,
        log=log,
    )
    return {"W": model.linear.weight.detach().cpu().view(-1), "b": 0.0, "history": history, **final}


def diffmean_layer(X: torch.Tensor, y: torch.Tensor, splits: list[str]) -> dict:
    """diffMean direction for one layer: unit(µ⁺ − µ⁻) on train rows only."""
    train_mask = torch.tensor([s == "train" for s in splits])
    val_mask = torch.tensor([s == "validation" for s in splits])
    w = diffmean_direction(X[train_mask], y[train_mask])
    final = {}
    if val_mask.any():
        final["dev_auroc"] = auroc(y[val_mask].numpy(), (X[val_mask] @ w).numpy())
    return {"W": w, "b": 0.0, "history": None, **final}


def probe_artifact_path(pca_artifact: str | Path, probe: str) -> Path:
    """outputs/probes/{pca stem}_{probe}.pt — the PCA stem already carries
    _pca{K}_fit-{split} (DESIGN §6); string concat, no Path.with_suffix."""
    return Path("outputs/probes") / f"{Path(pca_artifact).stem}_{probe}.pt"


def run_train(
    cfg: Config,
    *,
    pca_path: str | Path | None = None,
    fit_split: str | None = None,
    probes: list[str] | None = None,
    layers: list[int] | None = None,
    lr: float | None = None,
    log=print,
) -> list[Path]:
    """Train probes on all (or selected) layers of a PCA artifact.

    One output artifact per probe, named after the consumed PCA artifact (which
    encodes the fit variant); meta records the source artifact, its fit_split
    and hparams. Returns the saved artifact paths.
    """
    probes = list(probes) if probes is not None else list(cfg.probes)
    unknown = set(probes) - set(PROBE_NAMES)
    if unknown:
        raise ProbeError(f"unknown probes {sorted(unknown)} (expected {PROBE_NAMES})")

    if pca_path is None:
        fs = fit_split or cfg.pca.fit_split
        pca_path = pca_artifact_paths(artifact_path(cfg), cfg.pca.n_components, fs)[0]
    pca_path = Path(pca_path)
    if not pca_path.is_file():
        raise ProbeError(f"source PCA artifact not found: {pca_path} (run `rq-probe pca` first)")

    obj = torch.load(pca_path, map_location="cpu", weights_only=True)
    X = obj["embeddings"]
    if X.dim() != 3:
        raise ProbeError(f"expected [N, L, K] PCA embeddings, got shape {tuple(X.shape)}")
    X = X.float()
    N, L, K = X.shape
    if obj.get("ids") != list(range(N)):
        raise ProbeError("source artifact ids violate the row-alignment contract (ids must be 0..N-1)")
    labels = torch.tensor(obj["labels"], dtype=torch.long)
    splits = list(obj["splits"])
    if len(labels) != N or len(splits) != N:
        raise ProbeError(f"labels/splits length {len(labels)}/{len(splits)} != rows {N}")
    n_train = sum(1 for s in splits if s == "train")
    n_val = sum(1 for s in splits if s == "validation")
    if n_train == 0 or n_val == 0:
        raise ProbeError(
            f"train/validation splits must be non-empty for probing, got train={n_train}, validation={n_val}"
        )

    layer_idx = list(range(L)) if layers is None else list(layers)
    if not layer_idx or any(not 0 <= l < L for l in layer_idx):
        raise ProbeError(f"layers must be within 0..{L - 1}, got {layers}")

    hp = cfg.probe_hparams
    hparams = {
        "lr": lr if lr is not None else hp.lr,
        "weight_decay": hp.weight_decay,
        "batch_size": hp.batch_size,
        "max_epochs": hp.max_epochs,
        "patience": hp.patience,
        "early_stopping_metric": hp.early_stopping_metric,
    }

    src_meta = obj.get("meta", {})
    meta_common = {
        "source_artifact": str(pca_path),
        "pca_fit_split": src_meta.get("fit_split"),
        "pca_n_components": src_meta.get("n_components"),
        "source_meta": src_meta.get("source_meta"),
        "seed": cfg.experiment.seed,
        "layers": layer_idx,
        "device": "cpu",
        "torch": str(torch.__version__),  # str(): torch.__version__ is a TorchVersion object, which weights_only load rejects
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    saved: list[Path] = []
    for probe in probes:
        W = torch.full((L, K), float("nan"))
        b = torch.full((L,), float("nan"))
        dev_loss = torch.full((L,), float("nan"))
        dev_acc = torch.full((L,), float("nan"))
        dev_auroc = torch.full((L,), float("nan"))
        best_epoch = torch.full((L,), -1.0)
        history: dict[int, dict] = {}
        t0 = time.time()
        for l in layer_idx:
            X_l = X[:, l, :]
            if probe == "diffmean":
                res = diffmean_layer(X_l, labels, splits)
            else:
                res = train_layer(
                    X_l,
                    labels,
                    splits,
                    loss_type=probe,
                    seed=cfg.experiment.seed,
                    log=log,
                    **hparams,
                )
            W[l] = res["W"]
            b[l] = res["b"]
            dev_loss[l] = res.get("dev_loss", float("nan"))
            dev_acc[l] = res.get("dev_acc", float("nan"))
            dev_auroc[l] = res.get("dev_auroc", float("nan"))
            best_epoch[l] = res.get("best_epoch", -1.0)
            if res.get("history") is not None:
                history[l] = res["history"]
            if probe == "diffmean":
                log(f"{probe} layer {l}: dev_auroc={dev_auroc[l]:.4f}")
            else:
                log(
                    f"{probe} layer {l}: dev_auroc={dev_auroc[l]:.4f} "
                    f"dev_loss={dev_loss[l]:.4f} dev_acc={dev_acc[l]:.4f} "
                    f"best_epoch={int(best_epoch[l])}"
                )

        out_path = probe_artifact_path(pca_path, probe)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if probe == "diffmean":
            hparams_out: dict = {"kind": "diffmean"}
        else:
            hparams_out = dict(hparams, kind=probe)
        torch.save(
            {
                "W": W,
                "b": b,
                "probe": probe,
                "dev_loss": dev_loss,
                "dev_acc": dev_acc,
                "dev_auroc": dev_auroc,
                "best_epoch": best_epoch,
                "history": history,
                "meta": dict(meta_common, probe=probe, hparams=hparams_out),
            },
            out_path,
        )
        log(
            f"saved {probe} probes for {len(layer_idx)}/{L} layers to {out_path} "
            f"({time.time() - t0:.0f}s)"
        )
        saved.append(out_path)
    return saved
