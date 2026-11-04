"""Hidden-state extraction (Phase 3).

Batched, resumable extraction of per-layer hidden states from a decoder-only
model, mathematically identical to the upstream batch-1 scripts (proved by
tests/test_extraction.py against a batch-1 reference):

- last_token: hidden state of the final valid token. Upstream used h[0, -1, :]
  at batch=1; batched, we gather per-example index attention_mask.sum(1) - 1
  with padding forced to the right.
- mean_token: mask-weighted mean (upstream's unmasked mean is identical at
  batch=1 where no padding exists).

Artifacts: outputs/embeddings/{dataset}_{column}_{model}_{pooling}.pt holding
an fp32 tensor [N, L+1, D] (row i <-> id i) plus full provenance metadata.
Resume: a {artifact}.partial.pt checkpoint every `shard_every` examples;
resuming verifies a config fingerprint before continuing.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from .config import Config
from .data import load_dataset

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
PAD_SIDE = "right"


class ExtractionError(RuntimeError):
    pass


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def artifact_path(cfg: Config, max_examples: int | None = None) -> Path:
    model_short = cfg.model.checkpoint.split("/")[-1]
    name = f"{cfg.dataset.name}_{cfg.dataset.text_column}_{model_short}_{cfg.representation.pooling}"
    if max_examples is not None:
        name += f"_pilot{max_examples}"
    return Path("outputs/embeddings") / f"{name}.pt"


def pool_hidden_states(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    pooling: str,
    ids: list[int],
) -> torch.Tensor:
    """Pool per-layer hidden states to [B, L+1, D] fp32; raises on non-finite values."""
    if pooling == "last_token":
        if PAD_SIDE != "right":
            raise ExtractionError("last-token gather assumes right padding")
        last_idx = (attention_mask.sum(dim=1) - 1).long()  # [B]; cast guards float masks
        rows = torch.arange(attention_mask.size(0), device=attention_mask.device)
        pooled = torch.stack([h[rows, last_idx] for h in hidden_states], dim=1)
    elif pooling == "mean_token":
        mask = attention_mask.unsqueeze(-1).to(hidden_states[0].dtype)  # [B, T, 1]
        denom = mask.sum(dim=1)  # [B, 1]
        pooled = torch.stack([(h * mask).sum(dim=1) / denom for h in hidden_states], dim=1)
    else:
        raise ExtractionError(f"unknown pooling: {pooling!r}")
    pooled = pooled.float()
    if not torch.isfinite(pooled).all():
        bad = ids[0] if isinstance(ids, list) else None
        raise ExtractionError(f"non-finite hidden states in batch starting at id {bad}")
    return pooled


def _meta(cfg: Config, dataset_sha256: str, n_rows: int, truncated: int, shape: tuple[int, ...]) -> dict:
    return {
        "model_checkpoint": cfg.model.checkpoint,
        "model_revision": cfg.model.revision,
        "dtype_loaded": cfg.model.dtype,
        "pooling": cfg.representation.pooling,
        "max_length": cfg.representation.max_length,
        "batch_size": cfg.representation.batch_size,
        "dataset_path": str(cfg.dataset.file),
        "dataset_sha256": dataset_sha256,
        "text_column": cfg.dataset.text_column,
        "seed": cfg.experiment.seed,
        "n_rows": n_rows,
        "truncated_rows": truncated,
        "tensor_shape": list(shape),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_extraction(
    cfg: Config,
    *,
    tokenizer=None,
    model=None,
    max_examples: int | None = None,
    fresh: bool = False,
    stop_after: int | None = None,
    log=print,
) -> Path:
    """Extract representations for the configured dataset.

    tokenizer/model are injectable for tests; default loads from cfg.
    stop_after (test seam): stop after this many *new* examples, leaving the
    partial checkpoint in place to simulate an interruption.

    Returns the final artifact path.
    """
    if cfg.representation.pooling not in ("last_token", "mean_token"):
        raise ExtractionError(f"unknown pooling: {cfg.representation.pooling!r}")
    out_path = artifact_path(cfg, max_examples)
    partial_path = out_path.with_suffix(".partial.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(cfg.dataset.file)
    texts = dataset.texts if max_examples is None else dataset.texts[:max_examples]
    n_total = len(texts)
    fingerprint = {
        "model_checkpoint": cfg.model.checkpoint,
        "model_revision": cfg.model.revision,
        "pooling": cfg.representation.pooling,
        "max_length": cfg.representation.max_length,
        "dataset_sha256": dataset.sha256,
    }

    torch.manual_seed(cfg.experiment.seed)
    device = resolve_device(cfg.model.device)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint, revision=cfg.model.revision)
    tokenizer.padding_side = PAD_SIDE
    if model is None:
        model = AutoModel.from_pretrained(
            cfg.model.checkpoint, revision=cfg.model.revision, dtype=DTYPES[cfg.model.dtype]
        ).to(device)
    model.eval()

    # truncation statistics: token counts without truncation (cheap CPU pass)
    t0 = time.time()
    full_lengths = [len(tokenizer(t)["input_ids"]) for t in texts]
    truncated = sum(1 for n in full_lengths if n > cfg.representation.max_length)
    log(f"tokenized {n_total} texts in {time.time() - t0:.1f}s; {truncated} exceed max_length={cfg.representation.max_length}")

    # resume from partial checkpoint
    done = 0
    done_states = None
    if partial_path.exists():
        if fresh:
            partial_path.unlink()
            log("removed existing partial checkpoint (--fresh)")
        else:
            partial = torch.load(partial_path, map_location="cpu", weights_only=False)
            if partial.get("fingerprint") != fingerprint:
                raise ExtractionError(
                    f"partial checkpoint {partial_path} was written for a different configuration "
                    f"(stored {partial.get('fingerprint')}); delete it or pass --fresh"
                )
            done = int(partial["done"])
            done_states = partial["embeddings"]
            log(f"resuming from {done}/{n_total} completed examples")

    bs = cfg.representation.batch_size
    new_states: list[torch.Tensor] = []
    t0 = time.time()
    new_count = 0
    for start in range(done, n_total, bs):
        batch_ids = list(range(start, min(start + bs, n_total)))
        enc = tokenizer(
            [texts[i] for i in batch_ids],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.representation.max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            outputs = model(**enc, output_hidden_states=True, use_cache=False)
        pooled = pool_hidden_states(outputs.hidden_states, enc["attention_mask"], cfg.representation.pooling, batch_ids)
        new_states.append(pooled.cpu())
        new_count += len(batch_ids)

        if stop_after is not None and new_count >= stop_after:
            break

        if (start // bs) % max(1, cfg.representation.shard_every // bs) == 0 and (
            start + bs < n_total
        ):
            _save_partial(partial_path, fingerprint, batch_ids[-1] + 1, done_states, new_states)
        log(
            f"extracted {min(start + bs, n_total)}/{n_total} "
            f"({(time.time() - t0) / new_count:.2f}s/ex)"
        )

    total_done = done + new_count
    if total_done < n_total:
        _save_partial(partial_path, fingerprint, total_done, done_states, new_states)
        log(f"stopped early at {total_done}/{n_total}; partial checkpoint saved")
        return out_path

    embeddings = torch.cat([t for t in ([done_states] if done_states is not None else []) + new_states])
    assert embeddings.shape[0] == n_total
    if device.type == "cuda":
        # subprocess-safe VRAM report: the parent notebook process cannot see this
        log(f"peak VRAM during extraction: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    torch.save(
        {
            "embeddings": embeddings,
            "ids": list(range(n_total)),  # row i <-> id i by contract (data.py enforces it)
            "labels": dataset.labels if max_examples is None else dataset.labels[:max_examples],
            "splits": dataset.splits if max_examples is None else dataset.splits[:max_examples],
            "meta": _meta(cfg, dataset.sha256, n_total, truncated, tuple(embeddings.shape)),
        },
        out_path,
    )
    if partial_path.exists():
        partial_path.unlink()
    log(f"saved {tuple(embeddings.shape)} fp32 embeddings to {out_path}")
    return out_path


def _save_partial(path: Path, fingerprint: dict, done: int, done_states, new_states) -> None:
    parts = [t for t in ([done_states] if done_states is not None else []) + new_states]
    torch.save({"fingerprint": fingerprint, "done": done, "embeddings": torch.cat(parts)}, path)
