"""Tests for rq_probe.extraction — CPU-only, using a tiny random-config Qwen3
model and a whitespace stub tokenizer (no downloads, no GPU).

The reference in every correctness test is a batch-1, unpadded forward pass —
the upstream `generate_embeddings.py` behavior that our batched pipeline must
reproduce exactly (docs/EXPERIMENT_PROTOCOL.md §3)."""

import csv
from pathlib import Path

import pytest
import torch
from transformers import AutoModel, Qwen3Config

from rq_probe.config import Config, RepresentationConfig
from rq_probe.extraction import (
    ExtractionError,
    artifact_path,
    pool_hidden_states,
    run_extraction,
)

COLUMNS = ["context", "question", "label", "dataset", "id", "question_with_context", "binary_label"]
TEXTS = [
    "short text here",
    "a much longer sentence with many more words than the first one had originally",
    "medium length sentence with some words",
    "tiny",
    "another example with a handful of words for the padded batch case",
    "last row completes the tiny fixture dataset of six rows",
]
LABELS = [1, 0, 1, 0, 1, 0]


class StubTokenizer:
    """Whitespace tokenizer: no special tokens (matches Qwen3), right padding,
    id 0 reserved for pad."""

    padding_side = "right"

    def __init__(self):
        self.vocab = {}

    def _encode(self, text: str) -> list[int]:
        out = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 1
            out.append(self.vocab[word])
        return out

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False, max_length=None):
        single = isinstance(texts, str)
        seqs = [self._encode(t) for t in ([texts] if single else texts)]
        if truncation and max_length is not None:
            seqs = [s[:max_length] for s in seqs]
        if return_tensors is None:
            # real HF tokenizers return plain lists without return_tensors,
            # flat for a single string input
            if single:
                return {"input_ids": seqs[0], "attention_mask": [1] * len(seqs[0])}
            return {"input_ids": seqs, "attention_mask": [[1] * len(s) for s in seqs]}
        T = max(len(s) for s in seqs)
        input_ids = torch.tensor([s + [0] * (T - len(s)) for s in seqs])
        attention_mask = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return AutoModel.from_config(cfg).eval()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("data/raw/RQ.csv")
    path.parent.mkdir(parents=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i, (text, label) in enumerate(zip(TEXTS, LABELS)):
            writer.writerow(
                {
                    "context": f"c{i}", "question": f"q{i}?", "label": str(label),
                    "dataset": "train" if i < 4 else ("dev" if i == 4 else "test"),
                    "id": str(i), "question_with_context": text, "binary_label": str(label),
                }
            )
    return tmp_path


def make_cfg(pooling="last_token", shard_every=2, batch_size=2, max_length=8) -> Config:
    cfg = Config()
    cfg.dataset.file = "data/raw/RQ.csv"
    cfg.model.device = "cpu"
    cfg.model.dtype = "float32"
    cfg.representation = RepresentationConfig(
        pooling=pooling, batch_size=batch_size, max_length=max_length, shard_every=shard_every
    )
    return cfg


def batch1_reference(model, tok, texts, pooling, max_length):
    """Upstream-equivalent reference: one forward pass per text, no batch effects."""
    out = []
    for t in texts:
        enc = tok([t], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        with torch.inference_mode():
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        out.append(pool_hidden_states(hs, enc["attention_mask"], pooling, [0])[0])
    return torch.stack(out)


def test_last_token_matches_batch1_reference(workspace, tiny_model):
    tok = StubTokenizer()
    cfg = make_cfg(pooling="last_token", batch_size=2, max_length=8)
    out = run_extraction(cfg, tokenizer=tok, model=tiny_model, log=lambda *_: None)
    saved = torch.load(out, weights_only=False)
    ref = batch1_reference(tiny_model, tok, TEXTS, "last_token", 8)
    assert saved["embeddings"].dtype == torch.float32
    assert saved["embeddings"].shape == (6, 4, 32)  # N, L+1=3+1, D
    assert torch.allclose(saved["embeddings"], ref, atol=1e-6)


def test_mean_token_matches_batch1_reference(workspace, tiny_model):
    tok = StubTokenizer()
    cfg = make_cfg(pooling="mean_token", batch_size=3, max_length=8)
    out = run_extraction(cfg, tokenizer=tok, model=tiny_model, log=lambda *_: None)
    saved = torch.load(out, weights_only=False)
    ref = batch1_reference(tiny_model, tok, TEXTS, "mean_token", 8)
    assert torch.allclose(saved["embeddings"], ref, atol=1e-6)


def test_padded_batch_does_not_leak_into_pools(workspace, tiny_model):
    # explicit check: a short text inside a padded batch pools identically to alone
    tok = StubTokenizer()
    enc_batch = tok([TEXTS[1], TEXTS[3]], return_tensors="pt", padding=True, truncation=True, max_length=8)
    enc_alone = tok([TEXTS[3]], return_tensors="pt", padding=True, truncation=True, max_length=8)
    with torch.inference_mode():
        hs_batch = tiny_model(**enc_batch, output_hidden_states=True, use_cache=False).hidden_states
        hs_alone = tiny_model(**enc_alone, output_hidden_states=True, use_cache=False).hidden_states
    pooled_batch = pool_hidden_states(hs_batch, enc_batch["attention_mask"], "last_token", [0, 1])[1]
    pooled_alone = pool_hidden_states(hs_alone, enc_alone["attention_mask"], "last_token", [0])[0]
    assert torch.allclose(pooled_batch, pooled_alone, atol=1e-6)


def test_nan_detection(workspace):
    # mean_token reads every position, so a nan at any position must surface
    hs = [torch.randn(2, 5, 8) for _ in range(3)]
    hs[1][:, 2, 0] = float("nan")
    mask = torch.ones(2, 5)
    with pytest.raises(ExtractionError, match="non-finite"):
        pool_hidden_states(tuple(hs), mask, "mean_token", [7, 8])


def test_resume_produces_identical_artifact(workspace, tiny_model):
    tok = StubTokenizer()
    cfg = make_cfg(pooling="last_token", batch_size=2, shard_every=2)

    # first run: interrupt after 3 examples -> partial checkpoint left behind
    run_extraction(cfg, tokenizer=tok, model=tiny_model, stop_after=3, log=lambda *_: None)
    partial = Path("outputs/embeddings/RQ_question_with_context_Qwen3-4B_last_token.partial.pt")
    final = partial.with_suffix("")
    assert partial.exists() and not final.exists()

    # second run: resumes from the partial and completes
    out = run_extraction(cfg, tokenizer=tok, model=tiny_model, log=lambda *_: None)
    ref = batch1_reference(tiny_model, tok, TEXTS, "last_token", 8)
    saved = torch.load(out, weights_only=False)
    assert saved["embeddings"].shape[0] == 6
    assert torch.allclose(saved["embeddings"], ref, atol=1e-6)
    assert not partial.exists()  # cleaned up after successful finish


def test_fingerprint_mismatch_rejected(workspace, tiny_model):
    tok = StubTokenizer()
    cfg = make_cfg()
    run_extraction(cfg, tokenizer=tok, model=tiny_model, stop_after=2, log=lambda *_: None)
    partial = Path("outputs/embeddings/RQ_question_with_context_Qwen3-4B_last_token.partial.pt")
    # same partial, but now the config changes max_length -> fingerprint differs
    cfg2 = make_cfg(max_length=4)
    with pytest.raises(ExtractionError, match="different configuration"):
        run_extraction(cfg2, tokenizer=tok, model=tiny_model, log=lambda *_: None)


def test_truncation_counted_in_metadata(workspace, tiny_model):
    tok = StubTokenizer()
    cfg = make_cfg(max_length=4)  # TEXTS with >4 words: rows 1, 2, 4, 5 -> 4 truncated
    out = run_extraction(cfg, tokenizer=tok, model=tiny_model, log=lambda *_: None)
    meta = torch.load(out, weights_only=False)["meta"]
    assert meta["truncated_rows"] == 4
    for key in (
        "model_checkpoint", "pooling", "max_length", "dataset_sha256", "seed",
        "n_rows", "tensor_shape", "torch", "transformers", "python", "timestamp_utc",
    ):
        assert key in meta, f"missing metadata key: {key}"


def test_pilot_artifact_has_distinct_name():
    from rq_probe.config import ModelConfig, DatasetConfig

    cfg = Config(dataset=DatasetConfig(name="RQ"), model=ModelConfig(checkpoint="Qwen/Qwen3-4B"))
    cfg.representation = RepresentationConfig(pooling="last_token")
    assert artifact_path(cfg).name == "RQ_question_with_context_Qwen3-4B_last_token.pt"
    assert artifact_path(cfg, max_examples=200).name == (
        "RQ_question_with_context_Qwen3-4B_last_token_pilot200.pt"
    )
