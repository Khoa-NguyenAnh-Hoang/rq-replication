"""Experiment configuration (Phase 3): YAML → typed dataclasses with audited defaults.

Defaults follow docs/REPOSITORY_DESIGN.md §5; every value traces to
docs/EXPERIMENT_PROTOCOL.md or the upstream code audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    name: str = "rq_replication"
    seed: int = 42


@dataclass
class DatasetConfig:
    name: str = "RQ"
    file: str = "data/raw/RQ.csv"
    text_column: str = "question_with_context"


@dataclass
class ModelConfig:
    checkpoint: str = "Qwen/Qwen3-4B"
    revision: str | None = None  # pin the hash verified by the Colab smoke test
    dtype: str = "float16"  # T4 has no bf16; upstream "auto" would load bf16
    device: str = "auto"


@dataclass
class RepresentationConfig:
    pooling: str = "last_token"  # last_token | mean_token
    batch_size: int = 16
    max_length: int = 512  # upstream had none; truncation stats are reported
    shard_every: int = 500  # checkpoint frequency for resume


@dataclass
class PCAConfig:
    n_components: int = 64  # paper §3.2 / Fig. 1
    fit_split: str = "all"  # all = published protocol; train = leakage-controlled variant


PROBE_NAMES = ("diffmean", "logistic", "hinge")


@dataclass
class ProbeHparamsConfig:
    """Upstream train.py defaults (REPRODUCTION_DECISIONS D4)."""

    lr: float = 0.1
    weight_decay: float = 0.0
    batch_size: int = 256
    max_epochs: int = 1000
    patience: int = 20
    early_stopping_metric: str = "dev_loss"


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    probes: list[str] = field(default_factory=lambda: list(PROBE_NAMES))
    probe_hparams: ProbeHparamsConfig = field(default_factory=ProbeHparamsConfig)


def _build(cls, data: dict | None):
    if data is None:
        return cls()
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = Config(
        experiment=_build(ExperimentConfig, raw.get("experiment")),
        dataset=_build(DatasetConfig, raw.get("dataset")),
        model=_build(ModelConfig, raw.get("model")),
        representation=_build(RepresentationConfig, raw.get("representation")),
        pca=_build(PCAConfig, raw.get("pca")),
        probe_hparams=_build(ProbeHparamsConfig, raw.get("probe_hparams")),
    )
    probes = raw.get("probes")
    if probes is not None:
        unknown = set(probes) - set(PROBE_NAMES)
        if unknown:
            raise ValueError(f"unknown probes in config: {sorted(unknown)}")
        cfg.probes = list(probes)
    return cfg
