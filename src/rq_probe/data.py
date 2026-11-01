"""Dataset loading and validation (Phase 1).

Contract: the CSV's `id` column must be exactly 0..N-1 in row order, so an
embedding tensor row i corresponds to `id == i` — the alignment invariant the
whole pipeline relies on. Splits named `dev`/`val` are normalised to
`validation`; upstream's own loader expected `validation` and silently got an
empty split on the shipped CSV (docs/MASTER_PLAN.md §4.1 #8).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TEXT_COLUMN = "question_with_context"
LABEL_COLUMN = "binary_label"
RAW_LABEL_COLUMN = "label"
SPLIT_COLUMN = "dataset"
SPLIT_ALIASES = {"dev": "validation", "val": "validation"}
KNOWN_SPLITS = ("train", "validation", "test")


class DataValidationError(ValueError):
    """Raised when a dataset CSV fails an integrity check."""


@dataclass
class RQDataset:
    path: Path
    sha256: str
    texts: list[str]
    labels: list[int]
    splits: list[str]

    @property
    def n_rows(self) -> int:
        return len(self.texts)

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for split in KNOWN_SPLITS:
            rhetorical = sum(1 for s, y in zip(self.splits, self.labels) if s == split and y == 1)
            total = sum(1 for s in self.splits if s == split)
            out[split] = {"total": total, "rhetorical": rhetorical, "informational": total - rhetorical}
        return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(csv_path: str | Path, *, text_column: str = TEXT_COLUMN) -> RQDataset:
    """Load and validate a dataset CSV. Raises DataValidationError on any failure."""
    path = Path(csv_path)
    if not path.is_file():
        raise DataValidationError(f"dataset file not found: {path}")

    df = pd.read_csv(path)
    required = {"id", text_column, LABEL_COLUMN, RAW_LABEL_COLUMN, SPLIT_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"{path}: missing columns {sorted(missing)}")
    if len(df) == 0:
        raise DataValidationError(f"{path}: no rows")

    if not df["id"].is_unique:
        raise DataValidationError(f"{path}: duplicate ids")
    if df["id"].tolist() != list(range(len(df))):
        raise DataValidationError(
            f"{path}: ids must be exactly 0..{len(df) - 1} in row order (embedding row alignment contract)"
        )

    bad_labels = set(df[LABEL_COLUMN].unique()) - {0, 1}
    if bad_labels:
        raise DataValidationError(f"{path}: {LABEL_COLUMN} values outside {{0, 1}}: {sorted(bad_labels)}")
    if (df[RAW_LABEL_COLUMN] != df[LABEL_COLUMN]).any():
        raise DataValidationError(f"{path}: '{RAW_LABEL_COLUMN}' and '{LABEL_COLUMN}' disagree on some rows")

    text = df[text_column]
    empty = text.isna() | (text.astype(str).str.strip() == "")
    if empty.any():
        rows = df.index[empty].tolist()[:5]
        raise DataValidationError(f"{path}: empty '{text_column}' in rows {rows}")

    splits: list[str] = []
    unknown: set[str] = set()
    for raw in df[SPLIT_COLUMN]:
        name = SPLIT_ALIASES.get(str(raw).strip(), str(raw).strip())
        if name not in KNOWN_SPLITS:
            unknown.add(str(raw))
        splits.append(name)
    if unknown:
        raise DataValidationError(
            f"{path}: unknown split names {sorted(unknown)} (expected train/dev/validation/test)"
        )

    return RQDataset(
        path=path,
        sha256=_sha256(path),
        texts=text.astype(str).tolist(),
        labels=df[LABEL_COLUMN].astype(int).tolist(),
        splits=splits,
    )
