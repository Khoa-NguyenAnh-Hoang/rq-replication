"""Tests for rq_probe.data — synthetic fixtures (fast, no data/raw dependency),
plus one integration test against the real RQ.csv when it is present."""

import csv
from pathlib import Path

import pytest

from rq_probe.data import DataValidationError, load_dataset

COLUMNS = ["context", "question", "label", "dataset", "id", "question_with_context", "binary_label"]

# 6 rows: train 3 (2 rhetorical), validation 2 (1 rhetorical), test 1 (0 rhetorical)
SPLIT_LABEL_SPEC = [("train", 1), ("train", 0), ("train", 1), ("dev", 0), ("validation", 1), ("test", 0)]


def make_rows(**overrides):
    rows = []
    for i, (split, label) in enumerate(SPLIT_LABEL_SPEC):
        row = {
            "context": f"context {i}",
            "question": f"question {i}?",
            "label": str(label),
            "dataset": split,
            "id": str(i),
            "question_with_context": f"context {i} question {i}?",
            "binary_label": str(label),
        }
        row.update(overrides.get(i, {}))
        rows.append(row)
    return rows


@pytest.fixture
def dataset_file(tmp_path):
    path = tmp_path / "tiny.csv"
    write_rows(path, make_rows())
    return path


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_load_ok(dataset_file):
    ds = load_dataset(dataset_file)
    assert ds.n_rows == 6
    assert ds.labels == [1, 0, 1, 0, 1, 0]
    assert ds.counts()["train"] == {"total": 3, "rhetorical": 2, "informational": 1}
    assert ds.counts()["test"] == {"total": 1, "rhetorical": 0, "informational": 1}
    assert len(ds.sha256) == 64


def test_dev_mapped_to_validation(dataset_file):
    ds = load_dataset(dataset_file)
    assert "dev" not in ds.splits
    assert ds.splits.count("validation") == 2


def test_missing_column(dataset_file):
    path = dataset_file.parent / "missing.csv"
    write_rows(path, [{k: v for k, v in row.items() if k != "binary_label"} for row in make_rows()])
    with pytest.raises(DataValidationError, match="missing columns"):
        load_dataset(path)


def test_duplicate_id(dataset_file):
    path = dataset_file.parent / "dup.csv"
    rows = make_rows()
    rows[1]["id"] = "0"
    write_rows(path, rows)
    with pytest.raises(DataValidationError, match="duplicate ids"):
        load_dataset(path)


def test_non_sequential_id(dataset_file):
    path = dataset_file.parent / "gap.csv"
    rows = make_rows()
    rows[3]["id"] = "9"  # breaks the 0..N-1 contract
    write_rows(path, rows)
    with pytest.raises(DataValidationError, match="0..5"):
        load_dataset(path)


def test_empty_text(dataset_file):
    path = dataset_file.parent / "empty.csv"
    rows = make_rows()
    rows[2]["question_with_context"] = "   "
    write_rows(path, rows)
    with pytest.raises(DataValidationError, match="empty 'question_with_context'"):
        load_dataset(path)


def test_bad_label(dataset_file):
    path = dataset_file.parent / "badlabel.csv"
    rows = make_rows()
    rows[0]["binary_label"] = "2"
    write_rows(path, rows)
    with pytest.raises(DataValidationError, match="outside"):
        load_dataset(path)


def test_label_mismatch(dataset_file):
    path = dataset_file.parent / "mismatch.csv"
    rows = make_rows()
    rows[1]["label"] = "1"  # binary_label stays 0
    write_rows(path, rows)
    with pytest.raises(DataValidationError, match="disagree"):
        load_dataset(path)


def test_unknown_split(dataset_file):
    path = dataset_file.parent / "split.csv"
    rows = make_rows()
    rows[5]["dataset"] = "holdout"
    write_rows(path, rows)
    with pytest.raises(DataValidationError, match="unknown split"):
        load_dataset(path)


def test_missing_file(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        load_dataset(tmp_path / "nope.csv")


RQ_CSV = Path(__file__).parents[1] / "data" / "raw" / "RQ.csv"


@pytest.mark.skipif(not RQ_CSV.is_file(), reason="data/raw/RQ.csv not downloaded")
def test_real_rq_dataset():
    ds = load_dataset(RQ_CSV)
    # Paper §3.1: 4,997 instances, 2,332 rhetorical, 3,200/797/1,000 split
    assert ds.n_rows == 4997
    assert ds.counts()["train"] == {"total": 3200, "rhetorical": 1508, "informational": 1692}
    assert ds.counts()["validation"] == {"total": 797, "rhetorical": 367, "informational": 430}
    assert ds.counts()["test"] == {"total": 1000, "rhetorical": 457, "informational": 543}
    assert sum(c["rhetorical"] for c in ds.counts().values()) == 2332
