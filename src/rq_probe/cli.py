"""rq-probe command-line interface."""

from __future__ import annotations

import argparse
import sys

from .data import DataValidationError, load_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rq-probe", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="validate a dataset CSV and print statistics")
    p_val.add_argument("--file", default="data/raw/RQ.csv")

    args = parser.parse_args(argv)

    if args.command == "validate":
        try:
            ds = load_dataset(args.file)
        except DataValidationError as exc:
            print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
            return 1
        print(f"dataset : {ds.path}")
        print(f"sha256  : {ds.sha256}")
        print(f"rows    : {ds.n_rows}")
        print(f"{'split':<13}{'total':>7}{'rhetorical':>12}{'informational':>14}")
        for split, c in ds.counts().items():
            print(f"{split:<13}{c['total']:>7}{c['rhetorical']:>12}{c['informational']:>14}")
        print("all integrity checks passed")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
