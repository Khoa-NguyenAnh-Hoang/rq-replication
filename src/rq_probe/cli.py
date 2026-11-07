"""rq-probe command-line interface."""

from __future__ import annotations

import argparse
import sys

from .data import DataValidationError, load_dataset


def _layers_arg(text: str) -> list[int]:
    try:
        return [int(t) for t in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rq-probe", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="validate a dataset CSV and print statistics")
    p_val.add_argument("--file", default="data/raw/RQ.csv")

    p_ext = sub.add_parser("extract", help="extract per-layer hidden-state representations (GPU)")
    p_ext.add_argument("-c", "--config", default="configs/experiment.yaml")
    p_ext.add_argument("--max-examples", type=int, default=None, help="pilot subset (first N ids)")
    p_ext.add_argument("--batch-size", type=int, default=None)
    p_ext.add_argument("--pooling", choices=["last_token", "mean_token"], default=None)
    p_ext.add_argument("--device", default=None)
    p_ext.add_argument("--dtype", default=None)
    p_ext.add_argument("--fresh", action="store_true", help="ignore and delete any partial checkpoint")

    p_pca = sub.add_parser("pca", help="per-layer PCA compression of an embedding artifact (CPU)")
    p_pca.add_argument("-c", "--config", default="configs/experiment.yaml")
    p_pca.add_argument("--embedding", default=None, help="source .pt (default: the configured extraction artifact)")
    p_pca.add_argument("--fit-split", choices=["all", "train"], default=None)
    p_pca.add_argument("--n-components", type=int, default=None)

    p_train = sub.add_parser("train", help="train per-layer probes on a PCA artifact (CPU)")
    p_train.add_argument("-c", "--config", default="configs/experiment.yaml")
    p_train.add_argument(
        "--probe",
        action="append",
        choices=["diffmean", "logistic", "hinge"],
        default=None,
        help="repeatable; default: all probes in the config",
    )
    p_train.add_argument("--pca", default=None, help="source PCA artifact (default: the configured fit)")
    p_train.add_argument("--fit-split", choices=["all", "train"], default=None, help="select PCA fit variant")
    p_train.add_argument("--layers", type=_layers_arg, default=None, metavar="N[,N...]", help="layer indices (default: all)")
    p_train.add_argument("--lr", type=float, default=None, help="override probe_hparams.lr")

    p_ev = sub.add_parser("evaluate", help="layer-wise test AUROC + Hanley–McNeil CIs (CPU)")
    p_ev.add_argument("-c", "--config", default="configs/experiment.yaml")
    p_ev.add_argument("--pca", default=None, help="source PCA artifact (default: the configured fit)")
    p_ev.add_argument("--fit-split", choices=["all", "train"], default=None, help="select PCA fit variant")
    p_ev.add_argument("--probe", action="append", choices=["diffmean", "logistic", "hinge"], default=None)
    p_ev.add_argument("--negative-control", action="store_true", help="retrain logistic on shuffled train labels (expect ≈0.5)")

    p_an = sub.add_parser("analyze", help="probe alignment metrics + PCA map-back sanity (CPU)")
    p_an.add_argument("-c", "--config", default="configs/experiment.yaml")
    p_an.add_argument("--pca", default=None, help="source PCA artifact (default: the configured fit)")
    p_an.add_argument("--fit-split", choices=["all", "train"], default=None, help="select PCA fit variant")
    p_an.add_argument("--pair", action="append", default=None, metavar="A,B", help="probe pair (repeatable; default: all pairs)")

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

    if args.command == "extract":
        from .config import load_config
        from .extraction import run_extraction

        cfg = load_config(args.config)
        if args.batch_size:
            cfg.representation.batch_size = args.batch_size
        if args.pooling:
            cfg.representation.pooling = args.pooling
        if args.device:
            cfg.model.device = args.device
        if args.dtype:
            cfg.model.dtype = args.dtype
        run_extraction(cfg, max_examples=args.max_examples, fresh=args.fresh)
        return 0

    if args.command == "pca":
        from .config import load_config
        from .pca_step import run_pca

        run_pca(
            load_config(args.config),
            embedding_path=args.embedding,
            fit_split=args.fit_split,
            n_components=args.n_components,
        )
        return 0

    if args.command == "train":
        from .config import load_config
        from .probes.trainable import run_train

        run_train(
            load_config(args.config),
            pca_path=args.pca,
            fit_split=args.fit_split,
            probes=args.probe,
            layers=args.layers,
            lr=args.lr,
        )
        return 0

    if args.command == "evaluate":
        from .config import load_config
        from .evaluate import run_evaluate

        run_evaluate(
            load_config(args.config),
            pca_path=args.pca,
            fit_split=args.fit_split,
            probes=args.probe,
            negative_control=args.negative_control,
        )
        return 0

    if args.command == "analyze":
        from .config import load_config
        from .alignment import run_analyze

        pairs = [tuple(t.strip() for t in p.split(",")) for p in args.pair] if args.pair else None
        run_analyze(load_config(args.config), pca_path=args.pca, fit_split=args.fit_split, pairs=pairs)
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
