"""Command line interface for the anomaly detection toolkit.

Subcommands:

* ``simulate``  -- write labelled synthetic tabular data to CSV,
* ``detect``    -- run one or all detectors over a CSV and report flagged rows,
* ``evaluate``  -- compare every detector on labelled data and print metrics,
* ``report``    -- full pipeline (simulate + evaluate + markdown report).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import report as rep
from .classic import STATISTICAL_SCORERS
from .evaluate import best_f1_threshold, compare_detectors, evaluate_scores, threshold_sweep
from .generators import make_tabular
from .models import COPOD, HBOS, IsolationForest, LocalOutlierFactor, _flags_from_contamination

STATISTICAL_ALIASES = {"zscore": "z-score", "mad": "modified z-score", "iqr": "IQR"}
MODEL_NAMES = ["isolation_forest", "lof", "copod", "hbos"]
DETECTOR_NAMES = MODEL_NAMES + list(STATISTICAL_ALIASES)


def _detector_scores(name: str, X: np.ndarray, seed: Optional[int]) -> np.ndarray:
    if name == "isolation_forest":
        return IsolationForest(seed=seed).fit(X).score_samples(X)
    if name == "lof":
        return LocalOutlierFactor().fit(X).score_samples(X)
    if name == "copod":
        return COPOD().fit(X).score_samples(X)
    if name == "hbos":
        return HBOS().fit(X).score_samples(X)
    return STATISTICAL_SCORERS[STATISTICAL_ALIASES[name]](X)


def _write_csv(path: str, X: np.ndarray, y: np.ndarray, feature_prefix: str = "f") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join([f"{feature_prefix}{i}" for i in range(X.shape[1])] + ["y_true"])
    np.savetxt(out, np.column_stack([X, y]), delimiter=",", header=header, comments="", fmt="%.6f")


def _read_labeled(path: str):
    """Read a CSV written by ``simulate``; ``y_true`` is optional."""
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip()
    names = [n for n in header.split(",") if n]
    data = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    if names and names[-1] == "y_true":
        y = data[:, -1].astype(int)
        X = data[:, :-1]
        names = names[:-1]
    else:
        X = data
        y = None
    return X, y, names


def _cmd_simulate(args: argparse.Namespace) -> None:
    X, y = make_tabular(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_clusters=args.n_clusters,
        contamination=args.contamination,
        outlier_types=args.outliers,
        seed=args.seed,
    )
    _write_csv(args.out, X, y)
    print(f"wrote {args.out} ({X.shape[0]} rows, {X.shape[1]} features, {int(y.sum())} outliers)")


def _cmd_detect(args: argparse.Namespace) -> None:
    X, y, _ = _read_labeled(args.data)
    names = DETECTOR_NAMES if args.detector == "all" else [args.detector]
    for name in names:
        scores = _detector_scores(name, X, args.seed)
        flags = _flags_from_contamination(scores, args.contamination)
        flagged = np.flatnonzero(flags)
        line = f"{name}: flagged {len(flagged)} of {len(X)}"
        if y is not None:
            metrics = evaluate_scores(y, scores, contamination=args.contamination)
            line += (
                f" (precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f},"
                f" f1={metrics['f1']:.3f}, auc={metrics['auc']:.3f})"
            )
        print(line)
        if args.indices:
            print("  flagged indices:", " ".join(str(i) for i in flagged[: args.indices]))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    X, y, _ = _read_labeled(args.data)
    if y is None:
        raise SystemExit("evaluate needs a y_true column; generate data with `simulate` first")
    detectors = {name: (lambda X, name=name: _detector_scores(name, X, args.seed)) for name in DETECTOR_NAMES}
    rows = compare_detectors(X, y, detectors, contamination=args.contamination)
    print(rep.comparison_table(rows))
    if args.out:
        rep.write_report(args.out, "Detector comparison", [rep.comparison_table(rows), rep.caveats()])
        print(f"wrote {args.out}")


def _cmd_report(args: argparse.Namespace) -> None:
    X, y = make_tabular(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_clusters=args.n_clusters,
        contamination=args.contamination,
        outlier_types=args.outliers,
        seed=args.seed,
    )
    detectors = {name: (lambda X, name=name: _detector_scores(name, X, args.seed)) for name in DETECTOR_NAMES}
    rows = compare_detectors(X, y, detectors, contamination=args.contamination)
    best_name = str(rows[0]["detector"])
    scores = detectors[best_name](X)
    flags = _flags_from_contamination(scores, args.contamination)
    sweep = threshold_sweep(y, scores)
    best_t, best_f1 = best_f1_threshold(y, scores)

    sections = [
        "## Data",
        "",
        f"- {X.shape[0]} rows, {X.shape[1]} features, "
        f"{int(y.sum())} injected outliers ({args.contamination:.0%} contamination)",
        "",
        "## Detector comparison",
        "",
        rep.comparison_table(rows),
        "",
        "## Score summaries",
        "",
    ]
    for name in DETECTOR_NAMES:
        s = detectors[name](X)
        f = _flags_from_contamination(s, args.contamination)
        sections.append(rep.score_summary(name, s, int(f.sum())))
        sections.append("")
    sections += [
        f"## Flagged rows (best detector: {best_name})",
        "",
        rep.flagged_rows_table(X, scores, flags),
        "",
        f"## Threshold sweep ({best_name})",
        "",
        rep.sweep_table(sweep),
        "",
        f"Best F1 threshold: {best_t:.4f} (F1 = {best_f1:.4f})",
        "",
        rep.caveats(),
    ]
    rep.write_report(args.out, "Anomaly detection report", sections)
    print(rep.comparison_table(rows))
    print(f"wrote {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anomaly-detect",
        description="Unsupervised anomaly detection toolkit (numpy-only)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    simulate = sub.add_parser("simulate", help="write labelled synthetic tabular data")
    simulate.add_argument("--n-samples", type=int, default=500)
    simulate.add_argument("--n-features", type=int, default=4)
    simulate.add_argument("--n-clusters", type=int, default=3)
    simulate.add_argument("--contamination", type=float, default=0.05)
    simulate.add_argument("--outliers", nargs="*", choices=["shift", "scale"], default=["shift", "scale"])
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--out", default="data.csv")
    simulate.set_defaults(func=_cmd_simulate)

    detect = sub.add_parser("detect", help="flag anomalies in a CSV")
    detect.add_argument("--data", required=True)
    detect.add_argument("--detector", choices=DETECTOR_NAMES + ["all"], default="all")
    detect.add_argument("--contamination", type=float, default=0.1)
    detect.add_argument("--seed", type=int, default=0)
    detect.add_argument("--indices", type=int, metavar="N", default=None, help="print up to N flagged row indices")
    detect.set_defaults(func=_cmd_detect)

    evaluate = sub.add_parser("evaluate", help="compare all detectors on labelled data")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--contamination", type=float, default=0.1)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--out", default=None, help="optional markdown report path")
    evaluate.set_defaults(func=_cmd_evaluate)

    report = sub.add_parser("report", help="run the full pipeline and write a markdown report")
    report.add_argument("--n-samples", type=int, default=500)
    report.add_argument("--n-features", type=int, default=4)
    report.add_argument("--n-clusters", type=int, default=3)
    report.add_argument("--contamination", type=float, default=0.05)
    report.add_argument("--outliers", nargs="*", choices=["shift", "scale"], default=["shift", "scale"])
    report.add_argument("--seed", type=int, default=0)
    report.add_argument("--out", default="anomaly_report.md")
    report.set_defaults(func=_cmd_report)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


def entry() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entry()
