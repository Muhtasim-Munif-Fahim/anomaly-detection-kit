"""Command line interface for the anomaly detection toolkit.

Subcommands:

* ``simulate``  -- write labelled synthetic tabular data to CSV,
* ``detect``    -- run one or all detectors over a CSV and report flagged rows,
* ``evaluate``  -- compare every detector on labelled data and print metrics,
* ``discords``  -- top-k anomalous subsequences via the matrix profile,
* ``changepoints`` -- level/variance change points via BOCPD and CUSUM,
* ``ensemble``  -- combine several detectors (average/max/median/AOM/MOA/LSCP),
* ``report``    -- full pipeline (simulate + evaluate + markdown report).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import report as rep
from .changepoint import bocpd, changepoint_hits, cusum
from .combination import COMBINATION_METHODS, combine_scores, lscp_scores
from .classic import STATISTICAL_SCORERS, matrix_profile_discords
from .evaluate import best_f1_threshold, compare_detectors, evaluate_scores, threshold_sweep
from .generators import make_tabular
from .models import (
    COPOD,
    ECOD,
    HBOS,
    IsolationForest,
    KNN,
    LocalOutlierFactor,
    OneClassSVM,
    EllipticEnvelope,
    CBLOF,
    LODA,
    ABOD,
    COF,
    SOD,
    PCA,
    KDE,
    SOS,
    GMM,
    INNE,
    ExtendedIsolationForest,
    HalfSpaceTrees,
    _flags_from_contamination,
)

STATISTICAL_ALIASES = {"zscore": "z-score", "mad": "modified z-score", "iqr": "IQR"}
MODEL_NAMES = ["isolation_forest", "lof", "copod", "ecod", "hbos", "knn", "ocsvm", "elliptic", "cblof", "loda", "abod", "cof", "sod", "pca", "kde", "sos", "gmm", "inne", "eif", "hst"]
DETECTOR_NAMES = MODEL_NAMES + list(STATISTICAL_ALIASES)


def _detector_scores(name: str, X: np.ndarray, seed: Optional[int]) -> np.ndarray:
    if name == "isolation_forest":
        return IsolationForest(seed=seed).fit(X).score_samples(X)
    if name == "lof":
        return LocalOutlierFactor().fit(X).score_samples(X)
    if name == "copod":
        return COPOD().fit(X).score_samples(X)
    if name == "ecod":
        return ECOD().fit(X).score_samples(X)
    if name == "hbos":
        return HBOS().fit(X).score_samples(X)
    if name == "knn":
        return KNN().fit(X).score_samples(X)
    if name == "ocsvm":
        return OneClassSVM().fit(X).score_samples(X)
    if name == "elliptic":
        return EllipticEnvelope(seed=seed).fit(X).score_samples(X)
    if name == "cblof":
        return CBLOF(seed=seed).fit(X).score_samples(X)
    if name == "loda":
        return LODA(seed=seed).fit(X).score_samples(X)
    if name == "abod":
        return ABOD().fit(X).score_samples(X)
    if name == "cof":
        return COF().fit(X).score_samples(X)
    if name == "sod":
        return SOD().fit(X).score_samples(X)
    if name == "pca":
        return PCA().fit(X).score_samples(X)
    if name == "kde":
        return KDE().fit(X).score_samples(X)
    if name == "sos":
        return SOS(perplexity=min(30.0, max(1.0, X.shape[0] - 1))).fit(X).score_samples(X)
    if name == "gmm":
        return GMM(n_components=3, seed=0 if seed is None else seed).fit(X).score_samples(X)
    if name == "inne":
        return INNE(seed=0 if seed is None else seed).fit(X).score_samples(X)
    if name == "eif":
        return ExtendedIsolationForest(seed=0 if seed is None else seed).fit(X).score_samples(X)
    if name == "hst":
        return HalfSpaceTrees(seed=0 if seed is None else seed).fit(X).score_samples(X)
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


def _synthetic_series(n: int, period: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = np.sin(2 * np.pi * t / period) + 0.1 * rng.normal(size=n)
    start = n // 2
    width = max(3, period // 2)
    values[start:start + width] += np.linspace(0.0, 2.0, width)  # distorted cycle
    return values


def _series_from_csv(path: str, column: Optional[str]):
    X, _, names = _read_labeled(path)
    if column is None:
        col = X.shape[1] - 1
    elif column in names:
        col = names.index(column)
    else:
        try:
            col = int(column)
        except ValueError:
            raise SystemExit(f"unknown column {column!r}; available: {', '.join(names)}")
        if not 0 <= col < X.shape[1]:
            raise SystemExit(f"column index {col} out of range")
    return X[:, col], f"{path} [{names[col] if col < len(names) else col}]"


def _cmd_discords(args: argparse.Namespace) -> None:
    if args.data:
        values, source = _series_from_csv(args.data, args.column)
    else:
        values = _synthetic_series(args.n_points, args.window, args.seed)
        source = f"synthetic sine (n={args.n_points}, distorted cycle at {args.n_points // 2})"
    try:
        found = matrix_profile_discords(values, args.window, k=args.k)
    except ValueError as exc:
        raise SystemExit(f"discords: {exc}")
    print(f"matrix-profile discords: {source}, window={args.window}")
    for rank, (start, dist) in enumerate(found, 1):
        print(f"  #{rank}: start={start} end={start + args.window - 1} distance={dist:.4f}")


def _synthetic_regimes(n: int, seed: int):
    """Piecewise-constant mean with three level shifts at known positions."""
    rng = np.random.default_rng(seed)
    cuts = [n // 4, n // 2, (3 * n) // 4]
    levels = [0.0, 2.5, -0.5, 1.5]
    values = rng.normal(size=n)
    for start, end, level in zip([0] + cuts, cuts + [n], levels):
        values[start:end] += level
    return values, cuts


def _cmd_changepoints(args: argparse.Namespace) -> None:
    truth = None
    if args.data:
        values, source = _series_from_csv(args.data, args.column)
    else:
        values, truth = _synthetic_regimes(args.n_points, args.seed)
        source = f"synthetic regimes (n={args.n_points}, true change points {truth})"
    print(f"change points: {source}")
    try:
        if args.method in ("bocpd", "both"):
            res = bocpd(
                values,
                hazard=args.hazard,
                lag=args.lag,
                threshold=args.threshold,
                max_run_length=args.max_run_length,
            )
            print(f"  bocpd (hazard={args.hazard}, lag={args.lag}): {res.changepoints}")
            print(f"    log evidence = {res.log_evidence:.3f}")
            for cp in res.changepoints:
                print(f"    start={cp} probability={res.changepoint_score[cp]:.3f}")
            if truth is not None:
                p, r, f = changepoint_hits(res.changepoints, truth, tolerance=args.tolerance)
                print(f"    precision={p:.3f} recall={r:.3f} f1={f:.3f} (tolerance {args.tolerance})")
        if args.method in ("cusum", "both"):
            res_c = cusum(values, k=args.k, h=args.h)
            print(f"  cusum (k={args.k}, h={args.h}): {res_c.changepoints}")
            for alarm in res_c.alarms:
                print(f"    alarm={alarm.alarm_index} start={alarm.start_index} direction={alarm.direction}")
            if truth is not None:
                p, r, f = changepoint_hits(res_c.changepoints, truth, tolerance=args.tolerance)
                print(f"    precision={p:.3f} recall={r:.3f} f1={f:.3f} (tolerance {args.tolerance})")
    except ValueError as exc:
        raise SystemExit(f"changepoints: {exc}")


def _synthetic_spikes(n: int, seed: int, drift: float):
    """Gaussian noise with optional linear drift and four injected spikes."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=n) + np.linspace(0.0, drift, n)
    idx = sorted(int(i) for i in rng.choice(np.arange(n // 2, n), size=4, replace=False))
    for k, i in enumerate(idx):
        values[i] += 7.0 if k % 2 == 0 else -7.0
    return values, idx


def _cmd_evt(args: argparse.Namespace) -> None:
    from .evt import DSPOT, SPOT

    truth = None
    if args.data:
        values, source = _series_from_csv(args.data, args.column)
    else:
        values, truth = _synthetic_spikes(args.n_points, args.seed, args.drift)
        source = f"synthetic spikes (n={args.n_points}, drift={args.drift}, true anomalies {truth})"
    n_init = int(args.init_fraction * values.size)
    print(f"evt thresholds: {source}")
    try:
        if args.method == "dspot":
            model = DSPOT(depth=args.depth, q=args.q, level=args.level, side=args.side)
        else:
            model = SPOT(q=args.q, level=args.level, side=args.side)
        model.fit(values[:n_init])
        res = model.run(values[n_init:])
    except ValueError as exc:
        raise SystemExit(f"evt: {exc}")
    alarms = [a + n_init for a in res.alarms]
    print(f"  {args.method} (q={args.q}, level={args.level}, side={args.side}, calibration={n_init} points)")
    if res.upper_thresholds is not None:
        print(f"    final upper threshold = {res.upper_thresholds[-1]:.4f}")
    if res.lower_thresholds is not None:
        print(f"    final lower threshold = {res.lower_thresholds[-1]:.4f}")
    shown = alarms if len(alarms) <= 20 else alarms[:20] + ["..."]
    print(f"    alarms ({len(alarms)}): {shown}")
    if truth is not None:
        hit = len(set(alarms) & set(truth))
        print(f"    detected {hit}/{len(truth)} injected anomalies, {len(alarms) - hit} false alarms")


def _cmd_ensemble(args: argparse.Namespace) -> None:
    X, y, _ = _read_labeled(args.data)
    names = [n.strip() for n in args.detectors.split(",") if n.strip()]
    unknown = [n for n in names if n not in DETECTOR_NAMES]
    if unknown:
        raise SystemExit(f"ensemble: unknown detector(s) {', '.join(unknown)}; choose from {', '.join(DETECTOR_NAMES)}")
    if len(names) < 2:
        raise SystemExit("ensemble: need at least two detectors")
    M = np.column_stack([_detector_scores(n, X, args.seed) for n in names])
    try:
        if args.method == "lscp":
            combined = lscp_scores(
                X, M, n_neighbors=min(args.neighbors, X.shape[0] - 1),
                pseudo_target=args.pseudo_target, standardize=args.standardize,
            ).scores
        else:
            combined = combine_scores(
                M, args.method, standardize=args.standardize, n_buckets=args.buckets,
                bucket_method=args.bucket_method, seed=args.seed,
            )
    except ValueError as exc:
        raise SystemExit(f"ensemble: {exc}")
    flags = _flags_from_contamination(combined, args.contamination)
    line = f"ensemble[{args.method}] of {','.join(names)}: flagged {int(flags.sum())} of {len(X)}"
    if y is not None:
        m = evaluate_scores(y, combined, contamination=args.contamination)
        line += f" (precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f}, auc={m['auc']:.3f})"
    print(line)
    if y is not None:
        for j, n in enumerate(names):
            m = evaluate_scores(y, M[:, j], contamination=args.contamination)
            print(f"  {n}: f1={m['f1']:.3f}, auc={m['auc']:.3f}")


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

    discords = sub.add_parser(
        "discords", help="top-k anomalous subsequences of a series via the matrix profile"
    )
    discords.add_argument("--data", default=None, help="CSV with a header; default: synthetic series")
    discords.add_argument("--column", default=None, help="column name or index (default: last feature)")
    discords.add_argument("--window", type=int, default=40, help="subsequence length (default: 40)")
    discords.add_argument("--k", type=int, default=3, help="number of discords (default: 3)")
    discords.add_argument("--n-points", type=int, default=600, help="synthetic series length")
    discords.add_argument("--seed", type=int, default=0)
    discords.set_defaults(func=_cmd_discords)

    changepoints = sub.add_parser(
        "changepoints", help="level/variance change points via BOCPD and/or two-sided CUSUM"
    )
    changepoints.add_argument("--data", default=None, help="CSV with a header; default: synthetic regimes")
    changepoints.add_argument("--column", default=None, help="column name or index (default: last feature)")
    changepoints.add_argument("--method", choices=["bocpd", "cusum", "both"], default="both")
    changepoints.add_argument("--hazard", type=float, default=1.0 / 250.0, help="BOCPD change rate per step")
    changepoints.add_argument("--lag", type=int, default=10, help="BOCPD confirmation lag (points)")
    changepoints.add_argument("--threshold", type=float, default=0.5, help="BOCPD posterior threshold")
    changepoints.add_argument("--max-run-length", type=int, default=None, help="BOCPD run-length truncation")
    changepoints.add_argument("--k", type=float, default=0.5, help="CUSUM reference value (sigmas)")
    changepoints.add_argument("--h", type=float, default=8.0, help="CUSUM decision interval (sigmas)")
    changepoints.add_argument("--tolerance", type=int, default=5, help="match tolerance vs. synthetic truth")
    changepoints.add_argument("--n-points", type=int, default=800, help="synthetic series length")
    changepoints.add_argument("--seed", type=int, default=0)
    changepoints.set_defaults(func=_cmd_changepoints)

    evt = sub.add_parser("evt", help="streaming extreme-value thresholds (SPOT / DSPOT)")
    evt.add_argument("--data", default=None, help="CSV with a header; default: synthetic spikes")
    evt.add_argument("--column", default=None, help="column name or index (default: last feature)")
    evt.add_argument("--method", choices=["spot", "dspot"], default="spot")
    evt.add_argument("--q", type=float, default=1e-4, help="target risk (false-alarm probability)")
    evt.add_argument("--level", type=float, default=0.98, help="initial-threshold quantile")
    evt.add_argument("--side", choices=["upper", "lower", "both"], default="both")
    evt.add_argument("--depth", type=int, default=20, help="DSPOT moving-average window")
    evt.add_argument("--init-fraction", type=float, default=0.25, help="calibration share of the series")
    evt.add_argument("--drift", type=float, default=0.0, help="synthetic linear drift over the series")
    evt.add_argument("--n-points", type=int, default=4000, help="synthetic series length")
    evt.add_argument("--seed", type=int, default=0)
    evt.set_defaults(func=_cmd_evt)

    ensemble = sub.add_parser("ensemble", help="combine several detectors' scores into one ensemble score")
    ensemble.add_argument("--data", required=True)
    ensemble.add_argument("--detectors", default="knn,lof,hbos,ecod,isolation_forest",
                          help="comma-separated detector names")
    ensemble.add_argument("--method", choices=list(COMBINATION_METHODS) + ["lscp"], default="average")
    ensemble.add_argument("--standardize", choices=["zscore", "rank", "minmax", "none"], default="zscore")
    ensemble.add_argument("--buckets", type=int, default=5, help="AOM/MOA bucket count")
    ensemble.add_argument("--bucket-method", choices=["static", "dynamic"], default="static")
    ensemble.add_argument("--neighbors", type=int, default=30, help="LSCP local region size")
    ensemble.add_argument("--pseudo-target", choices=["max", "average"], default="max")
    ensemble.add_argument("--contamination", type=float, default=0.1)
    ensemble.add_argument("--seed", type=int, default=0)
    ensemble.set_defaults(func=_cmd_ensemble)

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
