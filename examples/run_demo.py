"""Run the anomaly detection toolkit end to end and write a markdown report.

Simulates labelled tabular data with known contamination, scores it with
every detector, prints the comparison and saves
``examples/output/demo_report.md``.
"""

from __future__ import annotations

from pathlib import Path

from anomaly_detection import report as rep
from anomaly_detection.classic import STATISTICAL_SCORERS
from anomaly_detection.evaluate import (
    best_f1_threshold,
    compare_detectors,
    threshold_sweep,
)
from anomaly_detection.generators import make_tabular
from anomaly_detection.models import COPOD, IsolationForest, LocalOutlierFactor, _flags_from_contamination

SEED = 42
CONTAMINATION = 0.05


def main() -> None:
    X, y = make_tabular(
        n_samples=800,
        n_features=5,
        n_clusters=3,
        contamination=CONTAMINATION,
        seed=SEED,
    )
    detectors = {
        "isolation forest": IsolationForest(
            n_estimators=200, max_samples=256, seed=SEED
        ).fit(X).score_samples,
        "local outlier factor": LocalOutlierFactor(n_neighbors=20).fit(X).score_samples,
        "COPOD": COPOD().fit(X).score_samples,
    }
    detectors.update(STATISTICAL_SCORERS)

    comparison = compare_detectors(X, y, detectors, contamination=CONTAMINATION)
    print(rep.comparison_table(comparison))

    best_name = str(comparison[0]["detector"])
    scores = detectors[best_name](X)
    flags = _flags_from_contamination(scores, CONTAMINATION)
    sweep = threshold_sweep(y, scores)
    best_t, best_f1 = best_f1_threshold(y, scores)

    sections = [
        "## Data",
        "",
        f"- {X.shape[0]} rows, {X.shape[1]} features, "
        f"{int(y.sum())} injected outliers ({CONTAMINATION:.0%} contamination)",
        "",
        "## Detector comparison",
        "",
        rep.comparison_table(comparison),
        "",
        "## Score summaries",
        "",
    ]
    for name, scorer in detectors.items():
        s = scorer(X)
        f = _flags_from_contamination(s, CONTAMINATION)
        sections.append(rep.score_summary(str(name), s, int(f.sum())))
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

    out_path = Path(__file__).resolve().parent / "output" / "demo_report.md"
    rep.write_report(str(out_path), "Anomaly detection demo", sections)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
