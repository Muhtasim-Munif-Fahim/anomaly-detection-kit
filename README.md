# anomaly-detection-kit

A small, dependency-light toolkit for **unsupervised anomaly / outlier
detection** in tabular data and time series. It ships classical statistical
baselines (z-score, median/MAD, IQR fences, generalized ESD) and four
self-contained models (isolation forest, local outlier factor, COPOD,
HBOS), plus a seeded synthetic-data generator, evaluation metrics, and a
markdown report renderer — all built on **numpy only** (no scipy, no sklearn).

Everything is implemented from first principles so the internals stay
readable and easy to extend.

## Features

- **Synthetic data** — gaussian clusters with injected mean-shift and
  variance outliers at a known contamination level, plus time series with
  point spikes and level shifts. All draws are seeded and reproducible.
- **Classical baselines** — z-score, modified z-score (median/MAD),
  Tukey IQR fences, and the generalized ESD test. GESD critical values use
  an in-house Student-t quantile (regularised incomplete beta).
- **Models** — an isolation forest with random feature/split trees and
  path-length scoring, a local outlier factor with kNN
  reachability-density ratios, COPOD (copula-based outlier detection
  from empirical left/right tail CDFs), and HBOS (histogram-based outlier
  score from independent univariate histograms).
- **Evaluation** — precision / recall / F1, rank-based ROC-AUC, threshold
  sweep with best-F1 selection, and a comparison table across detectors.
- **Reports** — markdown renderer with per-detector score summaries, top
  flagged rows, sweep tables and caveats.
- **CLI** — `simulate`, `detect`, `evaluate`, `report` subcommands.

## Installation

Requires Python 3.9+ and numpy.

```bash
pip install -r requirements.txt
pip install -e .        # optional, exposes the `anomaly-detect` command
```

## Quickstart

### Library

```python
from anomaly_detection.generators import make_tabular
from anomaly_detection.models import COPOD, HBOS, IsolationForest
from anomaly_detection.evaluate import compare_detectors

X, y_true = make_tabular(n_samples=600, contamination=0.05, seed=7)

detectors = {
    "isolation forest": IsolationForest(seed=7).fit(X).score_samples,
    "COPOD": COPOD().fit(X).score_samples,
    "HBOS": HBOS().fit(X).score_samples,
}
rows = compare_detectors(X, y_true, detectors, contamination=0.05)
print(rows[0]["f1"], rows[0]["auc"])
```

### Demo

```bash
python examples/run_demo.py
```

Simulates labelled data, scores it with every detector, prints the
comparison and writes `examples/output/demo_report.md`.

### CLI

```bash
anomaly-detect simulate --n-samples 800 --contamination 0.05 --seed 0 --out data.csv
anomaly-detect detect --data data.csv --contamination 0.05 --indices 10
anomaly-detect evaluate --data data.csv --contamination 0.05 --out comparison.md
anomaly-detect report --out anomaly_report.md
```

Or without installing: `python -m anomaly_detection <command> ...`.

## Detectors

| Detector          | Type          | Score meaning                          | Main knob                    |
| ----------------- | ------------- | -------------------------------------- | ---------------------------- |
| Isolation forest  | model         | higher = more anomalous                | `n_estimators`, `max_samples`|
| Local outlier factor | model     | higher = more anomalous                | `n_neighbors`               |
| COPOD             | model         | higher = more anomalous                | none (parameter-free)       |
| HBOS              | model         | higher = more anomalous                | `n_bins`, `alpha`, `tol`    |
| z-score           | statistical   | max abs z per row                       | `threshold` (default 3.0)   |
| Modified z-score  | statistical   | median/MAD robust z                    | `threshold` (default 3.5)   |
| IQR fences        | statistical   | distance beyond fence / IQR            | `k` (default 1.5)           |
| Generalized ESD   | time series   | standardised deviation vs t critical   | `k`, `alpha`                |

## Project layout

```
src/anomaly_detection/
├── generators.py   # seeded synthetic data (tabular + time series)
├── classic.py      # statistical baselines incl. GESD
├── models.py       # isolation forest, local outlier factor, COPOD, HBOS
├── evaluate.py     # metrics, threshold sweep, comparison
├── report.py       # markdown rendering
└── cli.py          # command line interface
examples/run_demo.py
tests/
```

## Caveats

- Detectors are unsupervised: how many rows get flagged depends on the
  assumed contamination fraction or on fixed statistical thresholds.
- Anomaly scores are only meaningful relative to one another on the same
  data.
- Isolation forest and LOF are stochastic; pass a `seed` for reproducible
  runs. COPOD (empirical CDFs) and HBOS (histograms) are deterministic.
- Labels come from the synthetic generator, so the metrics measure recovery
  of known-injected outliers, not performance on unlabelled real-world data.
- The generalized ESD test assumes a roughly normal baseline and can miss
  small level shifts relative to the noise.

## Testing

```bash
python -m pytest tests -q
```

## License

MIT
