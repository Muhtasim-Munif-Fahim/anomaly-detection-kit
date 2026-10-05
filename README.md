# anomaly-detection-kit

A small, dependency-light toolkit for **unsupervised anomaly / outlier
detection** in tabular data and time series. It ships classical statistical
baselines (z-score, median/MAD, IQR fences, generalized ESD) and fifteen
self-contained models (isolation forest, local outlier factor, k-nearest
neighbours, COPOD, ECOD, HBOS, one-class SVM, EllipticEnvelope / FAST-MCD,
CBLOF, LODA, ABOD, COF, SOD, PCA, KDE), plus a seeded synthetic-data generator,
evaluation metrics, and a markdown report renderer — all built on
**numpy only** (no scipy, no sklearn).

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
  reachability-density ratios, k-nearest neighbours (k-th neighbour
  distance, or the mean of the k distances), COPOD (copula-based outlier
  detection from empirical left/right tail CDFs), ECOD (univariate empirical
  CDF outlier detection with left/right/auto tails), HBOS
  (histogram-based outlier score from independent univariate histograms),
  a one-class SVM (Schölkopf dual, SMO) whose anomaly score is the
  negative decision function, an EllipticEnvelope whose score is the
  Mahalanobis distance under a FAST-MCD robust covariance estimate, and
  CBLOF (cluster-based local outlier factor via k-means large/small
  clusters and distance-to-large-centre scoring), and LODA (random 1-D
  projections with histogram density scores; higher = more anomalous),
  and ABOD (angle-based outlier detection via negative ABOF of weighted
  cosines to k nearest neighbours; higher = more anomalous), and COF
  (connectivity-based outlier factor via average chaining distance along
  the set-based nearest path; higher = more anomalous), and SOD
  (subspace outlier detection via normalised distance to the neighbour mean
  in a locally relevant axis-parallel subspace; higher = more anomalous), and PCA
  (reconstruction-error outlier detector via squared L2 residual after a low-rank
  PCA projection; higher = more anomalous), and KDE (Gaussian kernel density
  estimation scored by negative log density with Scott/Silverman/fixed
  bandwidth and leave-one-out training scores; higher = more anomalous).
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
from anomaly_detection.models import ABOD, CBLOF, COF, COPOD, ECOD, EllipticEnvelope, HBOS, IsolationForest, KDE, KNN, LODA, OneClassSVM, PCA, SOD
from anomaly_detection.evaluate import compare_detectors

X, y_true = make_tabular(n_samples=600, contamination=0.05, seed=7)

detectors = {
    "isolation forest": IsolationForest(seed=7).fit(X).score_samples,
    "KNN": KNN(n_neighbors=5).fit(X).score_samples,
    "COPOD": COPOD().fit(X).score_samples,
    "ECOD": ECOD().fit(X).score_samples,
    "HBOS": HBOS().fit(X).score_samples,
    "one-class SVM": OneClassSVM(nu=0.1).fit(X).score_samples,
    "elliptic envelope": EllipticEnvelope(seed=7).fit(X).score_samples,
    "CBLOF": CBLOF(n_clusters=8, seed=7).fit(X).score_samples,
    "LODA": LODA(n_bins=10, n_random_cuts=100, seed=7).fit(X).score_samples,
    "ABOD": ABOD(n_neighbors=10).fit(X).score_samples,
    "COF": COF(n_neighbors=20).fit(X).score_samples,
    "SOD": SOD(n_neighbors=20).fit(X).score_samples,
    "PCA": PCA(n_components=0.95).fit(X).score_samples,
    "KDE": KDE(bandwidth="scott").fit(X).score_samples,
}
rows = compare_detectors(X, y_true, detectors, contamination=0.05)
print(rows[0]["f1"], rows[0]["auc"])
```

kNN scores each row by how far it sits from its neighbours in the fitted
reference set. The default ``method="largest"`` is the distance to the
k-th neighbour; ``method="mean"`` averages those k distances. Higher
scores are more anomalous:

```python
from anomaly_detection.models import KNN

knn = KNN(n_neighbors=5, method="largest").fit(X)
scores = knn.score_samples(X)
flags = knn.predict(X, contamination=0.05)
```

One-class SVM fits a kernel half-space around the training rows. The
default RBF kernel uses ``gamma = 1 / (n_features * Var(X))`` and ranks
points far from the training mass as anomalous. Scores are the negative
decision function (inliers often fall below zero); higher scores are more
anomalous. The linear kernel instead treats the origin side of the
hyperplane as anomalous:

```python
from anomaly_detection.models import OneClassSVM

ocsvm = OneClassSVM(nu=0.1, kernel="rbf").fit(X)
scores = ocsvm.score_samples(X)
flags = ocsvm.predict(X, contamination=0.05)
```

Kernel density estimation scores each row by its negative log density
under an isotropic Gaussian KDE fitted to the (standardised) training rows.
``bandwidth`` accepts ``"scott"``, ``"silverman"`` or a positive float;
scoring the training matrix uses a leave-one-out density. Higher scores are
more anomalous:

```python
from anomaly_detection.models import KDE

kde = KDE(bandwidth="silverman").fit(X)
scores = kde.score_samples(X)
flags = kde.predict(X, contamination=0.05)
print(kde.bandwidth_)
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
| k-nearest neighbours | model     | higher = more anomalous                | `n_neighbors`, `method`     |
| COPOD             | model         | higher = more anomalous                | none (parameter-free)       |
| ECOD              | model         | higher = more anomalous                | none (parameter-free)       |
| HBOS              | model         | higher = more anomalous                | `n_bins`, `alpha`, `tol`    |
| One-class SVM     | model         | higher = more anomalous                | `nu`, `kernel`, `gamma`     |
| EllipticEnvelope  | model         | higher = more anomalous                | `support_fraction`, `seed`  |
| CBLOF             | model         | higher = more anomalous                | `n_clusters`, `alpha`, `beta`|
| LODA              | model         | higher = more anomalous                | `n_bins`, `n_random_cuts`   |
| ABOD              | model         | higher = more anomalous                | `n_neighbors`               |
| COF               | model         | higher = more anomalous                | `n_neighbors`               |
| SOD               | model         | higher = more anomalous                | `n_neighbors`, `alpha`      |
| PCA               | model         | higher = more anomalous                | `n_components`              |
| KDE               | model         | higher = more anomalous                | `bandwidth`, `standardize`  |
| z-score           | statistical   | max abs z per row                       | `threshold` (default 3.0)   |
| Modified z-score  | statistical   | median/MAD robust z                    | `threshold` (default 3.5)   |
| IQR fences        | statistical   | distance beyond fence / IQR            | `k` (default 1.5)           |
| Generalized ESD   | time series   | standardised deviation vs t critical   | `k`, `alpha`                |

## Project layout

```
src/anomaly_detection/
├── generators.py   # seeded synthetic data (tabular + time series)
├── classic.py      # statistical baselines incl. GESD
├── models.py       # isolation forest, LOF, kNN, COPOD, ECOD, HBOS, one-class SVM, EllipticEnvelope, CBLOF, LODA, ABOD, COF, SOD, PCA, KDE
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
  runs. COPOD / ECOD (empirical CDFs), HBOS (histograms), kNN (pairwise
  distances) and one-class SVM (SMO on a fixed kernel) are deterministic.
  CBLOF is stochastic in its k-means init; pass a `seed` for reproducible runs.
  LODA is stochastic in its random projections; pass a `seed` for reproducible runs.
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
