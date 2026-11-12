"""Distance-based detectors: LDOF and the Sampling outlier detector.

* **LDOF** -- Local Distance-based Outlier Factor (Zhang, Hutter & Jin,
  PAKDD 2009). For a point ``p`` with k nearest neighbours ``N_p``::

      d_bar(p) = mean_{q in N_p} d(p, q)              # kNN distance
      D_bar(p) = mean_{q != q' in N_p} d(q, q')       # inner neighbourhood spread
      LDOF(p)  = d_bar(p) / D_bar(p)

  ``LDOF < 1`` means ``p`` sits inside its neighbourhood's "cloud"; values
  well above 1 mean ``p`` is far from a neighbourhood that is itself tight,
  which makes LDOF robust in scattered, multi-density data.

* **Sampling** -- Sugiyama & Borgwardt (NeurIPS 2013): score each point by
  its distance to the nearest member of one small random subsample of the
  training data. With subsample sizes around 20 it is orders of magnitude
  faster than exact kNN and competitive in accuracy. Training points that
  are themselves in the sample are scored against the rest of the sample.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .models import _euclidean_distances, _flags_from_contamination

__all__ = ["LDOF", "SamplingOutlierDetector"]


def _check_X(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D array")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    return X


class LDOF:
    """Local Distance-based Outlier Factor.

    Parameters
    ----------
    n_neighbors : int
        Neighbourhood size ``k`` (>= 2; the paper recommends ``k`` at least
        the data dimension and reports stable results for ``k`` in 10-50).
    """

    def __init__(self, n_neighbors: int = 20) -> None:
        if isinstance(n_neighbors, bool) or int(n_neighbors) != n_neighbors or n_neighbors < 2:
            raise ValueError("n_neighbors must be an integer >= 2")
        self.n_neighbors = int(n_neighbors)
        self._X_train: Optional[np.ndarray] = None
        self.knn_distance_: Optional[np.ndarray] = None
        self.inner_distance_: Optional[np.ndarray] = None
        self.scores_: Optional[np.ndarray] = None

    def fit(self, X) -> "LDOF":
        X = _check_X(X)
        if X.shape[0] < 3:
            raise ValueError("LDOF needs at least 3 training rows")
        self._X_train = X
        self.scores_ = self.score_samples(X)
        return self

    def score_samples(self, X) -> np.ndarray:
        """Return LDOF scores (higher is more anomalous)."""
        if self._X_train is None:
            raise ValueError("LDOF must be fitted before scoring")
        X = _check_X(X)
        train = self._X_train
        if X.shape[1] != train.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but LDOF was fitted on {train.shape[1]}"
            )
        same = X.shape == train.shape and np.array_equal(X, train)
        n_ref = train.shape[0] - (1 if same else 0)
        k = min(self.n_neighbors, n_ref)
        if k < 2:
            raise ValueError("need at least 2 reference neighbours")
        D = _euclidean_distances(X, train)
        if same:
            np.fill_diagonal(D, np.inf)
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        rows = np.arange(X.shape[0])[:, None]
        d_bar = D[rows, idx].mean(axis=1)
        D_train = _euclidean_distances(train, train)
        inner = D_train[idx[:, :, None], idx[:, None, :]]  # (n, k, k)
        D_bar = inner.sum(axis=(1, 2)) / (k * (k - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(D_bar > 0, d_bar / D_bar, np.where(d_bar > 0, np.inf, 0.0))
        finite = np.isfinite(score)
        if not finite.all():
            cap = score[finite].max() * 10 + 1 if finite.any() else 1.0
            score = np.where(finite, score, cap)
        self.knn_distance_ = d_bar
        self.inner_distance_ = D_bar
        return score

    def predict(self, X, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)


class SamplingOutlierDetector:
    """Sugiyama-Borgwardt subsample nearest-neighbour outlier detector.

    Parameters
    ----------
    subsample_size : int
        Number of training rows drawn (without replacement) as the reference
        sample. Capped at ``n_samples``.
    seed : int or None
        Random seed for the subsample.
    """

    def __init__(self, subsample_size: int = 20, seed: Optional[int] = None) -> None:
        if isinstance(subsample_size, bool) or int(subsample_size) != subsample_size or subsample_size < 1:
            raise ValueError("subsample_size must be a positive integer")
        self.subsample_size = int(subsample_size)
        self.seed = seed
        self.sample_indices_: Optional[np.ndarray] = None
        self._sample: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None
        self.scores_: Optional[np.ndarray] = None

    def fit(self, X) -> "SamplingOutlierDetector":
        X = _check_X(X)
        n = X.shape[0]
        s = min(self.subsample_size, n)
        if s < 2 and n >= 2:
            s = 2
        if n < 2:
            raise ValueError("need at least 2 training rows")
        rng = np.random.default_rng(self.seed)
        self.sample_indices_ = np.sort(rng.choice(n, size=s, replace=False))
        self._sample = X[self.sample_indices_]
        self._X_train = X
        self.scores_ = self.score_samples(X)
        return self

    def score_samples(self, X) -> np.ndarray:
        """Distance to the nearest subsample point (self-matches excluded)."""
        if self._sample is None:
            raise ValueError("SamplingOutlierDetector must be fitted before scoring")
        X = _check_X(X)
        if X.shape[1] != self._sample.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but the detector was fitted on {self._sample.shape[1]}"
            )
        D = _euclidean_distances(X, self._sample)
        same = X.shape == self._X_train.shape and np.array_equal(X, self._X_train)
        if same:
            D[self.sample_indices_, np.arange(self.sample_indices_.size)] = np.inf
        return D.min(axis=1)

    def predict(self, X, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)
