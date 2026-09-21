"""Model-based anomaly detectors.

Five dependency-free models:

* :class:`IsolationForest` -- random feature + random split-point trees with
  path-length anomaly scoring (higher score = more anomalous).
* :class:`LocalOutlierFactor` -- kNN reachability-density ratio, computed by
  brute force on pairwise distances (higher score = more anomalous).
* :class:`KNN` -- k-nearest-neighbour distance (k-th neighbour or mean of
  the k distances; higher score = more anomalous).
* :class:`COPOD` -- copula-based outlier detection from empirical left/right
  tail CDFs and a skewness-corrected tail (higher score = more anomalous).
* :class:`HBOS` -- histogram-based outlier score from independent univariate
  histograms (higher score = more anomalous).

All expose the same interface: ``fit``, ``score_samples(X)`` returning
continuous anomaly scores, and ``fit_predict(X, contamination=...)``
returning binary flags for the top ``contamination`` fraction.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

EULER_GAMMA = 0.57721566490153286060651209


def _harmonic_correction(n: np.ndarray) -> np.ndarray:
    """Average path length of an unsuccessful BST search of ``n`` nodes."""
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    equal_two = n == 2
    greater = n > 2
    out[equal_two] = 1.0
    out[greater] = (
        2.0 * (np.log(n[greater] - 1.0) + EULER_GAMMA)
        - 2.0 * (n[greater] - 1.0) / n[greater]
    )
    return out


def _flags_from_contamination(scores: np.ndarray, contamination: float) -> np.ndarray:
    """Flag the top ``contamination`` fraction of anomaly scores."""
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if contamination <= 0.0:
        return np.zeros(n, dtype=int)
    if contamination >= 1.0:
        return np.ones(n, dtype=int)
    k = max(1, int(round(contamination * n)))
    threshold = np.partition(scores, n - k)[n - k]
    return (scores >= threshold).astype(int)


class IsolationForest:
    """Isolation forest anomaly detector.

    Each tree splits the (subsampled) data on a random feature and a random
    split point drawn uniformly between the feature's observed bounds. Points
    requiring short average paths to isolate are scored close to 1.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        max_depth: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_samples = int(max_samples)
        self.max_depth = max_depth
        self.seed = seed
        self.scores_: Optional[np.ndarray] = None
        self._trees: list = []
        self._ref_n = 0

    def fit(self, X: np.ndarray) -> "IsolationForest":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n = len(X)
        if n < 2:
            raise ValueError("IsolationForest needs at least two samples")
        max_samples = max(2, min(self.max_samples, n))
        if self.max_depth is None:
            max_depth = int(math.ceil(math.log2(max_samples)))
        else:
            max_depth = max(1, int(self.max_depth))
        self._ref_n = max_samples
        rng = np.random.default_rng(self.seed)
        self._trees = []
        for _ in range(self.n_estimators):
            sample_idx = rng.choice(n, size=max_samples, replace=False)
            self._trees.append(self._fit_tree(X[sample_idx], max_depth, rng))
        return self

    def _fit_tree(
        self, X: np.ndarray, max_depth: int, rng: np.random.Generator
    ) -> dict:
        n_cols = X.shape[1]
        left: list = []
        right: list = []
        split_feature: list = []
        split_value: list = []
        leaf_size: list = []

        def build(idx: np.ndarray, depth: int) -> int:
            node = len(left)
            left.append(-1)
            right.append(-1)
            split_feature.append(-1)
            split_value.append(0.0)
            leaf_size.append(int(idx.size))
            m = idx.size
            if m <= 1 or depth >= max_depth:
                return node
            feature = -1
            split_val = 0.0
            for _ in range(2 * n_cols):
                f = int(rng.integers(0, n_cols))
                lo = float(X[idx, f].min())
                hi = float(X[idx, f].max())
                if hi - lo > 1e-12:
                    feature = f
                    split_val = float(rng.uniform(lo, hi))
                    break
            if feature < 0:
                return node
            split_feature[node] = feature
            split_value[node] = split_val
            mask = X[idx, feature] < split_val
            left[node] = build(idx[mask], depth + 1)
            right[node] = build(idx[~mask], depth + 1)
            return node

        build(np.arange(len(X)), 0)
        return {
            "left": np.asarray(left, dtype=np.int64),
            "right": np.asarray(right, dtype=np.int64),
            "split_feature": np.asarray(split_feature, dtype=np.int64),
            "split_value": np.asarray(split_value, dtype=float),
            "leaf_sizes": np.asarray(leaf_size, dtype=np.int64),
        }

    def _path_lengths(self, tree: dict, X: np.ndarray) -> np.ndarray:
        sf = tree["split_feature"]
        sv = tree["split_value"]
        left = tree["left"]
        right = tree["right"]
        leaf_size = tree["leaf_sizes"]
        n = len(X)
        node = np.zeros(n, dtype=np.int64)
        depth = np.zeros(n)
        corr = np.zeros(n)
        active = np.ones(n, dtype=bool)
        while True:
            act = np.flatnonzero(active)
            cur = sf[node[active]]
            internal = cur >= 0
            leaf_pts = act[~internal]
            corr[leaf_pts] = _harmonic_correction(leaf_size[node[leaf_pts]])
            if not internal.any():
                break
            int_pts = act[internal]
            go_left = X[int_pts, cur[internal]] < sv[node[int_pts]]
            node[int_pts] = np.where(
                go_left, left[node[int_pts]], right[node[int_pts]]
            )
            depth[int_pts] += 1
            active[act] = sf[node[active]] >= 0
        return depth + corr

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if not self._trees:
            raise ValueError("IsolationForest must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        avg_path = np.zeros(len(X))
        for tree in self._trees:
            avg_path += self._path_lengths(tree, X)
        avg_path /= len(self._trees)
        c = _harmonic_correction(self._ref_n)
        self.scores_ = 2.0 ** (-avg_path / c)
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)


class LocalOutlierFactor:
    """Local Outlier Factor detector (batch mode).

    Neighbourhoods are built inside the scored set, so ``score_samples`` must
    receive the full data. Scores are the reachability-density ratio: values
    clearly above 1 indicate isolated points.
    """

    def __init__(self, n_neighbors: int = 20, seed: Optional[int] = None) -> None:
        self.n_neighbors = int(n_neighbors)
        self.seed = seed
        self.scores_: Optional[np.ndarray] = None
        self._n_features = None

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        sq = np.einsum("ij,ij->i", X, X)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2)

    def fit(self, X: np.ndarray) -> "LocalOutlierFactor":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        self._n_features = X.shape[1]
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n = len(X)
        if n == 0:
            self.scores_ = np.empty(0)
            return self.scores_
        k = min(self.n_neighbors, n - 1)
        if k < 1:
            self.scores_ = np.ones(n)
            return self.scores_
        D = self._pairwise_distances(X)
        scale = max(float(D.max()), 1.0)
        order = np.argsort(D, axis=1)[:, 1 : k + 1]
        kd = np.take_along_axis(D, order, axis=1)[:, -1]
        dists = np.take_along_axis(D, order, axis=1)
        rd = np.maximum(kd[:, None], dists)
        rd = np.maximum(rd, 1e-12 * scale)
        lrd = 1.0 / rd.mean(axis=1)
        lof = (lrd[order] / lrd[:, None]).mean(axis=1)
        self.scores_ = lof
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)


def _euclidean_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between rows of ``X`` and ``Y``."""
    sqx = np.einsum("ij,ij->i", X, X)
    sqy = np.einsum("ij,ij->i", Y, Y)
    d2 = sqx[:, None] + sqy[None, :] - 2.0 * (X @ Y.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


class KNN:
    """k-nearest-neighbour outlier detector (Ramaswamy, Rastogi & Shim, 2000).

    ``fit`` stores the training rows. ``score_samples`` is the distance from
    each query row to its neighbours in that reference set: ``method="largest"``
    (default) uses the k-th neighbour, ``method="mean"`` averages the k
    distances. Scoring the training matrix itself excludes each point as its
    own neighbour, matching the usual transductive kNN outlier ranking.

    Higher scores are more anomalous.
    """

    def __init__(self, n_neighbors: int = 5, method: str = "largest") -> None:
        n_neighbors = int(n_neighbors)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        method = str(method)
        if method not in ("largest", "mean"):
            raise ValueError("method must be 'largest' or 'mean'")
        self.n_neighbors = n_neighbors
        self.method = method
        self.scores_: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    def fit(self, X: np.ndarray) -> "KNN":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n = len(X)
        if n < 1:
            raise ValueError("KNN needs at least one sample")
        self._X_train = np.array(X, dtype=float, copy=True)
        self._n_features = X.shape[1]
        self.scores_ = self._score_against(self._X_train, exclude_self=True)
        return self

    def _score_against(self, X: np.ndarray, exclude_self: bool) -> np.ndarray:
        ref = self._X_train
        if ref is None:
            raise ValueError("KNN must be fitted before scoring")
        n_query = len(X)
        n_train = len(ref)
        if n_query == 0:
            return np.empty(0)
        max_k = n_train - 1 if exclude_self else n_train
        if max_k < 1:
            return np.zeros(n_query)
        k = min(self.n_neighbors, max_k)
        D = _euclidean_distances(X, ref)
        if exclude_self:
            np.fill_diagonal(D, np.inf)
        # k smallest distances per query row
        idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
        dists = np.take_along_axis(D, idx, axis=1)
        if self.method == "largest":
            return dists.max(axis=1)
        return dists.mean(axis=1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._X_train is None:
            raise ValueError("KNN must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but KNN was fitted on {self._n_features}"
            )
        same_train = X.shape == self._X_train.shape and np.array_equal(X, self._X_train)
        if same_train:
            self.scores_ = self._score_against(self._X_train, exclude_self=True)
        else:
            self.scores_ = self._score_against(X, exclude_self=False)
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)


def _column_skewness(X: np.ndarray) -> np.ndarray:
    """Per-column skewness as in Li et al. 2020 (COPOD, eq. 11).

    Numerator is the third central moment (``1/n``); the denominator is the
    sample standard deviation (``ddof=1``) cubed. Constant columns return 0.
    """
    n, n_features = X.shape
    if n < 2:
        return np.zeros(n_features)
    centered = X - X.mean(axis=0)
    m3 = np.mean(centered ** 3, axis=0)
    var = np.sum(centered * centered, axis=0) / (n - 1)
    std = np.sqrt(np.maximum(var, 0.0))
    out = np.zeros(n_features)
    nz = std > 0.0
    out[nz] = m3[nz] / (std[nz] ** 3)
    return out


def _ecdf_against(X: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    """Column-wise empirical CDF of ``X`` relative to a pre-sorted reference.

    ``sorted_ref`` must already be sorted along axis 0. Returns the fraction
    of reference values that are ``<=`` each entry of ``X`` (ties share the
    highest rank, matching ``searchsorted(..., side='right')``).
    """
    n_ref = sorted_ref.shape[0]
    n, n_features = X.shape
    U = np.empty((n, n_features), dtype=float)
    for j in range(n_features):
        U[:, j] = np.searchsorted(sorted_ref[:, j], X[:, j], side="right") / n_ref
    return U


class COPOD:
    """Copula-based outlier detector (Li, Zhao, Botta, Ionescu & Hu, 2020).

    Parameter-free. Each feature's empirical left-tail CDF and right-tail
    survival function are treated as copula observations; the row score is
    the most extreme of the three tail probabilities (left, right, and
    skewness-corrected), on the ``-log`` scale.

    ``fit`` stores the training columns so ``score_samples`` can score new
    rows against those ECDFs. Scoring the training matrix itself matches
    the original transductive Algorithm 1.
    """

    def __init__(self) -> None:
        self.scores_: Optional[np.ndarray] = None
        self._sorted: Optional[np.ndarray] = None
        self._sorted_neg: Optional[np.ndarray] = None
        self._n_train = 0
        self._n_features: Optional[int] = None
        self._skewness: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "COPOD":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n = len(X)
        if n < 1:
            raise ValueError("COPOD needs at least one sample")
        self._n_train = n
        self._n_features = X.shape[1]
        self._sorted = np.sort(X, axis=0)
        self._sorted_neg = np.sort(-X, axis=0)
        self._skewness = _column_skewness(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._sorted is None or self._sorted_neg is None or self._skewness is None:
            raise ValueError("COPOD must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but COPOD was fitted on {self._n_features}"
            )
        n = len(X)
        if n == 0:
            self.scores_ = np.empty(0)
            return self.scores_
        U = _ecdf_against(X, self._sorted)
        V = _ecdf_against(-X, self._sorted_neg)
        floor = 1.0 / (self._n_train + 1.0)
        U = np.clip(U, floor, 1.0)
        V = np.clip(V, floor, 1.0)
        W = np.where(self._skewness < 0.0, U, V)
        p_l = -np.log(U).sum(axis=1)
        p_r = -np.log(V).sum(axis=1)
        p_s = -np.log(W).sum(axis=1)
        self.scores_ = np.maximum(np.maximum(p_l, p_r), p_s)
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)


def _hbos_column_heights(
    values: np.ndarray,
    edges: np.ndarray,
    hist: np.ndarray,
    floor: float,
    tol: float,
) -> np.ndarray:
    """Histogram height for each value of one feature.

    In-range points use the bin they fall into. Points slightly outside the
    training range (within ``tol`` times the edge-bin width) inherit the
    edge bin; farther points inherit ``floor`` (the regularised empty-bin
    height), which scores them as rare.
    """
    n = values.size
    n_bins = hist.size
    heights = np.empty(n, dtype=float)
    in_range = (values >= edges[0]) & (values <= edges[-1])
    inds = np.searchsorted(edges, values, side="right") - 1
    inds = np.clip(inds, 0, n_bins - 1)
    heights[in_range] = hist[inds[in_range]]

    below = values < edges[0]
    if below.any():
        width = edges[1] - edges[0]
        near = (edges[0] - values[below]) <= width * tol
        heights[below] = np.where(near, hist[0], floor)

    above = values > edges[-1]
    if above.any():
        width = edges[-1] - edges[-2]
        near = (values[above] - edges[-1]) <= width * tol
        heights[above] = np.where(near, hist[-1], floor)

    return heights


class HBOS:
    """Histogram-based outlier detector (Goldstein & Dengel, 2012).

    Assumes feature independence. Each column is turned into a univariate
    histogram of ``n_bins`` equal-width bins. Heights are scaled so the
    tallest bin is 1 (equal feature weight) and empty bins are floored at
    ``alpha``. The row score is the sum of ``-log`` heights: sparse bins
    score high.

    ``fit`` stores the per-column bin edges and heights so ``score_samples``
    can score new rows. Values slightly outside the training range (within
    ``tol`` times the edge-bin width) inherit the edge bin; farther values
    inherit the empty-bin floor.
    """

    def __init__(
        self,
        n_bins: int = 10,
        alpha: float = 0.1,
        tol: float = 0.5,
    ) -> None:
        n_bins = int(n_bins)
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if not 0.0 <= float(alpha) < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        if not 0.0 <= float(tol) <= 1.0:
            raise ValueError("tol must be in [0, 1]")
        self.n_bins = n_bins
        self.alpha = float(alpha)
        self.tol = float(tol)
        self.scores_: Optional[np.ndarray] = None
        self._hist: Optional[np.ndarray] = None
        self._edges: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None
        self._floor = self.alpha if self.alpha > 0.0 else 1e-12

    def fit(self, X: np.ndarray) -> "HBOS":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n, n_features = X.shape
        if n < 1:
            raise ValueError("HBOS needs at least one sample")
        self._n_features = n_features
        hist = np.empty((n_features, self.n_bins), dtype=float)
        edges = np.empty((n_features, self.n_bins + 1), dtype=float)
        for j in range(n_features):
            counts, col_edges = np.histogram(X[:, j], bins=self.n_bins)
            peak = float(counts.max())
            if peak > 0.0:
                col_hist = counts.astype(float) / peak
            else:
                col_hist = np.ones(self.n_bins, dtype=float)
            hist[j] = np.maximum(col_hist, self._floor)
            edges[j] = col_edges
        self._hist = hist
        self._edges = edges
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._hist is None or self._edges is None:
            raise ValueError("HBOS must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but HBOS was fitted on {self._n_features}"
            )
        n = len(X)
        if n == 0:
            self.scores_ = np.empty(0)
            return self.scores_
        scores = np.zeros(n, dtype=float)
        for j in range(self._n_features):
            heights = _hbos_column_heights(
                X[:, j], self._edges[j], self._hist[j], self._floor, self.tol
            )
            scores += -np.log(heights)
        self.scores_ = scores
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)
