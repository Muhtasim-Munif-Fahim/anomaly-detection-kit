"""Model-based anomaly detectors.

Fifteen dependency-free models:

* :class:`IsolationForest` -- random feature + random split-point trees with
  path-length anomaly scoring (higher score = more anomalous).
* :class:`LocalOutlierFactor` -- kNN reachability-density ratio, computed by
  brute force on pairwise distances (higher score = more anomalous).
* :class:`KNN` -- k-nearest-neighbour distance (k-th neighbour or mean of
  the k distances; higher score = more anomalous).
* :class:`COPOD` -- copula-based outlier detection from empirical left/right
  tail CDFs and a skewness-corrected tail (higher score = more anomalous).
* :class:`ECOD` -- empirical CDF outlier detection from univariate left/right
  tail probabilities and a skewness-corrected auto tail (higher score =
  more anomalous).
* :class:`HBOS` -- histogram-based outlier score from independent univariate
  histograms (higher score = more anomalous).
* :class:`OneClassSVM` -- one-class SVM (Schölkopf et al.) scored by the
  negative decision function (higher score = more anomalous).
* :class:`EllipticEnvelope` -- FAST-MCD robust covariance / Mahalanobis
  distance (higher score = more anomalous).
* :class:`CBLOF` -- cluster-based local outlier factor: k-means clusters are
  labelled large/small, then points score by distance to their own large
  centre or to the nearest large centre (higher score = more anomalous).
* :class:`LODA` -- lightweight online detector of anomalies: random
  1-D projections with histogram density scores (higher = more anomalous).
* :class:`ABOD` -- angle-based outlier detection: negative variance of
  weighted cosines to k nearest neighbours (higher = more anomalous).
* :class:`COF` -- connectivity-based outlier factor: average chaining
  distance ratio along the set-based nearest path (higher = more anomalous).
* :class:`SOD` -- subspace outlier detection: normalised distance to the
  neighbour mean in a locally relevant axis-parallel subspace (higher =
  more anomalous).
* :class:`PCA` -- PCA reconstruction-error outlier detector: squared L2
  residual after projecting onto the leading principal components (higher =
  more anomalous).
* :class:`KDE` -- Gaussian kernel density estimation: negative log density
  under a Scott/Silverman/fixed-bandwidth isotropic KDE, leave-one-out on
  the training rows (higher = more anomalous).

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




class ECOD:
    """Empirical Cumulative Distribution Outlier Detection (Li et al., 2022).

    Parameter-free. Each feature's univariate left-tail CDF and right-tail
    survival function yield ``-log`` tail scores; the row score is the most
    extreme of the three aggregates (left, right, and skewness-corrected
    auto). Unlike COPOD this never builds a multivariate copula — every
    dimension is scored independently and the logs are summed.

    ``fit`` stores the training columns so ``score_samples`` can score new
    rows against those ECDFs. Scoring the training matrix itself matches
    the usual transductive ranking.
    """

    def __init__(self) -> None:
        self.scores_: Optional[np.ndarray] = None
        self._sorted: Optional[np.ndarray] = None
        self._sorted_neg: Optional[np.ndarray] = None
        self._n_train = 0
        self._n_features: Optional[int] = None
        self._skewness: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ECOD":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n = len(X)
        if n < 1:
            raise ValueError("ECOD needs at least one sample")
        self._n_train = n
        self._n_features = X.shape[1]
        self._sorted = np.sort(X, axis=0)
        self._sorted_neg = np.sort(-X, axis=0)
        self._skewness = _column_skewness(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._sorted is None or self._sorted_neg is None or self._skewness is None:
            raise ValueError("ECOD must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but ECOD was fitted on {self._n_features}"
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


def _scale_gamma(X: np.ndarray) -> float:
    """RBF / polynomial scale ``1 / (n_features * Var(X))``.

    Variance is the population variance of every entry (``ddof=0``). A
    zero or non-finite variance falls back to 1, matching the usual
    ``gamma="scale"`` heuristic.
    """
    if X.size == 0 or X.shape[1] == 0:
        return 1.0
    var = float(np.var(X))
    if not math.isfinite(var) or var <= 0.0:
        return 1.0
    return 1.0 / (X.shape[1] * var)


def _ocsvm_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str,
    gamma: float,
    degree: int,
    coef0: float,
) -> np.ndarray:
    """Gram matrix between rows of ``X`` and rows of ``Y``."""
    if kernel == "linear":
        return X @ Y.T
    dots = X @ Y.T
    if kernel == "poly":
        return (gamma * dots + coef0) ** degree
    x2 = np.einsum("ij,ij->i", X, X)
    y2 = np.einsum("ij,ij->i", Y, Y)
    d2 = x2[:, None] + y2[None, :] - 2.0 * dots
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-gamma * d2)


def _rho_from_gradient(G: np.ndarray, alpha: np.ndarray, C: float) -> float:
    """Decision offset from the dual gradient ``G = K @ alpha``.

    Unbounded support vectors (``0 < alpha < C``) sit on the margin, so
    ``rho`` is their mean gradient. If every weight is at a bound, ``rho``
    is the midpoint of the KKT bounds.
    """
    eps = 1e-6 * C
    free = (alpha > eps) & (alpha < C - eps)
    if np.any(free):
        return float(np.mean(G[free]))
    upper = G[alpha < C - eps]
    lower = G[alpha > eps]
    if upper.size and lower.size:
        return 0.5 * (float(np.min(upper)) + float(np.max(lower)))
    if upper.size:
        return float(np.min(upper))
    if lower.size:
        return float(np.max(lower))
    return float(np.mean(G))


def _solve_one_class_dual(
    K: np.ndarray, nu: float, tol: float, max_iter: int
) -> tuple[np.ndarray, float]:
    """SMO for the Schölkopf one-class dual.

    Minimises ``0.5 * alpha @ K @ alpha`` subject to ``sum(alpha) == 1``
    and ``0 <= alpha_i <= 1/(nu * n)``. Pair selection maximises the
    estimated decrease (second-order working set). Returns ``(alpha, rho)``.
    """
    n = int(K.shape[0])
    C = 1.0 / (nu * float(n))
    # nu == 1 leaves a single feasible point: every weight sits at the box.
    if nu >= 1.0:
        alpha = np.full(n, 1.0 / n)
        alpha[-1] = 1.0 - float(alpha[:-1].sum())
        return alpha, float(np.max(K @ alpha))

    alpha = np.full(n, 1.0 / n)
    alpha[-1] = 1.0 - float(alpha[:-1].sum())
    G = K @ alpha
    diag = np.diag(K).copy()
    bound_eps = 1e-10 * C

    for n_iter in range(1, max_iter + 1):
        up = np.flatnonzero(alpha < C - bound_eps)
        low = np.flatnonzero(alpha > bound_eps)
        if up.size == 0 or low.size == 0:
            break
        i = int(up[np.argmin(G[up])])
        viol = G - G[i]
        eta_row = diag[i] + diag - 2.0 * K[i]
        eligible = np.zeros(n, dtype=bool)
        eligible[low] = True
        eligible[i] = False
        eligible &= viol > 0.0
        if not np.any(eligible):
            break
        eta_m = eta_row[eligible]
        viol_m = viol[eligible]
        gain = np.empty(eta_m.size, dtype=float)
        positive_eta = eta_m > 1e-12
        gain[positive_eta] = (viol_m[positive_eta] ** 2) / eta_m[positive_eta]
        gain[~positive_eta] = np.inf
        j = int(np.flatnonzero(eligible)[int(np.argmax(gain))])
        gap = float(G[j] - G[i])
        scale = max(1.0, abs(float(G[i])), abs(float(G[j])))
        if gap <= tol * scale:
            break
        room = min(C - float(alpha[i]), float(alpha[j]))
        eta_ij = float(eta_row[j])
        if eta_ij <= 1e-12:
            delta = room
        else:
            delta = gap / eta_ij
            if delta > room:
                delta = room
            if delta < 0.0:
                delta = 0.0
        if delta <= 1e-14:
            break
        alpha[i] += delta
        alpha[j] -= delta
        G += delta * (K[:, i] - K[:, j])
        if n_iter % 40 == 0:
            G = K @ alpha

    return alpha, _rho_from_gradient(K @ alpha, alpha, C)


class OneClassSVM:
    """One-class SVM anomaly detector (Schölkopf, Platt, Shawe-Taylor, Smola, Williamson, 2001).

    ``fit`` solves the dual on the training Gram matrix with SMO. The
    anomaly score of a row is the negative decision function

    ``rho - sum_i alpha_i K(x_i, x)``

    so points outside the half-space score higher. ``nu`` is an upper bound
    on the fraction of training margin errors and a lower bound on the
    fraction of support vectors.

    ``kernel`` is ``"rbf"`` (default), ``"linear"``, or ``"poly"``. With
    ``gamma=None`` the scale is ``1 / (n_features * Var(X))`` from the
    training matrix (or 1 when that variance is 0). RBF with this scale is
    invariant to translating or rescaling the features and is the kernel
    that ranks Euclidean outliers. The linear kernel ignores ``gamma`` and
    separates the sample from the origin, so the anomalous side is toward
    the origin. A polynomial kernel likewise scores a feature-space
    half-space: points far from the origin can look ordinary.

    The solver is deterministic: the same matrix always yields the same scores.
    """

    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 5000,
    ) -> None:
        nu = float(nu)
        if not math.isfinite(nu) or not 0.0 < nu <= 1.0:
            raise ValueError("nu must be in (0, 1]")
        kernel = str(kernel)
        if kernel not in ("rbf", "linear", "poly"):
            raise ValueError("kernel must be 'rbf', 'linear' or 'poly'")
        if gamma is not None:
            gamma = float(gamma)
            if not math.isfinite(gamma) or gamma <= 0.0:
                raise ValueError(
                    "gamma must be a positive finite number, or None to use the scale heuristic"
                )
        degree = int(degree)
        if degree < 1:
            raise ValueError("degree must be at least 1")
        coef0 = float(coef0)
        if not math.isfinite(coef0):
            raise ValueError("coef0 must be finite")
        tol = float(tol)
        if not math.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be a positive finite number")
        max_iter = int(max_iter)
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.scores_: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._rho: Optional[float] = None
        self._gamma: Optional[float] = None
        self._n_features: Optional[int] = None

    def fit(self, X: np.ndarray) -> "OneClassSVM":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n = len(X)
        if n < 1:
            raise ValueError("OneClassSVM needs at least one sample")
        self._n_features = int(X.shape[1])
        self._X_train = np.array(X, dtype=float, copy=True)
        if self.gamma is None:
            self._gamma = _scale_gamma(self._X_train)
        else:
            self._gamma = float(self.gamma)
        K = _ocsvm_kernel(
            self._X_train,
            self._X_train,
            self.kernel,
            self._gamma,
            self.degree,
            self.coef0,
        )
        if not np.isfinite(K).all():
            raise ValueError("kernel matrix is not finite; check gamma, degree and coef0")
        self._alpha, self._rho = _solve_one_class_dual(K, self.nu, self.tol, self.max_iter)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if (
            self._X_train is None
            or self._alpha is None
            or self._rho is None
            or self._gamma is None
        ):
            raise ValueError("OneClassSVM must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but OneClassSVM was fitted on {self._n_features}"
            )
        n = len(X)
        if n == 0:
            self.scores_ = np.empty(0)
            return self.scores_
        K = _ocsvm_kernel(
            X, self._X_train, self.kernel, self._gamma, self.degree, self.coef0
        )
        if not np.isfinite(K).all():
            raise ValueError("kernel matrix is not finite; check gamma, degree and coef0")
        self.scores_ = self._rho - K @ self._alpha
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)



def _mcd_h_support(n: int, n_features: int, support_fraction: Optional[float]) -> int:
    """Subset size ``h`` for the Minimum Covariance Determinant."""
    n = int(n)
    p = int(n_features)
    if n < p + 1:
        raise ValueError(
            f"EllipticEnvelope needs at least n_features+1={p + 1} samples, got {n}"
        )
    h_min = (n + p + 1) // 2
    if support_fraction is None:
        return min(n, max(h_min, p + 1))
    frac = float(support_fraction)
    if not 0.5 <= frac <= 1.0:
        raise ValueError("support_fraction must be in [0.5, 1.0]")
    h = int(round(frac * n))
    return min(n, max(h_min, h, p + 1))


def _mcd_covariance(X: np.ndarray, idx: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Location, covariance and log-det of the subset ``idx``."""
    subset = X[idx]
    mean = subset.mean(axis=0)
    centered = subset - mean
    n_h = max(len(idx), 1)
    cov = (centered.T @ centered) / float(n_h)
    # Ridge keeps singular subsets invertible when h is barely above p.
    cov = cov + ridge * np.eye(cov.shape[0])
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0.0 or not math.isfinite(logdet):
        return mean, cov, float("inf")
    return mean, cov, float(logdet)


def _mahalanobis2(X: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    centered = X - mean
    return np.einsum("ij,jk,ik->i", centered, precision, centered)


def _mcd_cstep(
    X: np.ndarray,
    idx: np.ndarray,
    h: int,
    ridge: float,
    max_c_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Concentrate an initial subset to a local MCD minimum."""
    best_idx = np.asarray(idx, dtype=np.int64)
    mean, cov, logdet = _mcd_covariance(X, best_idx, ridge)
    if not math.isfinite(logdet):
        return best_idx, mean, cov, logdet
    for _ in range(max_c_steps):
        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return best_idx, mean, cov, float("inf")
        d2 = _mahalanobis2(X, mean, precision)
        new_idx = np.argpartition(d2, h - 1)[:h]
        new_idx.sort()
        if new_idx.size == best_idx.size and np.array_equal(new_idx, best_idx):
            break
        mean, cov, logdet = _mcd_covariance(X, new_idx, ridge)
        best_idx = new_idx
        if not math.isfinite(logdet):
            break
    return best_idx, mean, cov, logdet


def _fast_mcd(
    X: np.ndarray,
    h: int,
    n_trials: int,
    seed: Optional[int],
    ridge: float,
    max_c_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rousseeuw & Van Driessen FAST-MCD (simplified, exact C-steps).

    Draws ``n_trials`` raw subsets of size ``n_features + 1``, expands each
    to the ``h`` nearest points under the subset covariance, runs C-steps,
    and keeps the location / covariance with the smallest determinant.
    """
    n, p = X.shape
    rng = np.random.default_rng(seed)
    best_logdet = float("inf")
    best_mean = X.mean(axis=0)
    best_cov = np.cov(X.T) + ridge * np.eye(p)
    if best_cov.ndim == 0:
        best_cov = np.array([[float(best_cov)]])

    raw_size = min(n, p + 1)
    for _ in range(max(1, int(n_trials))):
        start = rng.choice(n, size=raw_size, replace=False)
        mean0, cov0, logdet0 = _mcd_covariance(X, start, ridge)
        if not math.isfinite(logdet0):
            continue
        try:
            precision0 = np.linalg.inv(cov0)
        except np.linalg.LinAlgError:
            continue
        d2 = _mahalanobis2(X, mean0, precision0)
        seed_idx = np.argpartition(d2, h - 1)[:h]
        idx, mean, cov, logdet = _mcd_cstep(X, seed_idx, h, ridge, max_c_steps)
        if logdet < best_logdet:
            best_logdet = logdet
            best_mean = mean
            best_cov = cov
    return best_mean, best_cov


class EllipticEnvelope:
    """Robust Gaussian envelope via the FAST Minimum Covariance Determinant.

    ``fit`` estimates a high-breakdown location and covariance with the
    Rousseeuw & Van Driessen FAST-MCD concentration steps. The anomaly score
    of a row is its Mahalanobis distance under that fit (square root of the
    quadratic form), so points far from the robust centre score higher.

    ``support_fraction`` controls the MCD subset size ``h``. The default
    ``None`` uses the classic ``(n + n_features + 1) // 2`` breakdown point.
    ``assume_centered`` skips location estimation and forces the mean to 0.

    The contamination argument on ``predict`` / ``fit_predict`` flags the
    top fraction of Mahalanobis scores, matching the other detectors.
    """

    def __init__(
        self,
        *,
        support_fraction: Optional[float] = None,
        assume_centered: bool = False,
        n_trials: int = 50,
        max_c_steps: int = 30,
        ridge: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        if support_fraction is not None:
            support_fraction = float(support_fraction)
            if not 0.5 <= support_fraction <= 1.0:
                raise ValueError("support_fraction must be in [0.5, 1.0]")
        n_trials = int(n_trials)
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        max_c_steps = int(max_c_steps)
        if max_c_steps < 1:
            raise ValueError("max_c_steps must be at least 1")
        ridge = float(ridge)
        if not math.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be a non-negative finite number")
        self.support_fraction = support_fraction
        self.assume_centered = bool(assume_centered)
        self.n_trials = n_trials
        self.max_c_steps = max_c_steps
        self.ridge = ridge
        self.seed = seed
        self.scores_: Optional[np.ndarray] = None
        self.location_: Optional[np.ndarray] = None
        self.covariance_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.support_: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    def fit(self, X: np.ndarray) -> "EllipticEnvelope":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n, p = X.shape
        if n < 1:
            raise ValueError("EllipticEnvelope needs at least one sample")
        self._n_features = p
        if self.assume_centered:
            mean = np.zeros(p, dtype=float)
            centered = X
            cov = (centered.T @ centered) / float(max(n, 1)) + self.ridge * np.eye(p)
            support = np.ones(n, dtype=bool)
        else:
            h = _mcd_h_support(n, p, self.support_fraction)
            mean, cov = _fast_mcd(
                X, h=h, n_trials=self.n_trials, seed=self.seed,
                ridge=self.ridge, max_c_steps=self.max_c_steps,
            )
            try:
                precision = np.linalg.inv(cov)
            except np.linalg.LinAlgError as exc:
                raise ValueError("MCD covariance is singular; increase ridge") from exc
            d2 = _mahalanobis2(X, mean, precision)
            support_idx = np.argpartition(d2, h - 1)[:h]
            support = np.zeros(n, dtype=bool)
            support[support_idx] = True
            # One final recompute on the concentrated subset
            mean, cov, _ = _mcd_covariance(X, support_idx, self.ridge)

        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("MCD covariance is singular; increase ridge") from exc
        self.location_ = np.asarray(mean, dtype=float)
        self.covariance_ = np.asarray(cov, dtype=float)
        self.precision_ = np.asarray(precision, dtype=float)
        self.support_ = support if not self.assume_centered else np.ones(n, dtype=bool)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.location_ is None or self.precision_ is None:
            raise ValueError("EllipticEnvelope must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but EllipticEnvelope was fitted on {self._n_features}"
            )
        n = len(X)
        if n == 0:
            self.scores_ = np.empty(0)
            return self.scores_
        d2 = _mahalanobis2(X, self.location_, self.precision_)
        self.scores_ = np.sqrt(np.maximum(d2, 0.0))
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)


def _kmeans(
    X: np.ndarray,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    tol: float,
    seed: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd k-means with multiple random reinits; returns (labels, centers)."""
    n, p = X.shape
    rng = np.random.default_rng(seed)
    best_inertia = math.inf
    best_labels = np.zeros(n, dtype=int)
    best_centers = np.zeros((n_clusters, p), dtype=float)

    for _ in range(n_init):
        # Forgy init: sample distinct rows when possible.
        if n >= n_clusters:
            idx = rng.choice(n, size=n_clusters, replace=False)
        else:
            idx = rng.integers(0, n, size=n_clusters)
        centers = X[idx].copy()
        labels = np.zeros(n, dtype=int)
        for _it in range(max_iter):
            # Assign
            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
            x2 = np.sum(X * X, axis=1, keepdims=True)
            c2 = np.sum(centers * centers, axis=1)
            d2 = np.maximum(x2 + c2 - 2.0 * (X @ centers.T), 0.0)
            new_labels = np.argmin(d2, axis=1).astype(int)
            # Update
            new_centers = centers.copy()
            for k in range(n_clusters):
                members = X[new_labels == k]
                if len(members) == 0:
                    # Re-seed empty cluster on a random point.
                    new_centers[k] = X[int(rng.integers(0, n))]
                else:
                    new_centers[k] = members.mean(axis=0)
            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            labels = new_labels
            if shift <= tol:
                break
        inertia = float(np.sum((X - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers


def _cblof_large_mask(
    sizes: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Decide which clusters are large given sorted-by-size heuristics.

    Clusters are processed from largest to smallest. A cut is placed at the
    first index where either (a) the cumulative size of the larger clusters
    already covers ``alpha`` of the data, or (b) the size ratio between the
    previous and current cluster exceeds ``beta``. Everything before the cut
    is large; the rest is small. At least one cluster is always large.
    """
    n_clusters = sizes.size
    order = np.argsort(sizes)[::-1]
    sorted_sizes = sizes[order]
    total = float(sorted_sizes.sum())
    if total <= 0.0:
        mask = np.zeros(n_clusters, dtype=bool)
        mask[order[0]] = True
        return mask

    cut = n_clusters  # all large by default
    cum = 0.0
    for i in range(n_clusters):
        cum += float(sorted_sizes[i])
        # After including cluster i as large, check whether we should stop.
        # Cut after i (so clusters 0..i inclusive stay large) when the alpha
        # mass is reached, or when the next cluster (if any) is beta-smaller.
        alpha_hit = cum >= alpha * total
        beta_hit = False
        if i + 1 < n_clusters and sorted_sizes[i + 1] > 0:
            beta_hit = (sorted_sizes[i] / sorted_sizes[i + 1]) >= beta
        if alpha_hit or beta_hit:
            cut = i + 1
            break
    cut = max(1, min(cut, n_clusters))
    large = np.zeros(n_clusters, dtype=bool)
    large[order[:cut]] = True
    return large


class CBLOF:
    """Cluster-Based Local Outlier Factor (He, Deng & Xu, 2003).

    ``fit`` runs Lloyd k-means, labels clusters as large or small with the
    alpha / beta heuristics used by PyOD, then scores each row by its
    Euclidean distance to a large-cluster centre:

    * points assigned to a **large** cluster use the distance to their own
      centre;
    * points assigned to a **small** cluster use the distance to the nearest
      large centre.

    When ``use_weights`` is true the distance is multiplied by the size of
    the point's own cluster, so outliers sitting in populous regions score
    higher. The contamination argument on ``predict`` / ``fit_predict``
    flags the top fraction of scores, matching the other detectors.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        alpha: float = 0.9,
        beta: float = 5.0,
        use_weights: bool = True,
        n_init: int = 10,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: Optional[int] = None,
    ) -> None:
        n_clusters = int(n_clusters)
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        alpha = float(alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        beta = float(beta)
        if not math.isfinite(beta) or beta < 1.0:
            raise ValueError("beta must be a finite number >= 1")
        n_init = int(n_init)
        if n_init < 1:
            raise ValueError("n_init must be at least 1")
        max_iter = int(max_iter)
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        tol = float(tol)
        if not math.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be a non-negative finite number")
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.use_weights = bool(use_weights)
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.scores_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.cluster_sizes_: Optional[np.ndarray] = None
        self.large_cluster_labels_: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    def fit(self, X: np.ndarray) -> "CBLOF":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n, p = X.shape
        if n < self.n_clusters:
            raise ValueError(
                f"CBLOF needs at least n_clusters={self.n_clusters} samples, got {n}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        self._n_features = p
        labels, centers = _kmeans(
            X,
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            seed=self.seed,
        )
        sizes = np.bincount(labels, minlength=self.n_clusters).astype(float)
        large_mask = _cblof_large_mask(sizes, self.alpha, self.beta)
        if not bool(large_mask.any()):
            large_mask[int(np.argmax(sizes))] = True
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.cluster_sizes_ = sizes
        self.large_cluster_labels_ = np.flatnonzero(large_mask)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if (
            self.cluster_centers_ is None
            or self.large_cluster_labels_ is None
            or self.cluster_sizes_ is None
            or self.labels_ is None
        ):
            raise ValueError("CBLOF must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but CBLOF was fitted on {self._n_features}"
            )
        n = len(X)
        if n == 0:
            self.scores_ = np.empty(0)
            return self.scores_

        centers = self.cluster_centers_
        large_idx = self.large_cluster_labels_
        large_centers = centers[large_idx]
        # Assign query rows to nearest of the fitted centres.
        x2 = np.sum(X * X, axis=1, keepdims=True)
        c2 = np.sum(centers * centers, axis=1)
        d2_all = np.maximum(x2 + c2 - 2.0 * (X @ centers.T), 0.0)
        assign = np.argmin(d2_all, axis=1)
        # Distance to own centre and to nearest large centre.
        own = np.sqrt(d2_all[np.arange(n), assign])
        lc2 = np.sum(large_centers * large_centers, axis=1)
        d2_large = np.maximum(x2 + lc2 - 2.0 * (X @ large_centers.T), 0.0)
        nearest_large = np.sqrt(np.min(d2_large, axis=1))

        large_set = set(int(i) for i in large_idx)
        scores = np.empty(n, dtype=float)
        for i in range(n):
            if int(assign[i]) in large_set:
                scores[i] = own[i]
            else:
                scores[i] = nearest_large[i]
            if self.use_weights:
                scores[i] *= float(self.cluster_sizes_[int(assign[i])])
        self.scores_ = scores
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)


def _loda_histogram_heights(
    values: np.ndarray,
    edges: np.ndarray,
    dens: np.ndarray,
    floor: float,
) -> np.ndarray:
    """Lookup histogram density for projected values.

    Points outside the training range inherit ``floor`` (rare).
    """
    n = values.size
    n_bins = dens.size
    heights = np.empty(n, dtype=float)
    in_range = (values >= edges[0]) & (values <= edges[-1])
    inds = np.searchsorted(edges, values, side="right") - 1
    inds = np.clip(inds, 0, n_bins - 1)
    heights[in_range] = dens[inds[in_range]]
    heights[~in_range] = floor
    return heights


class LODA:
    """Lightweight Online Detector of Anomalies (Pevný, 2016).

    Draws ``n_random_cuts`` sparse random 1-D projections of the data, fits
    an equal-width histogram on each projected axis, and scores a row as the
    mean of ``-log`` histogram densities across cuts. Sparse bins (and points
    outside the training range) score high, so higher scores are more
    anomalous. Pass ``seed`` for reproducible projection draws.
    """

    def __init__(
        self,
        n_bins: int = 10,
        n_random_cuts: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        n_bins = int(n_bins)
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        n_random_cuts = int(n_random_cuts)
        if n_random_cuts < 1:
            raise ValueError("n_random_cuts must be at least 1")
        self.n_bins = n_bins
        self.n_random_cuts = n_random_cuts
        self.seed = seed
        self.scores_: Optional[np.ndarray] = None
        self.projections_: Optional[np.ndarray] = None
        self._hist: Optional[np.ndarray] = None
        self._edges: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None
        self._floor = 1e-12

    def fit(self, X: np.ndarray) -> "LODA":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n, p = X.shape
        if n < 1:
            raise ValueError("LODA needs at least one sample")
        if p < 1:
            raise ValueError("LODA needs at least one feature")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        self._n_features = p
        rng = np.random.default_rng(self.seed)
        # Sparse random projections (~sqrt(p) nonzero ±1 entries per cut).
        n_nonzero = max(int(round(p ** 0.5)), 1)
        n_nonzero = min(n_nonzero, p)
        projections = np.zeros((self.n_random_cuts, p), dtype=float)
        for k in range(self.n_random_cuts):
            idx = rng.choice(p, size=n_nonzero, replace=False)
            signs = rng.choice(np.array([-1.0, 1.0]), size=n_nonzero)
            projections[k, idx] = signs
        self.projections_ = projections

        projected = X @ projections.T  # (n, n_cuts)
        hist = np.empty((self.n_random_cuts, self.n_bins), dtype=float)
        edges = np.empty((self.n_random_cuts, self.n_bins + 1), dtype=float)
        for k in range(self.n_random_cuts):
            col = projected[:, k]
            counts, col_edges = np.histogram(col, bins=self.n_bins)
            # Convert counts to probability densities (sum to 1).
            total = float(counts.sum())
            if total > 0.0:
                dens = counts.astype(float) / total
            else:
                dens = np.full(self.n_bins, 1.0 / self.n_bins)
            dens = np.maximum(dens, self._floor)
            hist[k] = dens
            edges[k] = col_edges
        self._hist = hist
        self._edges = edges
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.projections_ is None or self._hist is None or self._edges is None:
            raise ValueError("LODA must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but LODA was fitted on {self._n_features}"
            )
        n = len(X)
        if n == 0:
            self.scores_ = np.empty(0)
            return self.scores_
        projected = X @ self.projections_.T
        scores = np.zeros(n, dtype=float)
        for k in range(self.n_random_cuts):
            heights = _loda_histogram_heights(
                projected[:, k], self._edges[k], self._hist[k], self._floor
            )
            scores += -np.log(heights)
        scores /= float(self.n_random_cuts)
        self.scores_ = scores
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)


class ABOD:
    """Angle-Based Outlier Detection (Kriegel, Schubert & Zimek, 2008).

    Approximate ABOD: for each point, take its ``n_neighbors`` nearest
    neighbours (or all other points when ``n`` is small) and compute the
    variance of the weighted cosine

        ``<AB, AC> / (||AB||^2 · ||AC||^2)``

    over neighbour pairs ``(B, C)``. Outliers sit in directions with low
    angle variance (small ABOF). The returned score is ``-ABOF`` so that
    **higher scores are more anomalous**, matching the other detectors.

    ``fit`` stores the training rows. Scoring the training matrix excludes
    each point as its own neighbour. Pass ``seed`` only for API symmetry;
    the algorithm is deterministic given ``X``.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        n_neighbors = int(n_neighbors)
        if n_neighbors < 2:
            raise ValueError("n_neighbors must be at least 2")
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.scores_: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    def fit(self, X: np.ndarray) -> "ABOD":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n = len(X)
        if n < 3:
            raise ValueError("ABOD needs at least three samples")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        self._X_train = np.array(X, dtype=float, copy=True)
        self._n_features = X.shape[1]
        return self

    def _abof_rows(self, X: np.ndarray, ref: np.ndarray, exclude_self: bool) -> np.ndarray:
        n_query = len(X)
        n_ref = len(ref)
        if n_query == 0:
            return np.empty(0)
        max_k = n_ref - 1 if exclude_self else n_ref
        if max_k < 2:
            return np.zeros(n_query)
        k = min(self.n_neighbors, max_k)
        D = _euclidean_distances(X, ref)
        if exclude_self:
            np.fill_diagonal(D, np.inf)
        # k nearest neighbour indices per query
        nn_idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
        scores = np.empty(n_query, dtype=float)
        eps = 1e-12
        for i in range(n_query):
            nbrs = ref[nn_idx[i]]
            # Vectors from query point to each neighbour: shape (k, p)
            V = nbrs - X[i]
            # Squared norms
            sq = np.einsum("ij,ij->i", V, V)
            sq = np.maximum(sq, eps)
            # Weighted cosine weights: <Vb, Vc> / (||Vb||^2 · ||Vc||^2)
            # = (Vb · Vc) / (sq_b * sq_c)
            dots = V @ V.T
            inv_sq = 1.0 / sq
            W = dots * inv_sq[:, None] * inv_sq[None, :]
            # Collect off-diagonal pair weights (b < c)
            vals = W[np.triu_indices(k, k=1)]
            if vals.size < 2:
                scores[i] = 0.0
            else:
                abof = float(np.var(vals))
                scores[i] = -abof
        return scores

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._X_train is None:
            raise ValueError("ABOD must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but ABOD was fitted on {self._n_features}"
            )
        same_train = (
            X.shape == self._X_train.shape and np.array_equal(X, self._X_train)
        )
        self.scores_ = self._abof_rows(X, self._X_train, exclude_self=same_train)
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)


class COF:
    """Connectivity-based Outlier Factor (Tang, Chen, Fu & Cheung, 2002).

    For each point, a set-based nearest (SBN) path through its ``n_neighbors``
    nearest neighbours yields an average chaining distance (ACD). The COF
    score is the ratio of the point's ACD to the mean ACD of its neighbours.
    Points whose neighbourhood is loosely connected score high.

    ``fit`` stores the training rows. Scoring the training matrix excludes
    each point as its own neighbour. Higher scores are more anomalous.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        seed: Optional[int] = None,
    ) -> None:
        n_neighbors = int(n_neighbors)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.scores_: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    def fit(self, X: np.ndarray) -> "COF":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n = len(X)
        if n < 2:
            raise ValueError("COF needs at least two samples")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        self._X_train = np.array(X, dtype=float, copy=True)
        self._n_features = X.shape[1]
        return self

    def _acd_rows(
        self, X: np.ndarray, ref: np.ndarray, exclude_self: bool
    ) -> np.ndarray:
        """Average chaining distance for each query row against ``ref``."""
        n_query = len(X)
        n_ref = len(ref)
        if n_query == 0:
            return np.empty(0)
        max_k = n_ref - 1 if exclude_self else n_ref
        if max_k < 1:
            return np.zeros(n_query)
        k = min(self.n_neighbors, max_k)
        D = _euclidean_distances(X, ref)
        if exclude_self:
            np.fill_diagonal(D, np.inf)
        # k nearest neighbour indices per query
        nn_idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
        # Sort neighbours by distance for a stable start of the SBN path.
        nn_dists = np.take_along_axis(D, nn_idx, axis=1)
        order = np.argsort(nn_dists, axis=1)
        nn_idx = np.take_along_axis(nn_idx, order, axis=1)

        # Precompute pairwise distances among ref for SBN chaining.
        # For large n this is O(n^2); acceptable for the kit's demo sizes.
        ref_D = _euclidean_distances(ref, ref)
        np.fill_diagonal(ref_D, 0.0)

        acd = np.empty(n_query, dtype=float)
        # Geometric path weights: 2*(k-j+1) / (k*(k+1)) for edge j=1..k
        denom = float(k * (k + 1))
        weights = np.array(
            [2.0 * (k - j + 1) / denom for j in range(1, k + 1)], dtype=float
        )

        for i in range(n_query):
            nbrs = nn_idx[i].astype(np.int64)
            # Distances from query to each neighbour (for first hop)
            d_to_query = D[i, nbrs]
            # SBN path: start with nearest neighbour, then greedily add the
            # remaining neighbour closest to the current path set.
            remaining = list(range(k))
            path = []  # indices into nbrs
            # First point: nearest to query
            first = int(np.argmin(d_to_query))
            path.append(first)
            remaining.remove(first)
            edge_dists = [float(d_to_query[first])]

            # Distance from each neighbour to the query point (as set member 0)
            # and to other neighbours via ref_D.
            while remaining:
                best_r = remaining[0]
                best_d = float("inf")
                for r in remaining:
                    # Dist to query
                    d_min = float(d_to_query[r])
                    # Dist to any neighbour already on the path
                    for p in path:
                        d_min = min(d_min, float(ref_D[nbrs[r], nbrs[p]]))
                    if d_min < best_d:
                        best_d = d_min
                        best_r = r
                path.append(best_r)
                remaining.remove(best_r)
                edge_dists.append(best_d)

            acd[i] = float(np.dot(weights, edge_dists))
        return acd

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._X_train is None:
            raise ValueError("COF must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but COF was fitted on {self._n_features}"
            )
        same_train = (
            X.shape == self._X_train.shape and np.array_equal(X, self._X_train)
        )
        n = len(X)
        if n == 0:
            self.scores_ = np.empty(0)
            return self.scores_

        ref = self._X_train
        exclude_self = same_train
        max_k = len(ref) - 1 if exclude_self else len(ref)
        if max_k < 1:
            self.scores_ = np.ones(n)
            return self.scores_
        k = min(self.n_neighbors, max_k)

        acd = self._acd_rows(X, ref, exclude_self=exclude_self)

        # Neighbour indices for the COF ratio (same kNN as ACD).
        D = _euclidean_distances(X, ref)
        if exclude_self:
            np.fill_diagonal(D, np.inf)
        nn_idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]

        if same_train:
            # ACD of neighbours is just acd[nn_idx]
            nbr_acd = acd[nn_idx]
        else:
            # Score neighbours in the reference set
            ref_acd = self._acd_rows(ref, ref, exclude_self=True)
            nbr_acd = ref_acd[nn_idx]

        mean_nbr = nbr_acd.mean(axis=1)
        mean_nbr = np.where(mean_nbr > 1e-15, mean_nbr, 1e-15)
        self.scores_ = acd / mean_nbr
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)


class SOD:
    """Subspace Outlier Detection (Kriegel, Kröger, Schubert & Zimek, 2009).

    For each point, its ``n_neighbors`` nearest neighbours define a local
    reference set. Per-dimension variance among those neighbours marks a
    *relevant* subspace: dimensions whose variance is at most ``alpha`` times
    the mean variance. The SOD score is the Euclidean distance from the point
    to the neighbour mean in that subspace, normalised by
    ``sqrt(|relevant|)``. Points that deviate in a tight local subspace score
    high. Higher scores are more anomalous.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        alpha: float = 1.1,
        seed: Optional[int] = None,
    ) -> None:
        n_neighbors = int(n_neighbors)
        alpha = float(alpha)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if not (alpha > 0.0) or not np.isfinite(alpha):
            raise ValueError("alpha must be a positive finite float")
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.seed = seed
        self.scores_: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    def fit(self, X: np.ndarray) -> "SOD":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n = len(X)
        if n < 2:
            raise ValueError("SOD needs at least two samples")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        self._X_train = np.array(X, dtype=float, copy=True)
        self._n_features = X.shape[1]
        return self

    def _sod_scores(
        self, X: np.ndarray, ref: np.ndarray, exclude_self: bool
    ) -> np.ndarray:
        n_query = len(X)
        n_ref = len(ref)
        if n_query == 0:
            return np.empty(0)
        max_k = n_ref - 1 if exclude_self else n_ref
        if max_k < 1:
            return np.zeros(n_query)
        k = min(self.n_neighbors, max_k)
        D = _euclidean_distances(X, ref)
        if exclude_self:
            np.fill_diagonal(D, np.inf)
        nn_idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
        scores = np.empty(n_query, dtype=float)
        alpha = self.alpha
        for i in range(n_query):
            nbrs = ref[nn_idx[i]]
            # Per-dimension sample variance of the neighbourhood.
            if k == 1:
                var = np.zeros(ref.shape[1], dtype=float)
            else:
                var = nbrs.var(axis=0, ddof=0)
            mean_var = float(var.mean())
            if mean_var <= 1e-15:
                # Flat neighbourhood: fall back to full-space deviation.
                relevant = np.ones(ref.shape[1], dtype=bool)
            else:
                relevant = var <= (alpha * mean_var)
                if not np.any(relevant):
                    relevant = np.ones(ref.shape[1], dtype=bool)
            mu = nbrs[:, relevant].mean(axis=0)
            diff = X[i, relevant] - mu
            denom = math.sqrt(float(relevant.sum()))
            scores[i] = float(np.linalg.norm(diff) / denom) if denom > 0 else 0.0
        return scores

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._X_train is None:
            raise ValueError("SOD must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but SOD was fitted on {self._n_features}"
            )
        same_train = (
            X.shape == self._X_train.shape and np.array_equal(X, self._X_train)
        )
        self.scores_ = self._sod_scores(X, self._X_train, exclude_self=same_train)
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)

class PCA:
    """PCA reconstruction-error outlier detector.

    Fit centres the training matrix, computes a thin SVD, and keeps the
    leading ``n_components`` principal directions (an integer rank, or a
    float in ``(0, 1]`` interpreted as a cumulative explained-variance
    ratio). Anomaly scores are the squared Euclidean reconstruction error
    ``||x - x_hat||_2^2``; larger residuals are more anomalous.
    """

    def __init__(
        self,
        n_components: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        if n_components is not None:
            if isinstance(n_components, bool):
                raise ValueError("n_components must be an int or a float in (0, 1]")
            if isinstance(n_components, (int, np.integer)):
                if int(n_components) < 1:
                    raise ValueError("n_components must be a positive integer")
                n_components = int(n_components)
            else:
                n_components = float(n_components)
                if not (0.0 < n_components <= 1.0) or not np.isfinite(n_components):
                    raise ValueError(
                        "n_components float must be in (0, 1] (variance ratio)"
                    )
        self.n_components = n_components
        self.seed = seed
        self.scores_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None
        self._n_features: Optional[int] = None

    def _resolve_n_components(
        self, n_samples: int, n_features: int, singular_values: np.ndarray
    ) -> int:
        max_k = min(n_features, max(1, n_samples - 1), singular_values.size)
        req = self.n_components
        if req is None:
            return max_k
        if isinstance(req, (int, np.integer)):
            k = int(req)
            if k < 1:
                raise ValueError("n_components must be a positive integer")
            if k > max_k:
                raise ValueError(
                    f"n_components={k} exceeds max possible rank {max_k}"
                )
            return k
        # Variance-ratio float in (0, 1]
        total = float(np.sum(singular_values ** 2))
        if total <= 0.0:
            return 1
        ratios = (singular_values ** 2) / total
        cum = np.cumsum(ratios)
        k = int(np.searchsorted(cum, float(req), side="left") + 1)
        return min(max(k, 1), max_k)

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("PCA needs at least two samples")
        if n_features < 1:
            raise ValueError("PCA needs at least one feature")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        mean = X.mean(axis=0)
        Xc = X - mean
        # Economy SVD; Vt rows are principal directions.
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        k = self._resolve_n_components(n_samples, n_features, s)
        components = vt[:k]
        # Population-style variances along each component (match sklearn n_samples).
        ev = (s[:k] ** 2) / float(n_samples)
        total = float(np.sum(s ** 2)) / float(n_samples)
        evr = ev / total if total > 0.0 else np.zeros_like(ev)

        self.mean_ = mean
        self.components_ = components
        self.explained_variance_ = ev
        self.explained_variance_ratio_ = evr
        self.n_components_ = k
        self._n_features = n_features
        return self

    def _reconstruct(self, X: np.ndarray) -> np.ndarray:
        Xc = X - self.mean_
        # Project then lift: X_hat = (Xc @ V.T) @ V + mean
        codes = Xc @ self.components_.T
        return codes @ self.components_ + self.mean_

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but PCA was fitted on {self._n_features}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        X_hat = self._reconstruct(X)
        resid = X - X_hat
        self.scores_ = np.sum(resid * resid, axis=1)
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)


def _kde_bandwidth_factor(rule: str, n: int, d: int) -> float:
    """Scott / Silverman rule-of-thumb factor for a ``d``-dimensional KDE."""
    if rule == "scott":
        return float(n) ** (-1.0 / (d + 4))
    # Silverman: (n (d + 2) / 4) ** (-1 / (d + 4))
    return (float(n) * (d + 2) / 4.0) ** (-1.0 / (d + 4))


class KDE:
    """Gaussian kernel density estimation outlier detector.

    Fits an isotropic Gaussian KDE

        ``p(x) = 1 / (n h^d (2 pi)^{d/2}) * sum_i exp(-||x - x_i||^2 / (2 h^2))``

    on the training rows and scores each query by its **negative log
    density** ``-log p(x)``, so **higher scores are more anomalous**. The
    log-sum-exp trick keeps scores finite even far from the data.

    ``bandwidth`` is either ``"scott"`` (``n^{-1/(d+4)}``), ``"silverman"``
    (``(n (d+2) / 4)^{-1/(d+4)}``) or a positive float. When
    ``standardize=True`` (default) features are z-scored with the training
    mean/std first, so the rule-of-thumb bandwidths are scale-free; the
    density is still reported in the standardised space. Scoring the
    training matrix itself uses a leave-one-out estimate (each row is
    excluded from its own density) so training points are not flattered by
    their own kernel. ``seed`` is accepted only for API symmetry; the model
    is deterministic.
    """

    _RULES = ("scott", "silverman")

    def __init__(
        self,
        bandwidth="scott",
        standardize: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if isinstance(bandwidth, str):
            if bandwidth not in self._RULES:
                raise ValueError(
                    "bandwidth must be 'scott', 'silverman' or a positive float"
                )
        else:
            if isinstance(bandwidth, bool):
                raise ValueError(
                    "bandwidth must be 'scott', 'silverman' or a positive float"
                )
            bandwidth = float(bandwidth)
            if not np.isfinite(bandwidth) or bandwidth <= 0.0:
                raise ValueError("bandwidth must be a positive finite float")
        self.bandwidth = bandwidth
        self.standardize = bool(standardize)
        self.seed = seed
        self.scores_: Optional[np.ndarray] = None
        self.bandwidth_: Optional[float] = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None
        self._Z_train: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit(self, X: np.ndarray) -> "KDE":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n, d = X.shape
        if n < 2:
            raise ValueError("KDE needs at least two samples")
        if d < 1:
            raise ValueError("KDE needs at least one feature")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if self.standardize:
            mean = X.mean(axis=0)
            scale = X.std(axis=0)
            scale = np.where(scale > 0.0, scale, 1.0)
        else:
            mean = np.zeros(d)
            scale = np.ones(d)
        self.mean_ = mean
        self.scale_ = scale
        self._X_train = np.array(X, dtype=float, copy=True)
        self._Z_train = self._transform(self._X_train)
        self._n_features = d
        if isinstance(self.bandwidth, str):
            factor = _kde_bandwidth_factor(self.bandwidth, n, d)
            if self.standardize:
                h = factor
            else:
                # Use the mean per-feature std as the reference scale.
                h = factor * float(np.mean(X.std(axis=0)))
                if not h > 0.0:
                    h = factor
        else:
            h = float(self.bandwidth)
        self.bandwidth_ = float(h)
        return self

    def _neg_log_density(self, Z: np.ndarray, exclude_self: bool) -> np.ndarray:
        ref = self._Z_train
        n_ref, d = ref.shape
        h = self.bandwidth_
        sq = _euclidean_distances(Z, ref) ** 2
        logk = -sq / (2.0 * h * h)
        if exclude_self:
            np.fill_diagonal(logk, -np.inf)
            n_eff = n_ref - 1
        else:
            n_eff = n_ref
        mx = np.max(logk, axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.sum(np.exp(logk - mx), axis=1))
        log_norm = math.log(n_eff) + d * math.log(h) + 0.5 * d * math.log(2.0 * math.pi)
        return log_norm - lse

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._Z_train is None:
            raise ValueError("KDE must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but KDE was fitted on {self._n_features}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        same_train = (
            X.shape == self._X_train.shape and np.array_equal(X, self._X_train)
        )
        self.scores_ = self._neg_log_density(self._transform(X), exclude_self=same_train)
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)
