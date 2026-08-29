"""Model-based anomaly detectors.

Two dependency-free models:

* :class:`IsolationForest` -- random feature + random split-point trees with
  path-length anomaly scoring (higher score = more anomalous).
* :class:`LocalOutlierFactor` -- kNN reachability-density ratio, computed by
  brute force on pairwise distances (higher score = more anomalous).

Both expose the same interface: ``fit``, ``score_samples(X)`` returning
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
