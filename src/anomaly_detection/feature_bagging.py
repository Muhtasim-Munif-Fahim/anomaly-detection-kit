"""Feature bagging outlier ensembles (Lazarevic & Kumar, 2005).

Distance- and density-based detectors such as LOF lose contrast in high
dimensions, and irrelevant noise features hide anomalies that live in a
few attributes. Feature bagging trains ``n_estimators`` copies of a base
detector, each on a random subset of between ``d/2`` and ``d - 1``
features, and combines their (standardised) outlier scores.

Combination rules:

* ``"average"`` -- mean of z-scored detector outputs (the cumulative-sum
  rule of the paper, up to a constant factor);
* ``"max"`` -- maximum z-score across detectors;
* ``"breadth_first"`` -- the paper's breadth-first rule: points are
  ranked by their *best* rank in any detector (the top point of every
  detector first, then every detector's second point, ...). Ties are broken
  by the mean rank.

Reference: A. Lazarevic and V. Kumar, "Feature bagging for outlier
detection", KDD 2005.
"""

from __future__ import annotations

import copy
from typing import Callable, List, Optional, Union

import numpy as np

from .models import LocalOutlierFactor, _flags_from_contamination

__all__ = ["FeatureBagging"]

_COMBINATIONS = ("average", "max", "breadth_first")


class FeatureBagging:
    """Feature-bagging ensemble around any detector in this package.

    Parameters
    ----------
    base_estimator : detector instance or zero-argument callable, optional
        Template detector exposing ``fit`` and ``score_samples``. An
        instance is deep-copied for every member; a callable is called to
        build each member. Defaults to ``LocalOutlierFactor(n_neighbors=10)``
        as in the original paper.
    n_estimators : int
        Number of ensemble members.
    max_features : int, float or None
        Features per member. ``None`` draws a size uniformly from
        ``[ceil(d/2), d-1]`` for every member (the paper's rule). An int
        fixes the count; a float in (0, 1] is a fraction of ``d``.
    combination : {"average", "max", "breadth_first"}
        How member scores are combined.
    bootstrap_samples : bool
        Also bootstrap rows for each member (rows are fitted on a resample
        but always scored on the full input).
    seed : int, optional
        Seed for feature subsets and for member detectors that have a
        ``seed`` attribute left at ``None``.
    """

    def __init__(
        self,
        base_estimator: Optional[Union[object, Callable[[], object]]] = None,
        n_estimators: int = 10,
        max_features: Optional[Union[int, float]] = None,
        combination: str = "average",
        bootstrap_samples: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be at least 1")
        if combination not in _COMBINATIONS:
            raise ValueError(
                f"unknown combination {combination!r}; use one of {_COMBINATIONS}"
            )
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.combination = combination
        self.bootstrap_samples = bool(bootstrap_samples)
        self.seed = seed
        self.estimators_: List[object] = []
        self.features_: List[np.ndarray] = []
        self.train_scores_: Optional[np.ndarray] = None
        self.scores_: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    # ------------------------------------------------------------------ utils
    def _make_member(self, member_seed: int):
        base = self.base_estimator
        if base is None:
            est = LocalOutlierFactor(n_neighbors=10)
        elif callable(base) and not hasattr(base, "score_samples"):
            est = base()
        else:
            est = copy.deepcopy(base)
        if not (hasattr(est, "fit") and hasattr(est, "score_samples")):
            raise TypeError("base_estimator must provide fit and score_samples")
        if hasattr(est, "seed") and getattr(est, "seed") is None:
            est.seed = member_seed
        return est

    def _subset_size(self, d: int, rng: np.random.Generator) -> int:
        mf = self.max_features
        if mf is None:
            lo = int(np.ceil(d / 2))
            hi = max(lo, d - 1)
            return int(rng.integers(lo, hi + 1))
        if isinstance(mf, (float, np.floating)):
            if not 0.0 < float(mf) <= 1.0:
                raise ValueError("fractional max_features must be in (0, 1]")
            return max(1, int(round(float(mf) * d)))
        k = int(mf)
        if not 1 <= k <= d:
            raise ValueError(f"max_features must be between 1 and {d}")
        return k

    @staticmethod
    def _check_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        return X

    # -------------------------------------------------------------------- API
    def fit(self, X: np.ndarray) -> "FeatureBagging":
        X = self._check_X(X)
        n, d = X.shape
        if n < 2:
            raise ValueError("FeatureBagging needs at least two samples")
        rng = np.random.default_rng(self.seed)
        self._n_features = d
        self.estimators_ = []
        self.features_ = []
        member_scores = []
        for _ in range(self.n_estimators):
            k = self._subset_size(d, rng)
            feats = np.sort(rng.choice(d, size=k, replace=False))
            est = self._make_member(int(rng.integers(0, 2**31 - 1)))
            rows = rng.integers(0, n, size=n) if self.bootstrap_samples else slice(None)
            est.fit(X[rows][:, feats])
            member_scores.append(np.asarray(est.score_samples(X[:, feats]), dtype=float))
            self.estimators_.append(est)
            self.features_.append(feats)
        self.train_scores_ = np.column_stack(member_scores)
        return self

    def member_scores(self, X: np.ndarray) -> np.ndarray:
        """Raw scores of every member, shape ``(n_samples, n_estimators)``."""
        if not self.estimators_:
            raise ValueError("FeatureBagging must be fitted before scoring")
        X = self._check_X(X)
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features but the ensemble was fitted on {self._n_features}"
            )
        return np.column_stack(
            [
                np.asarray(est.score_samples(X[:, f]), dtype=float)
                for est, f in zip(self.estimators_, self.features_)
            ]
        )

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        S = self.member_scores(X)
        T = self.train_scores_
        if self.combination == "breadth_first":
            n = S.shape[0]
            # rank 0 = most anomalous within each member
            ranks = np.argsort(np.argsort(-S, axis=0, kind="stable"), axis=0)
            best = ranks.min(axis=1).astype(float)
            mean = ranks.mean(axis=1) / max(n, 1)
            self.scores_ = (n - best) - mean
            return self.scores_
        mu = T.mean(axis=0)
        sd = T.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Z = (S - mu) / sd
        self.scores_ = Z.mean(axis=1) if self.combination == "average" else Z.max(axis=1)
        return self.scores_

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)
