"""LOCI: Local Correlation Integral outlier detection.

Papadimitriou, Kitagawa, Gibbons & Faloutsos (ICDE 2003). For a point
``p``, a sampling radius ``r`` and a counting fraction ``alpha`` (default
0.5) define

* ``n(p, alpha*r)`` -- number of points within ``alpha*r`` of ``p``;
* ``N(p, r)`` -- the *sampling neighbourhood*, points within ``r`` of ``p``;
* ``n_hat(p, r, alpha)`` -- the mean of ``n(q, alpha*r)`` over ``q`` in
  ``N(p, r)`` and ``sigma_n_hat`` its standard deviation.

The multi-granularity deviation factor is
``MDEF = 1 - n(p, alpha*r) / n_hat`` with normalised deviation
``sigma_MDEF = sigma_n_hat / n_hat``. A point is flagged when
``MDEF > k_sigma * sigma_MDEF`` at any radius where the sampling
neighbourhood holds at least ``n_min`` points; the paper's default
``k_sigma = 3`` gives an automatic, data-driven cut-off (no contamination
rate needed). The anomaly score is ``max_r MDEF / sigma_MDEF``.

This implementation evaluates a shared geometric grid of radii, so each
radius costs ``O(n^2)`` via matrix products instead of per-point
neighbourhood scans.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .models import _euclidean_distances, _flags_from_contamination


class LOCI:
    """Local Correlation Integral (LOCI) outlier detector.

    Parameters
    ----------
    alpha : float
        Counting-to-sampling radius ratio in ``(0, 1)``.
    k_sigma : float
        Number of normalised deviations used by :meth:`flag` (``3`` in the
        paper).
    n_min : int
        Minimum sampling-neighbourhood size for a radius to count.
    n_radii : int
        Number of geometric radii between the smallest radius with ``n_min``
        neighbours and ``r_max = diameter / (2 * alpha)`` (the paper's
        ``alpha^-1 R_P``). Out-of-sample queries far from the data extend the
        grid up to their largest training distance.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        k_sigma: float = 3.0,
        n_min: int = 20,
        n_radii: int = 30,
    ) -> None:
        alpha = float(alpha)
        k_sigma = float(k_sigma)
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie strictly between 0 and 1")
        if not (k_sigma > 0.0) or not np.isfinite(k_sigma):
            raise ValueError("k_sigma must be a positive finite float")
        if isinstance(n_min, bool) or int(n_min) != n_min or n_min < 2:
            raise ValueError("n_min must be an integer >= 2")
        if isinstance(n_radii, bool) or int(n_radii) != n_radii or n_radii < 1:
            raise ValueError("n_radii must be a positive integer")
        self.alpha = alpha
        self.k_sigma = k_sigma
        self.n_min = int(n_min)
        self.n_radii = int(n_radii)
        self.radii_: Optional[np.ndarray] = None
        self.scores_: Optional[np.ndarray] = None
        self.mdef_: Optional[np.ndarray] = None
        self.sigma_mdef_: Optional[np.ndarray] = None
        self.critical_radius_: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None
        self._D_train: Optional[np.ndarray] = None

    @staticmethod
    def _check(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def fit(self, X: np.ndarray) -> "LOCI":
        X = self._check(X)
        n = len(X)
        if n < 3:
            raise ValueError("LOCI needs at least three samples")
        D = _euclidean_distances(X, X)
        np.fill_diagonal(D, 0.0)
        n_min = min(self.n_min, n)
        sorted_d = np.sort(D, axis=1)
        r_lo = float(sorted_d[:, n_min - 1].min())
        # Paper: r_max ~ alpha^-1 * R_P with R_P the point-set radius (~diameter / 2).
        r_hi = float(D.max()) / (2.0 * self.alpha)
        if r_hi <= 0.0:
            radii = np.array([1.0])
        else:
            r_lo = max(r_lo, r_hi * 1e-6)
            radii = np.geomspace(r_lo, r_hi, self.n_radii) if r_hi > r_lo else np.array([r_hi])
        self.radii_ = radii
        self._X_train = np.array(X, copy=True)
        self._D_train = D
        self._n_min_eff = n_min
        return self

    def _mdef(self, Dq: np.ndarray, query_is_train: bool, radii: Optional[np.ndarray] = None):
        D = self._D_train
        radii = self.radii_ if radii is None else radii
        n_q = Dq.shape[0]
        R = radii.size
        mdef = np.zeros((n_q, R))
        sigma = np.zeros((n_q, R))
        valid = np.zeros((n_q, R), dtype=bool)
        for j, r in enumerate(radii):
            ar = self.alpha * r
            c_train = (D <= ar).sum(axis=1).astype(float)  # includes self
            M = (Dq <= r).astype(float)
            if query_is_train:
                own = c_train
                size = M.sum(axis=1)
                s1 = M @ c_train
                s2 = M @ (c_train**2)
            else:
                own = (Dq <= ar).sum(axis=1).astype(float) + 1.0
                size = M.sum(axis=1) + 1.0
                s1 = M @ c_train + own
                s2 = M @ (c_train**2) + own**2
            n_hat = s1 / size
            var = np.maximum(s2 / size - n_hat**2, 0.0)
            mdef[:, j] = 1.0 - own / n_hat
            sigma[:, j] = np.sqrt(var) / n_hat
            valid[:, j] = size >= self._n_min_eff
        return mdef, sigma, valid

    def _score_from(self, mdef, sigma, valid):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sigma > 1e-12, mdef / sigma, 0.0)
        ratio = np.where(valid, ratio, -np.inf)
        best = np.argmax(ratio, axis=1)
        rows = np.arange(len(ratio))
        score = ratio[rows, best]
        score = np.where(np.isfinite(score), score, 0.0)
        return score, best, rows

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return ``max_r MDEF / sigma_MDEF`` (higher is more anomalous)."""
        if self._X_train is None:
            raise ValueError("LOCI must be fitted before scoring")
        X = self._check(X)
        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but LOCI was fitted on {self._X_train.shape[1]}"
            )
        same = X.shape == self._X_train.shape and np.array_equal(X, self._X_train)
        Dq = self._D_train if same else _euclidean_distances(X, self._X_train)
        radii = self.radii_
        far = float(Dq.max()) if Dq.size else 0.0
        if not same and far > radii[-1]:
            radii = np.concatenate([radii, np.geomspace(radii[-1], far, 6)[1:]])
        mdef, sigma, valid = self._mdef(Dq, same, radii)
        score, best, rows = self._score_from(mdef, sigma, valid)
        self.mdef_ = mdef[rows, best]
        self.sigma_mdef_ = sigma[rows, best]
        self.critical_radius_ = radii[best]
        self.scores_ = score
        return score

    def flag(self, X: np.ndarray) -> np.ndarray:
        """LOCI's automatic rule: flag ``MDEF > k_sigma * sigma_MDEF``."""
        return (self.score_samples(X) > self.k_sigma).astype(int)

    def predict(self, X: np.ndarray, contamination: Optional[float] = 0.1) -> np.ndarray:
        """Top-``contamination`` flags, or the k-sigma rule when ``None``."""
        if contamination is None:
            return self.flag(X)
        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X: np.ndarray, contamination: Optional[float] = 0.1) -> np.ndarray:
        return self.fit(X).predict(X, contamination=contamination)
