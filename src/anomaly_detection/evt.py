"""Extreme-value thresholds: peaks-over-threshold, SPOT and DSPOT.

A fixed contamination rate or a "3 sigma" cut-off assumes the user already
knows how many anomalies to expect. Extreme value theory gives a threshold
from a target *risk* ``q`` instead. The Pickands-Balkema-de Haan theorem says
excesses over a high initial threshold ``t`` follow a generalized Pareto
distribution (GPD). Fitting that GPD gives the ``1 - q`` quantile

    z_q = t + sigma/gamma * ((q n / N_t)^(-gamma) - 1)        (gamma != 0)
    z_q = t - sigma * log(q n / N_t)                           (gamma == 0)

which can lie far beyond the largest observed value.

* :func:`fit_gpd` -- GPD maximum likelihood with Grimshaw's (1993) one-dimensional
  reduction. Roots of ``u(t) v(t) = 1`` are bracketed on a grid and
  bisected, then the best candidate (including the exponential case) by
  log-likelihood is kept.
* :func:`pot` -- static peaks-over-threshold quantile of a sample.
* :class:`SPOT` / :class:`DSPOT` -- streaming POT (Siffer et al., KDD 2017).
  Normal peaks update the GPD and the threshold online. Values above ``z_q``
  raise alarms and are kept out of the model. DSPOT works on the deviation
  from a moving average of the last ``depth`` normal values, so it can follow
  drift.
* :func:`evt_flags` -- threshold any detector's anomaly scores with POT.

NumPy only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


def _as_1d(values, name: str = "values") -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim == 2 and 1 in x.shape:
        x = x.ravel()
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be finite")
    return x


def gpd_log_likelihood(excesses, gamma: float, sigma: float) -> float:
    """Log-likelihood of non-negative excesses under GPD(gamma, sigma)."""
    y = np.asarray(excesses, dtype=float)
    if sigma <= 0:
        return -np.inf
    if abs(gamma) < 1e-12:
        return float(-y.size * np.log(sigma) - y.sum() / sigma)
    z = 1.0 + gamma * y / sigma
    if np.any(z <= 0):
        return -np.inf
    return float(-y.size * np.log(sigma) - (1.0 + 1.0 / gamma) * np.log(z).sum())


def _grimshaw_w(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    s = 1.0 + np.outer(t, y)
    u = np.mean(1.0 / s, axis=1)
    v = 1.0 + np.mean(np.log(s), axis=1)
    return u * v - 1.0


def _bracket_roots(lo: float, hi: float, y: np.ndarray, n_grid: int, log_grid: bool) -> List[float]:
    if not lo < hi:
        return []
    if log_grid and lo > 0:
        grid = np.geomspace(lo, hi, n_grid)
    else:
        grid = np.linspace(lo, hi, n_grid)
    w = _grimshaw_w(grid, y)
    roots = []
    for i in np.flatnonzero(np.sign(w[:-1]) * np.sign(w[1:]) < 0):
        a, b = grid[i], grid[i + 1]
        wa = w[i]
        for _ in range(80):
            m = 0.5 * (a + b)
            wm = _grimshaw_w(np.array([m]), y)[0]
            if np.sign(wm) == np.sign(wa):
                a, wa = m, wm
            else:
                b = m
        roots.append(0.5 * (a + b))
    return roots


def fit_gpd(excesses, n_grid: int = 400) -> Tuple[float, float, float]:
    """Maximum-likelihood GPD fit (location 0) by Grimshaw's method.

    Returns ``(gamma, sigma, log_likelihood)``. ``gamma`` is the shape
    (tail index) and ``sigma`` the scale.
    """
    y = _as_1d(excesses, "excesses")
    if y.size < 2:
        raise ValueError("need at least two excesses")
    if np.any(y < 0):
        raise ValueError("excesses must be non-negative")
    ymax, ymin, ymean = float(y.max()), float(y.min()), float(y.mean())
    if ymax <= 0:
        raise ValueError("excesses must not all be zero")
    candidates = [(0.0, ymean)]
    eps = 1e-8 / ymax
    left = _bracket_roots(-1.0 / ymax + eps, -1e-6 / ymean, y, 4 * n_grid, False)
    # Grimshaw: a positive root (gamma > 0) lies below 2 (ymean - ymin) / ymin^2;
    # scan it on a log grid that starts well above the trivial root t = 0.
    hi = 2.0 * (ymean - ymin) / (ymin * ymin) if ymin > 0 else 1e8 / ymax
    pos = _bracket_roots(1e-6 / ymean, max(hi, 1e-5 / ymean), y, 4 * n_grid, True)
    for t in left + pos:
        gamma = float(np.mean(np.log1p(t * y)))
        if gamma != 0.0:
            candidates.append((gamma, gamma / t))
    best = max(candidates, key=lambda gs: gpd_log_likelihood(y, *gs))
    return best[0], best[1], gpd_log_likelihood(y, *best)


def gpd_quantile(t: float, gamma: float, sigma: float, n: int, n_peaks: int, q: float) -> float:
    """POT quantile ``z_q`` exceeded with probability ``q``."""
    r = q * n / n_peaks
    if abs(gamma) < 1e-12:
        return float(t - sigma * np.log(r))
    return float(t + (sigma / gamma) * (r ** (-gamma) - 1.0))


def _check_q_level(q: float, level: float) -> None:
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")


@dataclass
class POTResult:
    threshold: float
    initial_threshold: float
    gamma: float
    sigma: float
    n_peaks: int
    n: int


def pot(values, q: float = 1e-3, level: float = 0.98) -> POTResult:
    """Static peaks-over-threshold: the ``1 - q`` quantile from a GPD tail.

    ``level`` sets the initial threshold ``t`` as an empirical quantile. If
    ``q`` is not smaller than ``1 - level``, the empirical quantile is
    already enough. In that case the returned threshold is the empirical
    ``1 - q`` quantile.
    """
    x = _as_1d(values)
    _check_q_level(q, level)
    if x.size < 10:
        raise ValueError("need at least ten observations")
    t = float(np.quantile(x, level))
    peaks = x[x > t] - t
    if peaks.size < 2 or q >= 1.0 - level:
        return POTResult(float(np.quantile(x, 1.0 - q)), t, float("nan"), float("nan"), int(peaks.size), int(x.size))
    gamma, sigma, _ = fit_gpd(peaks)
    z = gpd_quantile(t, gamma, sigma, x.size, peaks.size, q)
    return POTResult(z, t, gamma, sigma, int(peaks.size), int(x.size))


def evt_flags(scores, q: float = 1e-2, level: float = 0.9) -> Tuple[np.ndarray, float]:
    """Flag anomaly scores above the POT threshold. Returns ``(flags, z_q)``."""
    s = _as_1d(scores, "scores")
    res = pot(s, q=q, level=level)
    return (s > res.threshold).astype(int), res.threshold


@dataclass
class SPOTResult:
    thresholds: np.ndarray
    alarms: List[int] = field(default_factory=list)
    upper_thresholds: Optional[np.ndarray] = None
    lower_thresholds: Optional[np.ndarray] = None


class _Tail:
    """One-sided streaming POT state for the upper tail of a stream."""

    def __init__(self, init: np.ndarray, q: float, level: float):
        self.q = q
        self.t = float(np.quantile(init, level))
        self.peaks = list(init[init > self.t] - self.t)
        self.n = int(init.size)
        if len(self.peaks) < 2:
            raise ValueError("calibration produced fewer than two peaks; lower level or add data")
        self._refit()

    def _refit(self):
        self.gamma, self.sigma, _ = fit_gpd(np.asarray(self.peaks))
        self.z = gpd_quantile(self.t, self.gamma, self.sigma, self.n, len(self.peaks), self.q)

    def step(self, x: float) -> bool:
        """Process one value. Returns True for an alarm (state unchanged)."""
        if x > self.z:
            return True
        self.n += 1
        if x > self.t:
            self.peaks.append(x - self.t)
            self._refit()
        return False


class SPOT:
    """Streaming peaks-over-threshold (Siffer et al., 2017).

    ``fit(init)`` calibrates on a clean warm-up sample. ``run(stream)``
    returns per-step thresholds and alarm indices. ``side`` is ``"upper"``,
    ``"lower"`` or ``"both"``.
    """

    def __init__(self, q: float = 1e-4, level: float = 0.98, side: str = "upper"):
        _check_q_level(q, level)
        if side not in ("upper", "lower", "both"):
            raise ValueError("side must be 'upper', 'lower' or 'both'")
        self.q, self.level, self.side = q, level, side
        self._up: Optional[_Tail] = None
        self._lo: Optional[_Tail] = None

    def fit(self, init) -> "SPOT":
        x = _as_1d(init, "init")
        if x.size < 10:
            raise ValueError("need at least ten calibration values")
        if self.side in ("upper", "both"):
            self._up = _Tail(x, self.q, self.level)
        if self.side in ("lower", "both"):
            self._lo = _Tail(-x, self.q, self.level)
        return self

    @property
    def upper_threshold(self) -> Optional[float]:
        return None if self._up is None else self._up.z

    @property
    def lower_threshold(self) -> Optional[float]:
        return None if self._lo is None else -self._lo.z

    def _check(self):
        if self._up is None and self._lo is None:
            raise RuntimeError("SPOT must be fit before run")

    def _step(self, x: float) -> bool:
        up = self._up is not None and x > self._up.z
        lo = self._lo is not None and -x > self._lo.z
        if up or lo:
            return True
        if self._up is not None:
            self._up.step(x)
        if self._lo is not None:
            self._lo.step(-x)
        return False

    def run(self, stream) -> SPOTResult:
        self._check()
        x = _as_1d(stream, "stream")
        ups, los, alarms = [], [], []
        for i, xi in enumerate(x):
            ups.append(self.upper_threshold if self._up is not None else np.nan)
            los.append(self.lower_threshold if self._lo is not None else np.nan)
            if self._step(float(xi)):
                alarms.append(i)
        ups, los = np.array(ups, float), np.array(los, float)
        main = ups if self.side != "lower" else los
        return SPOTResult(main, alarms, ups if self._up is not None else None, los if self._lo is not None else None)


class DSPOT(SPOT):
    """Drift-aware SPOT: SPOT on ``x_t - mean(last depth normal values)``."""

    def __init__(self, depth: int = 10, q: float = 1e-4, level: float = 0.98, side: str = "upper"):
        super().__init__(q=q, level=level, side=side)
        if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)) or depth < 1:
            raise ValueError("depth must be a positive integer")
        self.depth = int(depth)
        self._window: deque = deque(maxlen=self.depth)

    def fit(self, init) -> "DSPOT":
        x = _as_1d(init, "init")
        if x.size < self.depth + 10:
            raise ValueError("need at least depth + 10 calibration values")
        csum = np.concatenate([[0.0], np.cumsum(x)])
        ma = (csum[self.depth:-1] - csum[: -self.depth - 1]) / self.depth
        super().fit(x[self.depth:] - ma)
        self._window = deque(x[-self.depth:], maxlen=self.depth)
        return self

    def run(self, stream) -> SPOTResult:
        self._check()
        x = _as_1d(stream, "stream")
        ups, los, alarms = [], [], []
        for i, xi in enumerate(x):
            m = float(np.mean(self._window))
            ups.append(m + self.upper_threshold if self._up is not None else np.nan)
            los.append(m + self.lower_threshold if self._lo is not None else np.nan)
            if self._step(float(xi) - m):
                alarms.append(i)
            else:
                self._window.append(float(xi))
        ups, los = np.array(ups, float), np.array(los, float)
        main = ups if self.side != "lower" else los
        return SPOTResult(main, alarms, ups if self._up is not None else None, los if self._lo is not None else None)


__all__ = [
    "DSPOT",
    "POTResult",
    "SPOT",
    "SPOTResult",
    "evt_flags",
    "fit_gpd",
    "gpd_log_likelihood",
    "gpd_quantile",
    "pot",
]
