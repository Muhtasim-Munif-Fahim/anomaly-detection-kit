"""Classical statistical outlier detection methods.

Pure-numpy implementations of the standard baselines:

* z-score (per-column, combined with the max over columns),
* modified z-score based on the median absolute deviation,
* Tukey IQR fences,
* the generalized ESD test for univariate time series.

GESD needs Student-t critical values; the required regularised incomplete
beta function and its inverse are implemented in this module (``_ibeta``,
``_ibeta_inv``, ``_t_inv``) so the whole module depends only on numpy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

MAD_SCALE = 0.6745


def _as_matrix(X: np.ndarray) -> np.ndarray:
    """Return a 2-D array with rows as observations."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim == 2:
        return X
    raise ValueError("X must be 1-D or 2-D")


# ---------------------------------------------------------------------------
# z-score
# ---------------------------------------------------------------------------

def zscore_scores(X: np.ndarray) -> np.ndarray:
    """Per-row anomaly score as the max absolute z-score across columns."""
    X = _as_matrix(X)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    z = np.abs((X - mean) / std)
    return z.max(axis=1)


def zscore_flags(X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Flag rows whose max absolute z-score strictly exceeds ``threshold``."""
    return (zscore_scores(X) > threshold).astype(int)


# ---------------------------------------------------------------------------
# modified z-score (median / MAD)
# ---------------------------------------------------------------------------

def mad_scores(X: np.ndarray) -> np.ndarray:
    """Robust z-score using the median and median absolute deviation.

    Columns whose MAD collapses to zero fall back to the root-mean-square
    deviation, which keeps constant columns from producing ``NaN``.
    """
    X = _as_matrix(X)
    median = np.median(X, axis=0)
    dev = np.abs(X - median)
    mad = np.median(dev, axis=0)
    fallback = np.sqrt(np.mean(dev * dev, axis=0))
    scale = np.where(mad > 0.0, mad, fallback)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (MAD_SCALE * dev / scale).max(axis=1)


def mad_flags(X: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Flag rows whose modified z-score strictly exceeds ``threshold``."""
    return (mad_scores(X) > threshold).astype(int)


# ---------------------------------------------------------------------------
# Tukey IQR fences
# ---------------------------------------------------------------------------

def iqr_scores(X: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Per-row score: how far the row sits beyond the IQR fences.

    Points inside the fences score zero; points outside score the excess
    distance scaled by the IQR, combined across columns with the maximum.
    """
    X = _as_matrix(X)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    iqr = np.where(iqr == 0.0, 1.0, iqr)
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    above = (X - upper) / iqr
    below = (lower - X) / iqr
    score = np.maximum(np.maximum(above, below), 0.0)
    return score.max(axis=1)


def iqr_flags(X: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Flag rows that fall strictly outside the IQR fences."""
    return (iqr_scores(X, k=k) > 0.0).astype(int)


# ---------------------------------------------------------------------------
# special functions for the generalized ESD critical values
# ---------------------------------------------------------------------------

def _gammaln(z: float) -> float:
    """Log-gamma via the Lanczos approximation (g = 7, n = 9)."""
    if z <= 0.0:
        raise ValueError("z must be positive")
    cof = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    if z < 0.5:
        return np.log(np.pi) - np.log(np.sin(np.pi * z)) - _gammaln(1.0 - z)
    t = z - 1.0
    x = cof[0]
    for i in range(1, len(cof)):
        x += cof[i] / (t + i)
    g = t + 7.5
    return 0.5 * np.log(2.0 * np.pi) + (t + 0.5) * np.log(g) - g + np.log(x)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _ibeta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lnbt = (
        _gammaln(a + b)
        - _gammaln(a)
        - _gammaln(b)
        + a * np.log(x)
        + b * np.log1p(-x)
    )
    bt = np.exp(lnbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _ibeta_inv(y: float, a: float, b: float) -> float:
    """Inverse regularised incomplete beta (bisection on the monotone CDF)."""
    if y <= 0.0:
        return 0.0
    if y >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _ibeta(mid, a, b) < y:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0e-15:
            break
    return 0.5 * (lo + hi)


def _t_inv(p: float, df: int) -> float:
    """Student-t quantile: value t such that P(T <= t) = p."""
    if df <= 0:
        raise ValueError("df must be positive")
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    if p == 0.5:
        return 0.0
    q = 2.0 * min(p, 1.0 - p)
    u = _ibeta_inv(q, df / 2.0, 0.5)
    t = np.sqrt(df * (1.0 - u) / u)
    return -t if p < 0.5 else t


# ---------------------------------------------------------------------------
# generalized ESD
# ---------------------------------------------------------------------------

def gesd(
    values: np.ndarray,
    k: int = 5,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, List[int], List[float], List[float]]:
    """Generalized ESD test for up to ``k`` outliers in a univariate series.

    Iteratively removes the point with the largest standardised deviation,
    compares it against the Student-t critical value, and reports the points
    whose ``R`` statistic exceeds the bound. Returns
    ``(flags, anomaly_indices, R_stats, critical_values)``.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    empty: Tuple[np.ndarray, List[int], List[float], List[float]] = (
        np.zeros(n, dtype=int),
        [],
        [],
        [],
    )
    if n < 3:
        return empty
    k = max(1, int(k))
    k = min(k, n - 2)
    R: List[float] = []
    lambdas: List[float] = []
    anomalies: List[int] = []
    remaining = x.copy()
    original_idx = np.arange(n)
    for i in range(1, k + 1):
        mean = remaining.mean()
        std = remaining.std(ddof=1)
        if std <= 1.0e-12:
            break
        dev = np.abs(remaining - mean)
        r = float(dev.max() / std)
        pos = int(np.argmax(dev))
        p = 1.0 - alpha / (2.0 * (n - i + 1))
        df = n - i - 1
        t = _t_inv(p, df)
        lam = t * (n - i) / np.sqrt((df + t * t) * (n - i + 1))
        R.append(r)
        lambdas.append(float(lam))
        if r > lam:
            anomalies.append(int(original_idx[pos]))
        remaining = np.delete(remaining, pos)
        original_idx = np.delete(original_idx, pos)
    flags = np.zeros(n, dtype=int)
    flags[anomalies] = 1
    return flags, anomalies, R, lambdas


def gesd_indices(
    values: np.ndarray,
    k: int = 5,
    alpha: float = 0.05,
) -> List[int]:
    """Return the anomaly indices declared by the generalized ESD test."""
    _, anomalies, _, _ = gesd(values, k=k, alpha=alpha)
    return anomalies


# ---------------------------------------------------------------------------
# registry for the evaluation module
# ---------------------------------------------------------------------------

STATISTICAL_SCORERS: Dict[str, object] = {
    "z-score": zscore_scores,
    "modified z-score": mad_scores,
    "IQR": iqr_scores,
}
