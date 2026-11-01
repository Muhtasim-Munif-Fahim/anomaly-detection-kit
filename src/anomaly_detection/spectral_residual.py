"""Spectral Residual (SR) saliency detector for univariate time series.

Implements the unsupervised part of Ren et al., "Time-Series Anomaly
Detection Service at Microsoft" (KDD 2019), which borrows the visual
saliency idea of Hou & Zhang (CVPR 2007):

1. Take the FFT of the series: amplitude ``A(f)`` and phase ``P(f)``.
2. Log amplitude ``L(f) = log A(f)``; its "expected" smooth part is a
   moving average ``h_q * L(f)``. The **spectral residual** is
   ``R(f) = L(f) - (h_q * L)(f)``.
3. Back to the time domain with the original phase:
   ``S(x) = |IFFT(exp(R(f) + i P(f)))|`` is the **saliency map**.
4. A point is anomalous when its saliency stands out from the local
   average of the preceding saliency values:
   ``score_i = (S_i - mean(S_{i-z..i-1})) / mean(S_{i-z..i-1})``.

Because the FFT treats the series as periodic, the most recent points are
poorly estimated. Like the paper, we append ``extend`` synthetic points
before the transform. They continue the series from its last value with the
average gradient between the last point and the ``n_grad`` preceding
points (``x_{n+1} = x_{n-m+1} + g * m``), and are dropped again afterwards.

Everything is pure numpy. Higher scores mean more anomalous, as for the
other detectors in this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

__all__ = [
    "SpectralResidualResult",
    "extend_series",
    "saliency_map",
    "spectral_residual",
    "spectral_residual_scores",
]

_EPS = 1e-8


def _as_series(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("values must be a 1-D series")
    if x.size < 4:
        raise ValueError("series must contain at least 4 points")
    if not np.all(np.isfinite(x)):
        raise ValueError("series must not contain NaN or inf")
    return x


def _check_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _moving_average(a: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge-replicated padding (same length)."""
    if window <= 1:
        return a.copy()
    left = (window - 1) // 2
    right = window - 1 - left
    padded = np.concatenate([np.full(left, a[0]), a, np.full(right, a[-1])])
    csum = np.concatenate([[0.0], np.cumsum(padded)])
    return (csum[window:] - csum[:-window]) / window


def _trailing_mean(a: np.ndarray, window: int) -> np.ndarray:
    """Mean of the ``window`` values strictly before each index.

    The first element (no history) uses itself so its score is 0.
    """
    n = a.size
    csum = np.concatenate([[0.0], np.cumsum(a)])
    idx = np.arange(n)
    lo = np.maximum(idx - window, 0)
    counts = idx - lo
    out = np.empty(n)
    has = counts > 0
    out[has] = (csum[idx[has]] - csum[lo[has]]) / counts[has]
    out[~has] = a[~has]
    return out


def extend_series(values, extend: int = 5, n_grad: int = 5) -> np.ndarray:
    """Append ``extend`` points that continue the series past its end.

    With ``m = n_grad`` (capped at ``n - 1``) the extension value is
    ``x[n-m] + g * m`` where ``g`` is the mean of the gradients
    ``(x[n-1] - x[n-1-k]) / k`` for ``k = 1..m`` (the estimator used by Ren et
    al. 2019). All appended points share that value.
    """
    x = _as_series(values)
    if isinstance(extend, bool) or not isinstance(extend, (int, np.integer)) or extend < 0:
        raise ValueError("extend must be a non-negative integer")
    n_grad = _check_positive_int(n_grad, "n_grad")
    if extend == 0:
        return x.copy()
    m = min(n_grad, x.size - 1)
    last = x[-1]
    ks = np.arange(1, m + 1)
    grads = (last - x[-1 - ks]) / ks
    value = x[-m] + float(np.mean(grads)) * m
    return np.concatenate([x, np.full(int(extend), value)])


def saliency_map(values, amp_window: int = 3) -> np.ndarray:
    """Spectral-residual saliency map of a series (same length as ``values``)."""
    x = _as_series(values)
    q = _check_positive_int(amp_window, "amp_window")
    spectrum = np.fft.fft(x)
    amplitude = np.abs(spectrum)
    log_amp = np.log(amplitude + _EPS)
    residual = log_amp - _moving_average(log_amp, q)
    phase = np.angle(spectrum)
    return np.abs(np.fft.ifft(np.exp(residual + 1j * phase)))


def spectral_residual_scores(
    values,
    amp_window: int = 3,
    score_window: int = 21,
    extend: int = 5,
    n_grad: int = 5,
) -> np.ndarray:
    """Per-point SR anomaly score ``(S_i - local mean) / local mean``.

    ``score_window`` is the number of preceding saliency values averaged to
    form the local reference. Scores are non-negative after clipping below
    at 0 (points *less* salient than their neighbourhood are not anomalies).
    """
    x = _as_series(values)
    z = _check_positive_int(score_window, "score_window")
    ext = extend_series(x, extend=extend, n_grad=n_grad)
    sal = saliency_map(ext, amp_window=amp_window)[: x.size]
    ref = _trailing_mean(sal, z)
    score = (sal - ref) / np.maximum(ref, _EPS)
    return np.maximum(score, 0.0)


@dataclass
class SpectralResidualResult:
    """Output of :func:`spectral_residual`."""

    saliency: np.ndarray
    scores: np.ndarray
    flags: np.ndarray
    threshold: float

    @property
    def anomalies(self) -> List[int]:
        return [int(i) for i in np.flatnonzero(self.flags)]


def spectral_residual(
    values,
    threshold: float = 3.0,
    amp_window: int = 3,
    score_window: int = 21,
    extend: int = 5,
    n_grad: int = 5,
) -> SpectralResidualResult:
    """Run the SR detector and flag points whose score exceeds ``threshold``."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a positive finite number")
    x = _as_series(values)
    ext = extend_series(x, extend=extend, n_grad=n_grad)
    sal = saliency_map(ext, amp_window=amp_window)[: x.size]
    scores = spectral_residual_scores(
        x, amp_window=amp_window, score_window=score_window, extend=extend, n_grad=n_grad
    )
    return SpectralResidualResult(
        saliency=sal, scores=scores, flags=scores > threshold, threshold=float(threshold)
    )
