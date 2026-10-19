"""Change-point detection for univariate series: BOCPD and two-sided CUSUM.

Point detectors (z-score, GESD, S-H-ESD) and shape detectors (matrix-profile
discords) answer "which values look odd?".  Level shifts are different: every
value after the shift looks ordinary on its own, but the *regime* changed.
This module adds two change-point methods.

* :func:`bocpd` -- Bayesian online change-point detection (Adams & MacKay,
  2007) with a Normal-Gamma conjugate model (unknown mean and variance) and a
  constant hazard.  It tracks the posterior over the current run length and
  returns, per time step, the predictive surprise ``-log p(x_t | x_{<t})``, the
  MAP run length, and a lagged change-point probability
  ``P(run started at s | x_{<= s + lag})``.
* :func:`cusum` -- Page's two-sided CUSUM on robustly standardised values with
  reference value ``k`` and decision interval ``h``; each alarm also reports
  the estimated change start (the last time the statistic was zero).

All code is numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import lgamma, log, pi
from typing import List, Optional, Tuple

import numpy as np

_LGAMMA = np.vectorize(lgamma, otypes=[float])


def _validate_series(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 2 and 1 in values.shape:
        values = values.ravel()
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional series")
    if values.size < 2:
        raise ValueError("need at least two observations")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    return values


def _robust_location_scale(values: np.ndarray) -> Tuple[float, float]:
    center = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - center)))
    if scale <= 0.0:
        scale = float(np.std(values))
    if scale <= 0.0:
        scale = 1.0
    return center, scale


def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    if not np.isfinite(m):
        return m
    return m + log(float(np.sum(np.exp(a - m))))


def _student_t_logpdf(x: float, mu, kappa, alpha, beta) -> np.ndarray:
    """Normal-Gamma posterior predictive: Student-t(2*alpha, mu, beta(kappa+1)/(alpha kappa))."""
    df = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    z2 = (x - mu) ** 2 / scale2
    return (
        _LGAMMA((df + 1.0) / 2.0)
        - _LGAMMA(df / 2.0)
        - 0.5 * np.log(df * pi * scale2)
        - (df + 1.0) / 2.0 * np.log1p(z2 / df)
    )


@dataclass
class BOCPDResult:
    """Output of :func:`bocpd`."""

    surprise: np.ndarray
    map_run_length: np.ndarray
    expected_run_length: np.ndarray
    changepoint_probability: np.ndarray
    changepoint_score: np.ndarray
    log_evidence: float
    hazard: float
    lag: int
    changepoints: List[int] = field(default_factory=list)

    def segment_starts(self) -> np.ndarray:
        """Start index of the MAP current run at every time step."""
        t = np.arange(self.map_run_length.shape[0])
        return t - self.map_run_length + 1


def bocpd(
    values: np.ndarray,
    hazard: float = 1.0 / 250.0,
    mu0: Optional[float] = None,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: Optional[float] = None,
    lag: int = 10,
    threshold: float = 0.5,
    radius: int = 2,
    min_distance: Optional[int] = None,
    max_run_length: Optional[int] = None,
) -> BOCPDResult:
    """Bayesian online change-point detection with a Normal-Gamma model.

    Run length ``r_t`` counts the observations of the current regime
    *including* ``x_t`` (so the regime started at ``t - r_t + 1``).  The
    recursion is

        P(r_t = 1,     x_{<=t}) = H * pi_0(x_t) * sum_r P(r_{t-1} = r, x_{<t})
        P(r_t = r + 1, x_{<=t}) = (1 - H) * pi_r(x_t) * P(r_{t-1} = r, x_{<t})

    with ``pi_r`` the Student-t predictive of a regime that has absorbed the
    last ``r`` values and ``pi_0`` the prior predictive.

    Parameters
    ----------
    hazard:
        Constant change probability per step (expected regime length ``1/H``).
    mu0, kappa0, alpha0, beta0:
        Normal-Gamma prior.  ``mu0`` defaults to the series median and
        ``beta0`` to ``alpha0 * s^2`` with ``s`` the MAD-based scale, so the
        prior noise level matches the data.
    lag:
        ``changepoint_probability[s] = P(r_{s+lag} = lag + 1 | x_{<=s+lag})``,
        the posterior that a regime began at ``s`` once ``lag`` more points are
        seen.  The last ``lag`` entries use whatever data is available.
    threshold, radius, min_distance:
        The exact start of a regime is often uncertain by a step or two, so
        the posterior mass is spread over neighbouring ``s``.  The detection
        score ``changepoint_score[s]`` is the mass on starts within
        ``radius`` of ``s``; peaks above ``threshold`` that are at least
        ``min_distance`` apart (default ``max(lag, 5)``) are kept and each is
        placed at the most probable start inside its window.  Starts within
        ``min_distance`` of the beginning of the series are not reported.
    max_run_length:
        Optional truncation of the run-length support (``O(n * R)`` instead of
        ``O(n^2)``); mass beyond it is dropped and the posterior renormalised.
    """
    x = _validate_series(values)
    n = x.shape[0]
    if not 0.0 < hazard < 1.0:
        raise ValueError("hazard must be in (0, 1)")
    if kappa0 <= 0 or alpha0 <= 0:
        raise ValueError("kappa0 and alpha0 must be positive")
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if max_run_length is not None and max_run_length < lag + 2:
        raise ValueError("max_run_length must be at least lag + 2")
    center, scale = _robust_location_scale(x)
    mu0 = center if mu0 is None else float(mu0)
    beta0 = alpha0 * scale**2 if beta0 is None else float(beta0)
    if beta0 <= 0:
        raise ValueError("beta0 must be positive")
    if min_distance is None:
        min_distance = max(lag, 5)

    log_h = log(hazard)
    log_1mh = log(1.0 - hazard)

    # Index r-1 holds run length r (regime has absorbed r observations).
    mu = np.array([(kappa0 * mu0 + x[0]) / (kappa0 + 1.0)])
    kappa = np.array([kappa0 + 1.0])
    alpha = np.array([alpha0 + 0.5])
    beta = np.array([beta0 + kappa0 * (x[0] - mu0) ** 2 / (2.0 * (kappa0 + 1.0))])
    log_r = np.array([0.0])

    surprise = np.empty(n)
    surprise[0] = -float(_student_t_logpdf(x[0], mu0, kappa0, alpha0, beta0))
    log_evidence = -surprise[0]
    map_rl = np.empty(n, dtype=int)
    exp_rl = np.empty(n)
    map_rl[0] = 1
    exp_rl[0] = 1.0
    cp_prob = np.zeros(n)
    if lag == 0:
        cp_prob[0] = 1.0

    for t in range(1, n):
        xt = x[t]
        log_pred = _student_t_logpdf(xt, mu, kappa, alpha, beta)
        log_prior_pred = float(_student_t_logpdf(xt, mu0, kappa0, alpha0, beta0))
        growth = log_r + log_pred + log_1mh
        change = log_h + log_prior_pred  # sum_r P(r_{t-1}) = 1 after normalisation
        joint = np.concatenate(([change], growth))
        log_marginal = _logsumexp(joint)
        surprise[t] = -log_marginal
        log_evidence += log_marginal
        log_r = joint - log_marginal

        # Sufficient-statistic update: new regime from the prior, others grow.
        new_mu = np.concatenate(([(kappa0 * mu0 + xt) / (kappa0 + 1.0)], (kappa * mu + xt) / (kappa + 1.0)))
        new_beta = np.concatenate(
            (
                [beta0 + kappa0 * (xt - mu0) ** 2 / (2.0 * (kappa0 + 1.0))],
                beta + kappa * (xt - mu) ** 2 / (2.0 * (kappa + 1.0)),
            )
        )
        kappa = np.concatenate(([kappa0 + 1.0], kappa + 1.0))
        alpha = np.concatenate(([alpha0 + 0.5], alpha + 0.5))
        mu, beta = new_mu, new_beta

        if max_run_length is not None and log_r.shape[0] > max_run_length:
            keep = max_run_length
            log_r = log_r[:keep]
            log_r = log_r - _logsumexp(log_r)
            mu, kappa, alpha, beta = mu[:keep], kappa[:keep], alpha[:keep], beta[:keep]

        probs = np.exp(log_r)
        lengths = np.arange(1, probs.shape[0] + 1)
        map_rl[t] = int(lengths[int(np.argmax(probs))])
        exp_rl[t] = float(np.dot(lengths, probs))
        if t - lag >= 0 and lag < probs.shape[0]:
            cp_prob[t - lag] = float(probs[lag])

    # Tail: the last ``lag`` starts have fewer than ``lag`` follow-up points;
    # use the final posterior (P(r_{n-1} = n - s)).
    probs = np.exp(log_r)
    for s in range(max(1, n - lag), n):
        r = n - s
        if r - 1 < probs.shape[0]:
            cp_prob[s] = float(probs[r - 1])

    width = 2 * int(radius) + 1
    padded = np.concatenate((np.zeros(radius), cp_prob, np.zeros(radius)))
    cp_score = np.minimum(np.convolve(padded, np.ones(width), mode="valid"), 1.0)
    changepoints = []
    # The series start is a trivial regime boundary; ignore starts that close.
    cp_score[: int(min_distance)] = 0.0
    for peak in _pick_peaks(cp_score, threshold, int(min_distance)):
        lo, hi = max(1, peak - radius), min(n, peak + radius + 1)
        changepoints.append(int(lo + np.argmax(cp_prob[lo:hi])))
    changepoints = sorted(set(changepoints))
    return BOCPDResult(
        surprise=surprise,
        map_run_length=map_rl,
        expected_run_length=exp_rl,
        changepoint_probability=cp_prob,
        changepoint_score=cp_score,
        log_evidence=float(log_evidence),
        hazard=float(hazard),
        lag=int(lag),
        changepoints=changepoints,
    )


def _pick_peaks(score: np.ndarray, threshold: float, min_distance: int) -> List[int]:
    candidates = np.nonzero(score >= threshold)[0]
    candidates = candidates[candidates > 0]
    # Greedy non-maximum suppression, highest score first.
    order = candidates[np.argsort(-score[candidates], kind="stable")]
    chosen: List[int] = []
    for idx in order:
        if all(abs(int(idx) - c) >= min_distance for c in chosen):
            chosen.append(int(idx))
    return sorted(chosen)


def bocpd_changepoints(values: np.ndarray, **kwargs) -> List[int]:
    """Indices where a new regime starts according to :func:`bocpd`."""
    return bocpd(values, **kwargs).changepoints


@dataclass
class CusumAlarm:
    """One CUSUM alarm: when it fired, where the change likely began, and its sign."""

    alarm_index: int
    start_index: int
    direction: str


@dataclass
class CusumResult:
    """Output of :func:`cusum`."""

    upper: np.ndarray
    lower: np.ndarray
    alarms: List[CusumAlarm]
    center: float
    scale: float
    k: float
    h: float

    @property
    def changepoints(self) -> List[int]:
        return [alarm.start_index for alarm in self.alarms]

    def scores(self) -> np.ndarray:
        """Per-point score ``max(S+, S-)`` (higher = more evidence of a shift)."""
        return np.maximum(self.upper, self.lower)


def cusum(
    values: np.ndarray,
    k: float = 0.5,
    h: float = 8.0,
    center: Optional[float] = None,
    scale: Optional[float] = None,
    burn_in: Optional[int] = None,
    reset: bool = True,
    rewarm: Optional[int] = None,
) -> CusumResult:
    """Page's two-sided tabular CUSUM on standardised values.

    ``z_t = (x_t - center) / scale``; ``S+_t = max(0, S+_{t-1} + z_t - k)`` and
    ``S-_t = max(0, S-_{t-1} - z_t - k)``.  An alarm fires when either exceeds
    ``h``; its change start is one past the last time that statistic was 0.
    With ``reset`` both statistics restart at 0 after an alarm, are held at 0
    for ``rewarm`` further points (default ``min(burn_in, 20)``), and the in-control
    centre/scale are then re-estimated robustly from the new regime (from the
    estimated change start onwards) and refreshed every step until the regime
    has ``burn_in`` points (a self-starting CUSUM), unless fixed by the caller.

    ``center``/``scale`` default to the median and MAD-based scale of the first
    ``burn_in`` points (default ``max(10, n // 10)``).  ``k = 0.5`` targets a
    one-sigma shift.  The textbook ``h = 5`` gives a two-sided in-control
    ARL of only ~465 points, which on long series (and with estimated
    centre/scale) means regular false alarms; the default ``h = 8``
    (Siegmund approximation ~9.5k) costs a few points of detection delay.
    """
    x = _validate_series(values)
    n = x.shape[0]
    if k < 0:
        raise ValueError("k must be non-negative")
    if h <= 0:
        raise ValueError("h must be positive")
    if burn_in is None:
        burn_in = max(10, n // 10)
    burn_in = int(min(max(burn_in, 2), n))
    rewarm = min(burn_in, 20) if rewarm is None else int(rewarm)
    if rewarm < 0:
        raise ValueError("rewarm must be non-negative")
    est_center, est_scale = _robust_location_scale(x[:burn_in])
    fixed_center = center is not None
    fixed_scale = scale is not None
    mu = float(center) if fixed_center else est_center
    sigma = float(scale) if fixed_scale else est_scale
    if sigma <= 0:
        raise ValueError("scale must be positive")

    upper = np.zeros(n)
    lower = np.zeros(n)
    alarms: List[CusumAlarm] = []
    s_pos = s_neg = 0.0
    zero_pos = zero_neg = -1
    regime_start = 0
    warm_until = -1
    for t in range(n):
        if t <= warm_until:
            # Re-learn the in-control level of the new regime (online: only
            # points from the estimated change start up to now are used).
            if t == warm_until:
                seen = x[regime_start : t + 1]
                new_center, new_scale = _robust_location_scale(seen)
                if not fixed_center:
                    mu = new_center
                if not fixed_scale and seen.size >= 5:
                    sigma = new_scale
            zero_pos = zero_neg = t
            continue
        if alarms and reset and t - regime_start < burn_in:
            # Self-starting refinement: until the new regime has ``burn_in``
            # points, keep re-estimating its level from all of its past points.
            seen = x[regime_start:t]
            new_center, new_scale = _robust_location_scale(seen)
            if not fixed_center:
                mu = new_center
            if not fixed_scale and seen.size >= 5:
                sigma = new_scale
        z = (x[t] - mu) / sigma
        s_pos = max(0.0, s_pos + z - k)
        s_neg = max(0.0, s_neg - z - k)
        if s_pos == 0.0:
            zero_pos = t
        if s_neg == 0.0:
            zero_neg = t
        upper[t] = s_pos
        lower[t] = s_neg
        if s_pos > h or s_neg > h:
            if s_pos >= s_neg:
                alarms.append(CusumAlarm(t, zero_pos + 1, "up"))
                start = zero_pos + 1
            else:
                alarms.append(CusumAlarm(t, zero_neg + 1, "down"))
                start = zero_neg + 1
            if reset:
                s_pos = s_neg = 0.0
                zero_pos = zero_neg = t
                regime_start = start
                warm_until = t + rewarm
    return CusumResult(
        upper=upper,
        lower=lower,
        alarms=alarms,
        center=mu,
        scale=sigma,
        k=float(k),
        h=float(h),
    )


def changepoint_hits(
    detected: List[int], truth: List[int], tolerance: int = 5
) -> Tuple[float, float, float]:
    """Precision, recall and F1 of detected change points within ``tolerance`` steps.

    Each true change point can be matched by at most one detection.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    detected = sorted(int(d) for d in detected)
    remaining = sorted(int(t) for t in truth)
    tp = 0
    for d in detected:
        best = None
        for i, t in enumerate(remaining):
            if abs(d - t) <= tolerance and (best is None or abs(d - t) < abs(d - remaining[best])):
                best = i
        if best is not None:
            remaining.pop(best)
            tp += 1
    precision = tp / len(detected) if detected else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
