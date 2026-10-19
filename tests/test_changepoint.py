"""Tests for BOCPD and CUSUM change-point detection."""

from __future__ import annotations

import itertools
from math import lgamma, log, pi

import numpy as np
import pytest

from anomaly_detection.changepoint import bocpd, bocpd_changepoints, changepoint_hits, cusum
from anomaly_detection.cli import _synthetic_regimes, main
from anomaly_detection.evaluate import roc_auc


def _regimes(seed, cuts=(150, 300, 420), levels=(0.0, 3.0, -1.0, 1.5), n=560, sd=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=sd, size=n)
    for start, end, level in zip((0,) + cuts, cuts + (n,), levels):
        x[start:end] += level
    return x, list(cuts)


def _ng_log_marginal(seg, mu0, kappa0, alpha0, beta0):
    """Closed-form Normal-Gamma marginal likelihood of a segment."""
    m = len(seg)
    mean = float(np.mean(seg))
    kappa_n = kappa0 + m
    alpha_n = alpha0 + m / 2.0
    beta_n = (
        beta0
        + 0.5 * float(np.sum((seg - mean) ** 2))
        + kappa0 * m * (mean - mu0) ** 2 / (2.0 * kappa_n)
    )
    return (
        lgamma(alpha_n) - lgamma(alpha0)
        + alpha0 * log(beta0) - alpha_n * log(beta_n)
        + 0.5 * (log(kappa0) - log(kappa_n))
        - 0.5 * m * log(2 * pi)
    )


def test_bocpd_matches_brute_force_over_all_segmentations():
    x = np.array([0.3, -0.2, 2.9, 3.4, 3.1, -0.5])
    hazard, mu0, kappa0, alpha0, beta0 = 0.2, 0.0, 1.0, 2.0, 1.5
    res = bocpd(x, hazard=hazard, mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0, lag=0)
    n = len(x)
    log_terms, last_len = [], []
    for mask in itertools.product([0, 1], repeat=n - 1):
        bounds = [0] + [i + 1 for i, b in enumerate(mask) if b] + [n]
        lp = sum(log(hazard) if b else log(1 - hazard) for b in mask)
        for a, b in zip(bounds[:-1], bounds[1:]):
            lp += _ng_log_marginal(x[a:b], mu0, kappa0, alpha0, beta0)
        log_terms.append(lp)
        last_len.append(bounds[-1] - bounds[-2])
    log_terms = np.array(log_terms)
    top = log_terms.max()
    evidence = top + np.log(np.exp(log_terms - top).sum())
    assert res.log_evidence == pytest.approx(evidence, abs=1e-9)
    weights = np.exp(log_terms - evidence)
    expected_len = float(np.dot(weights, last_len))
    assert res.expected_run_length[-1] == pytest.approx(expected_len, abs=1e-9)
    # lag=0: P(regime starts at the last index) = P(final run length is 1).
    assert res.changepoint_probability[-1] == pytest.approx(
        float(weights[np.array(last_len) == 1].sum()), abs=1e-9
    )


def test_log_evidence_is_sum_of_predictive_log_densities():
    x, _ = _regimes(1)
    res = bocpd(x)
    assert res.log_evidence == pytest.approx(-res.surprise.sum(), rel=1e-12)


def test_bocpd_finds_level_shifts():
    x, cuts = _regimes(0)
    res = bocpd(x)
    precision, recall, _ = changepoint_hits(res.changepoints, cuts, tolerance=3)
    assert recall == 1.0 and precision == 1.0
    # MAP run length right before a shift spans the previous regime.
    assert res.segment_starts()[cuts[1] - 1] == pytest.approx(cuts[0], abs=3)
    assert bocpd_changepoints(x) == res.changepoints


def test_bocpd_finds_variance_change():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(0, 1, 250), rng.normal(0, 4, 250)])
    res = bocpd(x)
    assert any(abs(cp - 250) <= 8 for cp in res.changepoints)
    assert len(res.changepoints) <= 2


def test_bocpd_rarely_fires_on_stationary_noise():
    total = sum(len(bocpd(np.random.default_rng(50 + s).normal(size=400)).changepoints) for s in range(10))
    assert total <= 3


def test_bocpd_surprise_scores_spikes():
    rng = np.random.default_rng(7)
    x = rng.normal(size=400)
    y = np.zeros(400, dtype=int)
    spikes = rng.choice(np.arange(20, 400), 12, replace=False)
    x[spikes] += rng.choice([-1, 1], 12) * 6.0
    y[spikes] = 1
    assert roc_auc(y, bocpd(x).surprise) > 0.95


def test_truncated_run_length_agrees_on_short_regimes():
    x, cuts = _regimes(2)
    full = bocpd(x)
    short = bocpd(x, max_run_length=200)
    assert short.changepoints == full.changepoints
    assert short.map_run_length.max() <= 200


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"hazard": 0.0}, "hazard"),
        ({"hazard": 1.0}, "hazard"),
        ({"lag": -1}, "lag"),
        ({"threshold": 0.0}, "threshold"),
        ({"radius": -1}, "radius"),
        ({"max_run_length": 3, "lag": 5}, "max_run_length"),
        ({"beta0": -1.0}, "beta0"),
    ],
)
def test_bocpd_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bocpd(np.arange(20.0), **kwargs)


def test_series_validation():
    with pytest.raises(ValueError, match="two observations"):
        bocpd([1.0])
    with pytest.raises(ValueError, match="finite"):
        cusum([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        cusum(np.ones((3, 3)))


def test_cusum_detects_shift_with_known_parameters():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0, 1, 200), rng.normal(1.5, 1, 200)])
    res = cusum(x, center=0.0, scale=1.0, h=5.0, reset=False)
    first = res.alarms[0]
    assert first.direction == "up"
    assert 200 <= first.alarm_index <= 215
    assert abs(first.start_index - 200) <= 5
    assert res.scores().shape == x.shape
    assert np.all(res.upper >= 0) and np.all(res.lower >= 0)


def test_cusum_tracks_multiple_regimes_with_reset():
    x, cuts = _regimes(0)
    res = cusum(x)
    precision, recall, _ = changepoint_hits(res.changepoints, cuts, tolerance=5)
    assert recall == 1.0
    assert precision >= 0.75
    directions = [a.direction for a in res.alarms if any(abs(a.start_index - c) <= 5 for c in cuts)]
    assert directions == ["up", "down", "up"]


def test_cusum_downward_direction_and_statistic_definition():
    x = np.array([0.0] * 30 + [-3.0] * 10)
    res = cusum(x, center=0.0, scale=1.0, k=0.5, h=4.0, reset=False)
    # S- grows by 2.5 per step after the shift: 2.5, 5.0 -> alarm on the 2nd point.
    assert res.lower[30] == pytest.approx(2.5)
    assert res.alarms[0].alarm_index == 31
    assert res.alarms[0].start_index == 30
    assert res.alarms[0].direction == "down"


def test_cusum_validation():
    with pytest.raises(ValueError, match="k must"):
        cusum(np.arange(10.0), k=-1)
    with pytest.raises(ValueError, match="h must"):
        cusum(np.arange(10.0), h=0)
    with pytest.raises(ValueError, match="scale must"):
        cusum(np.arange(10.0), scale=0.0)


def test_changepoint_hits_matches_each_truth_once():
    assert changepoint_hits([100, 101, 300], [100, 200, 300], tolerance=3) == pytest.approx((2 / 3, 2 / 3, 2 / 3))
    assert changepoint_hits([], [10]) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        changepoint_hits([1], [1], tolerance=-1)


def test_synthetic_regimes_and_cli(capsys, tmp_path):
    values, truth = _synthetic_regimes(400, 0)
    assert truth == [100, 200, 300]
    assert main(["changepoints", "--n-points", "400"]) == 0
    out = capsys.readouterr().out
    assert "bocpd" in out and "cusum" in out and "f1=" in out
    path = tmp_path / "series.csv"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("t,value,label\n")
        for i, v in enumerate(values):
            handle.write(f"{i},{v},0\n")
    assert main(["changepoints", "--data", str(path), "--column", "value", "--method", "bocpd"]) == 0
    out = capsys.readouterr().out
    assert "cusum" not in out and "bocpd" in out
