"""Tests for matrix-profile (STOMP) discord discovery."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection.classic import (
    matrix_profile,
    matrix_profile_discords,
    matrix_profile_scores,
)
from anomaly_detection.cli import main
from anomaly_detection.evaluate import roc_auc


def _brute(x, m, excl):
    n = len(x) - m + 1
    W = np.array([x[i:i + m] for i in range(n)])
    Z = (W - W.mean(1, keepdims=True)) / W.std(1, keepdims=True)
    P, I = np.full(n, np.inf), np.full(n, -1)
    for i in range(n):
        d = np.sqrt(((Z - Z[i]) ** 2).sum(1))
        d[max(0, i - excl): i + excl + 1] = np.inf
        I[i] = int(np.argmin(d))
        P[i] = d[I[i]]
    return P, I


def _sine_with_anomaly(n=600, period=40, at=300, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sin(2 * np.pi * np.arange(n) / period) + 0.1 * rng.normal(size=n)
    x[at: at + 20] += np.linspace(0, 2, 20)
    return x


@pytest.mark.parametrize("seed, m", [(0, 8), (1, 25), (2, 40)])
def test_matches_brute_force(seed, m):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=300))
    P, I = matrix_profile(x, m)
    Pb, Ib = _brute(x, m, int(np.ceil(m / 4)))
    assert np.allclose(P, Pb, atol=1e-8)
    assert np.array_equal(I, Ib)
    assert np.all(np.abs(I - np.arange(I.size)) > int(np.ceil(m / 4)))


def test_custom_exclusion_zone():
    x = _sine_with_anomaly()
    P, I = matrix_profile(x, 40, exclusion_zone=40)
    assert np.all(np.abs(I - np.arange(I.size)) > 40)
    Pb, _ = _brute(x, 40, 40)
    assert np.allclose(P, Pb, atol=1e-8)


def test_top_discord_covers_injected_anomaly():
    x = _sine_with_anomaly()
    found = matrix_profile_discords(x, 40, k=3)
    start, dist = found[0]
    assert start <= 300 + 19 and start + 40 > 300
    assert dist == max(d for _, d in found)
    starts = [s for s, _ in found]
    assert all(abs(a - b) >= 40 for i, a in enumerate(starts) for b in starts[i + 1:])


def test_scores_rank_anomalous_points():
    x = _sine_with_anomaly()
    scores = matrix_profile_scores(x, 40)
    assert scores.shape == x.shape
    y = np.zeros(x.size, dtype=int)
    y[300:320] = 1
    assert roc_auc(y, scores) > 0.9


def test_constant_subsequences():
    x = np.concatenate([np.zeros(30), np.sin(np.arange(60) / 3.0), np.zeros(30)])
    P, I = matrix_profile(x, 10)
    assert P[0] == pytest.approx(0.0)  # matches the other flat stretch
    assert np.all(np.isfinite(P))


def test_too_many_discords_returns_available():
    x = _sine_with_anomaly(n=120, at=60)
    found = matrix_profile_discords(x, 40, k=10)
    assert 1 <= len(found) < 10


@pytest.mark.parametrize("kwargs, match", [
    (dict(values=np.arange(20.0), window=2), "window"),
    (dict(values=np.arange(10.0), window=8), "2 \\* window"),
    (dict(values=np.array([0.0, np.nan] * 20), window=4), "finite"),
])
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        matrix_profile(**kwargs)


def test_discords_validation():
    with pytest.raises(ValueError, match="k must"):
        matrix_profile_discords(np.arange(50.0), 5, k=0)


def test_cli_synthetic_and_csv(tmp_path, capsys):
    assert main(["discords", "--k", "2"]) == 0
    out = capsys.readouterr().out
    assert "#1: start=" in out and "#3" not in out
    x = _sine_with_anomaly()
    path = tmp_path / "series.csv"
    np.savetxt(path, np.column_stack([np.arange(x.size), x]), delimiter=",",
               header="t,value", comments="")
    assert main(["discords", "--data", str(path), "--column", "value", "--window", "40", "--k", "1"]) == 0
    line = [l for l in capsys.readouterr().out.splitlines() if "#1" in l][0]
    start = int(line.split("start=")[1].split()[0])
    assert start <= 319 and start + 40 > 300
