"""Tests for Seasonal Hybrid ESD."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection.classic import (
    seasonal_hybrid_esd,
    seasonal_hybrid_esd_indices,
)


def _seasonal_series(n=120, period=12, spike_at=50, spike=8.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = 3.0 * np.sin(2 * np.pi * t / period)
    trend = 0.02 * t
    noise = rng.normal(0, 0.3, size=n)
    y = seasonal + trend + noise
    y[spike_at] += spike
    return y


def test_shesd_finds_injected_spike():
    y = _seasonal_series()
    flags, anoms, seasonal, trend, resid = seasonal_hybrid_esd(y, period=12, k=3)
    assert 50 in anoms
    assert flags[50] == 1
    assert seasonal.shape == y.shape
    assert np.isfinite(resid).all()


def test_shesd_indices_helper():
    y = _seasonal_series(spike_at=80, spike=10.0)
    idxs = seasonal_hybrid_esd_indices(y, period=12, k=2)
    assert 80 in idxs


def test_shesd_rejects_short_series():
    y = np.arange(10.0)
    with pytest.raises(ValueError, match="two full seasons"):
        seasonal_hybrid_esd(y, period=12)


def test_shesd_rejects_bad_period():
    y = np.arange(40.0)
    with pytest.raises(ValueError, match="period"):
        seasonal_hybrid_esd(y, period=1)
