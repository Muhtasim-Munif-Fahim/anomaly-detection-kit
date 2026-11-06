import numpy as np
import pytest

from anomaly_detection.classic import hampel_filter, hampel_flags


def test_hampel_flags_spike():
    x = np.ones(30, dtype=float)
    x[15] = 50.0
    flags = hampel_flags(x, window=7, n_sigmas=3.0)
    assert flags[15] == 1
    assert flags.sum() == 1


def test_hampel_imputes_with_local_median():
    x = np.arange(20, dtype=float)
    x[10] = 1000.0
    out = hampel_filter(x, window=5, n_sigmas=3.0, impute=True)
    assert out["flags"][10] == 1
    assert out["filtered"][10] == pytest.approx(out["medians"][10])
    assert out["filtered"][10] != 1000.0


def test_clean_series_unflagged():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    flags = hampel_flags(x, window=11, n_sigmas=5.0)
    assert flags.sum() <= 2  # rare under Gaussian + high threshold


def test_rejects_even_window():
    with pytest.raises(ValueError):
        hampel_filter(np.arange(10.0), window=6)
