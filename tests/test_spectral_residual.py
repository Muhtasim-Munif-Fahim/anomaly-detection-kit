"""Tests for the Spectral Residual saliency detector."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection.cli import main
from anomaly_detection.evaluate import roc_auc
from anomaly_detection.spectral_residual import (
    extend_series,
    saliency_map,
    spectral_residual,
    spectral_residual_scores,
)


def _series(n=800, period=50, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.1 * rng.normal(size=n)
    idx = [i for i in (150, 400, 650) if i < n]
    x[idx] += [3.0, -3.0, 2.5][: len(idx)]
    y = np.zeros(n)
    y[idx] = 1
    return x, y, idx


def test_saliency_matches_reference_formula():
    rng = np.random.default_rng(1)
    x = rng.normal(size=64)
    spec = np.fft.fft(x)
    log_amp = np.log(np.abs(spec) + 1e-8)
    padded = np.concatenate([[log_amp[0]], log_amp, [log_amp[-1]]])
    smooth = np.array([padded[i:i + 3].mean() for i in range(64)])
    expected = np.abs(np.fft.ifft(np.exp(log_amp - smooth + 1j * np.angle(spec))))
    np.testing.assert_allclose(saliency_map(x, amp_window=3), expected, rtol=1e-10)


def test_amp_window_one_gives_flat_spectrum_impulse():
    # With no smoothing the residual is 0 everywhere: unit amplitude spectrum.
    x = np.random.default_rng(2).normal(size=32)
    sal = saliency_map(x, amp_window=1)
    assert sal.shape == x.shape
    np.testing.assert_allclose(np.sum(sal ** 2), 1.0, rtol=1e-8)  # Parseval: sum |S|^2 = 1


def test_extend_series_uses_average_gradient():
    x = np.arange(10.0)  # slope 1
    ext = extend_series(x, extend=3, n_grad=4)
    assert ext.size == 13
    np.testing.assert_allclose(ext[:10], x)
    np.testing.assert_allclose(ext[10:], 6.0 + 1.0 * 4)  # x[n-m] + g*m
    np.testing.assert_allclose(extend_series(x, extend=0), x)


def test_detects_injected_spikes():
    x, y, idx = _series()
    result = spectral_residual(x)
    assert set(idx) <= set(result.anomalies)
    assert roc_auc(y, result.scores) > 0.99
    assert result.flags.sum() < 20
    np.testing.assert_allclose(result.scores, spectral_residual_scores(x))


def test_scores_nonnegative_and_recent_spike_found():
    x, _, _ = _series()
    x[-8] += 4.0
    scores = spectral_residual_scores(x)
    assert np.all(scores >= 0)
    assert scores[-8] > 3.0
    assert int(np.argmax(scores[-50:])) == 42


@pytest.mark.parametrize(
    "kwargs",
    [{"amp_window": 0}, {"score_window": 0}, {"extend": -1}, {"n_grad": 0}],
)
def test_invalid_parameters(kwargs):
    x, _, _ = _series(n=100)
    with pytest.raises(ValueError):
        spectral_residual_scores(x, **kwargs)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        saliency_map(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        saliency_map([1.0, 2.0])
    with pytest.raises(ValueError):
        saliency_map([1.0, np.nan, 2.0, 3.0])
    with pytest.raises(ValueError):
        spectral_residual(np.arange(10.0), threshold=0)


def test_cli_sr_synthetic(capsys):
    assert main(["sr", "--n-points", "600"]) == 0
    out = capsys.readouterr().out
    assert "spectral residual" in out
    assert "recovered 3/3 injected spikes" in out


def test_cli_sr_csv(tmp_path, capsys):
    x, _, _ = _series(n=300)
    path = tmp_path / "s.csv"
    path.write_text("value\n" + "\n".join(f"{v:.6f}" for v in x) + "\n")
    assert main(["sr", "--data", str(path), "--column", "value", "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert "index=150" in out
