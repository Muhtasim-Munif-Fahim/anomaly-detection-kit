"""Tests for extreme-value thresholds: GPD fit, POT, SPOT and DSPOT."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import genpareto

from anomaly_detection.cli import _synthetic_spikes, main
from anomaly_detection.evt import (
    DSPOT,
    SPOT,
    evt_flags,
    fit_gpd,
    gpd_log_likelihood,
    gpd_quantile,
    pot,
)


class TestGPDFit:
    @pytest.mark.parametrize("shape", [-0.4, -0.2, 0.1, 0.3, 0.5, 1.0])
    @pytest.mark.parametrize("n", [50, 500])
    def test_matches_scipy_mle(self, shape, n):
        y = genpareto.rvs(shape, scale=2.0, size=n, random_state=n + int(shape * 100))
        gamma, sigma, ll = fit_gpd(y)
        c, _, scale = genpareto.fit(y, floc=0)
        assert ll == pytest.approx(gpd_log_likelihood(y, gamma, sigma))
        assert ll >= gpd_log_likelihood(y, c, scale) - 1e-6
        assert gamma == pytest.approx(c, abs=2e-3)
        assert sigma == pytest.approx(scale, rel=2e-3)

    def test_log_likelihood_matches_scipy_logpdf(self):
        y = np.array([0.1, 0.5, 1.2, 3.0])
        for g in (-0.3, 0.0, 0.4):
            expected = genpareto.logpdf(y, g, scale=1.5).sum()
            assert gpd_log_likelihood(y, g, 1.5) == pytest.approx(expected)

    def test_outside_support_is_minus_inf(self):
        assert gpd_log_likelihood([1.0, 5.0], -0.5, 1.0) == -np.inf
        assert gpd_log_likelihood([1.0], 0.2, -1.0) == -np.inf

    def test_large_sample_recovers_parameters(self):
        y = genpareto.rvs(0.25, scale=1.0, size=20000, random_state=1)
        gamma, sigma, _ = fit_gpd(y)
        assert gamma == pytest.approx(0.25, abs=0.03)
        assert sigma == pytest.approx(1.0, abs=0.04)

    @pytest.mark.parametrize("bad", [[1.0], [-1.0, 2.0], [0.0, 0.0, 0.0], [[1.0, 2.0], [3.0, 4.0]], [1.0, np.nan]])
    def test_validation(self, bad):
        with pytest.raises(ValueError):
            fit_gpd(bad)


class TestPOT:
    def test_quantile_formula(self):
        # P(X > z) = (Nt/n) * (1 + gamma (z - t) / sigma)^(-1/gamma) must equal q.
        t, g, s, n, nt, q = 2.0, 0.3, 1.5, 1000, 50, 1e-3
        z = gpd_quantile(t, g, s, n, nt, q)
        assert nt / n * (1 + g * (z - t) / s) ** (-1 / g) == pytest.approx(q)
        z0 = gpd_quantile(t, 0.0, s, n, nt, q)
        assert nt / n * np.exp(-(z0 - t) / s) == pytest.approx(q)

    def test_exponential_tail_quantile(self):
        x = np.random.default_rng(0).standard_exponential(100_000)
        res = pot(x, q=1e-4, level=0.98)
        assert res.threshold == pytest.approx(-np.log(1e-4), rel=0.05)
        assert res.gamma == pytest.approx(0.0, abs=0.05)
        assert res.n_peaks == pytest.approx(2000, abs=2)

    def test_pareto_tail_beyond_sample(self):
        x = genpareto.rvs(0.3, size=5000, random_state=3)
        res = pot(x, q=1e-5, level=0.95)
        true = genpareto.ppf(1 - 1e-5, 0.3)
        assert res.threshold > x.max() * 0.8
        assert res.threshold == pytest.approx(true, rel=0.35)

    def test_falls_back_to_empirical_quantile(self):
        x = np.arange(100.0)
        res = pot(x, q=0.1, level=0.95)
        assert res.threshold == pytest.approx(np.quantile(x, 0.9))
        assert np.isnan(res.gamma)

    def test_evt_flags(self):
        rng = np.random.default_rng(4)
        scores = rng.normal(size=2000)
        scores[[10, 500, 1500]] = 9.0
        flags, z = evt_flags(scores, q=1e-3, level=0.95)
        assert flags.dtype.kind == "i" and set(np.unique(flags)) <= {0, 1}
        assert all(flags[[10, 500, 1500]] == 1)
        assert flags.sum() <= 10
        assert z > 2.5

    @pytest.mark.parametrize("kw", [{"q": 0.0}, {"q": 1.0}, {"level": 1.0}, {"level": 0.0}])
    def test_validation(self, kw):
        with pytest.raises(ValueError):
            pot(np.arange(50.0), **kw)
        with pytest.raises(ValueError):
            pot(np.arange(5.0))


class TestSPOT:
    def test_false_alarm_rate_is_controlled(self):
        x = np.random.default_rng(0).standard_exponential(21_000)
        res = SPOT(q=1e-3, level=0.98).fit(x[:1000]).run(x[1000:])
        rate = len(res.alarms) / 20_000
        assert rate < 3e-3
        assert res.thresholds.shape == (20_000,)
        assert res.thresholds[-1] == pytest.approx(-np.log(1e-3), rel=0.15)

    def test_detects_spikes_both_sides(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=4000)
        y[2500] += 8
        y[3000] -= 8
        res = SPOT(q=1e-4, side="both").fit(y[:1000]).run(y[1000:])
        alarms = [a + 1000 for a in res.alarms]
        assert 2500 in alarms and 3000 in alarms
        assert len(alarms) <= 5
        assert res.upper_thresholds is not None and res.lower_thresholds is not None
        assert np.all(res.lower_thresholds < 0) and np.all(res.upper_thresholds > 0)

    def test_side_selection(self):
        rng = np.random.default_rng(2)
        y = rng.normal(size=3000)
        y[2000] -= 9
        upper = SPOT(q=1e-4, side="upper").fit(y[:1000]).run(y[1000:])
        lower = SPOT(q=1e-4, side="lower").fit(y[:1000]).run(y[1000:])
        assert 1000 not in upper.alarms and lower.lower_thresholds is not None
        assert 1000 in lower.alarms and lower.upper_thresholds is None
        assert np.array_equal(lower.thresholds, lower.lower_thresholds)

    def test_alarms_do_not_update_threshold(self):
        rng = np.random.default_rng(3)
        y = rng.normal(size=1500)
        s = SPOT(q=1e-3).fit(y[:1000])
        before = s.upper_threshold
        s.run([50.0, 60.0])
        assert s.upper_threshold == before

    def test_run_before_fit(self):
        with pytest.raises(RuntimeError):
            SPOT().run([1.0, 2.0])

    @pytest.mark.parametrize("kw", [{"side": "middle"}, {"q": 2.0}, {"level": -0.1}])
    def test_validation(self, kw):
        with pytest.raises(ValueError):
            SPOT(**kw)
        with pytest.raises(ValueError):
            SPOT().fit(np.arange(5.0))


class TestDSPOT:
    def test_tracks_drift_where_spot_fails(self):
        rng = np.random.default_rng(5)
        z = rng.normal(size=4000) + np.linspace(0, 20, 4000)
        z[3000] += 7
        d = DSPOT(depth=20, q=1e-4).fit(z[:1000]).run(z[1000:])
        s = SPOT(q=1e-4).fit(z[:1000]).run(z[1000:])
        assert 3000 in [a + 1000 for a in d.alarms]
        assert len(d.alarms) <= 3
        assert len(s.alarms) > 1000
        assert d.thresholds[-1] > d.thresholds[0] + 10

    def test_validation(self):
        for depth in (0, 2.5, True):
            with pytest.raises(ValueError):
                DSPOT(depth=depth)
        with pytest.raises(ValueError):
            DSPOT(depth=20).fit(np.arange(25.0))


class TestCLI:
    def test_synthetic_spikes(self):
        values, idx = _synthetic_spikes(1000, 0, 0.0)
        assert values.shape == (1000,) and len(idx) == 4
        assert all(i >= 500 for i in idx)

    def test_evt_spot(self, capsys):
        assert main(["evt", "--seed", "0"]) == 0
        out = capsys.readouterr().out
        assert "spot (q=0.0001" in out and "detected 4/4" in out

    def test_evt_dspot_with_drift(self, capsys):
        assert main(["evt", "--method", "dspot", "--drift", "15"]) == 0
        out = capsys.readouterr().out
        assert "detected 4/4 injected anomalies, 0 false alarms" in out

    def test_evt_csv(self, tmp_path, capsys):
        rng = np.random.default_rng(0)
        x = rng.normal(size=800)
        x[700] = 10.0
        path = tmp_path / "s.csv"
        np.savetxt(path, np.column_stack([np.arange(800), x]), delimiter=",", header="t,value", comments="", fmt="%.6f")
        assert main(["evt", "--data", str(path), "--column", "value", "--side", "upper", "--q", "1e-3"]) == 0
        assert "700" in capsys.readouterr().out
