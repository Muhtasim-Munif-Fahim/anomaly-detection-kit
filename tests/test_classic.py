import numpy as np
import pytest

from anomaly_detection import classic as c
from anomaly_detection import generators as g


class TestZScore:
    def test_scores_shape_2d(self):
        X = np.random.default_rng(0).normal(size=(100, 4))
        s = c.zscore_scores(X)
        assert s.shape == (100,)

    def test_scores_1d_input(self):
        s = c.zscore_scores(np.arange(10.0))
        assert s.shape == (10,)

    def test_flags_detect_obvious_outlier(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=100)
        values[40] = 40.0
        flags = c.zscore_flags(values, threshold=4.0)
        assert flags[40] == 1
        assert flags.sum() == 1

    def test_constant_column_produces_no_flags(self):
        X = np.full((50, 3), 5.0)
        s = c.zscore_scores(X)
        assert np.all(s == 0.0)
        assert c.zscore_flags(X).sum() == 0

    def test_threshold_boundary_strict(self):
        X = np.array([[0.0], [0.0], [0.0], [0.0], [2.0]])
        scores = c.zscore_scores(X)
        t = float(scores.max())
        assert c.zscore_flags(X, threshold=t).sum() == 0
        assert c.zscore_flags(X, threshold=t - 1e-12).sum() == 1

    def test_single_row(self):
        s = c.zscore_scores(np.array([[3.0, 4.0]]))
        assert s.shape == (1,)
        assert np.all(s == 0.0)

    def test_detects_shift_outliers_from_generator(self):
        X, y = g.make_tabular(600, 4, 3, contamination=0.05, outlier_types="shift", seed=9)
        flags = c.zscore_flags(X, threshold=3.0)
        tp = int(((flags == 1) & (y == 1)).sum())
        assert tp >= int(y.sum()) * 0.8


class TestModifiedZScore:
    def test_scores_scale_invariant(self):
        X = np.random.default_rng(1).normal(size=(100, 3))
        s1 = c.mad_scores(X)
        s2 = c.mad_scores(X * 100.0)
        assert np.allclose(s1, s2)

    def test_flags_detect_obvious_outlier(self):
        values = np.arange(10.0)
        values[3] = 45.0
        flags = c.mad_flags(values, threshold=3.5)
        assert flags[3] == 1
        assert flags.sum() == 1

    def test_constant_column_no_nan(self):
        X = np.full((40, 2), 7.0)
        s = c.mad_scores(X)
        assert np.all(s == 0.0)
        assert np.all(np.isfinite(s))

    def test_more_than_half_identical_fallback(self):
        values = np.array([1.0] * 8 + [50.0, 60.0])
        s = c.mad_scores(values)
        assert np.all(np.isfinite(s))
        assert s[8] > s[:8].max()
        assert s[9] > s[:8].max()
        assert c.mad_flags(values, threshold=1.3).sum() == 2

    def test_shape_matches_input(self):
        X = np.random.default_rng(2).normal(size=(60, 5))
        assert c.mad_scores(X).shape == (60,)

    def test_flags_detect_obvious_outlier(self):
        values = np.array([0.0, 1.0, 2.0, 100.0])
        flags = c.mad_flags(values, threshold=3.5)
        assert flags[3] == 1
        assert flags[:3].sum() == 0


class TestIQR:
    def test_flags_detect_far_outliers(self):
        values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, -30]).astype(float)
        flags = c.iqr_flags(values, k=1.5)
        assert flags[10] == 1
        assert flags[11] == 1
        assert flags[:10].sum() == 0

    def test_scores_zero_inside_fences(self):
        values = np.arange(20.0)
        s = c.iqr_scores(values)
        assert (s == 0.0).sum() == len(values)

    def test_boundary_exactly_at_fence(self):
        values = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.5, 2.6])
        assert c.iqr_flags(values, k=1.5)[8] == 0
        assert c.iqr_flags(values, k=1.5)[9] == 1

    def test_constant_column(self):
        X = np.full((30, 2), 3.0)
        assert c.iqr_scores(X).sum() == 0.0
        assert c.iqr_flags(X).sum() == 0

    def test_k_changes_sensitivity(self):
        values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20.0])
        assert c.iqr_flags(values, k=1.5).sum() == 1
        assert c.iqr_flags(values, k=3.0).sum() == 0

    def test_scores_combined_across_columns(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [100.0, 0.0]])
        s = c.iqr_scores(X)
        assert s[3] > s[0]


class TestSpecialFunctions:
    def test_gammaln_known_values(self):
        assert abs(c._gammaln(1.0) - 0.0) < 1e-12
        assert abs(c._gammaln(2.0) - 0.0) < 1e-12
        assert abs(c._gammaln(5.0) - np.log(24.0)) < 1e-10
        assert abs(c._gammaln(0.5) - 0.5 * np.log(np.pi)) < 1e-10

    def test_ibeta_symmetry_at_half(self):
        assert abs(c._ibeta(0.5, 1.0, 1.0) - 0.5) < 1e-12
        assert abs(c._ibeta(0.5, 2.0, 2.0) - 0.5) < 1e-12
        assert abs(c._ibeta(0.5, 0.5, 0.5) - 0.5) < 1e-12

    def test_ibeta_complement_symmetry(self):
        for a, b in [(3.0, 2.0), (0.5, 4.0), (2.0, 5.0)]:
            assert abs(c._ibeta(0.3, a, b) - (1.0 - c._ibeta(0.7, b, a))) < 1e-12

    def test_ibeta_endpoints(self):
        assert c._ibeta(0.0, 2.0, 3.0) == 0.0
        assert c._ibeta(1.0, 2.0, 3.0) == 1.0

    def test_ibeta_inv_roundtrip(self):
        for a, b in [(0.5, 0.5), (2.0, 3.0), (1.0, 5.0), (5.0, 0.5)]:
            u = c._ibeta_inv(0.3, a, b)
            assert abs(c._ibeta(u, a, b) - 0.3) < 1e-10

    def test_ibeta_inv_endpoints(self):
        assert c._ibeta_inv(0.0, 2.0, 2.0) == 0.0
        assert c._ibeta_inv(1.0, 2.0, 2.0) == 1.0

    def test_t_inv_known_quantiles(self):
        assert abs(c._t_inv(0.975, 10) - 2.228139) < 1e-4
        assert abs(c._t_inv(0.95, 20) - 1.724718) < 1e-4
        assert abs(c._t_inv(0.995, 30) - 2.750) < 1e-4

    def test_t_inv_symmetry_and_centre(self):
        assert c._t_inv(0.5, 10) == 0.0
        assert abs(c._t_inv(0.025, 10) + c._t_inv(0.975, 10)) < 1e-12


class TestGESD:
    def test_detects_point_spikes(self):
        rng = np.random.default_rng(11)
        values = rng.normal(0.0, 1.0, size=300)
        spikes = np.array([10, 60, 150, 230])
        values[spikes] += np.array([8.0, -9.0, 8.5, -8.0])
        flags, anomalies, R, lambdas = c.gesd(values, k=8, alpha=0.05)
        assert sorted(anomalies) == sorted(spikes.tolist())
        assert flags.sum() == 4
        assert len(R) == 8
        assert len(lambdas) == 8
        assert all(r >= 0 for r in R)
        assert all(lam > 0 for lam in lambdas)

    def test_no_anomaly_on_clean_data(self):
        rng = np.random.default_rng(5)
        values = rng.normal(0.0, 1.0, size=500)
        flags, anomalies, _, _ = c.gesd(values, k=5, alpha=0.05)
        assert flags.sum() == 0
        assert anomalies == []

    def test_constant_series(self):
        values = np.ones(100)
        flags, anomalies, _, _ = c.gesd(values, k=5)
        assert flags.sum() == 0
        assert anomalies == []

    def test_small_series(self):
        for n in (0, 1, 2):
            flags, anomalies, R, lam = c.gesd(np.zeros(n), k=5)
            assert flags.shape == (n,)
            assert anomalies == []
            assert R == []
            assert lam == []

    def test_k_is_clamped(self):
        values = np.random.default_rng(3).normal(size=10)
        flags, _, R, _ = c.gesd(values, k=50)
        assert flags.shape == (10,)
        assert len(R) == 8

    def test_lower_alpha_flags_fewer_or_equal(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=400)
        values[50] += 6.0
        _, a_low, _, _ = c.gesd(values, k=5, alpha=0.001)
        _, a_high, _, _ = c.gesd(values, k=5, alpha=0.5)
        assert len(a_low) <= len(a_high)
        assert 50 in a_high

    def test_deterministic(self):
        rng = np.random.default_rng(13)
        values = rng.normal(size=200)
        values[[4, 90, 120]] += [7.0, -6.0, 8.0]
        f1, a1, R1, l1 = c.gesd(values, k=5)
        f2, a2, R2, l2 = c.gesd(values, k=5)
        assert np.array_equal(f1, f2)
        assert a1 == a2
        assert R1 == R2
        assert l1 == l2

    def test_gesd_indices_matches_gesd(self):
        rng = np.random.default_rng(17)
        values = rng.normal(size=200)
        values[33] += 9.0
        _, anomalies, _, _ = c.gesd(values, k=3)
        assert c.gesd_indices(values, k=3) == anomalies

    def test_level_shift_flags_within_segment(self):
        rng = np.random.default_rng(23)
        values = rng.normal(size=300)
        values[285:] += 25.0
        flags, anomalies, _, _ = c.gesd(values, k=10)
        assert flags.sum() >= 1
        assert all(i >= 285 for i in anomalies)


class TestRegistry:
    def test_statistical_scorers_keys(self):
        assert set(c.STATISTICAL_SCORERS) == {"z-score", "modified z-score", "IQR"}

    def test_registry_callables_return_scores(self):
        X = np.random.default_rng(4).normal(size=(80, 3))
        for fn in c.STATISTICAL_SCORERS.values():
            s = fn(X)
            assert s.shape == (80,)
            assert np.all(np.isfinite(s))
