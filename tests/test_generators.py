import numpy as np
import pytest

from anomaly_detection import generators as g


class TestGaussianClusters:
    def test_shape(self):
        X, ids = g.gaussian_clusters(300, 4, 3, seed=7)
        assert X.shape == (300, 4)
        assert ids.shape == (300,)
        assert set(np.unique(ids)) == {0, 1, 2}

    def test_deterministic(self):
        X1, _ = g.gaussian_clusters(200, 3, 2, seed=11)
        X2, _ = g.gaussian_clusters(200, 3, 2, seed=11)
        assert np.array_equal(X1, X2)

    def test_seed_changes_draws(self):
        X1, _ = g.gaussian_clusters(200, 3, 2, seed=11)
        X2, _ = g.gaussian_clusters(200, 3, 2, seed=12)
        assert not np.array_equal(X1, X2)

    def test_explicit_centers_used(self):
        centers = np.array([[-5.0, 0.0], [5.0, 0.0]])
        X, ids = g.gaussian_clusters(2000, 2, 2, centers=centers, seed=3)
        for cid, c in enumerate(centers):
            assert np.linalg.norm(X[ids == cid].mean(axis=0) - c) < 0.2

    def test_invalid_centers_shape(self):
        with pytest.raises(ValueError):
            g.gaussian_clusters(50, 4, 3, centers=np.zeros((2, 4)), seed=1)

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError):
            g.gaussian_clusters(0, 4, 2, seed=1)

    def test_negative_cluster_std(self):
        with pytest.raises(ValueError):
            g.gaussian_clusters(50, 2, 2, cluster_std=-1.0, seed=1)

    def test_accepts_generator_seed(self):
        rng = np.random.default_rng(42)
        X1, _ = g.gaussian_clusters(100, 2, 2, seed=42)
        X2, _ = g.gaussian_clusters(100, 2, 2, seed=rng)
        assert X1.shape == X2.shape


class TestInjectShiftedOutliers:
    def test_count_and_shape(self):
        X, _ = g.gaussian_clusters(200, 4, 2, seed=1)
        X2, idx = g.inject_shifted_outliers(X, 10, shift=3.0, seed=5)
        assert X2.shape == X.shape
        assert len(idx) == 10
        assert X2[idx].shape == (10, 4)

    def test_outliers_strictly_further_than_inliers(self):
        X, _ = g.gaussian_clusters(400, 4, 2, seed=1)
        X2, idx = g.inject_shifted_outliers(X, 20, shift=3.0, seed=5)
        center = X2.mean(axis=0)
        d_out = np.linalg.norm(X2[idx] - center, axis=1)
        d_in = np.linalg.norm(X2[np.setdiff1d(np.arange(len(X2)), idx)] - center, axis=1)
        assert d_out.min() > d_in.max()

    def test_deterministic(self):
        X, _ = g.gaussian_clusters(200, 4, 2, seed=1)
        A, idx1 = g.inject_shifted_outliers(X, 10, seed=5)
        B, idx2 = g.inject_shifted_outliers(X, 10, seed=5)
        assert np.array_equal(A, B)
        assert np.array_equal(idx1, idx2)

    def test_too_many_outliers_raises(self):
        X, _ = g.gaussian_clusters(10, 2, 2, seed=1)
        with pytest.raises(ValueError):
            g.inject_shifted_outliers(X, 11, seed=1)

    def test_zero_outliers_no_change(self):
        X, _ = g.gaussian_clusters(100, 3, 2, seed=1)
        X2, idx = g.inject_shifted_outliers(X, 0, seed=1)
        assert np.array_equal(X, X2)
        assert len(idx) == 0


class TestInjectScaleOutliers:
    def test_count_and_dtype(self):
        X, _ = g.gaussian_clusters(200, 3, 2, seed=2)
        X2, idx = g.inject_scale_outliers(X, 10, scale=3.0, seed=9)
        assert X2.shape == X.shape
        assert len(idx) == 10
        assert X2.dtype == X.dtype

    def test_outliers_stretched_away_from_centre(self):
        X, _ = g.gaussian_clusters(400, 4, 2, seed=2)
        X2, idx = g.inject_scale_outliers(X, 20, scale=3.0, seed=9)
        center = X2.mean(axis=0)
        d_out = np.linalg.norm(X2[idx] - center, axis=1)
        d_in = np.linalg.norm(X2[np.setdiff1d(np.arange(len(X2)), idx)] - center, axis=1)
        assert d_out.mean() > d_in.mean()

    def test_deterministic(self):
        X, _ = g.gaussian_clusters(200, 4, 2, seed=2)
        A, _ = g.inject_scale_outliers(X, 10, scale=3.0, seed=9)
        B, _ = g.inject_scale_outliers(X, 10, scale=3.0, seed=9)
        assert np.array_equal(A, B)

    def test_too_many_outliers_raises(self):
        X, _ = g.gaussian_clusters(10, 2, 2, seed=2)
        with pytest.raises(ValueError):
            g.inject_scale_outliers(X, 12, seed=1)

    def test_works_on_constant_column(self):
        X = np.zeros((100, 2))
        X[:, 0] = np.arange(100)
        X2, idx = g.inject_shifted_outliers(X, 5, seed=1)
        assert np.all(np.isfinite(X2))
        assert len(idx) == 5


class TestMakeTabular:
    def test_shape_and_labels(self):
        X, y = g.make_tabular(500, 4, 3, contamination=0.05, seed=10)
        assert X.shape == (500, 4)
        assert y.shape == (500,)
        assert set(np.unique(y)) <= {0, 1}

    def test_contamination_match(self):
        X, y = g.make_tabular(1000, 4, 3, contamination=0.10, seed=10)
        assert int(y.sum()) == 100

    def test_zero_contamination(self):
        X, y = g.make_tabular(200, 3, 2, contamination=0.0, seed=10)
        assert y.sum() == 0

    def test_deterministic(self):
        X1, y1 = g.make_tabular(300, 4, 3, seed=12)
        X2, y2 = g.make_tabular(300, 4, 3, seed=12)
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)

    def test_seed_changes_data(self):
        X1, y1 = g.make_tabular(300, 4, 3, seed=12)
        X2, _ = g.make_tabular(300, 4, 3, seed=13)
        assert not np.array_equal(X1, X2)

    def test_only_shift_outliers(self):
        X, y = g.make_tabular(400, 4, 3, contamination=0.05, outlier_types="shift", seed=4)
        assert int(y.sum()) == 20
        center = X.mean(axis=0)
        d_out = np.linalg.norm(X[y == 1] - center, axis=1)
        d_in = np.linalg.norm(X[y == 0] - center, axis=1)
        assert d_out.min() > d_in.max()

    def test_only_scale_outliers(self):
        X, y = g.make_tabular(400, 4, 3, contamination=0.05, outlier_types="scale", seed=4)
        assert int(y.sum()) == 20

    def test_invalid_outlier_type_raises(self):
        with pytest.raises(ValueError):
            g.make_tabular(100, 3, 2, outlier_types=("bogus",), seed=1)

    def test_invalid_contamination_raises(self):
        with pytest.raises(ValueError):
            g.make_tabular(100, 3, 2, contamination=1.5, seed=1)

    def test_explicit_centers(self):
        centers = np.array([[-8.0, -8.0], [8.0, 8.0]])
        X, y = g.make_tabular(300, 2, 2, centers=centers, contamination=0.05, seed=6)
        assert X.shape == (300, 2)

    def test_works_with_numpy_seed(self):
        X, y = g.make_tabular(100, 3, 2, seed=np.random.default_rng(1))
        assert X.shape == (100, 3)


class TestMakeTimeSeries:
    def test_shape(self):
        X, y, values = g.make_time_series(400, contamination=0.03, seed=20)
        assert X.shape == (400, 2)
        assert y.shape == (400,)
        assert values.shape == (400,)
        assert np.allclose(X[:, 0], np.arange(400))
        assert np.allclose(X[:, 1], values)

    def test_contamination_match(self):
        X, y, _ = g.make_time_series(1000, contamination=0.04, seed=20)
        assert int(y.sum()) == 40

    def test_deterministic(self):
        X1, y1, v1 = g.make_time_series(300, seed=21)
        X2, y2, v2 = g.make_time_series(300, seed=21)
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)
        assert np.array_equal(v1, v2)

    def test_no_outliers_when_zero_contamination(self):
        X, y, _ = g.make_time_series(200, contamination=0.0, seed=21)
        assert y.sum() == 0

    def test_point_spikes_are_extreme(self):
        X, y, values = g.make_time_series(1000, contamination=0.02, noise=1.0, seed=30)
        baseline = np.median(np.abs(values))
        flagged = np.abs(values[y == 1] - np.median(values))
        clean = np.abs(values[y == 0] - np.median(values))
        assert flagged.max() > clean.max()

    def test_level_shift_segment_flagged(self):
        X, y, _ = g.make_time_series(600, contamination=0.03, noise=1.0, seed=31)
        flagged_pos = np.flatnonzero(y == 1)
        assert len(flagged_pos) == 18
        diff = np.diff(flagged_pos)
        assert (diff == 1).any()

    def test_trend_is_preserved(self):
        X, y, values = g.make_time_series(500, contamination=0.0, seed=32)
        first, last = values[0], values[-1]
        assert last > first + 0.3

    def test_zero_noise_constant_series(self):
        X, y, values = g.make_time_series(200, contamination=0.03, noise=0.0, seed=33)
        assert np.all(np.isfinite(values))

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            g.make_time_series(0, seed=1)
        with pytest.raises(ValueError):
            g.make_time_series(100, contamination=1.5, seed=1)
        with pytest.raises(ValueError):
            g.make_time_series(100, noise=-1.0, seed=1)
