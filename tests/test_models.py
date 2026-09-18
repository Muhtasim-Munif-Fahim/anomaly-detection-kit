import numpy as np
import pytest

from anomaly_detection import generators as g
from anomaly_detection import models as m


class TestIsolationForest:
    def test_scores_in_unit_range(self):
        X, _ = g.make_tabular(300, 4, 3, contamination=0.05, seed=1)
        iforest = m.IsolationForest(n_estimators=50, max_samples=128, seed=7).fit(X)
        s = iforest.score_samples(X)
        assert s.shape == (300,)
        assert s.min() >= 0.0 and s.max() <= 1.0

    def test_outliers_score_higher_than_inliers(self):
        X, y = g.make_tabular(600, 4, 3, contamination=0.05, outlier_types="shift", seed=2)
        iforest = m.IsolationForest(n_estimators=100, max_samples=256, seed=7).fit(X)
        s = iforest.score_samples(X)
        assert s[y == 1].mean() > s[y == 0].mean()

    def test_predict_flags_exact_contamination(self):
        X, _ = g.make_tabular(500, 4, 3, contamination=0.05, seed=3)
        iforest = m.IsolationForest(n_estimators=50, max_samples=128, seed=7)
        flags = iforest.fit_predict(X, contamination=0.05)
        assert flags.sum() == 25

    def test_predict_zero_and_full_contamination(self):
        X, _ = g.make_tabular(200, 4, 3, contamination=0.05, seed=4)
        iforest = m.IsolationForest(n_estimators=20, max_samples=64, seed=7)
        assert iforest.fit_predict(X, contamination=0.0).sum() == 0
        assert iforest.fit_predict(X, contamination=1.0).sum() == 200

    def test_deterministic_with_seed(self):
        X, _ = g.make_tabular(300, 4, 3, contamination=0.05, seed=5)
        s1 = m.IsolationForest(n_estimators=40, seed=9).fit(X).score_samples(X)
        s2 = m.IsolationForest(n_estimators=40, seed=9).fit(X).score_samples(X)
        assert np.array_equal(s1, s2)

    def test_different_seeds_give_different_scores(self):
        X, _ = g.make_tabular(300, 4, 3, contamination=0.05, seed=6)
        s1 = m.IsolationForest(n_estimators=40, seed=9).fit(X).score_samples(X)
        s2 = m.IsolationForest(n_estimators=40, seed=10).fit(X).score_samples(X)
        assert not np.allclose(s1, s2)

    def test_constant_columns_give_uniform_scores(self):
        X = np.full((100, 3), 4.0)
        iforest = m.IsolationForest(n_estimators=10, seed=1).fit(X)
        s = iforest.score_samples(X)
        assert np.allclose(s, s[0])
        assert s[0] == 0.5

    def test_max_samples_larger_than_data_clamps(self):
        X, _ = g.make_tabular(50, 4, 2, contamination=0.05, seed=8)
        iforest = m.IsolationForest(n_estimators=10, max_samples=1000, seed=1).fit(X)
        assert iforest.score_samples(X).shape == (50,)

    def test_single_estimator_works(self):
        X, _ = g.make_tabular(100, 3, 2, contamination=0.05, seed=9)
        s = m.IsolationForest(n_estimators=1, seed=1).fit(X).score_samples(X)
        assert s.shape == (100,)

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError):
            m.IsolationForest(seed=1).fit(np.zeros((1, 3)))

    def test_scoring_before_fit_raises(self):
        with pytest.raises(ValueError):
            m.IsolationForest(seed=1).score_samples(np.zeros((5, 3)))

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            m.IsolationForest(seed=1).fit(np.arange(10.0))

    def test_fit_predict_matches_predict(self):
        X, _ = g.make_tabular(300, 4, 3, contamination=0.05, seed=11)
        iforest = m.IsolationForest(n_estimators=30, seed=3)
        f1 = iforest.fit(X).predict(X, contamination=0.1)
        f2 = iforest.fit_predict(X, contamination=0.1)
        assert np.array_equal(f1, f2)


class TestLocalOutlierFactor:
    def test_scores_finite_and_positive(self):
        X, _ = g.make_tabular(300, 4, 3, contamination=0.05, seed=12)
        lof = m.LocalOutlierFactor(n_neighbors=20).fit(X)
        s = lof.score_samples(X)
        assert s.shape == (300,)
        assert np.all(np.isfinite(s))
        assert (s > 0).all()

    def test_outliers_score_higher_than_inliers(self):
        X, y = g.make_tabular(600, 4, 3, contamination=0.05, outlier_types="shift", seed=13)
        lof = m.LocalOutlierFactor(n_neighbors=20)
        s = lof.fit(X).score_samples(X)
        assert s[y == 1].mean() > s[y == 0].mean()

    def test_scale_invariance(self):
        X, _ = g.make_tabular(200, 4, 3, contamination=0.05, seed=14)
        s1 = m.LocalOutlierFactor(n_neighbors=15).fit(X).score_samples(X)
        s2 = m.LocalOutlierFactor(n_neighbors=15).fit(X * 100.0).score_samples(X * 100.0)
        assert np.allclose(s1, s2, rtol=1e-6)

    def test_predict_flags_exact_contamination(self):
        X, _ = g.make_tabular(500, 4, 3, contamination=0.05, seed=15)
        lof = m.LocalOutlierFactor(n_neighbors=20)
        flags = lof.fit_predict(X, contamination=0.05)
        assert flags.sum() == 25

    def test_predict_zero_contamination(self):
        X, _ = g.make_tabular(200, 4, 3, contamination=0.05, seed=16)
        lof = m.LocalOutlierFactor(n_neighbors=10)
        assert lof.fit_predict(X, contamination=0.0).sum() == 0

    def test_duplicate_points_no_nan(self):
        X = np.repeat(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), 10, axis=0)
        lof = m.LocalOutlierFactor(n_neighbors=5)
        s = lof.fit(X).score_samples(X)
        assert np.all(np.isfinite(s))
        duplicates = s[0:10]
        assert duplicates.mean() < 2.0

    def test_n_neighbors_clamped(self):
        X, _ = g.make_tabular(10, 3, 2, contamination=0.0, seed=17)
        s = m.LocalOutlierFactor(n_neighbors=100).fit(X).score_samples(X)
        assert np.all(np.isfinite(s))

    def test_single_point_scores_one(self):
        s = m.LocalOutlierFactor(n_neighbors=5).fit(np.array([[1.0, 2.0]])).score_samples(
            np.array([[1.0, 2.0]])
        )
        assert np.allclose(s, 1.0)

    def test_deterministic(self):
        X, _ = g.make_tabular(300, 4, 3, contamination=0.05, seed=18)
        s1 = m.LocalOutlierFactor(n_neighbors=15).fit(X).score_samples(X)
        s2 = m.LocalOutlierFactor(n_neighbors=15).fit(X).score_samples(X)
        assert np.array_equal(s1, s2)

    def test_constant_data_all_scores_one(self):
        X = np.full((50, 3), 2.0)
        s = m.LocalOutlierFactor(n_neighbors=10).fit(X).score_samples(X)
        assert np.allclose(s, 1.0)

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            m.LocalOutlierFactor().fit(np.arange(10.0))

    def test_fit_predict_matches_predict(self):
        X, _ = g.make_tabular(300, 4, 3, contamination=0.05, seed=19)
        lof = m.LocalOutlierFactor(n_neighbors=15)
        f1 = lof.fit(X).predict(X, contamination=0.1)
        f2 = lof.fit_predict(X, contamination=0.1)
        assert np.array_equal(f1, f2)

    def test_empty_input(self):
        s = m.LocalOutlierFactor().fit(np.empty((0, 3))).score_samples(np.empty((0, 3)))
        assert s.shape == (0,)


class TestCOPOD:
    def test_scores_finite_and_nonnegative(self):
        X, _ = g.make_tabular(300, 4, 3, contamination=0.05, seed=20)
        s = m.COPOD().fit(X).score_samples(X)
        assert s.shape == (300,)
        assert np.all(np.isfinite(s))
        assert (s >= 0).all()

    def test_outliers_score_higher_than_inliers(self):
        X, y = g.make_tabular(600, 4, 3, contamination=0.05, outlier_types="shift", seed=21)
        s = m.COPOD().fit(X).score_samples(X)
        assert s[y == 1].mean() > s[y == 0].mean()

    def test_recovers_shift_outliers_from_generator(self):
        X, y = g.make_tabular(600, 4, 3, contamination=0.05, outlier_types="shift", seed=9)
        flags = m.COPOD().fit(X).predict(X, contamination=0.05)
        tp = int(((flags == 1) & (y == 1)).sum())
        assert tp >= int(y.sum()) * 0.8

    def test_recovers_mixed_outliers_from_generator(self):
        X, y = g.make_tabular(600, 4, 3, contamination=0.05, seed=22)
        flags = m.COPOD().fit(X).predict(X, contamination=0.05)
        tp = int(((flags == 1) & (y == 1)).sum())
        assert tp >= int(y.sum()) * 0.6

    def test_predict_flags_exact_contamination(self):
        X, _ = g.make_tabular(500, 4, 3, contamination=0.05, seed=23)
        flags = m.COPOD().fit_predict(X, contamination=0.05)
        assert flags.sum() == 25

    def test_predict_zero_and_full_contamination(self):
        X, _ = g.make_tabular(200, 4, 3, contamination=0.05, seed=24)
        copod = m.COPOD().fit(X)
        assert copod.predict(X, contamination=0.0).sum() == 0
        assert copod.predict(X, contamination=1.0).sum() == 200

    def test_positive_scale_invariance(self):
        X, _ = g.make_tabular(200, 4, 3, contamination=0.05, seed=25)
        s1 = m.COPOD().fit(X).score_samples(X)
        s2 = m.COPOD().fit(X * 100.0).score_samples(X * 100.0)
        assert np.allclose(s1, s2)

    def test_translation_invariance(self):
        X, _ = g.make_tabular(200, 4, 3, contamination=0.05, seed=26)
        s1 = m.COPOD().fit(X).score_samples(X)
        s2 = m.COPOD().fit(X + 50.0).score_samples(X + 50.0)
        assert np.allclose(s1, s2)

    def test_both_tails_score_high(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 2))
        X[0] = [-12.0, -12.0]
        X[1] = [12.0, 12.0]
        s = m.COPOD().fit(X).score_samples(X)
        assert s[0] > np.median(s)
        assert s[1] > np.median(s)

    def test_obvious_univariate_outlier_is_highest(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 1))
        X[40, 0] = 50.0
        s = m.COPOD().fit(X).score_samples(X)
        assert s[40] == s.max()

    def test_constant_data_zero_scores(self):
        X = np.full((50, 3), 2.0)
        s = m.COPOD().fit(X).score_samples(X)
        assert np.allclose(s, 0.0)

    def test_deterministic(self):
        X, _ = g.make_tabular(300, 4, 3, contamination=0.05, seed=27)
        s1 = m.COPOD().fit(X).score_samples(X)
        s2 = m.COPOD().fit(X).score_samples(X)
        assert np.array_equal(s1, s2)

    def test_out_of_sample_extreme_scores_high(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(200, 3))
        copod = m.COPOD().fit(X)
        s_in = copod.score_samples(X)
        s_ext = copod.score_samples(np.array([[80.0, 80.0, 80.0]]))
        assert s_ext[0] > s_in.max()

    def test_empty_score_after_fit(self):
        X, _ = g.make_tabular(20, 3, 2, contamination=0.0, seed=28)
        s = m.COPOD().fit(X).score_samples(np.empty((0, 3)))
        assert s.shape == (0,)

    def test_single_sample_zero_score(self):
        s = m.COPOD().fit(np.array([[1.0, 2.0]])).score_samples(np.array([[1.0, 2.0]]))
        assert np.allclose(s, 0.0)

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError):
            m.COPOD().fit(np.zeros((0, 3)))

    def test_scoring_before_fit_raises(self):
        with pytest.raises(ValueError):
            m.COPOD().score_samples(np.zeros((5, 3)))

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            m.COPOD().fit(np.arange(10.0))

    def test_feature_mismatch_raises(self):
        copod = m.COPOD().fit(np.zeros((10, 3)))
        with pytest.raises(ValueError):
            copod.score_samples(np.zeros((4, 2)))

    def test_fit_predict_matches_predict(self):
        X, _ = g.make_tabular(300, 4, 3, contamination=0.05, seed=29)
        copod = m.COPOD()
        f1 = copod.fit(X).predict(X, contamination=0.1)
        f2 = copod.fit_predict(X, contamination=0.1)
        assert np.array_equal(f1, f2)

    def test_ties_share_ecdf_rank(self):
        X = np.array([[1.0], [1.0], [1.0], [2.0]])
        s = m.COPOD().fit(X).score_samples(X)
        assert np.allclose(s[0], s[1])
        assert np.allclose(s[1], s[2])
        assert s[3] > s[0]


class TestHBOS:
    def test_scores_finite_and_nonnegative(self):
        X, _ = g.make_tabular(300, 4, 3, contamination=0.05, seed=30)
        s = m.HBOS().fit(X).score_samples(X)
        assert s.shape == (300,)
        assert np.all(np.isfinite(s))
        assert (s >= 0).all()

    def test_outliers_score_higher_than_inliers(self):
        X, y = g.make_tabular(600, 4, 3, contamination=0.05, outlier_types="shift", seed=31)
        s = m.HBOS().fit(X).score_samples(X)
        assert s[y == 1].mean() > s[y == 0].mean()

    def test_recovers_shift_outliers_from_generator(self):
        X, y = g.make_tabular(600, 4, 3, contamination=0.05, outlier_types="shift", seed=9)
        flags = m.HBOS().fit(X).predict(X, contamination=0.05)
        tp = int(((flags == 1) & (y == 1)).sum())
        assert tp >= int(y.sum()) * 0.8

    def test_recovers_mixed_outliers_from_generator(self):
        X, y = g.make_tabular(600, 4, 3, contamination=0.05, seed=32)
        flags = m.HBOS().fit(X).predict(X, contamination=0.05)
        tp = int(((flags == 1) & (y == 1)).sum())
        assert tp >= int(y.sum()) * 0.6

    def test_predict_flags_at_least_contamination(self):
        X, _ = g.make_tabular(500, 4, 3, contamination=0.05, seed=33)
        hbos = m.HBOS().fit(X)
        scores = hbos.score_samples(X)
        flags = hbos.predict(X, contamination=0.05)
        k = 25
        threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
        # Equal-width bins create tied scores, so rows sharing the cutoff
        # are all flagged (same rule as ``_flags_from_contamination``).
        assert flags.sum() >= k
        assert np.all(scores[flags == 1] >= threshold)
        assert np.all(scores[flags == 0] < threshold)

    def test_predict_zero_and_full_contamination(self):
        X, _ = g.make_tabular(200, 4, 3, contamination=0.05, seed=34)
        hbos = m.HBOS().fit(X)
        assert hbos.predict(X, contamination=0.0).sum() == 0
        assert hbos.predict(X, contamination=1.0).sum() == 200

    def test_positive_scale_invariance(self):
        X, _ = g.make_tabular(200, 4, 3, contamination=0.05, seed=35)
        s1 = m.HBOS().fit(X).score_samples(X)
        s2 = m.HBOS().fit(X * 100.0).score_samples(X * 100.0)
        assert np.allclose(s1, s2)

    def test_translation_invariance(self):
        X, _ = g.make_tabular(200, 4, 3, contamination=0.05, seed=36)
        s1 = m.HBOS().fit(X).score_samples(X)
        s2 = m.HBOS().fit(X + 50.0).score_samples(X + 50.0)
        assert np.allclose(s1, s2)

    def test_both_tails_score_high(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 2))
        X[0] = [-12.0, -12.0]
        X[1] = [12.0, 12.0]
        s = m.HBOS().fit(X).score_samples(X)
        assert s[0] > np.median(s)
        assert s[1] > np.median(s)

    def test_obvious_univariate_outlier_is_highest(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 1))
        X[40, 0] = 50.0
        s = m.HBOS().fit(X).score_samples(X)
        assert s[40] == s.max()

    def test_constant_data_zero_scores(self):
        X = np.full((50, 3), 2.0)
        s = m.HBOS().fit(X).score_samples(X)
        assert np.allclose(s, 0.0)

    def test_deterministic(self):
        X, _ = g.make_tabular(300, 4, 3, contamination=0.05, seed=37)
        s1 = m.HBOS().fit(X).score_samples(X)
        s2 = m.HBOS().fit(X).score_samples(X)
        assert np.array_equal(s1, s2)

    def test_out_of_sample_extreme_scores_high(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(200, 3))
        hbos = m.HBOS().fit(X)
        s_in = hbos.score_samples(X)
        s_ext = hbos.score_samples(np.array([[80.0, 80.0, 80.0]]))
        assert s_ext[0] > s_in.max()

    def test_empty_score_after_fit(self):
        X, _ = g.make_tabular(20, 3, 2, contamination=0.0, seed=38)
        s = m.HBOS().fit(X).score_samples(np.empty((0, 3)))
        assert s.shape == (0,)

    def test_single_sample_zero_score(self):
        s = m.HBOS().fit(np.array([[1.0, 2.0]])).score_samples(np.array([[1.0, 2.0]]))
        assert np.allclose(s, 0.0)

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError):
            m.HBOS().fit(np.zeros((0, 3)))

    def test_scoring_before_fit_raises(self):
        with pytest.raises(ValueError):
            m.HBOS().score_samples(np.zeros((5, 3)))

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            m.HBOS().fit(np.arange(10.0))

    def test_feature_mismatch_raises(self):
        hbos = m.HBOS().fit(np.zeros((10, 3)))
        with pytest.raises(ValueError):
            hbos.score_samples(np.zeros((4, 2)))

    def test_fit_predict_matches_predict(self):
        X, _ = g.make_tabular(300, 4, 3, contamination=0.05, seed=39)
        hbos = m.HBOS()
        f1 = hbos.fit(X).predict(X, contamination=0.1)
        f2 = hbos.fit_predict(X, contamination=0.1)
        assert np.array_equal(f1, f2)

    def test_invalid_n_bins_raises(self):
        with pytest.raises(ValueError):
            m.HBOS(n_bins=1)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            m.HBOS(alpha=1.0)
        with pytest.raises(ValueError):
            m.HBOS(alpha=-0.1)

    def test_invalid_tol_raises(self):
        with pytest.raises(ValueError):
            m.HBOS(tol=1.5)

    def test_sparse_bin_scores_higher_than_dense(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(400, 1))
        X[0, 0] = X.max() + 8.0
        s = m.HBOS(n_bins=10).fit(X).score_samples(X)
        assert s[0] > np.median(s)

    def test_n_bins_still_runs_when_large(self):
        X, _ = g.make_tabular(80, 3, 2, contamination=0.05, seed=40)
        s = m.HBOS(n_bins=40).fit(X).score_samples(X)
        assert s.shape == (80,)
        assert np.all(np.isfinite(s))


class TestFlagsHelper:
    def test_boundary_behaviour(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        flags = m._flags_from_contamination(scores, 0.2)
        assert flags.sum() == 2
        assert flags[-1] == 1 and flags[-2] == 1

    def test_all_equal_scores_flags_all_at_positive_contamination(self):
        flags = m._flags_from_contamination(np.full(10, 0.5), 0.1)
        assert flags.sum() == 10
