"""Tests for streaming Half-Space Trees."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection import cli
from anomaly_detection.evaluate import roc_auc
from anomaly_detection.generators import make_tabular
from anomaly_detection.models import HalfSpaceTrees


def test_tree_structure_is_data_independent_and_covers_range():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 3, size=(100, 3))
    a = HalfSpaceTrees(n_estimators=5, max_depth=4, seed=1).fit(X)
    b = HalfSpaceTrees(n_estimators=5, max_depth=4, seed=1).fit(X + 0.01 * rng.normal(size=X.shape).clip(-0.001, 0.001))
    assert a.split_dim_.shape == (5, 2**4 - 1)
    assert a.r_mass_.shape == (5, 2**5 - 1)
    np.testing.assert_array_equal(a.split_dim_, b.split_dim_)
    # root holds every reference point; each level partitions it
    np.testing.assert_allclose(a.r_mass_[:, 0], 100)
    for k in range(1, 5):
        level = slice(2**k - 1, 2 ** (k + 1) - 1)
        np.testing.assert_allclose(a.r_mass_[:, level].sum(axis=1), 100)


def test_batch_detects_injected_outliers():
    X, y = make_tabular(n_samples=400, n_features=4, contamination=0.05, seed=3)
    hst = HalfSpaceTrees(n_estimators=50, max_depth=8, seed=0).fit(X)
    scores = hst.scores_
    assert scores.shape == (400,) and np.all(scores >= 0)
    assert roc_auc(y, scores) > 0.85
    flags = hst.predict(X, contamination=0.05)
    assert flags.sum() == pytest.approx(20, abs=3)
    far = hst.score_samples(np.full((1, 4), 50.0))[0]
    assert far >= scores.max() - 1e-9


def test_mass_matches_manual_single_tree():
    X = np.array([[0.1], [0.2], [0.3], [0.9]])
    hst = HalfSpaceTrees(n_estimators=1, max_depth=3, size_limit=0.0, seed=0).fit(X)
    paths = hst._paths(np.array([[0.25]]))[0, :, 0]
    expected_mass = hst.r_mass_[0, paths[-1]] * 2**3
    max_mass = 4 * 2**3
    expected = np.log2(1 + max_mass) - np.log2(1 + expected_mass)
    assert hst.score_one([0.25]) == pytest.approx(expected)


def test_size_limit_stops_descent_early():
    X = np.random.default_rng(1).normal(size=(200, 2))
    loose = HalfSpaceTrees(n_estimators=10, max_depth=10, size_limit=0.0, seed=0).fit(X)
    strict = HalfSpaceTrees(n_estimators=10, max_depth=10, size_limit=1e9, seed=0).fit(X)
    # with an impossible limit every point stops at the root: mass = reference size
    s = strict.score_samples(X)
    np.testing.assert_allclose(s, s[0])
    assert np.ptp(loose.score_samples(X)) > 0


def test_streaming_windows_roll_and_track_drift():
    rng = np.random.default_rng(4)
    before = rng.normal(0.0, 1.0, size=(300, 2))
    after = rng.normal(6.0, 1.0, size=(300, 2))
    stream = np.vstack([before, after])
    hst = HalfSpaceTrees(n_estimators=30, max_depth=8, window_size=100, feature_range=(-5, 11), seed=0)
    scores = hst.score_stream(stream)
    assert np.all(np.isnan(scores[:100])) and np.all(np.isfinite(scores[100:]))
    assert hst.n_windows_ == 6 and hst.n_seen_in_window_ == 0
    # right after the shift the new regime looks anomalous ...
    assert scores[300:320].mean() > scores[200:300].mean() + 1.0
    # ... and once a window of the new regime is the reference it looks normal again
    assert scores[500:600].mean() < scores[300:320].mean() - 1.0


def test_learn_one_requires_range_and_validation():
    with pytest.raises(ValueError):
        HalfSpaceTrees().learn_one([0.0, 1.0])
    with pytest.raises(ValueError):
        HalfSpaceTrees().score_samples(np.zeros((2, 2)))
    for bad in ({"n_estimators": 0}, {"max_depth": 0}, {"max_depth": 25}, {"window_size": 0}, {"size_limit": -1}):
        with pytest.raises(ValueError):
            HalfSpaceTrees(**bad)
    hst = HalfSpaceTrees(n_estimators=3, max_depth=3).fit(np.zeros((5, 2)))
    with pytest.raises(ValueError):
        hst.score_samples(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        hst.score_samples(np.array([[np.nan, 0.0]]))
    with pytest.raises(ValueError):
        HalfSpaceTrees(feature_range=(1.0, 0.0)).learn_one([0.5])
    fresh = HalfSpaceTrees(n_estimators=3, max_depth=3, window_size=5, feature_range=(0, 1))
    fresh.learn_one([0.5])
    assert np.isnan(fresh.score_one([0.5]))


def test_cli_hst_detector():
    X, y = make_tabular(n_samples=200, n_features=3, contamination=0.05, seed=0)
    scores = cli._detector_scores("hst", X, seed=0)
    assert scores.shape == (200,) and roc_auc(y, scores) > 0.8
