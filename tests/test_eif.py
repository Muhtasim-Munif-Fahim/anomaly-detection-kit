"""Tests for the Extended Isolation Forest detector."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection import cli
from anomaly_detection.evaluate import roc_auc
from anomaly_detection.models import ExtendedIsolationForest, IsolationForest


def _two_blobs(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(200, 2)) * 0.5 + [-3.0, -3.0]
    b = rng.normal(size=(200, 2)) * 0.5 + [3.0, 3.0]
    return np.vstack([a, b])


def test_scores_in_unit_interval_and_detect_outliers():
    rng = np.random.default_rng(0)
    inliers = rng.normal(size=(300, 4))
    outliers = rng.normal(size=(15, 4)) * 0.3 + 6.0
    X = np.vstack([inliers, outliers])
    y = np.r_[np.zeros(300), np.ones(15)]
    model = ExtendedIsolationForest(n_estimators=100, seed=0).fit(X)
    s = model.score_samples(X)
    assert s.shape == (315,)
    assert np.all((s > 0) & (s <= 1))
    assert np.allclose(model.scores_, s)
    assert roc_auc(y, s) > 0.95
    assert model.extension_level_ == 3
    assert model.max_samples_ == 256


def test_extension_level_controls_normal_sparsity():
    X = np.random.default_rng(1).normal(size=(120, 5))
    for level in (0, 2, 4):
        model = ExtendedIsolationForest(n_estimators=5, extension_level=level, seed=0).fit(X)
        for tree in model._trees:
            internal = tree["left"] >= 0
            nnz = (tree["normals"][internal] != 0).sum(axis=1)
            assert np.all(nnz == level + 1)


def test_full_extension_removes_ghost_region():
    X = _two_blobs()
    q = np.array([[0.0, 0.0], [3.0, -3.0], [-3.0, 3.0]])
    gaps = {}
    for level in (0, 1):
        s = np.mean(
            [
                ExtendedIsolationForest(n_estimators=150, extension_level=level, seed=sd).fit(X).score_samples(q)
                for sd in range(3)
            ],
            axis=0,
        )
        # how much more "normal" the empty corners look than the gap at the origin
        gaps[level] = s[0] - s[1:].mean()
    assert gaps[0] > 0.04
    assert gaps[1] < gaps[0] - 0.03


def test_axis_parallel_mode_tracks_classic_forest():
    X = _two_blobs(1)
    q = np.array([[3.0, 3.0], [0.0, 0.0], [8.0, 8.0]])
    e0 = ExtendedIsolationForest(n_estimators=200, extension_level=0, seed=0).fit(X).score_samples(q)
    assert e0[0] < e0[1] < e0[2]
    iso = IsolationForest(n_estimators=200, seed=0).fit(X).score_samples(q)
    assert np.argsort(e0).tolist() == np.argsort(iso).tolist()


def test_deterministic_and_predict():
    X = np.random.default_rng(2).normal(size=(80, 3))
    a = ExtendedIsolationForest(n_estimators=20, seed=5).fit(X).score_samples(X)
    b = ExtendedIsolationForest(n_estimators=20, seed=5).fit(X).score_samples(X)
    assert np.array_equal(a, b)
    flags = ExtendedIsolationForest(n_estimators=20, seed=0).fit_predict(X, contamination=0.1)
    assert flags.sum() == 8
    one_d = ExtendedIsolationForest(n_estimators=20, seed=0).fit(X[:, :1])
    assert one_d.extension_level_ == 0
    assert one_d.score_samples(X[:5, :1]).shape == (5,)


def test_constant_data_and_small_samples():
    X = np.ones((10, 3))
    s = ExtendedIsolationForest(n_estimators=5, seed=0).fit(X).score_samples(X)
    assert np.allclose(s, s[0])
    s2 = ExtendedIsolationForest(n_estimators=5, max_samples=2, seed=0).fit(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert s2.max_samples_ == 2


def test_validation():
    with pytest.raises(ValueError):
        ExtendedIsolationForest(n_estimators=0)
    with pytest.raises(ValueError):
        ExtendedIsolationForest(max_samples=1)
    with pytest.raises(ValueError):
        ExtendedIsolationForest(extension_level=-1)
    with pytest.raises(ValueError):
        ExtendedIsolationForest(max_depth=0)
    with pytest.raises(ValueError):
        ExtendedIsolationForest(extension_level=3).fit(np.zeros((10, 2)))
    with pytest.raises(ValueError):
        ExtendedIsolationForest().fit(np.zeros((1, 2)))
    with pytest.raises(ValueError):
        ExtendedIsolationForest().fit(np.array([[0.0, np.nan], [1.0, 2.0]]))
    with pytest.raises(ValueError):
        ExtendedIsolationForest().score_samples(np.zeros((3, 2)))
    model = ExtendedIsolationForest(n_estimators=3).fit(np.random.default_rng(0).normal(size=(20, 2)))
    with pytest.raises(ValueError):
        model.score_samples(np.zeros((3, 3)))


def test_cli_eif_detector():
    X = np.random.default_rng(3).normal(size=(60, 3))
    scores = cli._detector_scores("eif", X, seed=0)
    assert scores.shape == (60,)
    assert np.all((scores > 0.0) & (scores <= 1.0))
