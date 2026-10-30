"""Tests for the Robust Random Cut Forest."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection import cli
from anomaly_detection.evaluate import roc_auc
from anomaly_detection.generators import make_tabular
from anomaly_detection.rrcf import RandomCutTree, RobustRandomCutForest, shingle


def test_tree_build_invariants_and_codisp_of_isolated_point():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(60, 2)), [[25.0, 25.0]]])
    tree = RandomCutTree(np.random.default_rng(1)).build(X)
    tree.check()
    assert len(tree) == 61
    # the far point is cut off near the root, displacing a large mass
    assert tree.depth(60) <= 3
    assert tree.codisp(60) >= 20.0
    assert tree.codisp(60) > np.median([tree.codisp(i) for i in range(60)])
    # a cut that isolates a leaf at the root has CoDisp == size of the rest
    t2 = RandomCutTree(np.random.default_rng(0)).build(np.array([[0.0], [0.1], [0.2], [100.0]]))
    assert t2.depth(3) == 1
    assert t2.codisp(3) == pytest.approx(3.0)


def test_insert_delete_roundtrip_keeps_invariants():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    tree = RandomCutTree(np.random.default_rng(3)).build(X)
    extra = rng.normal(size=(25, 3))
    for i, x in enumerate(extra):
        tree.insert(x, ("e", i))
        tree.check()
    assert len(tree) == 65
    for i in range(0, 40, 2):
        tree.forget(i)
        tree.check()
    for i in range(25):
        tree.forget(("e", i))
    tree.check()
    assert len(tree) == 20
    assert set(tree.leaves) == set(range(1, 40, 2))
    for k in list(tree.leaves):
        tree.forget(k)
    assert tree.root is None and len(tree) == 0


def test_duplicates_share_a_leaf_and_count():
    tree = RandomCutTree(np.random.default_rng(0))
    for k in range(3):
        tree.insert([1.0, 2.0], k)
    tree.insert([5.0, 2.0], "b")
    tree.check()
    assert tree.leaves[0] is tree.leaves[2] and tree.leaves[0].n == 3
    assert tree.codisp("b") == pytest.approx(3.0)
    assert tree.codisp(0) == pytest.approx(1 / 3)
    tree.forget(1)
    tree.check()
    assert tree.leaves[0].n == 2
    with pytest.raises(KeyError):
        tree.insert([0.0, 0.0], "b")


def test_insertion_matches_batch_distribution_on_root_cut():
    # For points {0, 1, 3} on a line the root cut lies in (0, 1) w.p. 1/3.
    # A tree grown by insertion must reproduce that law.
    hits = 0
    trials = 3000
    rng = np.random.default_rng(4)
    for _ in range(trials):
        tree = RandomCutTree(np.random.default_rng(rng.integers(2**32)))
        order = rng.permutation(3)
        pts = [[0.0], [1.0], [3.0]]
        for i in order:
            tree.insert(pts[i], i)
        hits += tree.root.val < 1.0
    assert hits / trials == pytest.approx(1 / 3, abs=0.03)


def test_forest_detects_injected_outliers_and_is_reproducible():
    X, y = make_tabular(n_samples=300, n_features=4, contamination=0.05, seed=3)
    a = RobustRandomCutForest(n_estimators=30, tree_size=128, seed=0).fit(X)
    b = RobustRandomCutForest(n_estimators=30, tree_size=128, seed=0).fit(X)
    np.testing.assert_allclose(a.scores_, b.scores_)
    assert a.scores_.shape == (300,) and np.all(a.scores_ >= 0)
    assert roc_auc(y, a.scores_) > 0.85
    assert a.fit_predict(X, contamination=0.05).sum() == pytest.approx(15, abs=3)


def test_score_samples_leaves_forest_unchanged():
    X, _ = make_tabular(n_samples=120, n_features=3, contamination=0.0, seed=1)
    f = RobustRandomCutForest(n_estimators=10, tree_size=64, seed=5).fit(X)
    sizes = [len(t) for t in f.trees_]
    s = f.score_samples(np.vstack([X[:5], np.full((1, 3), 40.0)]))
    assert [len(t) for t in f.trees_] == sizes
    for t in f.trees_:
        t.check()
    assert s[-1] > s[:5].max()
    assert f.predict(X).shape == (120,)


def test_streaming_window_and_spike():
    rng = np.random.default_rng(7)
    series = np.sin(np.arange(600) * 2 * np.pi / 50) + 0.05 * rng.normal(size=600)
    series[400] += 3.0
    S = shingle(series, 4)
    f = RobustRandomCutForest(n_estimators=20, tree_size=128, seed=0)
    scores = f.score_stream(S)
    assert scores.shape == (S.shape[0],)
    assert all(len(t) == 128 for t in f.trees_)
    # every shingle containing index 400 starts at 397..400
    assert int(np.argmax(scores[200:])) + 200 in range(397, 401)


def test_shingle_and_validation():
    np.testing.assert_array_equal(shingle([1, 2, 3, 4], 3), [[1, 2, 3], [2, 3, 4]])
    with pytest.raises(ValueError):
        shingle([1, 2], 3)
    with pytest.raises(ValueError):
        RobustRandomCutForest(n_estimators=0)
    with pytest.raises(ValueError):
        RobustRandomCutForest().score_samples(np.zeros((2, 2)))
    f = RobustRandomCutForest(n_estimators=3, tree_size=8).fit(np.random.default_rng(0).normal(size=(10, 2)))
    with pytest.raises(ValueError):
        f.score_samples(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        f.score_samples([[np.nan, 0.0]])


def test_cli_rrcf_detector():
    X, y = make_tabular(n_samples=150, n_features=3, contamination=0.05, seed=2)
    scores = cli._detector_scores("rrcf", X, seed=0)
    assert scores.shape == (150,) and roc_auc(y, scores) > 0.8
