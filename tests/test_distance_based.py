import itertools

import numpy as np
import pytest

from anomaly_detection import LDOF, SamplingOutlierDetector
from anomaly_detection.cli import _detector_scores
from anomaly_detection.evaluate import roc_auc


def _data(seed=0):
    rng = np.random.default_rng(seed)
    dense = rng.normal(0.0, 0.3, size=(150, 2))
    sparse = rng.normal(6.0, 1.5, size=(80, 2))
    out = np.array([[2.0, 0.0], [0.0, -2.2], [-1.8, 1.6], [13.0, 13.0]])
    X = np.vstack([dense, sparse, out])
    y = np.r_[np.zeros(230, dtype=int), np.ones(4, dtype=int)]
    return X, y


def _brute_ldof(X, i, k):
    d = np.linalg.norm(X - X[i], axis=1)
    d[i] = np.inf
    nn = np.argsort(d)[:k]
    d_bar = d[nn].mean()
    pairs = [np.linalg.norm(X[a] - X[b]) for a, b in itertools.permutations(nn, 2)]
    return d_bar / np.mean(pairs)


def test_ldof_matches_definition():
    X = np.random.default_rng(1).normal(size=(40, 3))
    model = LDOF(n_neighbors=6).fit(X)
    for i in (0, 11, 39):
        assert model.scores_[i] == pytest.approx(_brute_ldof(X, i, 6))


def test_ldof_new_points_use_all_training_neighbours():
    X = np.random.default_rng(2).normal(size=(30, 2))
    model = LDOF(n_neighbors=5).fit(X)
    q = np.array([[0.1, -0.2], [8.0, 8.0]])
    Xa = np.vstack([X, q[:1]])
    expected = _brute_ldof(Xa, 30, 5)
    assert model.score_samples(q)[0] == pytest.approx(expected)
    assert model.score_samples(q)[1] > 5 * model.score_samples(q)[0]


def test_ldof_and_sampling_rank_outliers_high():
    X, y = _data()
    ldof = LDOF(n_neighbors=10).fit(X)
    assert roc_auc(y, ldof.scores_) > 0.95
    flags = ldof.predict(X, contamination=4 / 234)
    assert flags[-1] == 1
    samp = SamplingOutlierDetector(subsample_size=30, seed=0).fit(X)
    assert roc_auc(y, samp.scores_) > 0.9
    assert samp.scores_.argmax() == X.shape[0] - 1


def test_sampling_definition_and_self_exclusion():
    X = np.random.default_rng(3).normal(size=(50, 2))
    m = SamplingOutlierDetector(subsample_size=8, seed=5).fit(X)
    S = X[m.sample_indices_]
    for i in range(50):
        d = np.linalg.norm(S - X[i], axis=1)
        d[m.sample_indices_ == i] = np.inf
        assert m.scores_[i] == pytest.approx(d.min())
    assert np.all(m.scores_ > 0)
    # reproducible
    m2 = SamplingOutlierDetector(subsample_size=8, seed=5).fit(X)
    np.testing.assert_array_equal(m.sample_indices_, m2.sample_indices_)
    # subsample capped at n
    tiny = SamplingOutlierDetector(subsample_size=100, seed=0).fit(X[:5])
    assert tiny.sample_indices_.size == 5


def test_duplicates_do_not_produce_nan():
    X = np.vstack([np.zeros((10, 2)), [[5.0, 5.0]]])
    s = LDOF(n_neighbors=3).fit(X).scores_
    assert np.all(np.isfinite(s))
    assert s.argmax() == 10


def test_validation_and_cli():
    with pytest.raises(ValueError):
        LDOF(n_neighbors=1)
    with pytest.raises(ValueError):
        LDOF().score_samples(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        LDOF(n_neighbors=3).fit(np.zeros((5, 2))).score_samples(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        SamplingOutlierDetector(subsample_size=0)
    with pytest.raises(ValueError):
        SamplingOutlierDetector().fit(np.array([[np.nan, 1.0]]))
    X = np.random.default_rng(4).normal(size=(40, 3))
    for name in ("ldof", "sampling"):
        scores = _detector_scores(name, X, seed=0)
        assert scores.shape == (40,)
        assert np.all(np.isfinite(scores))
