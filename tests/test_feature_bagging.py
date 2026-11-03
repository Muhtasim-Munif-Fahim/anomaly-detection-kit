import numpy as np
import pytest

from anomaly_detection.evaluate import roc_auc
from anomaly_detection.feature_bagging import FeatureBagging
from anomaly_detection.generators import make_tabular
from anomaly_detection.models import HBOS, KNN, LocalOutlierFactor


def _data(seed=0, n=300, d=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.zeros(n, dtype=int)
    idx = rng.choice(n, 15, replace=False)
    y[idx] = 1
    X[idx, :3] += rng.choice([-1.0, 1.0], size=(15, 3)) * 3.5
    return X, y


def test_default_subset_sizes_follow_paper_rule():
    X, _ = _data()
    fb = FeatureBagging(n_estimators=25, seed=1).fit(X)
    sizes = [len(f) for f in fb.features_]
    assert min(sizes) >= 6 and max(sizes) <= 11
    assert all(len(np.unique(f)) == len(f) for f in fb.features_)
    assert fb.train_scores_.shape == (300, 25)


@pytest.mark.parametrize("mf,expected", [(4, 4), (0.5, 6), (1.0, 12)])
def test_max_features_int_and_fraction(mf, expected):
    X, _ = _data()
    fb = FeatureBagging(n_estimators=3, max_features=mf, seed=0).fit(X)
    assert all(len(f) == expected for f in fb.features_)


@pytest.mark.parametrize("combination", ["average", "max", "breadth_first"])
def test_detects_planted_outliers(combination):
    X, y = _data()
    fb = FeatureBagging(n_estimators=15, combination=combination, seed=0)
    scores = fb.fit(X).score_samples(X)
    assert scores.shape == (300,)
    assert roc_auc(y, scores) > 0.9


def test_seed_is_deterministic():
    X, _ = _data()
    a = FeatureBagging(n_estimators=5, seed=3).fit(X).score_samples(X)
    b = FeatureBagging(n_estimators=5, seed=3).fit(X).score_samples(X)
    np.testing.assert_allclose(a, b)


def test_instance_base_is_copied_and_callable_base_works():
    X, y = _data()
    base = KNN(n_neighbors=5)
    fb = FeatureBagging(base_estimator=base, n_estimators=4, seed=0).fit(X)
    assert all(est is not base for est in fb.estimators_)
    fb2 = FeatureBagging(base_estimator=lambda: HBOS(), n_estimators=4, seed=0).fit(X)
    assert isinstance(fb2.estimators_[0], HBOS)
    assert roc_auc(y, fb2.score_samples(X)) > 0.8


def test_scores_new_data_with_knn_members():
    X, _ = _data()
    fb = FeatureBagging(base_estimator=KNN(n_neighbors=5), n_estimators=6, seed=0).fit(X)
    X_new = np.vstack([np.zeros((1, 12)), np.full((1, 12), 8.0)])
    s = fb.score_samples(X_new)
    assert s[1] > s[0]


def test_average_is_mean_of_standardised_members():
    X, _ = _data()
    fb = FeatureBagging(n_estimators=4, seed=0).fit(X)
    T = fb.train_scores_
    Z = (T - T.mean(0)) / T.std(0)
    np.testing.assert_allclose(fb.score_samples(X), Z.mean(1))


def test_breadth_first_puts_each_members_top_point_first():
    X, _ = _data()
    fb = FeatureBagging(n_estimators=5, combination="breadth_first", seed=0).fit(X)
    s = fb.score_samples(X)
    tops = set(np.argmax(fb.train_scores_, axis=0))
    top_k = set(np.argsort(-s)[: len(tops)])
    assert tops == top_k


def test_fit_predict_contamination_and_make_tabular():
    X, y = make_tabular(n_samples=300, n_features=6, contamination=0.05, seed=2)
    flags = FeatureBagging(n_estimators=8, seed=0).fit_predict(X, contamination=0.05)
    assert flags.sum() == 15
    assert roc_auc(y, FeatureBagging(n_estimators=8, seed=0).fit(X).score_samples(X)) > 0.8


def test_validation():
    X, _ = _data()
    with pytest.raises(ValueError):
        FeatureBagging(n_estimators=0)
    with pytest.raises(ValueError):
        FeatureBagging(combination="vote")
    with pytest.raises(ValueError):
        FeatureBagging(max_features=13).fit(X)
    with pytest.raises(ValueError):
        FeatureBagging(max_features=1.5).fit(X)
    with pytest.raises(ValueError):
        FeatureBagging().score_samples(X)
    with pytest.raises(ValueError):
        FeatureBagging(n_estimators=2).fit(X).score_samples(X[:, :5])
    with pytest.raises(ValueError):
        FeatureBagging().fit(X[:1])
    with pytest.raises(TypeError):
        FeatureBagging(base_estimator=object(), n_estimators=1).fit(X)
