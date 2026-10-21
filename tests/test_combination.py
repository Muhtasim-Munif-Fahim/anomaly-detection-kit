import numpy as np
import pytest

from anomaly_detection.cli import main
from anomaly_detection.combination import (
    aom_scores,
    average_scores,
    bucket_assignments,
    combine_scores,
    lscp_scores,
    max_scores,
    median_scores,
    moa_scores,
    standardize_scores,
)
from anomaly_detection.evaluate import roc_auc
from anomaly_detection.generators import make_tabular
from anomaly_detection.models import HBOS, KNN, LocalOutlierFactor

S = np.array(
    [
        [0.1, 2.0, -1.0, 0.5, 3.0, 1.0],
        [1.5, -0.3, 0.2, 2.2, 0.0, -1.1],
        [-0.7, 0.9, 1.8, -0.2, 0.4, 2.5],
        [2.1, 1.1, 0.3, 0.8, -1.5, 0.6],
    ]
)
# Buckets that combo/PyOD build for static AOM/MOA with n_buckets=3 and random_state=7
# (sklearn shuffle -> [3, 5, 0, 2, 1, 4]).
COMBO_BUCKETS = [[3, 5], [0, 2], [1, 4]]


def test_matches_combo_reference_values():
    # Reference values from combo.models.score_comb (PyOD's combination backend).
    np.testing.assert_allclose(aom_scores(S, buckets=COMBO_BUCKETS), [1.36666667, 1.23333333, 1.73333333, 1.33333333], atol=1e-8)
    np.testing.assert_allclose(moa_scores(S, buckets=COMBO_BUCKETS), [2.5, 0.85, 1.15, 1.2], atol=1e-12)
    np.testing.assert_allclose(average_scores(S, [1, 2, 3, 1, 1, 2]), [0.66, 0.15, 1.17, 0.57], atol=1e-12)
    np.testing.assert_allclose(max_scores(S), S.max(axis=1))
    np.testing.assert_allclose(median_scores(S), np.median(S, axis=1))
    np.testing.assert_allclose(average_scores(S), S.mean(axis=1))


def test_standardize_zscore_uses_training_stats():
    rng = np.random.default_rng(0)
    T = rng.normal(3.0, 2.0, size=(50, 3))
    T[:, 2] = 4.0  # constant column -> scale 1
    X = rng.normal(size=(7, 3))
    Z = standardize_scores(X, T)
    sd = T.std(axis=0)
    sd[2] = 1.0
    np.testing.assert_allclose(Z, (X - T.mean(axis=0)) / sd)
    R = standardize_scores(np.array([[T[:, 0].min()], [1e9]]), T[:, :1], method="rank")
    np.testing.assert_allclose(R.ravel(), [1 / 50, 1.0])
    M = standardize_scores(T[:, :2], method="minmax")
    assert M.min() == 0.0 and M.max() == 1.0
    with pytest.raises(ValueError):
        standardize_scores(X, T[:, :2])
    with pytest.raises(ValueError):
        standardize_scores(X, method="bogus")


def test_bucket_assignments_static_and_dynamic():
    b = bucket_assignments(6, 3, "static", seed=1)
    assert sorted(np.concatenate(b).tolist()) == list(range(6))
    assert all(len(x) == 2 for x in b)
    d = bucket_assignments(10, 4, "dynamic", seed=2)
    assert len(d) == 4 and all(2 <= len(x) <= 5 and len(set(x)) == len(x) for x in d)
    assert [x.tolist() for x in bucket_assignments(10, 4, "dynamic", seed=2)] == [x.tolist() for x in d]
    with pytest.raises(ValueError):
        bucket_assignments(7, 3, "static")
    with pytest.raises(ValueError):
        bucket_assignments(4, 5)


def test_aom_moa_bounds():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 8))
    for seed in range(5):
        a = aom_scores(X, n_buckets=4, seed=seed)
        m = moa_scores(X, n_buckets=4, seed=seed)
        assert np.all(a <= X.max(axis=1) + 1e-12) and np.all(a >= X.mean(axis=1) - 1e-12)
        assert np.all(m >= X.mean(axis=1) - 1e-12) and np.all(m <= X.max(axis=1) + 1e-12)


def test_combine_scores_dispatch_and_bucket_fallback():
    Z = standardize_scores(S)
    np.testing.assert_allclose(combine_scores(S, "average"), Z.mean(axis=1))
    np.testing.assert_allclose(combine_scores(S, "max"), Z.max(axis=1))
    np.testing.assert_allclose(combine_scores(S, "aom", buckets=COMBO_BUCKETS), aom_scores(Z, buckets=COMBO_BUCKETS))
    # 6 detectors, requested 5 buckets -> falls back to 3 static buckets
    assert combine_scores(S, "moa", n_buckets=5, seed=0).shape == (4,)
    assert combine_scores(S[:, :3], "aom", n_buckets=2, seed=0).shape == (4,)
    with pytest.raises(ValueError):
        combine_scores(S, "bogus")


def _detector_matrix(X):
    return np.column_stack([
        KNN().fit(X).score_samples(X),
        LocalOutlierFactor().fit(X).score_samples(X),
        HBOS().fit(X).score_samples(X),
    ])


def test_ensemble_is_competitive_on_synthetic_data():
    X, y = make_tabular(n_samples=300, n_features=4, contamination=0.05, seed=4)
    M = _detector_matrix(X)
    single = [roc_auc(y, M[:, j]) for j in range(M.shape[1])]
    for method in ("average", "aom", "moa", "median"):
        auc = roc_auc(y, combine_scores(M, method, n_buckets=3, bucket_method="dynamic", seed=0))
        assert auc >= min(single) - 1e-9
        assert auc > 0.8


def test_lscp_selects_the_locally_informative_detector():
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, size=(200, 2))
    signal = np.abs(X[:, 0]) * 3
    noise = rng.normal(size=200)
    train = np.column_stack([signal, noise] + [signal + 0.05 * rng.normal(size=200) for _ in range(3)])
    res = lscp_scores(X, train, n_neighbors=60, pseudo_target="average")
    assert res.scores.shape == (200,)
    assert res.local_correlation.shape == (200, 5)
    # the noise detector (column 1) is almost never selected
    assert np.mean(res.selected[:, 0] == 1) < 0.1
    Xt = rng.uniform(-1, 1, size=(10, 2))
    tt = np.column_stack([np.abs(Xt[:, 0]) * 3, rng.normal(size=10)] + [np.abs(Xt[:, 0]) * 3] * 3)
    r2 = lscp_scores(X, train, Xt, tt, n_neighbors=20, n_select=2, aggregate="max")
    assert r2.selected.shape == (10, 2)
    Zt = standardize_scores(tt, train)
    np.testing.assert_allclose(r2.scores, [Zt[i, r2.selected[i]].max() for i in range(10)])


def test_lscp_validation_and_constant_detector():
    X = np.arange(20, dtype=float)[:, None]
    train = np.column_stack([np.ones(20), X.ravel() ** 2])
    res = lscp_scores(X, train, n_neighbors=5, pseudo_target="average")
    assert np.all(res.selected[:, 0] == 1)  # constant detector never chosen
    with pytest.raises(ValueError):
        lscp_scores(X, train, n_neighbors=20)  # self-scoring allows at most n-1
    with pytest.raises(ValueError):
        lscp_scores(X, train, n_select=3)
    with pytest.raises(ValueError):
        lscp_scores(X, train, X_test=X)
    with pytest.raises(ValueError):
        lscp_scores(X, train, pseudo_target="median")


def test_cli_ensemble(tmp_path, capsys):
    data = tmp_path / "d.csv"
    main(["simulate", "--n-samples", "150", "--out", str(data), "--seed", "1"])
    capsys.readouterr()
    assert main(["ensemble", "--data", str(data), "--detectors", "knn,lof,hbos,ecod", "--method", "aom",
                 "--buckets", "2", "--seed", "0", "--contamination", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "ensemble[aom] of knn,lof,hbos,ecod" in out and "auc=" in out
    for name in ("knn", "lof", "hbos", "ecod"):
        assert f"  {name}:" in out
    assert main(["ensemble", "--data", str(data), "--detectors", "knn,hbos", "--method", "lscp",
                 "--neighbors", "15"]) == 0
    assert "ensemble[lscp]" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["ensemble", "--data", str(data), "--detectors", "knn,nope"])
