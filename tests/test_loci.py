import numpy as np
import pytest

from anomaly_detection import LOCI
from anomaly_detection.cli import DETECTOR_NAMES, _detector_scores
from anomaly_detection.evaluate import roc_auc


def _two_density_data(seed=0):
    rng = np.random.default_rng(seed)
    dense = rng.normal(0.0, 0.3, size=(150, 2))
    sparse = rng.normal(6.0, 1.5, size=(80, 2))
    # Outliers just outside the dense cluster: global methods miss these.
    out = np.array([[1.6, 0.0], [0.0, -1.7], [-1.5, 1.2], [12.5, 12.5]])
    X = np.vstack([dense, sparse, out])
    y = np.r_[np.zeros(230, dtype=int), np.ones(4, dtype=int)]
    return X, y


def _brute_force_mdef(D, i, r, alpha):
    # Reuse the model's distance matrix so ties at r are resolved identically.
    nbrs = np.where(D[i] <= r)[0]
    counts = np.array([(D[q] <= alpha * r).sum() for q in nbrs], dtype=float)
    n_hat = counts.mean()
    own = (D[i] <= alpha * r).sum()
    return 1 - own / n_hat, counts.std() / n_hat


def test_matches_brute_force_definition():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 2))
    model = LOCI(n_min=5, n_radii=6).fit(X)
    mdef, sigma, valid = model._mdef(model._D_train, True)
    for i in (0, 7, 19):
        for j, r in enumerate(model.radii_):
            m, s = _brute_force_mdef(model._D_train, i, r, 0.5)
            assert mdef[i, j] == pytest.approx(m)
            assert sigma[i, j] == pytest.approx(s)


def test_finds_local_outliers_with_ksigma_rule():
    X, y = _two_density_data()
    model = LOCI().fit(X)
    scores = model.score_samples(X)
    assert roc_auc(y, scores) > 0.99
    flags = model.flag(X)
    assert flags[-4:].sum() == 4
    assert flags[:150].mean() < 0.05  # dense cluster interior
    assert flags[:230].mean() < 0.10  # Gaussian tails of the sparse cluster may trip k=3
    assert model.critical_radius_.shape == (234,)
    assert np.all(model.mdef_ <= 1.0)


def test_predict_contamination_and_none():
    X, _ = _two_density_data()
    model = LOCI().fit(X)
    assert model.predict(X, contamination=0.05).sum() == pytest.approx(round(0.05 * 234), abs=1)
    np.testing.assert_array_equal(model.predict(X, contamination=None), model.flag(X))


def test_out_of_sample_scoring():
    X, _ = _two_density_data()
    model = LOCI().fit(X)
    q = np.array([[0.0, 0.0], [1.7, 0.1], [20.0, -20.0]])
    s = model.score_samples(q)
    assert s.shape == (3,)
    assert s[1] > s[0]
    assert s[2] > s[0]


def test_uniform_grid_has_no_flags():
    g = np.stack(np.meshgrid(np.arange(12.0), np.arange(12.0)), -1).reshape(-1, 2)
    model = LOCI(n_min=10).fit(g)
    assert model.flag(g).sum() == 0


def test_cli_registers_loci():
    X, _ = _two_density_data()
    assert "loci" in DETECTOR_NAMES
    assert _detector_scores("loci", X, seed=0).shape == (234,)


@pytest.mark.parametrize(
    "kwargs", [{"alpha": 1.0}, {"alpha": 0.0}, {"k_sigma": 0}, {"n_min": 1}, {"n_radii": 0}]
)
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        LOCI(**kwargs)


def test_input_validation():
    with pytest.raises(ValueError):
        LOCI().fit(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        LOCI().score_samples(np.zeros((5, 2)))
    model = LOCI().fit(np.random.default_rng(0).normal(size=(30, 2)))
    with pytest.raises(ValueError):
        model.score_samples(np.zeros((3, 3)))
