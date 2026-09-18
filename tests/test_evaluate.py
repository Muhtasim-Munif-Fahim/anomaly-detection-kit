import math

import numpy as np
import pytest

from anomaly_detection import classic as cl
from anomaly_detection import evaluate as ev
from anomaly_detection import generators as g
from anomaly_detection import models as m


class TestPrecisionRecallF1:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        out = ev.precision_recall_f1(y_true, y_pred)
        assert out["tp"] == 5 and out["fp"] == 0 and out["fn"] == 0 and out["tn"] == 3
        assert out["precision"] == 1.0
        assert out["recall"] == 1.0
        assert out["f1"] == 1.0

    def test_no_predictions(self):
        y_true = np.array([0, 1, 1, 0])
        out = ev.precision_recall_f1(y_true, np.zeros(4))
        assert out["recall"] == 0.0
        assert out["precision"] == 0.0
        assert out["f1"] == 0.0

    def test_no_positives_in_truth(self):
        out = ev.precision_recall_f1(np.zeros(6), np.array([1, 0, 0, 1, 0, 1]))
        assert out["precision"] == 0.0
        assert out["recall"] == 0.0
        assert out["f1"] == 0.0

    def test_half_precision_half_recall(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        out = ev.precision_recall_f1(y_true, y_pred)
        assert out["precision"] == 1.0
        assert out["recall"] == 0.5
        assert abs(out["f1"] - 2 / 3) < 1e-12


class TestRankdata:
    def test_no_ties(self):
        assert np.array_equal(ev._rankdata(np.array([3.0, 1.0, 2.0])), [3.0, 1.0, 2.0])

    def test_ties_averaged(self):
        assert np.array_equal(ev._rankdata(np.array([1.0, 1.0, 2.0])), [1.5, 1.5, 3.0])

    def test_all_equal(self):
        assert np.all(ev._rankdata(np.ones(5)) == 3.0)


class TestRocAuc:
    def test_perfect_separation(self):
        y = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
        assert ev.roc_auc(y, scores) == 1.0

    def test_reversed_separation(self):
        y = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.95, 0.9, 0.3, 0.2, 0.1])
        assert ev.roc_auc(y, scores) == 0.0

    def test_random_scores(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        assert ev.roc_auc(y, scores) == 0.5

    def test_ties_do_not_break(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        auc = ev.roc_auc(y, scores)
        assert auc == 0.5
        assert not math.isnan(auc)

    def test_single_class_returns_nan(self):
        assert math.isnan(ev.roc_auc(np.zeros(5), np.arange(5.0)))
        assert math.isnan(ev.roc_auc(np.ones(5), np.arange(5.0)))


class TestThresholdSweep:
    def test_best_f1_threshold(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        threshold, f1 = ev.best_f1_threshold(y, scores)
        assert f1 == 1.0
        assert 0.8 < threshold <= 0.9

    def test_sweep_contains_best_row(self):
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        rows = ev.threshold_sweep(y, scores)
        assert any(r["f1"] == 1.0 for r in rows)
        assert rows[0]["threshold"] == scores.min()
        assert rows[-1]["threshold"] == scores.max()

    def test_sweep_uses_given_thresholds(self):
        y = np.array([0, 1])
        scores = np.array([0.1, 0.9])
        rows = ev.threshold_sweep(y, scores, thresholds=[0.5, 0.95])
        assert [r["threshold"] for r in rows] == [0.5, 0.95]

    def test_sweep_caps_large_grids(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=1000)
        y = (rng.random(1000) < 0.05).astype(int)
        rows = ev.threshold_sweep(y, scores)
        assert len(rows) <= 105

    def test_best_f1_no_rows(self):
        assert ev.best_f1_threshold(np.array([0, 1]), np.array([1.0, 0.0]), thresholds=[]) == (
            None,
            0.0,
        )


class TestEvaluateScores:
    def test_metrics_keys_present(self):
        X, y = g.make_tabular(400, 4, 3, contamination=0.05, seed=1)
        iforest = m.IsolationForest(n_estimators=40, seed=2).fit(X)
        metrics = ev.evaluate_scores(y, iforest.score_samples(X), contamination=0.05)
        for key in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "auc", "n_flagged"):
            assert key in metrics
        assert metrics["n_flagged"] == 20
        assert 0.0 <= metrics["auc"] <= 1.0

    def test_perfect_detector(self):
        y = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.95, 0.9])
        metrics = ev.evaluate_scores(y, scores, contamination=0.4)
        assert metrics["f1"] == 1.0
        assert metrics["auc"] == 1.0


class TestCompareDetectors:
    def test_rows_sorted_by_f1_and_named(self):
        X, y = g.make_tabular(300, 4, 3, contamination=0.05, seed=3)
        detectors = {
            "iforest": m.IsolationForest(n_estimators=30, seed=4).fit(X).score_samples,
            "lof": m.LocalOutlierFactor(n_neighbors=15).fit(X).score_samples,
            "copod": m.COPOD().fit(X).score_samples,
            "zscore": cl.zscore_scores,
        }
        rows = ev.compare_detectors(X, y, detectors, contamination=0.05)
        assert sorted(r["detector"] for r in rows) == ["copod", "iforest", "lof", "zscore"]
        f1s = [r["f1"] for r in rows]
        assert f1s == sorted(f1s, reverse=True)
        assert all("detector" in r for r in rows)

    def test_empty_detectors(self):
        assert ev.compare_detectors(np.zeros((5, 2)), np.zeros(5), {}) == []

    def test_contamination_passthrough(self):
        X, y = g.make_tabular(200, 3, 2, contamination=0.1, seed=5)
        detectors = {"z": lambda X: np.abs(X[:, 0] - X[:, 0].mean())}
        rows = ev.compare_detectors(X, y, detectors, contamination=0.1)
        assert rows[0]["n_flagged"] == 20

    def test_copod_recovers_synthetic_shift_outliers(self):
        X, y = g.make_tabular(400, 4, 3, contamination=0.05, outlier_types="shift", seed=8)
        rows = ev.compare_detectors(
            X, y, {"copod": m.COPOD().fit(X).score_samples}, contamination=0.05
        )
        assert rows[0]["detector"] == "copod"
        assert rows[0]["auc"] > 0.9
        assert rows[0]["f1"] > 0.7
