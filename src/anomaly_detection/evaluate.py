"""Evaluation metrics for anomaly detectors on labelled synthetic data.

All metrics are computed from first principles (no sklearn): precision /
recall / F1 from contingency counts, ROC-AUC from averaged ranks
(Mann-Whitney U), a threshold sweep with best-F1 selection, and a comparison
table across detectors.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .models import _flags_from_contamination


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, object]:
    """Confusion counts plus precision, recall and F1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Rank values with ties averaged, like scipy.stats.rankdata."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    vals = a[order]
    rank = np.empty(len(a), dtype=float)
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1] == vals[i]:
            j += 1
        rank[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return rank


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    sum_pos = float(ranks[y_true == 1].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def threshold_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
) -> List[Dict[str, object]]:
    """Score metrics for every candidate threshold (``score >= threshold``)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if thresholds is None:
        uniq = np.unique(scores)
        if uniq.size > 100:
            step = uniq.size // 100
            uniq = uniq[::step]
            uniq = np.unique(np.concatenate([uniq, [scores.min(), scores.max()]]))
        thresholds = uniq
    rows: List[Dict[str, object]] = []
    for t in thresholds:
        pred = (scores >= float(t)).astype(int)
        metrics = precision_recall_f1(y_true, pred)
        metrics["threshold"] = float(t)
        rows.append(metrics)
    return rows


def best_f1_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
) -> tuple:
    """Return ``(threshold, f1)`` maximising F1 (ties broken by precision)."""
    rows = threshold_sweep(y_true, scores, thresholds=thresholds)
    if not rows:
        return None, 0.0
    best = max(rows, key=lambda r: (r["f1"], r["precision"]))
    return best["threshold"], best["f1"]


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    contamination: float = 0.1,
) -> Dict[str, object]:
    """Full metric set for one score vector at a fixed contamination."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    flags = _flags_from_contamination(scores, contamination)
    metrics = precision_recall_f1(y_true, flags)
    metrics["auc"] = roc_auc(y_true, scores)
    metrics["n_flagged"] = int(flags.sum())
    return metrics


def compare_detectors(
    X: np.ndarray,
    y_true: np.ndarray,
    detectors: Dict[str, Callable[[np.ndarray], np.ndarray]],
    contamination: float = 0.1,
) -> List[Dict[str, object]]:
    """Score each detector on ``X`` and return metrics sorted by F1.

    ``detectors`` maps a display name to a callable that turns ``X`` into an
    anomaly score vector (higher = more anomalous).
    """
    rows: List[Dict[str, object]] = []
    for name, scorer in detectors.items():
        scores = np.asarray(scorer(X), dtype=float)
        metrics = evaluate_scores(y_true, scores, contamination=contamination)
        metrics["detector"] = name
        rows.append(metrics)
    rows.sort(key=lambda r: r["f1"], reverse=True)
    return rows
