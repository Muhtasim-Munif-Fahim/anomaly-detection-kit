"""Synthetic data generators for unsupervised anomaly detection.

All generators are seeded and deterministic: the same ``seed`` produces the
same output arrays. Tabular mode produces a labelled matrix ``(X, y_true)``
where ``y_true == 1`` marks injected outliers (mean-shifted and/or variance
inflation). Time-series mode produces a ``(t, values)`` matrix plus labels
marking point spikes and level shifts.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

TABULAR_OUTLIER_TYPES = ("shift", "scale")

DEFAULT_SHIFT = 3.0
DEFAULT_SCALE = 3.0
TIME_SERIES_IMPULSE = 5.0
TIME_SERIES_LEVEL = 4.0


def gaussian_clusters(
    n_samples: int,
    n_features: int,
    n_clusters: int,
    centers: Optional[np.ndarray] = None,
    cluster_std: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw ``n_samples`` points from ``n_clusters`` isotropic gaussians.

    Cluster assignment is uniform at random. Returns ``(X, cluster_ids)``.
    """
    if n_samples <= 0 or n_features <= 0 or n_clusters <= 0:
        raise ValueError("n_samples, n_features and n_clusters must be positive")
    if cluster_std < 0:
        raise ValueError("cluster_std must be non-negative")
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = rng.uniform(low=-10.0, high=10.0, size=(n_clusters, n_features))
    else:
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 2 or centers.shape != (n_clusters, n_features):
            raise ValueError("centers must have shape (n_clusters, n_features)")
    cluster_ids = rng.integers(0, n_clusters, size=n_samples)
    counts = np.bincount(cluster_ids, minlength=n_clusters)
    X = np.empty((n_samples, n_features), dtype=float)
    for idx, count in enumerate(counts):
        if count:
            X[cluster_ids == idx] = rng.normal(
                loc=centers[idx], scale=cluster_std, size=(count, n_features)
            )
    return X, cluster_ids


def _centroid(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0)


def _radii(X: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X - center, axis=1)


def inject_shifted_outliers(
    X: np.ndarray,
    n_outliers: int,
    shift: float = DEFAULT_SHIFT,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Push ``n_outliers`` rows far from the data centre (mean shift).

    Each outlier is placed ``shift * max_inlier_radius`` away from the data
    centroid along a random unit direction, so the injected rows are strictly
    more distant than every inlier. Returns ``(X_augmented, indices)``.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n_outliers < 0 or n_outliers > n:
        raise ValueError("n_outliers must be in [0, len(X)]")
    rng = np.random.default_rng(seed)
    center = _centroid(X)
    radius_max = float(_radii(X, center).max()) if n else 0.0
    idx = rng.choice(n, size=n_outliers, replace=False)
    if n_outliers:
        offset = rng.normal(size=(n_outliers, X.shape[1]))
        offset = offset / np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-12)
        X = X.copy()
        X[idx] = center + offset * (shift * max(radius_max, 1e-12))
    return X, idx


def inject_scale_outliers(
    X: np.ndarray,
    n_outliers: int,
    scale: float = DEFAULT_SCALE,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inflate the deviation of ``n_outliers`` rows around the centroid.

    Each chosen row is stretched away from the data centre by ``scale``,
    modelling variance/scale outliers. Returns ``(X_augmented, indices)``.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n_outliers < 0 or n_outliers > n:
        raise ValueError("n_outliers must be in [0, len(X)]")
    rng = np.random.default_rng(seed)
    center = _centroid(X)
    idx = rng.choice(n, size=n_outliers, replace=False)
    if n_outliers:
        X = X.copy()
        X[idx] = center + (X[idx] - center) * scale
    return X, idx


def make_tabular(
    n_samples: int = 500,
    n_features: int = 4,
    n_clusters: int = 3,
    contamination: float = 0.05,
    outlier_types: Sequence[str] = TABULAR_OUTLIER_TYPES,
    centers: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create labelled tabular data with known injected outliers.

    Points are drawn from ``n_clusters`` gaussians; a fraction
    ``contamination`` of rows is then made anomalous with a mean shift and/or
    inflated variance. Returns ``(X, y_true)`` with ``y_true[i] == 1`` for
    outliers.
    """
    if not 0.0 <= contamination <= 1.0:
        raise ValueError("contamination must be in [0, 1]")
    if isinstance(outlier_types, str):
        outlier_types = (outlier_types,)
    outlier_types = tuple(outlier_types)
    for kind in outlier_types:
        if kind not in TABULAR_OUTLIER_TYPES:
            raise ValueError(
                f"unknown outlier type {kind!r}; expected one of {TABULAR_OUTLIER_TYPES}"
            )
    rng = np.random.default_rng(seed)
    X, _ = gaussian_clusters(
        n_samples, n_features, n_clusters, centers=centers, cluster_std=1.0, seed=rng
    )
    y = np.zeros(n_samples, dtype=int)
    n_outliers = int(round(contamination * n_samples))
    if n_outliers:
        idx = rng.choice(n_samples, size=n_outliers, replace=False)
        rng.shuffle(idx)
        n_types = len(outlier_types)
        base = n_outliers // n_types
        remainder = n_outliers % n_types
        cursor = 0
        for j, kind in enumerate(outlier_types):
            count = base + (1 if j < remainder else 0)
            subset = idx[cursor : cursor + count]
            cursor += count
            if kind == "shift":
                X = _shift_rows(X, subset, rng)
            else:
                X = _scale_rows(X, subset)
        y[idx] = 1
    return X, y


def _shift_rows(X: np.ndarray, subset: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = _centroid(X)
    radius_max = float(_radii(X, center).max())
    offset = rng.normal(size=(len(subset), X.shape[1]))
    offset = offset / np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-12)
    X = X.copy()
    X[subset] = center + offset * (DEFAULT_SHIFT * max(radius_max, 1e-12))
    return X


def _scale_rows(X: np.ndarray, subset: np.ndarray) -> np.ndarray:
    center = _centroid(X)
    X = X.copy()
    X[subset] = center + (X[subset] - center) * DEFAULT_SCALE
    return X


def make_time_series(
    n_points: int = 400,
    contamination: float = 0.03,
    base_level: float = 0.0,
    noise: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a noisy univariate series with injected point and level-shift anomalies.

    Returns ``(X, y_true, values)`` where ``X[:, 0]`` is the time index,
    ``X[:, 1]`` the observed values, and ``y_true`` marks point spikes and the
    shifted segment of level shifts.
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    if not 0.0 <= contamination <= 1.0:
        raise ValueError("contamination must be in [0, 1]")
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=float)
    trend = base_level + 0.01 * t
    values = trend + rng.normal(loc=0.0, scale=noise, size=n_points)
    y = np.zeros(n_points, dtype=int)
    n_outliers = int(round(contamination * n_points))
    if n_outliers:
        n_spikes = n_outliers // 2
        n_shift = n_outliers - n_spikes
        block = np.empty(0, dtype=int)
        if n_shift:
            n_shift = min(n_shift, n_points)
            start = int(rng.integers(0, n_points - n_shift + 1))
            block = np.arange(start, start + n_shift)
            direction = rng.choice((-1.0, 1.0))
            values[block] += direction * TIME_SERIES_LEVEL * noise
            y[block] = 1
        if n_spikes:
            free = np.setdiff1d(np.arange(n_points), block, assume_unique=True)
            spike_idx = rng.choice(free, size=n_spikes, replace=False)
            direction = rng.choice((-1.0, 1.0), size=n_spikes)
            magnitude = TIME_SERIES_IMPULSE * noise * rng.uniform(1.0, 1.5, size=n_spikes)
            values[spike_idx] += direction * magnitude
            y[spike_idx] = 1
    X = np.column_stack([t, values])
    return X, y, values
