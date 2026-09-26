"""Unsupervised anomaly detection toolkit.

A dependency-light (numpy-only) set of detectors, evaluators and report
utilities for spotting unusual rows in tabular data and point/level-shift
anomalies in time series. Model-based detectors include isolation forest,
local outlier factor, k-nearest neighbours, COPOD, HBOS, one-class SVM and EllipticEnvelope (FAST-MCD).
"""

from . import classic, evaluate, generators, models, report
from .generators import (
    gaussian_clusters,
    inject_scale_outliers,
    inject_shifted_outliers,
    make_tabular,
    make_time_series,
)

__version__ = "0.1.0"

__all__ = [
    "classic",
    "evaluate",
    "generators",
    "models",
    "report",
    "gaussian_clusters",
    "inject_scale_outliers",
    "inject_shifted_outliers",
    "make_tabular",
    "make_time_series",
    "__version__",
]
