"""Unsupervised anomaly detection toolkit.

A dependency-light (numpy-only) set of detectors, evaluators and report
utilities for spotting unusual rows in tabular data and point/level-shift
anomalies in time series, plus BOCPD and CUSUM change-point detection, extreme-value (POT/SPOT/DSPOT) thresholds and outlier-score ensembles
(average/max/median/AOM/MOA/LSCP). Model-based detectors include isolation forest,
local outlier factor, k-nearest neighbours, COPOD, ECOD, HBOS, one-class SVM, EllipticEnvelope (FAST-MCD), CBLOF, LODA, ABOD, COF, SOD, PCA, KDE, SOS, GMM, INNE, Extended Isolation Forest, streaming Half-Space Trees and Robust Random Cut Forest (CoDisp).
"""

from . import changepoint, classic, combination, evaluate, evt, generators, models, report, rrcf
from .generators import (
    gaussian_clusters,
    inject_scale_outliers,
    inject_shifted_outliers,
    make_tabular,
    make_time_series,
)

__version__ = "0.1.0"

__all__ = [
    "changepoint",
    "classic",
    "combination",
    "evaluate",
    "evt",
    "generators",
    "models",
    "report",
    "rrcf",
    "gaussian_clusters",
    "inject_scale_outliers",
    "inject_shifted_outliers",
    "make_tabular",
    "make_time_series",
    "__version__",
]
