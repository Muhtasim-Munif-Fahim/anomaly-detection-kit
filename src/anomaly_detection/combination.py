"""Outlier-score combination: build an ensemble out of several detectors.

Each detector reports its scores on its own scale. Before combining them,
put them on a common scale with :func:`standardize_scores`: column-wise
z-scores (the default), ranks scaled to (0, 1], or min-max. The scaling
statistics can come from a separate *training* score matrix, so test scores
are scaled using the training mean, std, range or ranks.

Combiners (Aggarwal & Sathe, 2015, "Theoretical foundations and algorithms
for outlier ensembles"), each mapping an ``(n_samples, n_detectors)`` matrix
to one combined score per row:

* :func:`average_scores`, optionally weighted,
* :func:`max_scores`,
* :func:`median_scores`,
* :func:`aom_scores`: *average of maximum*. Split the detectors into
  buckets, take the max inside each bucket, then average across buckets,
* :func:`moa_scores`: *maximum of average*, the dual of AOM,
* :func:`lscp_scores`: LSCP-style locally selective combination (Zhao et
  al., 2019). For each test point, take its k nearest training points,
  build a pseudo ground truth (the max or average of the standardised
  training scores), and keep the detector(s) whose local training scores
  correlate best (Pearson) with that target.

:func:`combine_scores` dispatches to any of the above by name. Everything
uses numpy only, and higher scores always mean more anomalous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

__all__ = [
    "COMBINATION_METHODS",
    "LSCPResult",
    "aom_scores",
    "average_scores",
    "bucket_assignments",
    "combine_scores",
    "lscp_scores",
    "max_scores",
    "median_scores",
    "moa_scores",
    "standardize_scores",
]

COMBINATION_METHODS = ("average", "max", "median", "aom", "moa")
Buckets = Sequence[Sequence[int]]


def _as_matrix(scores, name: str = "scores") -> np.ndarray:
    S = np.asarray(scores, dtype=float)
    if S.ndim == 1:
        S = S[:, None]
    if S.ndim != 2 or S.shape[0] == 0 or S.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty (n_samples, n_detectors) matrix")
    if not np.all(np.isfinite(S)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return S


def _rank01(column: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Fraction of ``reference`` values <= each entry of ``column`` (ECDF)."""
    ref = np.sort(reference)
    return np.searchsorted(ref, column, side="right") / ref.size


def standardize_scores(
    scores,
    train_scores=None,
    method: str = "zscore",
) -> np.ndarray:
    """Put every detector's scores (columns) on a common scale.

    Parameters
    ----------
    scores : array (n_samples, n_detectors)
        Scores to transform.
    train_scores : array (n_train, n_detectors), optional
        The scaling statistics come from this matrix (defaults to ``scores``
        itself). With ``method="zscore"`` this is exactly
        ``StandardScaler().fit(train).transform(scores)``, the same as PyOD's
        ``standardizer``. Constant columns get a scale of 1.
    method : {"zscore", "rank", "minmax", "none"}
        ``rank`` is the empirical CDF of ``train_scores``, in (0, 1].
    """
    S = _as_matrix(scores)
    T = S if train_scores is None else _as_matrix(train_scores, "train_scores")
    if T.shape[1] != S.shape[1]:
        raise ValueError(
            f"train_scores has {T.shape[1]} detectors but scores has {S.shape[1]}"
        )
    if method == "none":
        return S.copy()
    if method == "zscore":
        mu = T.mean(axis=0)
        sd = T.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        return (S - mu) / sd
    if method == "minmax":
        lo = T.min(axis=0)
        span = T.max(axis=0) - lo
        span = np.where(span > 0, span, 1.0)
        return (S - lo) / span
    if method == "rank":
        return np.column_stack([_rank01(S[:, j], T[:, j]) for j in range(S.shape[1])])
    raise ValueError(f"unknown standardisation {method!r}; use zscore, rank, minmax or none")


def average_scores(scores, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """(Weighted) mean across detectors."""
    S = _as_matrix(scores)
    if weights is None:
        return S.mean(axis=1)
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != S.shape[1]:
        raise ValueError(f"weights has {w.size} entries but there are {S.shape[1]} detectors")
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    return (S * w).sum(axis=1) / w.sum()


def max_scores(scores) -> np.ndarray:
    """Maximum across detectors: sensitive, but rewards any single detector's spike."""
    return _as_matrix(scores).max(axis=1)


def median_scores(scores) -> np.ndarray:
    """Median across detectors: robust to a minority of bad detectors."""
    return np.median(_as_matrix(scores), axis=1)


def bucket_assignments(
    n_detectors: int,
    n_buckets: int = 5,
    method: str = "static",
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Random detector groups for AOM/MOA.

    * ``static`` shuffles the detectors and splits them into ``n_buckets``
      disjoint, equal-sized buckets, which requires ``n_buckets`` to divide
      ``n_detectors``.
    * ``dynamic`` draws each bucket independently, with a random size between
      2 and ``max(2, n_detectors // 2)`` and no replacement inside a bucket.
    """
    if n_detectors < 2:
        raise ValueError("AOM/MOA need at least 2 detectors")
    if not 2 <= n_buckets <= n_detectors:
        raise ValueError(f"n_buckets must be in [2, {n_detectors}], got {n_buckets}")
    rng = np.random.default_rng(seed)
    if method == "static":
        if n_detectors % n_buckets:
            raise ValueError(
                f"static buckets need n_detectors ({n_detectors}) divisible by n_buckets ({n_buckets})"
            )
        perm = rng.permutation(n_detectors)
        return [np.sort(b) for b in np.split(perm, n_buckets)]
    if method == "dynamic":
        hi = max(2, n_detectors // 2)
        out = []
        for _ in range(n_buckets):
            size = int(rng.integers(2, hi + 1))
            out.append(np.sort(rng.choice(n_detectors, size=size, replace=False)))
        return out
    raise ValueError(f"unknown bucket method {method!r}; use 'static' or 'dynamic'")


def _bucketed(S: np.ndarray, buckets, n_buckets, method, seed) -> List[np.ndarray]:
    if buckets is None:
        return bucket_assignments(S.shape[1], n_buckets, method, seed)
    out = []
    for b in buckets:
        idx = np.asarray(b, dtype=int).ravel()
        if idx.size == 0 or idx.min() < 0 or idx.max() >= S.shape[1]:
            raise ValueError("each bucket must be a non-empty list of valid detector indices")
        out.append(idx)
    if not out:
        raise ValueError("buckets must not be empty")
    return out


def aom_scores(
    scores,
    n_buckets: int = 5,
    method: str = "static",
    seed: Optional[int] = None,
    buckets: Optional[Buckets] = None,
) -> np.ndarray:
    """Average of Maximum: the max within each bucket, averaged over buckets.

    Pass explicit ``buckets`` (lists of column indices) to make the grouping
    deterministic. Otherwise they come from :func:`bucket_assignments`.
    """
    S = _as_matrix(scores)
    groups = _bucketed(S, buckets, n_buckets, method, seed)
    return np.column_stack([S[:, g].max(axis=1) for g in groups]).mean(axis=1)


def moa_scores(
    scores,
    n_buckets: int = 5,
    method: str = "static",
    seed: Optional[int] = None,
    buckets: Optional[Buckets] = None,
) -> np.ndarray:
    """Maximum of Average: the mean within each bucket, max over buckets."""
    S = _as_matrix(scores)
    groups = _bucketed(S, buckets, n_buckets, method, seed)
    return np.column_stack([S[:, g].mean(axis=1) for g in groups]).max(axis=1)


def combine_scores(
    scores,
    method: str = "average",
    standardize: str = "zscore",
    train_scores=None,
    weights: Optional[Sequence[float]] = None,
    n_buckets: int = 5,
    bucket_method: str = "static",
    seed: Optional[int] = None,
    buckets: Optional[Buckets] = None,
) -> np.ndarray:
    """Standardise a score matrix, then combine it with ``method``.

    ``method`` is one of :data:`COMBINATION_METHODS`. If ``n_buckets`` does not
    fit (for example, more buckets than detectors), AOM/MOA fall back to
    ``min(n_buckets, n_detectors)`` buckets. With the static scheme they use
    the largest divisor of ``n_detectors`` that does not exceed that.
    """
    Z = standardize_scores(scores, train_scores=train_scores, method=standardize)
    if method == "average":
        return average_scores(Z, weights)
    if method == "max":
        return max_scores(Z)
    if method == "median":
        return median_scores(Z)
    if method in ("aom", "moa"):
        fn = aom_scores if method == "aom" else moa_scores
        if buckets is None:
            m = Z.shape[1]
            nb = max(2, min(n_buckets, m))
            if bucket_method == "static":
                while nb > 2 and m % nb:
                    nb -= 1
                if m % nb:
                    nb = m
            return fn(Z, n_buckets=nb, method=bucket_method, seed=seed)
        return fn(Z, buckets=buckets)
    raise ValueError(f"unknown combination {method!r}; use one of {COMBINATION_METHODS}")


@dataclass
class LSCPResult:
    """Output of :func:`lscp_scores`."""

    scores: np.ndarray            # combined score per test point
    selected: np.ndarray          # (n_test, n_select) chosen detector indices
    local_correlation: np.ndarray # (n_test, n_detectors) Pearson r in the local region


def _pairwise_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def _rowwise_pearson(Y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pearson r between each column of ``Y`` (k, m) and target ``t`` (k,)."""
    Yc = Y - Y.mean(axis=0)
    tc = t - t.mean()
    den = np.sqrt((Yc * Yc).sum(axis=0) * (tc * tc).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Yc * tc[:, None]).sum(axis=0) / den
    return np.where(den > 0, r, -np.inf)


def lscp_scores(
    X_train,
    train_scores,
    X_test=None,
    test_scores=None,
    n_neighbors: int = 30,
    pseudo_target: str = "max",
    n_select: int = 1,
    aggregate: str = "average",
    standardize: str = "zscore",
) -> LSCPResult:
    """LSCP-style locally selective combination of detectors.

    For each test point, the local region is its ``n_neighbors`` nearest
    training points in feature space (Euclidean). When ``X_test`` is omitted
    the training points score themselves, and each point is left out of its
    own region. Inside the region, every detector's standardised training
    scores are correlated (Pearson) with a pseudo ground truth: the row-wise
    ``max`` or ``average`` of the standardised training scores. The
    ``n_select`` best detectors are kept and their test scores are combined
    with ``aggregate`` (``average`` or ``max``). ``pseudo_target="max"`` with
    ``n_select=1`` is LSCP_M, and ``"average"`` is LSCP_A. Detectors that are
    constant within the region are never selected unless every detector is.

    Unlike the reference implementation, this uses a single full-feature kNN
    region instead of a random-subspace kNN ensemble, so the result is
    deterministic.
    """
    Xtr = np.asarray(X_train, dtype=float)
    if Xtr.ndim == 1:
        Xtr = Xtr[:, None]
    Str = _as_matrix(train_scores, "train_scores")
    if Str.shape[0] != Xtr.shape[0]:
        raise ValueError("X_train and train_scores must have the same number of rows")
    self_scoring = X_test is None
    if self_scoring:
        if test_scores is not None:
            raise ValueError("test_scores given without X_test")
        Xte, Ste = Xtr, Str
    else:
        Xte = np.asarray(X_test, dtype=float)
        if Xte.ndim == 1:
            Xte = Xte[:, None]
        if test_scores is None:
            raise ValueError("X_test requires test_scores")
        Ste = _as_matrix(test_scores, "test_scores")
        if Ste.shape != (Xte.shape[0], Str.shape[1]) or Xte.shape[1] != Xtr.shape[1]:
            raise ValueError("X_test/test_scores shapes do not match the training inputs")
    n_train, m = Str.shape
    max_k = n_train - 1 if self_scoring else n_train
    if not 2 <= n_neighbors <= max_k:
        raise ValueError(f"n_neighbors must be in [2, {max_k}], got {n_neighbors}")
    if not 1 <= n_select <= m:
        raise ValueError(f"n_select must be in [1, {m}], got {n_select}")
    if pseudo_target not in ("max", "average"):
        raise ValueError("pseudo_target must be 'max' or 'average'")
    if aggregate not in ("average", "max"):
        raise ValueError("aggregate must be 'average' or 'max'")

    Ztr = standardize_scores(Str, method=standardize)
    Zte = standardize_scores(Ste, train_scores=Str, method=standardize)
    target = Ztr.max(axis=1) if pseudo_target == "max" else Ztr.mean(axis=1)

    D = _pairwise_sq(Xte, Xtr)
    if self_scoring:
        np.fill_diagonal(D, np.inf)
    nbrs = np.argsort(D, axis=1, kind="stable")[:, :n_neighbors]

    corr = np.empty((Xte.shape[0], m))
    selected = np.empty((Xte.shape[0], n_select), dtype=int)
    combined = np.empty(Xte.shape[0])
    for i, idx in enumerate(nbrs):
        r = _rowwise_pearson(Ztr[idx], target[idx])
        corr[i] = r
        order = np.argsort(-r, kind="stable")[:n_select]
        selected[i] = order
        vals = Zte[i, order]
        combined[i] = vals.mean() if aggregate == "average" else vals.max()
    return LSCPResult(scores=combined, selected=selected, local_correlation=corr)
