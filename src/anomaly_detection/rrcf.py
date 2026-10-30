"""Robust Random Cut Forest (Guha, Mishra, Roy & Schrijvers, ICML 2016).

A robust random cut tree (RRCT) is built by repeatedly choosing a cut
dimension with probability proportional to its bounding-box span and a cut
value uniformly inside that span. Unlike an isolation tree, the tree is a
*dynamic* data structure: points can be inserted and deleted exactly, and the
resulting tree has the same distribution as one built from scratch on the new
point set. That makes the forest a natural streaming detector with a sliding
window.

The anomaly score is the *collusive displacement* (CoDisp). For a leaf ``x``
we walk from the leaf to the root and, at each step, compare the size of the
sibling subtree with the size of the subtree that holds ``x``:

    CoDisp(x) = max over ancestors  |sibling| / |subtree containing x|

Points that are cut off high in the tree, away from a large mass of other
points, get a large displacement. Scores are averaged over the forest, and
**higher means more anomalous**, as for the other detectors in this package.

Everything is pure numpy plus small Python node objects. That is fine for the
usual RRCF sizes (tens of trees with a few hundred points each).
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

__all__ = ["RandomCutTree", "RobustRandomCutForest", "shingle"]


class _Node:
    __slots__ = ("left", "right", "parent", "dim", "val", "lo", "hi", "n", "point")

    def __init__(self) -> None:
        self.left: Optional["_Node"] = None
        self.right: Optional["_Node"] = None
        self.parent: Optional["_Node"] = None
        self.dim = -1
        self.val = 0.0
        self.lo: np.ndarray
        self.hi: np.ndarray
        self.n = 0
        self.point: Optional[np.ndarray] = None

    @property
    def is_leaf(self) -> bool:
        return self.point is not None


def _leaf(point: np.ndarray, count: int = 1) -> _Node:
    node = _Node()
    node.point = point
    node.lo = point
    node.hi = point
    node.n = count
    return node


class RandomCutTree:
    """A single robust random cut tree with exact insert / delete.

    Leaves are keyed by an arbitrary hashable ``key``. Inserting a point that
    is identical to an existing leaf increments that leaf's count (it is the
    same leaf and can be keyed several times).
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.root: Optional[_Node] = None
        self.leaves: Dict[object, _Node] = {}

    def __len__(self) -> int:
        return 0 if self.root is None else self.root.n

    # -- construction --------------------------------------------------
    def _cut(self, lo: np.ndarray, hi: np.ndarray):
        span = hi - lo
        total = span.sum()
        r = self.rng.uniform(0.0, total)
        cum = np.cumsum(span)
        dim = int(np.searchsorted(cum, r, side="right"))
        dim = min(dim, span.size - 1)
        while span[dim] <= 0:  # guard against r landing on a zero-width boundary
            dim = (dim + 1) % span.size
        prev = cum[dim - 1] if dim > 0 else 0.0
        val = lo[dim] + min(max(r - prev, 0.0), span[dim])
        return dim, float(val)

    def build(self, X: np.ndarray, keys: Optional[List[object]] = None) -> "RandomCutTree":
        """Build the tree from scratch on the rows of ``X``."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 1:
            raise ValueError("X must be a non-empty 2-D array")
        keys = list(range(X.shape[0])) if keys is None else list(keys)
        if len(keys) != X.shape[0] or len(set(keys)) != len(keys):
            raise ValueError("keys must be unique and match the number of rows")
        self.leaves = {}
        self.root = self._build(X, np.arange(X.shape[0]), keys, None)
        return self

    def _build(self, X, idx, keys, parent) -> _Node:
        pts = X[idx]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        if np.all(hi <= lo):  # all identical -> one leaf with multiplicity
            node = _leaf(pts[0].copy(), count=idx.size)
            node.parent = parent
            for i in idx:
                self.leaves[keys[i]] = node
            return node
        while True:
            dim, val = self._cut(lo, hi)
            go_left = pts[:, dim] <= val
            if go_left.any() and not go_left.all():
                break
        node = _Node()
        node.parent, node.dim, node.val = parent, dim, val
        node.lo, node.hi, node.n = lo, hi, idx.size
        node.left = self._build(X, idx[go_left], keys, node)
        node.right = self._build(X, idx[~go_left], keys, node)
        return node

    # -- dynamic updates -------------------------------------------------
    def _find_leaf(self, p: np.ndarray) -> Optional[_Node]:
        node = self.root
        while node is not None and not node.is_leaf:
            node = node.left if p[node.dim] <= node.val else node.right
        return node

    def insert(self, point, key) -> _Node:
        """Insert ``point`` under ``key``; returns its leaf."""
        p = np.asarray(point, dtype=float).ravel().copy()
        if key in self.leaves:
            raise KeyError(f"key {key!r} already in tree")
        if self.root is None:
            self.root = _leaf(p)
            self.leaves[key] = self.root
            return self.root
        if p.size != self.root.lo.size:
            raise ValueError("point dimension does not match the tree")
        same = self._find_leaf(p)
        if same is not None and np.array_equal(same.point, p):
            same.n += 1
            anc = same.parent
            while anc is not None:
                anc.n += 1
                anc = anc.parent
            self.leaves[key] = same
            return same
        node = self.root
        while True:
            lo = np.minimum(node.lo, p)
            hi = np.maximum(node.hi, p)
            dim, val = self._cut(lo, hi)
            if p[dim] <= val < node.lo[dim] or node.hi[dim] <= val < p[dim]:
                break
            if node.is_leaf:  # cut fell on a boundary; draw again
                continue
            node = node.left if p[node.dim] <= node.val else node.right
        leaf = _leaf(p)
        branch = _Node()
        branch.dim, branch.val = dim, val
        branch.lo, branch.hi = lo, hi
        branch.n = node.n + 1
        parent = node.parent
        if p[dim] <= val:
            branch.left, branch.right = leaf, node
        else:
            branch.left, branch.right = node, leaf
        leaf.parent = branch
        node.parent = branch
        branch.parent = parent
        if parent is None:
            self.root = branch
        elif parent.left is node:
            parent.left = branch
        else:
            parent.right = branch
        anc = parent
        while anc is not None:
            anc.n += 1
            np.minimum(anc.lo, p, out=anc.lo)
            np.maximum(anc.hi, p, out=anc.hi)
            anc = anc.parent
        self.leaves[key] = leaf
        return leaf

    def forget(self, key) -> None:
        """Delete the point stored under ``key``."""
        leaf = self.leaves.pop(key)
        if leaf.n > 1:
            leaf.n -= 1
            anc = leaf.parent
            while anc is not None:
                anc.n -= 1
                anc = anc.parent
            return
        parent = leaf.parent
        if parent is None:
            self.root = None
            return
        sibling = parent.right if parent.left is leaf else parent.left
        grand = parent.parent
        sibling.parent = grand
        if grand is None:
            self.root = sibling
            return
        if grand.left is parent:
            grand.left = sibling
        else:
            grand.right = sibling
        anc = grand
        while anc is not None:
            anc.n -= 1
            anc.lo = np.minimum(anc.left.lo, anc.right.lo)
            anc.hi = np.maximum(anc.left.hi, anc.right.hi)
            anc = anc.parent

    # -- scores ----------------------------------------------------------
    def codisp(self, key) -> float:
        """Collusive displacement of the leaf stored under ``key``."""
        node = self.leaves[key]
        best = 0.0
        while node.parent is not None:
            parent = node.parent
            sibling = parent.right if parent.left is node else parent.left
            best = max(best, sibling.n / node.n)
            node = parent
        return best

    def depth(self, key) -> int:
        node, d = self.leaves[key], 0
        while node.parent is not None:
            node, d = node.parent, d + 1
        return d

    def check(self) -> None:
        """Assert the structural invariants (used by the tests)."""
        if self.root is None:
            assert not self.leaves
            return
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                continue
            l, r = node.left, node.right
            assert l.parent is node and r.parent is node
            assert node.n == l.n + r.n
            np.testing.assert_allclose(node.lo, np.minimum(l.lo, r.lo))
            np.testing.assert_allclose(node.hi, np.maximum(l.hi, r.hi))
            assert l.hi[node.dim] <= node.val < r.lo[node.dim]
            stack.extend([l, r])
        assert self.root.n == len(self.leaves)


def shingle(series, size: int) -> np.ndarray:
    """Sliding windows of length ``size`` over a 1-D series (rows = windows).

    RRCF on shingles turns a univariate stream into a detector for anomalous
    *shapes* as well as single extreme values.
    """
    x = np.asarray(series, dtype=float).ravel()
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size must be a positive integer")
    if x.size < size:
        raise ValueError("series is shorter than the shingle size")
    return np.lib.stride_tricks.sliding_window_view(x, size).copy()


class RobustRandomCutForest:
    """Forest of robust random cut trees scored by mean CoDisp.

    Batch use: :meth:`fit` builds each tree on a random subsample of
    ``tree_size`` rows (all rows when ``n <= tree_size``). A training row's
    score is its mean CoDisp over the trees that contain it; rows that landed
    in no tree are scored like new data. :meth:`score_samples` scores new rows
    by inserting each one into every tree, reading its CoDisp, and deleting it
    again, so the fitted forest is left unchanged.

    Streaming use: :meth:`score_stream` (or :meth:`update`) keeps a sliding
    window. Each tree holds the latest ``tree_size`` points; the oldest point
    is forgotten before a new one is inserted, and the new point's mean CoDisp
    is returned. Combine with :func:`shingle` for univariate series.

    Parameters
    ----------
    n_estimators:
        Number of trees.
    tree_size:
        Points per tree (batch subsample size and streaming window).
    seed:
        RNG seed for subsampling and cuts.
    """

    def __init__(self, n_estimators: int = 40, tree_size: int = 256, seed: Optional[int] = 0) -> None:
        for name, value, low in (("n_estimators", n_estimators, 1), ("tree_size", tree_size, 2)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
                raise ValueError(f"{name} must be an integer >= {low}")
        self.n_estimators = int(n_estimators)
        self.tree_size = int(tree_size)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.trees_: List[RandomCutTree] = []
        self._queues: List[Deque[object]] = []
        self._n_features: Optional[int] = None
        self._t = 0
        self.scores_: Optional[np.ndarray] = None

    def _check_X(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] < 1:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but the forest was built for {self._n_features}"
            )
        return X

    def fit(self, X) -> "RobustRandomCutForest":
        self._n_features = None
        X = self._check_X(X)
        n = X.shape[0]
        if n < 2:
            raise ValueError("fit needs at least two rows")
        self._n_features = X.shape[1]
        size = min(self.tree_size, n)
        self.trees_ = []
        self._queues = []
        total = np.zeros(n)
        hits = np.zeros(n)
        for _ in range(self.n_estimators):
            idx = np.sort(self._rng.choice(n, size=size, replace=False)) if size < n else np.arange(n)
            keys = [int(i) for i in idx]
            tree = RandomCutTree(np.random.default_rng(self._rng.integers(2**63))).build(X[idx], keys)
            self.trees_.append(tree)
            self._queues.append(deque(keys))
            for k in keys:
                total[k] += tree.codisp(k)
                hits[k] += 1
        scores = np.divide(total, hits, out=np.zeros(n), where=hits > 0)
        missing = np.flatnonzero(hits == 0)
        if missing.size:
            scores[missing] = self.score_samples(X[missing])
        self.scores_ = scores
        self._t = n
        return self

    def score_samples(self, X) -> np.ndarray:
        """Mean CoDisp of each row if it were inserted (forest left unchanged)."""
        if not self.trees_:
            raise ValueError("RobustRandomCutForest must be fitted before scoring")
        X = self._check_X(X)
        out = np.zeros(X.shape[0])
        probe = ("__probe__",)
        for i, x in enumerate(X):
            s = 0.0
            for tree in self.trees_:
                tree.insert(x, probe)
                s += tree.codisp(probe)
                tree.forget(probe)
            out[i] = s / len(self.trees_)
        return out

    def update(self, x) -> float:
        """Streaming step: slide every tree's window by ``x``; return its score."""
        x = self._check_X(x)
        if x.shape[0] != 1:
            raise ValueError("update takes a single point")
        if self._n_features is None:
            self._n_features = x.shape[1]
        if not self.trees_:
            self.trees_ = [
                RandomCutTree(np.random.default_rng(self._rng.integers(2**63)))
                for _ in range(self.n_estimators)
            ]
            self._queues = [deque() for _ in range(self.n_estimators)]
        key = ("stream", self._t)
        s = 0.0
        for tree, queue in zip(self.trees_, self._queues):
            while len(queue) >= self.tree_size:
                tree.forget(queue.popleft())
            queue.append(key)
            tree.insert(x[0], key)
            s += tree.codisp(key)
        self._t += 1
        return s / len(self.trees_)

    def score_stream(self, X) -> np.ndarray:
        """Run :meth:`update` over every row of ``X`` in order."""
        X = self._check_X(X)
        return np.array([self.update(x) for x in X])

    def predict(self, X, contamination: float = 0.1) -> np.ndarray:
        from .models import _flags_from_contamination

        return _flags_from_contamination(self.score_samples(X), contamination)

    def fit_predict(self, X, contamination: float = 0.1) -> np.ndarray:
        from .models import _flags_from_contamination

        return _flags_from_contamination(self.fit(X).scores_, contamination)
