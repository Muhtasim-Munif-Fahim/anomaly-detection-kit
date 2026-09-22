"""Markdown report rendering for anomaly detection experiments."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


def score_summary(
    name: str, scores: np.ndarray, n_flagged: int, decimals: int = 4
) -> str:
    """One markdown section summarising a detector's score distribution."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return "\n".join([f"### {name}", "", "- no data", ""])
    lines = [
        f"### {name}",
        "",
        f"- score range: {scores.min():.{decimals}f} .. {scores.max():.{decimals}f}",
        f"- mean score: {scores.mean():.{decimals}f}",
        f"- median score: {float(np.median(scores)):.{decimals}f}",
        f"- flagged: {n_flagged} of {len(scores)}",
        "",
    ]
    return "\n".join(lines)


def flagged_rows_table(
    X: np.ndarray,
    scores: np.ndarray,
    flags: np.ndarray,
    max_rows: int = 10,
    decimals: int = 3,
) -> str:
    """Markdown table of the highest-scoring flagged rows."""
    X = np.asarray(X, dtype=float)
    scores = np.asarray(scores, dtype=float)
    flags = np.asarray(flags)
    idx = np.flatnonzero(flags)
    order = idx[np.argsort(scores[idx])[::-1]][:max_rows]
    header = ["row"] + [f"f{i}" for i in range(X.shape[1])] + ["score"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for i in order:
        cells = [str(int(i))]
        cells += [f"{v:.{decimals}f}" for v in X[i]]
        cells.append(f"{scores[i]:.{decimals}f}")
        lines.append("| " + " | ".join(cells) + " |")
    if len(idx) > max_rows:
        lines.append(f"_… {len(idx) - max_rows} more flagged rows omitted_")
    return "\n".join(lines)


def sweep_table(rows: Sequence[dict], decimals: int = 4) -> str:
    """Markdown table from the threshold-sweep rows of :mod:`evaluate`."""
    if not rows:
        return ""
    header = ["threshold", "precision", "recall", "f1", "tp", "fp", "fn"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in rows:
        lines.append(
            f"| {r['threshold']:.{decimals}f}"
            f" | {r['precision']:.{decimals}f}"
            f" | {r['recall']:.{decimals}f}"
            f" | {r['f1']:.{decimals}f}"
            f" | {r['tp']} | {r['fp']} | {r['fn']} |"
        )
    return "\n".join(lines)


def comparison_table(rows: Sequence[dict], decimals: int = 4) -> str:
    """Markdown table comparing detectors (input: ``compare_detectors`` rows)."""
    header = ["detector", "precision", "recall", "f1", "auc", "flagged"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in rows:
        lines.append(
            f"| {r['detector']}"
            f" | {r['precision']:.{decimals}f}"
            f" | {r['recall']:.{decimals}f}"
            f" | {r['f1']:.{decimals}f}"
            f" | {r['auc']:.{decimals}f}"
            f" | {r['n_flagged']} |"
        )
    return "\n".join(lines)


def caveats() -> str:
    """Fixed caveats section shipped with every report."""
    return """## Caveats

- Detectors are unsupervised: flagged counts depend on the assumed
  contamination fraction or on fixed statistical thresholds.
- Anomaly scores are only meaningful relative to one another on the same data.
- Isolation forest and LOF are stochastic; pass a ``seed`` for reproducible
  runs. COPOD, HBOS, kNN and one-class SVM are deterministic.
- Labels come from the synthetic generator, so metrics measure recovery of
  known-injected outliers, not performance on unlabelled real-world data.
- The generalized ESD test assumes a roughly normal baseline and can miss
  small level shifts relative to the noise.
"""


def write_report(path: str, title: str, sections: Sequence[str]) -> str:
    """Write ``sections`` under ``title`` to a markdown file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = [f"# {title}", f"_generated: {date.today().isoformat()}_"] + list(sections)
    text = "\n\n".join(blocks) + "\n"
    path.write_text(text, encoding="utf-8")
    return str(path)
