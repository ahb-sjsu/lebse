"""Embedding-geometry metrics (numpy) — pure, unit-tested.

These quantify how well an encoder organizes legal text:
  - anisotropy: mean cosine of random pairs (lower = more isotropic = better).
  - alignment / uniformity: Wang & Isola (2020) diagnostics on L2-normalized embeddings.
  - auroc: separation of positive vs negative cosine scores.
  - bootstrap_dauroc: paired bootstrap CI on an AUROC difference between two models.

``auroc``/``bootstrap_dauroc`` import scikit-learn lazily so ``import lebse.metrics`` stays light.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def anisotropy(emb: NDArray, i: NDArray, j: NDArray) -> float:
    """Mean cosine over index pairs (i, j). Assumes rows of ``emb`` are L2-normalized."""
    return float(np.sum(emb[i] * emb[j], axis=1).mean())


def alignment(pa: NDArray, pb: NDArray) -> float:
    """E||a - b||^2 over positive pairs (normalized). Lower = positives are closer."""
    return float(np.mean(np.sum((pa - pb) ** 2, axis=1)))


def uniformity(emb: NDArray, i: NDArray, j: NDArray) -> float:
    """log E exp(-2 ||xi - xj||^2) over random pairs. Lower = embeddings more spread out."""
    d2 = np.sum((emb[i] - emb[j]) ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-2.0 * d2))))


def auroc(pos: NDArray, neg: NDArray) -> float:
    """AUROC separating positive scores from negative scores (higher = better)."""
    from sklearn.metrics import roc_auc_score

    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def bootstrap_dauroc(
    pos_a: NDArray,
    neg_a: NDArray,
    pos_b: NDArray,
    neg_b: NDArray,
    n_boot: int = 2000,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Paired bootstrap on auroc(model_a) - auroc(model_b). Resamples the SAME pair indices for both
    models. Returns (mean_diff, lo_2.5%, hi_97.5%)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    npos, nneg = len(pos_a), len(neg_a)
    y = np.r_[np.ones(npos), np.zeros(nneg)]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pi = rng.integers(0, npos, npos)
        ni = rng.integers(0, nneg, nneg)
        da = roc_auc_score(y, np.r_[pos_a[pi], neg_a[ni]])
        db = roc_auc_score(y, np.r_[pos_b[pi], neg_b[ni]])
        diffs[b] = da - db
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
