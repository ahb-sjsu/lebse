import numpy as np

from lebse import metrics


def _norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_alignment_zero_for_identical_pairs():
    a = _norm(np.random.default_rng(0).normal(size=(50, 16)))
    assert metrics.alignment(a, a) == 0.0


def test_alignment_positive_for_different_pairs():
    rng = np.random.default_rng(0)
    a = _norm(rng.normal(size=(50, 16)))
    b = _norm(rng.normal(size=(50, 16)))
    assert metrics.alignment(a, b) > 0.0


def test_anisotropy_bounds():
    emb = _norm(np.random.default_rng(1).normal(size=(200, 32)))
    i = np.arange(100)
    j = np.arange(100, 200)
    val = metrics.anisotropy(emb, i, j)
    assert -1.0 <= val <= 1.0
    assert abs(val) < 0.3  # random high-dim unit vectors are near-orthogonal


def test_auroc_perfect_separation():
    pos = np.array([0.9, 0.8, 0.95, 0.85])
    neg = np.array([0.1, 0.2, 0.05, 0.15])
    assert metrics.auroc(pos, neg) == 1.0


def test_auroc_chance_is_half():
    x = np.array([0.5, 0.5, 0.5, 0.5])
    assert metrics.auroc(x, x) == 0.5


def test_bootstrap_dauroc_sign_and_ci_order():
    rng = np.random.default_rng(3)
    # model A separates better than model B -> positive diff, CI ordered
    pos_a = rng.normal(1.0, 0.2, 300)
    neg_a = rng.normal(0.0, 0.2, 300)
    pos_b = rng.normal(0.4, 0.2, 300)
    neg_b = rng.normal(0.0, 0.2, 300)
    mean, lo, hi = metrics.bootstrap_dauroc(pos_a, neg_a, pos_b, neg_b, n_boot=300)
    assert lo <= mean <= hi
    assert mean > 0
