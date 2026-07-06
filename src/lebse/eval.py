"""Held-out comparison of LeBSE vs a base encoder on legal-semantic geometry.

Evaluates on opinions past the training id cut (never seen in SimCSE), using the CourtListener
citation graph as the positive signal (which SimCSE does not use) — so this measures generalization,
not memorization. Reports citation / co-citation AUROC, anisotropy, Wang-Isola alignment/uniformity,
and a paired bootstrap CI on the citation-AUROC gain.

    lebse-eval --thr 9400 --n 3000 --pairs 2000 --lebse ./lebse
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from lebse import db, metrics, sql

MIN_SNIP = 500  # min snippet chars (filtered in Python, not SQL, to avoid full TOAST detoast)


def _pairs_from_edges(edge_rows: list[str]) -> tuple[list[str], list[str]]:
    edges: list[tuple[int, int]] = []
    for ln in edge_rows:
        p = ln.split(db.SEP)
        if len(p) == 2:
            edges.append((int(p[0]), int(p[1])))
    ids = sorted({i for e in edges for i in e})
    snips: dict[int, str] = {}
    if ids:
        for ln in db.rows(sql.snippets_by_id_sql(ids)):
            if db.SEP in ln:
                k, v = ln.split(db.SEP, 1)
                snips[int(k)] = v
    a_txt, b_txt = [], []
    for a, b in edges:
        sa, sb = snips.get(a), snips.get(b)
        if sa and sb and len(sa) >= MIN_SNIP and len(sb) >= MIN_SNIP:
            a_txt.append(sa)
            b_txt.append(sb)
    return a_txt, b_txt


def load_pool(thr: int, n: int) -> list[str]:
    texts = []
    for ln in db.rows(sql.pool_sql(thr, n * 2)):
        if db.SEP in ln:
            _, t = ln.split(db.SEP, 1)
            if len(t) >= MIN_SNIP:
                texts.append(t)
    return texts[:n]


def _encode(model, texts):
    return model.encode(
        texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def eval_model(name, model, pool, posA, posB, coA, coB):
    rng = np.random.default_rng(1)
    emb = _encode(model, pool)
    n = len(emb)
    i, j = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
    keep = i != j
    i, j = i[keep], j[keep]
    rand_cos = np.sum(emb[i] * emb[j], axis=1)
    pa, pb = _encode(model, posA), _encode(model, posB)
    pos_cos = np.sum(pa * pb, axis=1)
    neg_cos = rand_cos[: len(pos_cos)]
    if coA:  # co-citation is secondary; may be empty -> report NaN, don't crash
        ca, cb = _encode(model, coA), _encode(model, coB)
        co_cos = np.sum(ca * cb, axis=1)
        cocite = metrics.auroc(co_cos, rand_cos[: len(co_cos)])
    else:
        cocite = float("nan")
    res = {
        "name": name,
        "anisotropy": float(rand_cos.mean()),
        "cite_auroc": metrics.auroc(pos_cos, neg_cos),
        "cocite_auroc": cocite,
        "align": metrics.alignment(pa, pb),
        "uniform": metrics.uniformity(emb, i, j),
        "pos_cos": pos_cos,
        "neg_cos": neg_cos,
    }
    print(
        f"[{name}] anisotropy={res['anisotropy']:+.4f} cite_auroc={res['cite_auroc']:.4f} "
        f"cocite_auroc={res['cocite_auroc']:.4f} align={res['align']:.4f} "
        f"uniform={res['uniform']:.4f}"
    )
    return res


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lebse-eval")
    ap.add_argument("--thr", type=int, required=True, help="held-out id cut (opinions with id>thr)")
    ap.add_argument("--n", type=int, default=3000, help="held-out pool size")
    ap.add_argument("--pairs", type=int, default=2000, help="citation / co-citation positive pairs")
    ap.add_argument("--lebse", default="./lebse")
    ap.add_argument("--base", default="sentence-transformers/LaBSE")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    from lebse.model import load

    pool = load_pool(a.thr, a.n)
    posA, posB = _pairs_from_edges(db.rows(sql.citation_edge_sql(a.thr, a.pairs * 2)))
    coA, coB = _pairs_from_edges(db.rows(sql.cocitation_edge_sql(a.thr, a.pairs * 2)))
    print(f"[eval] pool={len(pool)} citation_pairs={len(posA)} cocitation_pairs={len(coA)}")
    if not pool or len(posA) < 50:
        print("[eval] ERROR: too little held-out data; widen --n / --pairs or lower --thr")
        return 1

    base = load(a.base)
    lebse = load(a.lebse)
    rb = eval_model("LaBSE", base, pool, posA, posB, coA, coB)
    rl = eval_model("LeBSE", lebse, pool, posA, posB, coA, coB)

    md, lo, hi = metrics.bootstrap_dauroc(
        rl["pos_cos"], rl["neg_cos"], rb["pos_cos"], rb["neg_cos"]
    )
    sig = lo > 0 or hi < 0
    print("\n==================== VERDICT ====================")
    print(f"  cite_auroc    LaBSE {rb['cite_auroc']:.4f} -> LeBSE {rl['cite_auroc']:.4f}")
    print(f"  cocite_auroc  LaBSE {rb['cocite_auroc']:.4f} -> LeBSE {rl['cocite_auroc']:.4f}")
    print(
        f"  anisotropy    LaBSE {rb['anisotropy']:+.4f} -> LeBSE {rl['anisotropy']:+.4f} "
        "(lower=better)"
    )
    print(
        f"  d(cite_auroc) = {md:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
        f"{'SIGNIFICANT' if sig else 'not significant'}"
    )
    print(
        f"  => LeBSE {'IMPROVES' if (md > 0 and lo > 0) else 'does NOT clearly improve'} "
        "held-out legal retrieval."
    )
    print("=================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
