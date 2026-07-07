"""Database-free held-out evaluation from a relation-pairs JSONL.

The evaluation that produced LeBSE-v2's published numbers, for **any** legal relation given as
pairs: citation retrieval (held-out citing/cited pairs) and docket-lineage retrieval (a district
opinion and its appellate reviewer — an *independent* relation the model never trained on). Point it
at a different pairs file to measure a different relation.

Each JSONL line is ``{"a": text_a, "b": text_b}`` — a true related pair. Positives are the true
pairs; negatives are a derangement (random cross-pairing, i != j). AUROC = P(true pair above a
random pair). Reports base vs. adapted and a paired bootstrap CI on the gain (reuses
:mod:`lebse.metrics`). numpy-only stats; needs sentence-transformers only to encode.

    lebse-eval-pairs --pairs pairs_docket.jsonl --lebse ahbond/lebse
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from lebse import metrics


def load_pairs(path: str, limit: int = 0) -> tuple[list[str], list[str]]:
    a_txt, b_txt = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("a") and d.get("b"):
                a_txt.append(d["a"])
                b_txt.append(d["b"])
            if limit and len(a_txt) >= limit:
                break
    return a_txt, b_txt


def deranged(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation with no fixed point (so every negative is a genuine cross-pairing)."""
    p = rng.permutation(n)
    fixed = np.where(p == np.arange(n))[0]
    if len(fixed) > 1:
        p[fixed] = p[np.roll(fixed, 1)]
    elif len(fixed) == 1:
        k = int(fixed[0])
        j = (k + 1) % n
        p[k], p[j] = p[j], p[k]
    return p


def _scores(model, A, B, rng):
    ea = model.encode(
        A, batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    eb = model.encode(
        B, batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    pos = np.sum(ea * eb, axis=1)
    neg = np.sum(ea * eb[deranged(len(B), rng)], axis=1)
    return pos, neg


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lebse-eval-pairs")
    ap.add_argument("--pairs", required=True, help="JSONL of {'a':..., 'b':...} true related pairs")
    ap.add_argument("--base", default="sentence-transformers/LaBSE")
    ap.add_argument("--lebse", default="ahbond/lebse")
    ap.add_argument("--limit", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    from lebse.model import load

    A, B = load_pairs(a.pairs, a.limit)
    if len(A) < 50:
        print(f"[eval] only {len(A)} pairs — need >= 50 for a stable estimate")
        return 1
    print(f"[eval] {len(A)} held-out pairs from {a.pairs}", flush=True)
    base = load(a.base)
    lebse = load(a.lebse)
    bp, bn = _scores(base, A, B, np.random.default_rng(1))
    lp, ln = _scores(lebse, A, B, np.random.default_rng(1))
    base_auroc, lebse_auroc = metrics.auroc(bp, bn), metrics.auroc(lp, ln)
    md, lo, hi = metrics.bootstrap_dauroc(lp, ln, bp, bn, seed=7)
    sig = lo > 0 or hi < 0
    print(f"[base ] auroc={base_auroc:.4f}")
    print(f"[lebse] auroc={lebse_auroc:.4f}")
    print("==================== VERDICT ====================")
    print(f"  auroc   base {base_auroc:.4f}  ->  LeBSE {lebse_auroc:.4f}")
    print(
        f"  d(auroc) = {md:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
        f"{'SIGNIFICANT' if sig else 'not significant'}"
    )
    verdict = "IMPROVES" if (md > 0 and lo > 0) else "does NOT clearly improve"
    print(f"  => LeBSE {verdict} this relation.")
    print("=================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
