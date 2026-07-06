"""Train LeBSE by unsupervised SimCSE on legal sentences from a CourtListener replica.

SimCSE (Gao et al., 2021): each sentence is its own positive pair via two dropout views; other
in-batch sentences are negatives (MultipleNegativesRankingLoss). No labels. Base: LaBSE.

    lebse-train --n-opinions 6000 --epochs 1 --dsn "dbname=courtlistener" --out ./lebse
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys

from lebse.text import split_sentences


def legal_sentences(n_opinions: int, max_sent: int, dsn: str) -> list[str]:
    """Sample sentences from the first ``n_opinions`` CourtListener opinions (by id)."""
    # standard string '\s+' (NOT E'\s+', which mis-parses \s as literal 's' and leaves newlines)
    sql = (
        "SELECT regexp_replace(plain_text, '\\s+', ' ', 'g') FROM search_opinion "
        f"WHERE length(plain_text) BETWEEN 1500 AND 25000 ORDER BY id LIMIT {n_opinions};"
    )
    proc = subprocess.run(
        ["psql", dsn, "-tA", "-c", sql], capture_output=True, text=True, timeout=600
    )
    if proc.returncode != 0:
        raise SystemExit("psql failed: " + proc.stderr[:500])
    sents: list[str] = []
    for line in proc.stdout.split("\n"):
        sents.extend(split_sentences(line))
    random.Random(0).shuffle(sents)
    return sents[:max_sent]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lebse-train")
    ap.add_argument("--n-opinions", type=int, default=6000)
    ap.add_argument("--max-sent", type=int, default=250_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--base", default="sentence-transformers/LaBSE")
    ap.add_argument("--dsn", default="dbname=courtlistener")
    ap.add_argument("--out", default="./lebse")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)

    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[lebse] device={dev} base={a.base}")

    sents = legal_sentences(a.n_opinions, a.max_sent, a.dsn)
    print(f"[lebse] {len(sents):,} legal sentences from {a.n_opinions} opinions")
    train = [InputExample(texts=[s, s]) for s in sents]  # SimCSE: same sentence = positive pair
    model = SentenceTransformer(a.base, device=dev)
    # a list of InputExample is a valid map-style dataset at runtime; the torch stub is stricter.
    loader = DataLoader(
        train,  # ty: ignore[invalid-argument-type]
        batch_size=a.batch,
        shuffle=True,
        drop_last=True,
    )
    loss = losses.MultipleNegativesRankingLoss(model)
    steps = (len(train) // a.batch) * a.epochs
    print(f"[lebse] training {a.epochs} epoch(s), {steps} steps, batch {a.batch} -> {a.out}")
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=a.epochs,
        warmup_steps=min(200, steps // 10),
        show_progress_bar=True,
        output_path=a.out,
        use_amp=(dev == "cuda"),
    )
    print(f"[lebse] DONE -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
