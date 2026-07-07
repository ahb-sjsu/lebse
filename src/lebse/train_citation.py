"""LeBSE v2 — citation-supervised (SPECTER-style) fine-tune of LaBSE.

Positive pair = (citing opinion body, cited opinion body) from the CourtListener citation graph:
real legal-relatedness supervision (vs v1's unsupervised SimCSE dropout). MultipleNegativesRanking
Loss uses the rest of the batch as negatives, so batch size = #negatives = the main quality lever.

Reads pairs JSONL: one ``{"a": citing_body, "b": cited_body}`` per line (build it from a
CourtListener Postgres replica: sample citation edges, fetch body snippets). Runs on any CUDA GPU.

    lebse-train-citation --pairs pairs.jsonl --batch 96 --max-seq-len 128 --epochs 2 --out ./lebse
"""

from __future__ import annotations

import argparse
import json
import sys


def load_pairs(path, limit, InputExample):
    ex = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            a, b = d.get("a"), d.get("b")
            if a and b:
                ex.append(InputExample(texts=[a, b]))
            if limit and len(ex) >= limit:
                break
    return ex


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lebse-train-citation")
    ap.add_argument("--pairs", required=True, help="JSONL of {'a':..., 'b':...} citation pairs")
    ap.add_argument("--limit", type=int, default=0, help="cap pairs (0 = all)")
    ap.add_argument("--base", default="sentence-transformers/LaBSE")
    ap.add_argument("--batch", type=int, default=96, help="= #in-batch negatives; bigger is better")
    ap.add_argument("--max-seq-len", type=int, default=128, help="token cap (memory vs context)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", default="./lebse")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)

    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[v2] device={dev} base={a.base}", flush=True)

    train = load_pairs(a.pairs, a.limit, InputExample)
    print(f"[v2] {len(train):,} citation pairs; batch={a.batch} epochs={a.epochs}", flush=True)
    model = SentenceTransformer(a.base, device=dev)
    model.max_seq_length = a.max_seq_len
    loader = DataLoader(
        train,  # ty: ignore[invalid-argument-type]
        batch_size=a.batch,
        shuffle=True,
        drop_last=True,
    )
    loss = losses.MultipleNegativesRankingLoss(model)
    steps = (len(train) // a.batch) * a.epochs
    print(f"[v2] {steps} steps, max_seq_len={model.max_seq_length} -> {a.out}", flush=True)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=a.epochs,
        warmup_steps=min(500, max(1, steps // 10)),
        optimizer_params={"lr": a.lr},
        show_progress_bar=True,
        output_path=a.out,
        use_amp=(dev == "cuda"),
    )
    print(f"[v2] DONE -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
