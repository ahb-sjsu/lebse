# LeBSE — Legal-BERT Sentence Embeddings (a legal-domain LaBSE)

[![CI](https://github.com/ahb-sjsu/lebse/actions/workflows/ci.yml/badge.svg)](https://github.com/ahb-sjsu/lebse/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with ty](https://img.shields.io/badge/types-ty-261230.svg)](https://github.com/astral-sh/ty)

**LeBSE** adapts Google's [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) to the
register of U.S. case law, so that legal sentences and opinions land in a better-organized
embedding space than the general-purpose base model — while keeping LaBSE's 109-language alignment.

It is a drop-in `sentence-transformers` model: same API, same 768-dim output, same tokenizer.

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("ahb-sjsu/lebse")          # or a local path
v = m.encode(["The district court lacked subject-matter jurisdiction over the claim."])
```

## Why

General sentence encoders are trained on web text; their geometry is anisotropic (embeddings pile
into a narrow cone) and tuned to everyday semantics, not the dense, formulaic register of judicial
writing. LeBSE re-tunes the geometry on real opinions so that legally-related text is measurably
closer and the space is more isotropic — useful for legal retrieval, clustering, near-duplicate and
boilerplate detection, citation recommendation, and as a feature extractor for downstream legal NLP.

## How it's trained (v2, citation-supervised)

Citation-**supervised** contrastive fine-tuning, SPECTER-style: positive pairs are (citing opinion
body, cited opinion body) drawn from the CourtListener citation graph, and `MultipleNegativesRanking
Loss` uses the rest of the batch as negatives. Base model: `sentence-transformers/LaBSE`. Corpus:
U.S. federal opinions from the [CourtListener / Free Law Project](https://www.courtlistener.com/help/api/bulk-data/)
bulk data (public domain). No case outcomes are used.

```bash
# lebse-train = SimCSE (v1) baseline; lebse-train-citation = the v2 citation-supervised model.
lebse-train-citation --pairs pairs_train.jsonl --batch 96 --max-seq-len 128 --epochs 2 --out ./lebse
```

> **v1** was unsupervised SimCSE and did *not* beat base LaBSE (it's kept in `MODEL_CARD.md` for the
> record). The published weights are **v2** (citation-supervised), which does — decisively.

## Rigorous evaluation

Two held-out protocols with **opinion-level** splits (train/eval opinions are disjoint), so any gain
is generalization: (1) **citation retrieval** — the trained relation, on held-out opinions; and
(2) **docket-lineage retrieval** — an *independent* relation (a district opinion and its appellate
reviewer, matched by docket number) the model never trained on. Each reports a paired bootstrap 95%
CI on the AUROC gain.

**Held-out results (v2, citation-supervised):**

| eval | base LaBSE | **LeBSE-v2** | Δ AUROC (95% CI) |
|------|-----------|--------------|------------------|
| citation retrieval (trained relation, held-out) | 0.765 | **0.971** | **+0.206 [+0.190, +0.223]** |
| docket-lineage (independent relation, unseen) | 0.545 | **0.562** | **+0.018 [+0.004, +0.031]** |

LeBSE-v2 dramatically improves citation-type relatedness and transfers a small-but-significant amount
to an independent legal relation (district↔appellate lineage) it never trained on. See
[`MODEL_CARD.md`](MODEL_CARD.md) for the full protocol and honest caveats, and
[`PAPER.md`](PAPER.md) for the technical report. (v1 used unsupervised SimCSE and did *not* beat
LaBSE — kept for the record.)

Reproduce **either** relation from a pairs file with the same command (both published numbers came
from this evaluator — point it at citation pairs or docket-lineage pairs):

```bash
lebse-eval-pairs --pairs pairs_citation.jsonl --lebse ahbond/lebse   # trained relation
lebse-eval-pairs --pairs pairs_docket.jsonl   --lebse ahbond/lebse   # independent relation
```

## Files

| file | purpose |
|------|---------|
| `train.py` / `train_citation.py` | SimCSE (v1 baseline) / citation-supervised (v2) fine-tune |
| `eval.py` | held-out citation eval driven from a CourtListener Postgres replica |
| `eval_pairs.py` | DB-free eval from a pairs JSONL — reproduces **both** the citation and docket-lineage numbers |
| `MODEL_CARD.md` / `PAPER.md` | intended use + full protocol / technical report |

## License

Apache-2.0 (matching the LaBSE base). Training data is U.S. federal case law (public domain).
See `LICENSE`.

## Citation

```bibtex
@software{bond_lebse_2026,
  author = {Bond, Andrew H.},
  title  = {LeBSE: Legal-Domain Sentence Embeddings adapted from LaBSE},
  year   = {2026},
  url    = {https://github.com/ahb-sjsu/lebse}
}
```
