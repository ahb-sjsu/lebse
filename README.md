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

## How it's trained

Unsupervised **SimCSE** (Gao et al., 2021): each sentence is its own positive pair via two dropout
views; other in-batch sentences are negatives (`MultipleNegativesRankingLoss`). No labels required.
Base model: `sentence-transformers/LaBSE`. Corpus: sentences sampled from U.S. federal opinions in
the [CourtListener / Free Law Project](https://www.courtlistener.com/help/api/bulk-data/) bulk data
(public domain).

> A citation-**supervised** variant (SPECTER-style, using the 77 M-edge citation graph as positive
> pairs) is the planned v2 and is expected to beat SimCSE; see `ROADMAP`.

```bash
pip install -r requirements.txt
python train.py --n-opinions 6000 --epochs 1 --dsn "dbname=courtlistener" --out ./lebse
```

## Rigorous evaluation

`eval.py` compares LeBSE against base LaBSE on opinions the model **never saw during training**
(held out by opinion id). The positive signal is the CourtListener **citation graph**, which the
SimCSE run does not use — so this measures genuine legal-semantic generalization, not memorization.

Metrics: citation-pair retrieval AUROC (headline), co-citation AUROC, embedding anisotropy
(Wang-Isola isotropy), alignment/uniformity, and a paired bootstrap 95% CI on the AUROC gain.

**First held-out result (honest):** v1 unsupervised SimCSE did **not** beat base LaBSE — citation-pair
AUROC 0.484 → 0.340 (Δ −0.145, 95% CI [−0.214, −0.081], significant). It lowered anisotropy as SimCSE
is known to, but that did not help retrieval. See [`MODEL_CARD.md`](MODEL_CARD.md) for the full table
and caveats. The path to a real gain is citation-**supervised** contrastive training — see
[`ROADMAP.md`](ROADMAP.md). v1 weights are a reproducible baseline, not a recommended encoder.

```bash
LEBSE_DSN="dbname=courtlistener" python eval.py --n 4000 --pairs 3000 --lebse ./lebse
```

## Files

| file | purpose |
|------|---------|
| `train.py` | SimCSE fine-tune of LaBSE on legal sentences |
| `eval.py`  | held-out, citation-graph-based comparison vs base LaBSE |
| `MODEL_CARD.md` | intended use, training data, metrics, limitations |
| `requirements.txt` | pinned deps |

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
