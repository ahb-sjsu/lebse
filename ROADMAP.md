# LeBSE Roadmap

## v1 (current) — unsupervised SimCSE
Light, label-free legal adaptation of LaBSE. Establishes the training/eval harness and a held-out,
citation-graph-based benchmark. Deterministic at inference.

## v2 — citation-supervised (SPECTER-style) — DONE ✅ (shipped as 0.2.0)
Uses the CourtListener citation graph as **supervised positive pairs** (citing ↔ cited opinion
bodies); in-batch opinions are negatives. Real legal-relatedness supervision, not dropout noise.
**Result:** held-out citation-AUROC 0.765 → 0.971 (Δ +0.206), plus a small significant gain on an
independent docket-lineage task. See `MODEL_CARD.md`. Weights: `ahbond/lebse` on the HF Hub.

**Design constraints:**
- **Never train on case outcomes** (affirmed/reversed/vacated). Citation supervision encodes
  *topical relatedness*, which is safe for downstream outcome-prediction studies. Training on the
  outcome would make any such study circular.
- **Train/test hygiene:** if the model is trained on citations, the evaluation must switch its
  positive signal to an *independent* structure (e.g. docket lineage, or a held-out legal STS set) —
  you cannot train and test on the same citation edges.
- **Scale knob:** contrastive quality is dominated by the number of in-batch negatives = batch size.
  Large-batch training (A100/H100, gradient checkpointing, or GradCache) is the main quality lever.

## v3 — longer context & pooling
Move beyond the 256-token window (hierarchical or long-context pooling) so whole opinions embed
without truncation.

## Publishing
- **v2 cleared the bar**, so 0.2.0 is published: package on PyPI, weights on the HF Hub
  (`ahbond/lebse`). Model artifact stays out of git (see `.gitignore`).

## v3 — next
- Bigger batch (more negatives) on an A100/L40 (or GradCache on the A10); more pairs; hard negatives.
- Longer context / hierarchical pooling to embed whole opinions, not a 128-token paragraph.
- More independent downstream evals (legal-doctrine clustering, statute retrieval).

## Developing
Run the exact CI checks locally. Note: the `dev` extra intentionally omits `torch`/
`sentence-transformers`, so run the type check in that same torch-free environment (CI does):

```bash
pip install -e ".[dev]"
black --check --no-cache .   # --no-cache: black's cache can hide drift
ruff check .
ty check src
pytest
```
