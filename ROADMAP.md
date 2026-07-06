# LeBSE Roadmap

## v1 (current) — unsupervised SimCSE
Light, label-free legal adaptation of LaBSE. Establishes the training/eval harness and a held-out,
citation-graph-based benchmark. Deterministic at inference.

## v2 — citation-supervised (SPECTER-style)
Use the CourtListener citation graph (~77 M edges) as **supervised positive pairs**: a citing
opinion and the opinion it cites are a related pair; in-batch (and hard-mined) opinions are
negatives. This is real legal-relatedness supervision, not dropout noise, and is expected to beat
SimCSE on retrieval.

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
- **PyPI is on hold until a version actually beats base LaBSE** on the held-out benchmark. v1 SimCSE
  does not (see `MODEL_CARD.md`), so it ships as source on GitHub only, not on PyPI. The
  trusted-publishing workflow is wired and will fire on the first GitHub release once v2 clears the
  bar.
- When ready: push weights to the Hugging Face Hub (`ahb-sjsu/lebse`) with the model card; keep
  training/eval code here; keep the model artifact out of git (see `.gitignore`).

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
