# Model Card — LeBSE (Legal-domain LaBSE)

## Model summary
- **Base:** `sentence-transformers/LaBSE` (Feng et al., 2022) — 471 M params, 768-dim, 109 languages.
- **Adaptation:** unsupervised **SimCSE** (Gao et al., 2021) fine-tune on U.S. federal case-law
  sentences. `MultipleNegativesRankingLoss`, each sentence its own dropout-augmented positive.
- **Output:** 768-dim L2-normalized sentence embedding. Drop-in `sentence-transformers` model.
- **Intended use:** legal sentence/opinion retrieval, clustering, near-duplicate & boilerplate
  detection, citation recommendation, and as a frozen feature extractor for downstream legal NLP.
- **Not intended for:** legal advice, outcome prediction on its own, or any decision about a real
  case. It is a text-similarity encoder, not a legal reasoner.

## Training data
Sentences sampled from opinions in the [CourtListener / Free Law Project](https://www.courtlistener.com)
bulk dump (U.S. federal case law, public domain). No labels. No case outcomes, dispositions, or
metadata are used in training — this matters for downstream causal/predictive studies, which can use
LeBSE as a text encoder without leaking outcome information.

## Evaluation
Held-out protocol (`eval.py`): opinions with id beyond the training cut, so no evaluated text was
seen in training. The positive signal is the CourtListener **citation graph**, which SimCSE does not
use — measuring generalization to legal relatedness, not memorization.

Run: CourtListener replica, held-out cut id>9400, pool=3000, first held-out run (n=77 citation
pairs, 413 co-citation pairs; opinion snippets = first ~3000 chars).

| metric | base LaBSE | LeBSE (v1 SimCSE) | better if |
|--------|-----------|-------------------|-----------|
| citation-pair AUROC | **0.484** | 0.340 | higher |
| co-citation AUROC | **0.713** | 0.680 | higher |
| anisotropy (rand-pair cos) | 0.614 | **0.578** | lower |
| alignment | **0.767** | 0.889 | lower |
| uniformity | −1.510 | **−1.650** | lower |
| Δ citation-AUROC (95% CI) | — | **−0.145 [−0.214, −0.081]** | CI excludes 0 |

**Honest interpretation — v1 SimCSE did NOT beat the base model.** The one thing it changed as
expected (lower anisotropy / better uniformity — the documented SimCSE effect) did **not** translate
into better retrieval; citation-pair separation got significantly *worse*. Two caveats temper how
strong a negative this is: (1) both models sit near chance on citation-AUROC because a 3000-char
snippet is dominated by the standardized caption/header, not legal substance; (2) n=77 citation pairs
is small. But the direction is clear and consistent with theory: a light, small-batch, *unsupervised*
SimCSE pass on sentences degrades LaBSE's tuned retrieval geometry rather than improving it.

**Takeaway:** unsupervised SimCSE is the wrong recipe. The path to a real improvement is
citation-**supervised** contrastive training (SPECTER-style) at large batch — see `ROADMAP.md`. v1
weights are published as a reproducible baseline, not as a recommended encoder.

## Limitations
- Single legal system (U.S. federal); not validated on state, contractual, or non-English legal text
  (though the LaBSE base retains multilingual alignment).
- SimCSE is a light adaptation; a **citation-supervised** variant (v2, SPECTER-style) is expected to
  be stronger and is on the roadmap.
- Truncates at LaBSE's 256-token window — it encodes a sentence/short passage, not a whole opinion.
- Inherits LaBSE's pretraining biases.

## Reproducibility
Deterministic given fixed weights: same input → same embedding (no sampling). Training is seeded
(`random.Random(0)` sentence shuffle). See `train.py` / `eval.py`.

## Citation
See `README.md`.
