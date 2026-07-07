# Model Card — LeBSE (Legal-domain LaBSE)

## Model summary
- **Base:** `sentence-transformers/LaBSE` (Feng et al., 2022) — 471 M params, 768-dim, 109 languages.
- **Adaptation (v2, current):** citation-**supervised** contrastive fine-tune (SPECTER-style). Positive
  pairs are (citing opinion body, cited opinion body) from the CourtListener citation graph;
  `MultipleNegativesRankingLoss` treats the rest of the batch as negatives.
- **Output:** 768-dim L2-normalized sentence embedding. Drop-in `sentence-transformers` model.
- **Intended use:** legal opinion/paragraph retrieval, citation recommendation, clustering,
  near-duplicate detection, and as a frozen feature extractor / strong legal-domain baseline for
  downstream legal NLP.
- **Not intended for:** legal advice, or any decision about a real case. It is a text-similarity
  encoder, not a legal reasoner.

## Training data
100,344 (citing, cited) opinion-body pairs from the
[CourtListener / Free Law Project](https://www.courtlistener.com) bulk dump (U.S. federal case law,
public domain). Body snippets skip the caption/header. **No case outcomes** (affirmed/reversed) are
used — the signal is citation relatedness only — so LeBSE can be used as a text encoder in downstream
outcome-prediction studies without leaking the outcome.

## Evaluation
Two held-out protocols. Splits are **opinion-level** (disjoint opinions in train vs eval), so gains
are generalization, not memorization.

**1. Citation retrieval** (the trained relation, held-out opinions, id %% 10 == 7; 1,293 pairs).
Positive = true (citing, cited) pair; negative = random cross-pairing. AUROC = P(true > random).

**2. Docket-lineage retrieval** (an *independent* relation the model never trained on: a district
opinion and its appellate reviewer, linked by matching docket numbers; 4,406 pairs, party names
stripped so it tests body semantics). District opinions are pre-2015 (below the id>10M training
floor) — unseen.

| eval | base LaBSE | **LeBSE-v2** | Δ AUROC (95% CI) |
|------|-----------|--------------|------------------|
| citation retrieval (trained relation) | 0.765 | **0.971** | **+0.206 [+0.190, +0.223]** |
| docket-lineage (independent relation) | 0.545 | **0.562** | **+0.018 [+0.004, +0.031]** |
| anisotropy (rand-pair cos, lower=better) | +0.570 | **+0.259** | — |

**Honest interpretation.** LeBSE-v2 **dramatically** improves the relatedness it was trained on
(citation AUROC 0.765 → 0.971) and also improves an **independent** legal relation by a **small but
statistically significant** margin (docket-lineage +0.018). Both models sit near chance on
docket-lineage (~0.55) because a district merits opinion and its appellate reviewer, with party
names removed, are only weakly similar in the body — a genuinely hard task. So: strong on
citation-type relatedness, with modest real transfer elsewhere. It also markedly improves embedding
isotropy (0.570 → 0.259).

### v1 (unsupervised SimCSE) — superseded, kept for the record
The first attempt fine-tuned LaBSE with unsupervised SimCSE (dropout positives). On held-out citation
retrieval it was significantly **worse** than base LaBSE (Δ AUROC −0.145, 95% CI [−0.214, −0.081]).
Lesson: a light, small-batch, unsupervised SimCSE pass degrades LaBSE's tuned geometry. Citation
supervision (v2) reverses this decisively. v1 is not distributed.

## Limitations
- Single legal system (U.S. federal); not validated on state, contractual, or non-English legal text
  (though the LaBSE base retains multilingual alignment).
- Specialized to citation-type relatedness; transfer to very different relations is real but small.
- Trained/evaluated at `max_seq_length = 128` tokens — encodes a paragraph, not a whole opinion.
- Inherits LaBSE's pretraining biases.

## Reproducibility
Deterministic given fixed weights: same input → same embedding (no sampling). Trained on one NVIDIA
A10 (NRP), `MultipleNegativesRankingLoss`, batch 96, `max_seq_length` 128, 2 epochs (loss 2.30 → 1.38).
Data extraction, training, and both evals are scripted; see `train.py` / `eval.py` and `ROADMAP.md`.

## Citation
See `README.md`.
