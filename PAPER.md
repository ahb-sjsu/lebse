# LeBSE: Citation-Supervised Legal-Domain Sentence Embeddings

**Andrew H. Bond**, San José State University · Technical report, 2026 · Code: https://github.com/ahb-sjsu/lebse · Model: https://huggingface.co/ahbond/lebse

## Abstract

We adapt LaBSE, a general-purpose multilingual sentence encoder, to U.S. case law by
citation-supervised contrastive fine-tuning, and evaluate it honestly on two held-out relations. A
first, unsupervised attempt (SimCSE) *degraded* the base encoder; citation supervision reversed this
decisively. On held-out citation retrieval, LeBSE improves AUROC from 0.765 to **0.971** (Δ +0.206,
95% CI [+0.190, +0.223]). On an **independent** relation the model never trained on—docket lineage,
linking a district opinion to its appellate reviewer by docket number—it improves AUROC from 0.545 to
**0.562** (Δ +0.018, 95% CI [+0.004, +0.031]): small but statistically significant transfer. LeBSE is
a strong legal-relatedness encoder; the gap between the two results bounds how much of the gain is
transferable structure versus specialization to the trained relation.

## 1. Motivation

General sentence encoders are trained on web text; their geometry is anisotropic and tuned to everyday
semantics rather than the dense, formulaic register of judicial writing. We ask whether a light
domain adaptation of LaBSE improves legal-text retrieval, and—critically—*which form of adaptation*,
selected by held-out performance rather than training loss. No case outcomes are used in training, so
LeBSE can serve as a text encoder in downstream outcome-prediction studies without leaking the outcome.

## 2. Method

Base model: `sentence-transformers/LaBSE` (471M params, 768-dim, 109 languages). We compare two
adaptations, both `MultipleNegativesRankingLoss` with in-batch negatives:

- **v1 — unsupervised SimCSE** (Gao et al., 2021): each legal sentence is its own positive pair via
  two dropout views. No labels.
- **v2 — citation-supervised** (SPECTER-style; Cohan et al., 2020): positive pairs are (citing opinion
  body, cited opinion body) drawn from the CourtListener citation graph. Real legal-relatedness
  supervision rather than dropout noise.

**Data.** Opinions from the [CourtListener / Free Law Project](https://www.courtlistener.com) bulk dump
(U.S. federal case law, public domain), loaded to a local Postgres replica. Training uses 100,344
(citing, cited) body-snippet pairs; body snippets skip the standardized caption. v2 was trained on one
NVIDIA A10 (batch 96, `max_seq_length` 128, 2 epochs; contrastive loss 2.30 → 1.38).

## 3. Evaluation

Two held-out protocols, both with **opinion-level** splits (train and eval opinions disjoint), so any
gain is generalization rather than memorization. Positives are true related pairs; negatives are a
derangement (random cross-pairing). AUROC is the probability a true pair scores above a random pair;
we report a paired bootstrap 95% CI on the model-vs-base difference.

1. **Citation retrieval** — the *trained* relation, on held-out opinions (id mod 10 == 7); 1,293 pairs.
2. **Docket-lineage retrieval** — an *independent* relation the model never trained on: a district
   opinion and the appellate opinion that reviews it, linked by matching docket numbers in the court's
   originating-case metadata (not by citation). 4,406 pairs, party names stripped so the task tests
   body semantics. The district opinions predate the electronic-text era and fall below the training
   sampling threshold, so they are unseen at the *entity* level, not merely the pair level.

### 3.1 Results

| relation | base LaBSE | **LeBSE-v2** | Δ AUROC (95% CI) |
|----------|:----------:|:------------:|------------------|
| citation retrieval (**trained**) | 0.765 | **0.971** | **+0.206 [+0.190, +0.223]** |
| docket lineage (**independent**) | 0.545 | **0.562** | **+0.018 [+0.004, +0.031]** |
| anisotropy (rand-pair cos, lower better) | +0.570 | **+0.259** | — |

The unsupervised v1 baseline reached only 0.340 on citation retrieval—**worse** than base LaBSE
(Δ −0.145, 95% CI [−0.214, −0.081]). A light, small-batch, unsupervised contrastive pass collapses
LaBSE's tuned geometry; its training loss falls while held-out retrieval falls with it. We report this
because it is the most useful data point for anyone tempted to reach for the cheap objective first.

### 3.2 A probe-calibration note

The first citation probe encoded each opinion's first ~3000 characters and returned **base LaBSE at
0.48—chance.** A 471M-parameter encoder cannot be that weak; the defect was in the *probe*. An
opinion's first 3000 characters are almost entirely the standardized caption ("UNITED STATES COURT OF
APPEALS FOR THE ... CIRCUIT ..."), a surface feature shared by all pairs. Skipping the caption and
encoding the body moved base LaBSE to a sensible 0.765 and made the comparison trustworthy. Before
trusting a probe to rank models, verify that a known-strong reference clears chance; a strong reference
at chance means the probe is measuring surface, not the target.

## 4. Interpretation

LeBSE-v2 dramatically improves the relatedness it was trained on and transfers a small,
statistically-significant amount to an independent legal relation. Both models sit near chance on
docket lineage (~0.55) because a district merits opinion and its appellate reviewer, with party names
removed, are only weakly similar in the body—a genuinely hard task, not a model failure. The
independent probe disciplines the claim: without it, 0.971 invites "a legal reasoning model"; with it,
the transfer ceiling is measured, not assumed. The methodology (objective-space search, probe
calibration, and the cross-relation generalization gap) is developed further as a case study in the
*structural-fuzzing* book, and the gap statistic is available as `structural_fuzzing.cross_relation_gap`.

## 5. Limitations

Single legal system (U.S. federal); specialized to citation-type relatedness with real-but-small
transfer elsewhere; encodes a ~128-token paragraph, not a whole opinion; inherits LaBSE's pretraining
biases. Not for legal advice or case-outcome decisions.

## 6. Reproducibility

Training: `lebse-train-citation --pairs pairs_train.jsonl --batch 96 --max-seq-len 128 --epochs 2`.
Evaluation (either relation): `lebse-eval-pairs --pairs <citation|docket>_pairs.jsonl --lebse ahbond/lebse`.
Building the pairs requires a CourtListener Postgres replica: citation pairs are sampled from the
`search_opinionscited` edge table (both endpoints with populated `plain_text`, i.e. recent opinions);
docket-lineage pairs match district (`jurisdiction='FD'`) to appellate (`'F'`) opinions on normalized
docket numbers via `search_originatingcourtinformation`, recovering old-opinion text from
`html_with_citations` when `plain_text` is empty. See `MODEL_CARD.md` for the full protocol.

## References

- Feng et al. (2022), *Language-agnostic BERT Sentence Embedding* (LaBSE).
- Gao, Yao, Chen (2021), *SimCSE: Simple Contrastive Learning of Sentence Embeddings*.
- Cohan et al. (2020), *SPECTER: Document-level Representation Learning using Citation-informed Transformers*.
- CourtListener / Free Law Project bulk data (public domain).
