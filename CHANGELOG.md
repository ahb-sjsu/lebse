# Changelog

## 0.2.0
- **LeBSE-v2: citation-supervised model — beats base LaBSE.** SPECTER-style contrastive fine-tune on
  100k (citing, cited) opinion-body pairs from the CourtListener citation graph (NRP A10 GPU).
  - Held-out **citation retrieval**: AUROC 0.765 → **0.971** (Δ +0.206, 95% CI [+0.190, +0.223]).
  - Independent **docket-lineage** retrieval (never trained on): 0.545 → 0.562 (Δ +0.018,
    95% CI [+0.004, +0.031]) — small but significant transfer.
- New `lebse-train-citation` CLI; `DEFAULT_MODEL` → `ahbond/lebse` (weights on the HF Hub).
- Model card rewritten with both evals; v1 SimCSE kept for the record as the superseded baseline.

## 0.1.0 (unreleased)
- Initial release: `src/`-layout package with `lebse.load()`, training (`lebse-train`) and held-out
  evaluation (`lebse-eval`) CLIs, and pure, unit-tested `text` / `sql` / `metrics` modules.
- CI: black, ruff, ty, pytest across Python 3.9 / 3.11 / 3.12.
- Held-out benchmark against base LaBSE via the CourtListener citation graph.
- **Honest finding:** v1 unsupervised SimCSE does not beat base LaBSE on citation retrieval
  (Δ AUROC −0.145, 95% CI [−0.214, −0.081]). Weights ship as a reproducible baseline; the roadmap
  targets citation-supervised training.
