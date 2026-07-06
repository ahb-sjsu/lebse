# Changelog

## 0.1.0 (unreleased)
- Initial release: `src/`-layout package with `lebse.load()`, training (`lebse-train`) and held-out
  evaluation (`lebse-eval`) CLIs, and pure, unit-tested `text` / `sql` / `metrics` modules.
- CI: black, ruff, ty, pytest across Python 3.9 / 3.11 / 3.12.
- Held-out benchmark against base LaBSE via the CourtListener citation graph.
- **Honest finding:** v1 unsupervised SimCSE does not beat base LaBSE on citation retrieval
  (Δ AUROC −0.145, 95% CI [−0.214, −0.081]). Weights ship as a reproducible baseline; the roadmap
  targets citation-supervised training.
