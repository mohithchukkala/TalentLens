# Figures generated from the stored evaluation artifacts

Regenerate with `python docs/figures/make_figures.py` (pure Python, no extra packages). The script reads
`artifacts/evaluation_results.csv` and `artifacts/evaluation_scores.csv` and types in no number by hand. All data are
**synthetic** (5 invented jobs, 46 invented resumes); the figures say nothing about real hiring. Open the `.svg` files in a
browser; for slides, insert them as pictures or export a PNG from the browser.

| File | Shows | Data source | Table view |
|---|---|---|---|
| `fig_evaluation_macro.svg` | macro Precision@3, Recall@3, MRR, nDCG@3 for TF-IDF, semantic, combined | `evaluation_results.csv` (scope MACRO) | README.md and EXPERIMENT_RESULTS.md evaluation table |
| `fig_ndcg_per_job.svg` | nDCG@3 per job for the three methods | `evaluation_results.csv` (per-job rows) | EXPERIMENT_RESULTS.md per-job table |
| `fig_component_influence.svg` | each weighted term's share of the variance of the combined index (skill coverage 0.746, semantic 0.201, TF-IDF 0.053) | `evaluation_scores.csv` | EXPERIMENT_RESULTS.md "Why skill coverage dominates" |

Colours (light and dark variants are built in): TF-IDF blue, semantic orange, combined green, taken from the first three slots of a
categorical palette whose all-pairs colour-vision separation was checked with a validator; the green is below 3:1 contrast on a
light background, so every bar carries its printed value. Each bar also has a hover tooltip in a browser.

## Architecture diagrams (Phase 13)

Regenerate with `python docs/figures/make_diagrams.py` (pure Python, no extra packages). The script is the editable source:
change a box, label or arrow in it and run it again. Each diagram is drawn from the implementation (`src/`, `app.py`), not
from the proposal.

| File | Shows |
|---|---|
| `diagram_architecture.svg` | System architecture: PDF upload, extraction, preprocessing, the three selectable methods (A TF-IDF, B semantic, C combined), skill coverage as a **component only (not selectable)**, ranking, CSV export, in-memory/localhost privacy boundary |
| `diagram_pipeline.svg` | End-to-end pipeline for one job description and a batch of resumes, including the method selector, the exclusion of unreadable files, the formula 0.50·max(0,S) + 0.20·T + 0.30·K and the renormalisation when coverage is unavailable |

PNG renders for slides are written to `presentation/images/` by `node presentation/tools/render_images.js` (tooling that is
not part of `requirements.txt`; it needs Playwright and an installed Edge or Chrome). The diagrams make no accuracy claim and
do not show anything the code does not do (no OCR, no database, no network service, no trained model).

Not figures of this project: the application screenshots are in `docs/screenshots/` (status in
`docs/SCREENSHOT_CHECKLIST.md`). The Mermaid sketches that used to be in `docs/PRESENTATION_OUTLINE.md` are superseded by the SVG
diagrams above.
