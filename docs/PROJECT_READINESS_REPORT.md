# Project readiness report (end of Phase 11)

Scope of this report: what has been verified, what has not, and what could still go wrong when this project is demonstrated
or assessed. It does not claim real-world validation, production readiness, OCR support or deployment. It was written from the
repository as it is now; every number is in `EXPERIMENT_RESULTS.md`.

## 1. Verdict
Ready for a **local academic demonstration and viva**, with the limitations below stated openly. Not ready, and not claimed to be
ready, for real recruitment use.

## 2. Verified achievements (with the evidence)
| Claim | Evidence | Verified how |
|---|---|---|
| The whole pipeline works end to end (PDF -> text -> skills -> three scores -> ranking -> details -> CSV) | Phase 10 probe with real generated PDFs and the real model, and 39 validation tests | run |
| Bad files are handled and reported (blank, corrupted, password-protected, image-only, empty, oversized, duplicate) and never stop other files | parser tests, validation tests, probe | run |
| Each method calls only its intended existing module; the interface contains no matching algorithm | spy tests; AST test of `app.py` / `app_logic.py`; app results equal direct module calls with the real model | run |
| The combined formula and weights are unchanged (0.50 / 0.20 / 0.30; missing coverage renormalises) | guard test; formula tests | run |
| The synthetic evaluation is reproducible and unchanged | `python -m src.evaluate` re-run: results CSVs byte-identical, max score difference 0.0; data and dictionary hashes equal the recorded ones (re-checked in Phase 11) | run |
| README evaluation table equals the stored artifacts | test `test_documented_evaluation_table_matches_the_stored_results` (the values are hard-coded in the test) and a direct comparison | run |
| CSV: only applicable columns, blank for unavailable, excluded files without scores, formula prefixes neutralised, parses with `csv` and pandas | tests for each method | run |
| Resume text and file names are displayed as plain text (markdown-rendering defect fixed) | regression test; fails when the fix is disabled | run |
| Caches are keyed by content and configuration; changes recompute; clear button works; data caches are session-scoped | tests (session scope shown by switching the session-id lookup) and Streamlit source reading | run + inspection |
| No files written and no resume text logged by this project's code during processing or a full interface run | tests (open-spy, empty working folder, captured logs) | run |
| No networking library imported by project code; server bound to localhost; telemetry off | AST test; config test | run |
| Full suite: 389 passed, 0 failed, 0 skipped (Windows 11, Python 3.12.10, 207 s) | Phase 10 final run, kept as the verified result | run (not re-run in Phase 11: no code or test changed) |
| Model loads offline from the local cache; a never-downloaded model fails with a clear error | run by hand | run (not in the automated suite) |

## 3. Combined-score behaviour that must be explained honestly
On the 46-judgment evaluation set the skill-coverage term dominates the combined index: standard deviations 0.346 (coverage),
0.089 (semantic), 0.051 (TF-IDF); about 75% of the variance of the index comes from the coverage term (semantic 20%, TF-IDF 5%);
the index correlates r = 0.93 with coverage and 0.59 with semantic (within-job Spearman 0.91 vs 0.62). It follows that the combined ranking
inherits the skill dictionary's weaknesses. This is a measured property of this data with weights fixed in advance; nothing was
changed in response. It corrects a Phase 7 statement made from a smaller inspection set.

## 4. Documentation consistency check performed in Phase 11
- Every file path written in `README.md` was checked against the repository (all exist; the only non-files are two `.gitignore` entries and the model id).
- The three command families in the docs (`streamlit run app.py`, `python -m pytest -q`, `python -m src.evaluate`) match the code; the offline variant and the failure for a never-downloaded model were run.
- Test counts per file add to 389 (app 28, app_logic 23, evaluate 23, pdf_parser 25, preprocessing 20, ranker 64, semantic_matcher 60, skill_extractor 57, tfidf_matcher 50, validation 39).
- Hashes of the evaluation data and dictionary equal the values in `artifacts/evaluation_config.json`. Hash prefixes of the stored artifacts at Phase 11, for future comparison: `evaluation_config.json` 4cfcf4bb, `evaluation_exploratory.csv` da5d0a12, `evaluation_results.csv` 892e7a1f, `evaluation_scores.csv` e0df87c8, `evaluation_tie_bounds.csv` 448fd5c3, `skill_extraction_eval.json` 3ef335ac.
- **Discrepancy found:** the Phase 11 brief (and earlier wording) lists "skill coverage, TF-IDF and semantic similarity" as the three ranking methods and asks for a screenshot of "rankings using skill coverage". The application's three selectable methods are **TF-IDF, semantic and combined**; coverage is a component of the combined index (and an exploratory ablation in the evaluation), not a selectable method.
  **Minimum safe correction applied:** documentation only. README and the screenshot checklist now say so and explain how to show coverage (sort the combined table by the coverage column). No code was changed and no new mode was added.

## 5. Remaining manual checks (not done by me)
1. Upload real PDFs in a real browser (file upload was exercised only through Streamlit's test harness). Procedure: README "Manual check with real PDFs".
2. Upload a few of your own text-based resumes (with consent, locally) and check the extracted-text panel for correct reading order, particularly multi-column or table layouts.
3. Try a genuinely scanned paper resume and confirm it is reported as needing OCR.
4. Optional: real window screenshots for items 10 to 12 of `docs/SCREENSHOT_CHECKLIST.md` (items 1 to 9 were captured in Phase 13; 10 to 12 exist only as pictures rendered from real text).
5. Run the demo script once, including the first-run model load time on the machine you will present on.
6. Run `python -m pytest -q` on the presentation machine if you want to show 389 passed (3 to 4 minutes).
7. Decide whether to exclude `.venv` from OneDrive sync or move the project (slows everything; open item since Phase 2).
8. Phase 12 items not yet done: the presentation outline and a related-work section. References verified so far (checked against the publisher, arXiv or the scikit-learn documentation): Sentence-BERT (EMNLP-IJCNLP 2019), MiniLM (arXiv:2002.10957), scikit-learn TF-IDF documentation. The "about 1B sentence pairs" fine-tuning detail of `all-MiniLM-L6-v2` is from model-card mirrors only.

## 6. Limitations
- No OCR; English only; skills from a dictionary (no unlisted skills, no negation, no role understanding, "A or B" read as both); coverage counts mentions, not use.
- Evaluation: 46 synthetic resumes, 5 synthetic jobs, one author/annotator, subjective labels; says nothing about real hiring effectiveness; differences between methods are not shown to be significant. The combined index was not shown to beat semantic similarity alone.
- Semantic matching: a single chunk can decide the score; a document just under the 254-token limit is embedded whole; scores are not calibrated and not comparable with TF-IDF scores.
- Duplicates are detected by identical bytes only; names and other personal details (other than e-mail, phone, URLs) stay in the text that is matched.
- Tested only on Windows 11 / Python 3.12.10 / CPU.
- Not deployed; localhost only; Streamlit holds uploads in server memory.

## 7. Outstanding risks
| Risk | Likelihood / impact | Mitigation |
|---|---|---|
| First semantic run is slow during a live demo | high / embarrassing | warm the model first (script step 2) |
| Examiner asks why the combined score is no better than semantic alone | high / fine if answered honestly | viva guide Q34, Q42 |
| Examiner treats 389 passed tests as proof of accuracy | medium | viva guide Q58 |
| A real PDF layout reads in the wrong order (multi-column) | medium / affects scores | manual check 2; stated as untested |
| Student quotes a number that is not in the logs | medium | "Things not to say" in the viva guide |
| Examiner asks about the chunk size vs the model's 128-token training length (found in Phase 12 from the model card) | medium | viva guide Q24b: not evaluated, unchanged, stated openly |
| Environment differs on the presentation machine | medium | `requirements-lock.txt`; run setup and a smoke test beforehand |
| Hugging Face unreachable on a new machine at first run | medium | download the model beforehand; offline mode |
| Privacy overclaim | low if the wording in the docs is used | say what was tested and what was not |

## 7b. Presentation-readiness checklist (updated in Phase 13, 2026-10-24)
| Item | Status |
|---|---|
| Related work with verified, linked sources | done: `docs/RELATED_WORK.md` (about 15 sources; most read at abstract / listing level, DPR text read from its PDF; not a systematic review; no novelty or research-gap claim) |
| Presentation outline (12 slides, notes, likely questions) | done: `docs/PRESENTATION_OUTLINE.md` |
| Evaluation figures from the stored artifacts | done: `docs/figures/*.svg`, regenerated by `python docs/figures/make_figures.py`; values come from the CSV artifacts (re-checked against the CSV in Phase 13 by `presentation/tools/verify_deck.py`) |
| Architecture and pipeline diagrams | done: `docs/figures/diagram_architecture.svg`, `diagram_pipeline.svg` (editable source `make_diagrams.py`; PNG renders in `presentation/images/`); show three selectable methods, coverage as a component only |
| TF-IDF vs semantic illustration, combined-score worked example | done on slides 8 and 9 (illustration labelled as such; worked example is a crop of a real screenshot) |
| Application screenshots (12) | items 1 to 9 captured from the running app with synthetic data; items 10 to 12 are rendered from real text (CSV, server console, pytest output, folder listing) and say so; real window screenshots of 10 to 12 remain manual (`docs/SCREENSHOT_CHECKLIST.md`) |
| Slides built | done: `presentation/Resume_Screening_Project.pptx` (12 editable slides, speaker notes on every slide, citations, synthetic-data disclosure on each result slide) |
| Deck accuracy review | done: formula and weights, evaluation table, chart values, variance shares, model facts, citations and the 128-token vs 200-token limitation checked by script and by reading; no combined-beats-semantic, novelty, real-world or production claim on any slide |
| Presenter name, guide, institution | to add by you on slide 1 (intentionally absent) |
| Demo rehearsed on the presentation machine, model pre-warmed | not done |
| 389-passed | fresh run on 2026-10-24: 389 passed in 191.69 s (`docs/screenshots/evidence/pytest_run.txt`); not a screenshot of a terminal |

**Facts found during Phase 13 that earlier documents had wrong or missing.** The 37-page resume is readable and **is ranked**: with the nine
backend resumes plus the six edge-case files, ten files are ranked and five are excluded (not "only nine ranked"); README step 3 of the
manual check and the screenshot checklist were corrected. Streamlit prints benign torchvision file-watcher tracebacks to the console when it
scans optional modules (315 in the captured run); they did not affect any result. The deck tooling (pptxgenjs, Playwright, python-pptx, Pillow)
is outside `requirements.txt` and is only needed to rebuild the deck, the screenshots or the PNG renders.

## 8b. What was not changed in Phase 13
No source code, test, ranking formula, weight, model setting, dataset, label, evaluation protocol or stored evaluation result was changed. Phase 13 added
`presentation/`, `docs/screenshots/`, the two diagram files and their generator in `docs/figures/`, and edited documentation only.

## 8. What was not changed in Phase 11
No source code, test, ranking formula, weight, model setting, dataset, label, evaluation protocol or Phase 8 result was changed. Only documentation was
created or edited (`README.md`, `PROJECT_LOG.md`, `EXPERIMENT_RESULTS.md`, and the four documents in `docs/`).
