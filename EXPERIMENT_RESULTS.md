# EXPERIMENT_RESULTS.md

Only measured values go here. Sections are in the order the work happened; the newest are the Phase 8 results and the Phase 10 validation section. Where an early note was later superseded it says so.

## Environment checks (measured 2026-10-24, Phase 2)
| Check | Result |
|---|---|
| Python in `.venv` | 3.12.10 |
| `pip check` | No broken requirements found |
| `import fitz` (PyMuPDF) | OK, PyMuPDF 2.0 |
| CUDA available | False (CPU only, as intended) |
| Model `all-MiniLM-L6-v2` loads | Yes |
| `max_seq_length` | 256 |
| Embedding dimension | 384 |
| 1000-word input vs its first 254 words | Identical embedding (truncation confirmed) |

## Unit test results (software tests, not evaluation metrics)
| Date | Command | Result |
|---|---|---|
| 2026-10-24 | `python -m pytest tests/test_pdf_parser.py -v` | 24 passed (first run) |
| 2026-10-24 | `python -m pytest -q` (after tightening tests, adding truncated-PDF test, repair warning) | 25 passed in 2.56 s |

| 2026-10-24 | `python -m pytest -q` (Phase 4, first run) | 96 passed, 1 failed (JD one-paragraph demotion bug) |
| 2026-10-24 | `python -m pytest -q` (Phase 4, after fixes) | 99 passed, then 100 passed after adding the same-sentence context test |

| 2026-10-24 | `python -m pytest tests/test_tfidf_matcher.py -q` (Phase 5, first run) | 49 passed, 1 failed (test bug, corrected) |
| 2026-10-24 | `python -m pytest -q` (Phase 5, after `go` stop-word fix + regression test) | **151 passed, 0 failed** in 7.23 s |

| 2026-10-24 | `python -m pytest -q` (Phase 6, after adding `src/inputs.py` refactor, before semantic tests) | 151 passed |
| 2026-10-24 | `python -m pytest tests/test_semantic_matcher.py -q` (first run, real model loaded) | 59 passed, 1 failed (model cache-key defect, fixed) |
| 2026-10-24 | `python -m pytest -q -rs` (Phase 6, final) | **211 passed, 0 failed, 0 skipped** in 29.23 s |

Phase 6 breakdown: `test_pdf_parser.py` 25, `test_preprocessing.py` 20, `test_skill_extractor.py` 56, `test_tfidf_matcher.py` 50, `test_semantic_matcher.py` 60 (50 with fake models, 10 with the real pretrained model). The real model was available in the local cache, so none of the real-model tests were skipped. Model configuration observed: `all-MiniLM-L6-v2`, CPU, `max_seq_length` 256, embedding dimension 384.

### Phase 6 illustrations with the real model (SYNTHETIC texts, not evaluations)
No relevance labels exist and no ranking metric was computed; these show behaviour on a few invented examples only.

1. **Demo (`python -m src.semantic_matcher`)**, SYNTHETIC job description (machine-learning engineer) and 4 SYNTHETIC resumes. Raw cosine similarities, not percentages:

| Rank | Candidate | Cosine |
|---|---|---|
| 1 | synthetic_A (trained transformer language models, PyTorch) | +0.4585 |
| 2 | synthetic_D (data analyst, SQL/Python) | +0.2801 |
| 3 | synthetic_C (frontend developer, React) | +0.1569 |
| 4 | synthetic_B (pastry chef) | +0.1334 |

   Note that the unrelated pastry-chef resume still gets +0.13: embedding cosines of unrelated texts are not near zero, unlike TF-IDF.
2. **Synonyms, no shared words:** job description "Machine learning engineer"; resume "Trained neural networks for image classification and speech recognition" scored 0.3074, resume "Managed restaurant staff schedules and food inventory" scored 0.1380. The TF-IDF baseline gave 0.0 to both (no shared word).
3. **Why chunking matters (one constructed case):** a 492-token resume with 30 lines of baking/hiking filler followed by a skills line at the end. The single-embedding cosine, where the model only reads the first 254 tokens, was 0.0339; the chunked score (3 chunks of 192, 192 and 172 tokens) was 0.3017, and the best-matching chunk was the one with the skills line. This shows what truncation hides in one invented example; it says nothing about how often this occurs in real resumes.

| 2026-10-24 | `python -m pytest tests/test_ranker.py -q` (Phase 7, first run) | 62 passed, 2 failed (fixture errors: PDF text too short; separately generated PDFs are not byte-identical) |
| 2026-10-24 | `python -m pytest -q -rs` (Phase 7, final) | **276 passed, 0 failed, 0 skipped** in 34.25 s |

| 2026-10-24 | `python -m pytest tests/test_evaluate.py -q` (Phase 8, first run) | 22 passed, 1 failed (my hand-calculated expectation was wrong; corrected) |
| 2026-10-24 | `python -m pytest -q -rs` (Phase 8, final) | **299 passed, 0 failed, 0 skipped** in 91.85 s |

| 2026-10-24 | `python -m pytest tests/test_app_logic.py -q` (Phase 9) | 23 passed |
| 2026-10-24 | `python -m pytest tests/test_app.py -q` (Phase 9, first run) | 24 passed, 4 failed (test assumptions, see PROJECT_LOG.md) |
| 2026-10-24 | `python -m pytest tests/test_app.py -q` (Phase 9, after correcting the tests) | 27 passed, 1 failed (radio option labels), then fixed |
| 2026-10-24 | `python -m pytest -q -rs` (Phase 9, final) | **350 passed, 0 failed, 0 skipped** in 143.86 s |

Phase 9 breakdown: `test_app.py` 28 (26 with a fake embedding model injected, 2 with the real pretrained model: all three methods through the interface, and the synthetic backend demo end to end), `test_app_logic.py` 23, `test_evaluate.py` 23, `test_pdf_parser.py` 25, `test_preprocessing.py` 20, `test_ranker.py` 64, `test_semantic_matcher.py` 60, `test_skill_extractor.py` 57, `test_tfidf_matcher.py` 50. These are software tests, not evaluation results; the Phase 8 results above are unchanged. After the final run only a code comment in `app.py` was edited.

Phase 8 breakdown: `test_evaluate.py` 23, `test_pdf_parser.py` 25, `test_preprocessing.py` 20, `test_ranker.py` 64, `test_semantic_matcher.py` 60, `test_skill_extractor.py` 57, `test_tfidf_matcher.py` 50. The metric tests use small hand-calculated examples (they do not load any model); the measured evaluation itself is the separate `python -m src.evaluate` run reported under "Phase 8 results".

Phase 7 breakdown: `test_pdf_parser.py` 25, `test_preprocessing.py` 20, `test_skill_extractor.py` 57, `test_tfidf_matcher.py` 50, `test_semantic_matcher.py` 60, `test_ranker.py` 64 (3 of them use the real pretrained model, which was available, so nothing was skipped).

### Phase 7 observations (SYNTHETIC data; not an evaluation, no labels, no ranking metrics)
Raw score ranges on 10 synthetic resumes against one synthetic job description: semantic cosine +0.008 to +0.763 (mean 0.32, sd 0.21); TF-IDF (pairwise) 0.000 to 0.213 (mean 0.043, sd 0.068). Semantic cosine was negative only for very short unrelated strings (-0.026, -0.023), never for resume-like texts in this set. Batch-mode TF-IDF changed one resume's score from 0.2134 (alone) to 0.2553 (batch of 10), which is why the ranker uses pairwise fitting.

Demo (`python -m src.ranker`, real model, SYNTHETIC job description with Requirements and Nice-to-have sections, weights 0.50 / 0.20 / 0.30, TF-IDF scope pairwise). Required skills found: Python, SQL, PostgreSQL, Docker, REST API, CI/CD; preferred: Kubernetes. The index orders candidates; it is not a percentage or probability.

| Rank | Candidate | Index | Semantic cosine | TF-IDF cosine | Coverage | Matched required | Missing required |
|---|---|---|---|---|---|---|---|
| 1 | synthetic_A | 0.6916 | +0.7059 | 0.1933 | 1.00 | all 6 | none |
| 2 | synthetic_B | 0.3652 | +0.3972 | 0.0831 | 0.50 | Python, SQL, Docker | PostgreSQL, REST API, CI/CD |
| 3 | synthetic_C (frontend) | 0.1665 | +0.3330 | 0.0000 | 0.00 | none | all 6 |
| 4 | synthetic_D (pastry chef) | 0.0773 | +0.1547 | 0.0000 | 0.00 | none | all 6 |
| 5 | synthetic_E (blank) | 0.0000 | +0.0000 | 0.0000 | 0.00 | none | all 6 (flagged: no usable text, ranked last) |

Independence from the batch (real model): adding 3 unrelated candidates to {A, B} changed A's index by 5.6e-8 and B's by 1.7e-8; TF-IDF similarity and coverage were identical, the tiny semantic difference is floating-point noise.

Phase 5 breakdown: `test_pdf_parser.py` 25, `test_preprocessing.py` 20, `test_skill_extractor.py` 56, `test_tfidf_matcher.py` 50.

### Synthetic demo output of the TF-IDF baseline (`python -m src.tfidf_matcher`, batch mode, 99-term vocabulary)
SYNTHETIC job description and 4 SYNTHETIC resumes written for this project. This is an illustration of the interface, **not an evaluation**: no relevance labels exist and no ranking metric was computed.

| Rank | Candidate | Raw cosine |
|---|---|---|
| 1 | synthetic_A (backend, Python/PostgreSQL/Docker/CI-CD) | 0.2248 |
| 2 | synthetic_D (data analyst, Python/SQL) | 0.0546 |
| 3 | synthetic_C (C++/C#/.NET engineer) | 0.0310 |
| 4 | synthetic_B (frontend, React) | 0.0000 |

What this single example illustrates (an observation on 4 invented documents, not a finding): the clearly matching resume gets only 0.22, which shows why the score must not be read as a percentage; and C ranks above B only because both the job description and C contain the generic word "engineer" (only shared term), which shows how keyword overlap can be driven by a generic word.

Final Phase 4 breakdown (from `pytest --collect-only`): `test_pdf_parser.py` 25, `test_preprocessing.py` 19, `test_skill_extractor.py` 56, total 100, all passed in 8.18 s.

### Known limitations of the skill extractor (observed by probing, not measured on a dataset)
Probe text: one line mixing real and ordinary-English uses of ambiguous words. After the fixes the extractor correctly ignored "go to meetings", "R&D", "React quickly", "Excel at teamwork", "Spark of joy" and "JS Bach", and found C, C++, Java, Go, PL/SQL, T-SQL and MS SQL. It still reported `Oracle` for "Used Oracle for payroll" because the word "database" appears in the same sentence. This shows that context rules reduce but do not remove false positives.
Skill-extraction precision and recall were **not** measured at the time of Phase 4. They were measured later, on 17 hand-annotated synthetic resumes: see "Skill extraction" under Phase 8 results (very optimistic; read the caveats there).

Test PDFs are generated in memory with PyMuPDF (synthetic). These tests show the parser behaves as specified on those synthetic cases only. They say nothing about extraction quality on real-world resumes with multi-column layouts, tables, or unusual fonts. That has not been tested.

## Phase 8 protocol (recorded 2026-10-24, before the evaluation dataset existed and before any ranking on it was run)
Full details are in PROJECT_LOG.md ("Phase 8 evaluation protocol"). In short: methods A (pairwise TF-IDF cosine), B (MiniLM semantic cosine, chunk 200 / overlap 40 / top_k 1) and C (index 0.50 / 0.20 / 0.30) are taken from one `CandidateRanker` run per job on 5 synthetic job descriptions with 9 to 10 synthetic resumes each; relevant = label >= 2; metrics Precision@3, Recall@3, MRR, nDCG@3 (gain 2^label - 1); macro average over jobs; undefined metrics are excluded and flagged; ties keep a label-independent hashed order, with best/worst-case bounds reported separately. Settings are frozen; ablations are exploratory and reported apart. Measured results appear in the "Phase 8 results" section below, once they exist.

## Phase 8 results (measured 2026-10-24 with `python -m src.evaluate`, run once with the frozen protocol)
**SYNTHETIC data only: 5 invented job descriptions, 46 invented resumes, one annotator's labels. These numbers are not real-world recruitment accuracy, not hiring probabilities, and with 5 jobs they support no claim of statistical significance.** Raw per-resume component scores are in `artifacts/evaluation_scores.csv`; the exact configuration, package versions and file hashes are in `artifacts/evaluation_config.json`.

Configurations (frozen before the data existed): A = pairwise TF-IDF cosine; B = MiniLM cosine (chunk 200 / overlap 40 / top_k 1), ranked by raw cosine; C = 0.50 x max(0, semantic) + 0.20 x TF-IDF + 0.30 x required-skill coverage. K = 3; relevant = label >= 2; gain 2^label - 1; macro average over the 5 jobs (all 5 jobs defined for every metric).

### Per-job and macro results (Precision@3 / Recall@3 / MRR / nDCG@3)
| Job | A TF-IDF | B Semantic | C Combined |
|---|---|---|---|
| JD-BE backend | 0.333 / 0.200 / 0.500 / 0.330 | 0.333 / 0.200 / 1.000 / 0.503 | 0.667 / 0.400 / 1.000 / 0.746 |
| JD-FE frontend | 0.333 / 0.250 / 0.333 / 0.348 | 0.667 / 0.500 / 1.000 / 0.813 | 0.667 / 0.500 / 1.000 / 0.727 |
| JD-DA data analyst | 0.667 / 0.400 / 0.500 / 0.464 | 0.667 / 0.400 / 1.000 / 0.612 | 0.667 / 0.400 / 1.000 / 0.746 |
| JD-ML machine learning | 0.333 / 0.250 / 0.500 / 0.419 | 0.333 / 0.250 / 1.000 / 0.630 | 0.333 / 0.250 / 0.500 / 0.419 |
| JD-FS full-stack | 0.667 / 0.400 / 1.000 / 0.746 | 1.000 / 0.600 / 1.000 / 0.697 | 0.667 / 0.400 / 1.000 / 0.630 |
| **Macro average** | **0.467 / 0.300 / 0.567 / 0.461** | **0.600 / 0.390 / 1.000 / 0.651** | **0.600 / 0.390 / 0.900 / 0.654** |

Results table in the requested layout (macro values; interpretation is descriptive, not a significance claim):

| Method | Metric | Measured value | Interpretation |
|---|---|---|---|
| A TF-IDF | Precision@3 | 0.467 | Under half of the top 3 were relevant. |
| A TF-IDF | Recall@3 | 0.300 | Capped at 0.6 to 0.75 by pools with 4 to 5 relevant candidates. |
| A TF-IDF | MRR | 0.567 | First relevant candidate on average around rank 2. |
| A TF-IDF | nDCG@3 | 0.461 | Lowest of the three. |
| B Semantic | Precision@3 | 0.600 | Best or tied-best. |
| B Semantic | Recall@3 | 0.390 | Same as C. |
| B Semantic | MRR | 1.000 | A relevant candidate was ranked first in all 5 jobs. |
| B Semantic | nDCG@3 | 0.651 | Practically equal to C (difference 0.003). |
| C Combined | Precision@3 | 0.600 | Same as B. |
| C Combined | Recall@3 | 0.390 | Same as B. |
| C Combined | MRR | 0.900 | Lower than B: in JD-ML a keyword-heavy resume ranked first. |
| C Combined | nDCG@3 | 0.654 | Practically equal to B. |

Tie sensitivity: no job had tied scores at the rank-3 boundary for any method, and the best-case and worst-case tie-breaking bounds are identical to the main results (`artifacts/evaluation_tie_bounds.csv`). The results do not depend on tie order.

### Top 3 per job (label in brackets; label >= 2 is relevant)
| Job | A TF-IDF | B Semantic | C Combined |
|---|---|---|---|
| BE | BE-07 (0), BE-01 (3), BE-06 (1) | BE-01 (3), BE-07 (0), BE-06 (1) | BE-01 (3), BE-06 (1), BE-09 (3) |
| FE | FE-06 (1), FE-07 (0), FE-01 (3) | FE-01 (3), FE-07 (0), FE-02 (3) | FE-01 (3), FE-04 (2), FE-06 (1) |
| DA | DA-06 (1), DA-01 (3), DA-04 (2) | DA-01 (3), DA-06 (1), DA-03 (2) | DA-01 (3), DA-06 (1), DA-09 (3) |
| ML | ML-06 (1), ML-01 (3), ML-08 (0) | ML-01 (3), ML-06 (1), ML-05 (1) | ML-06 (1), ML-01 (3), ML-08 (0) |
| FS | FS-01 (3), FS-06 (1), FS-09 (3) | FS-01 (3), FS-03 (2), FS-04 (2) | FS-01 (3), FS-04 (2), FS-06 (1) |

### What the results show (observations on this small synthetic set)
1. **Combined vs TF-IDF:** C is higher than A on all four macro metrics (Precision@3 0.600 vs 0.467, MRR 0.900 vs 0.567, nDCG@3 0.654 vs 0.461) and equal or better in 4 of 5 jobs on nDCG (JD-ML is identical to A; in JD-FS A has the higher nDCG, 0.746 vs 0.630). With 5 jobs this is a pattern, not evidence of significance.
2. **Combined vs semantic alone: not clearly better.** Macro Precision@3 and Recall@3 are equal (0.600, 0.390); nDCG@3 differs by 0.003 (0.654 vs 0.651), which is noise at this size; MRR is lower for C (0.900 vs 1.000). Per job on nDCG: C is better in BE (+0.243) and DA (+0.134), worse in FE (-0.086), ML (-0.210) and FS (-0.067). So the combined index did not improve ranking quality over the semantic score alone here. The two macro nDCG values are close only because two large gains offset three losses.
3. **Keyword-heavy candidates expose TF-IDF.** The keyword-list resumes (label 1: FE-06, DA-06, ML-06, BE-06, FS-06) have the highest or second-highest TF-IDF score in several jobs: TF-IDF ranked FE-06, DA-06 and ML-06 first, above the genuinely strong candidates. The negation resumes also score high because they repeat the job's words (BE-07 had the highest TF-IDF score in its pool, FE-07 the second highest). TF-IDF cannot tell "has Python" from "has not worked with Python".
4. **Semantically related but irrelevant candidates expose the semantic matcher.** Negation resumes rank second in the semantic ranking for BE (BE-07, label 0, ahead of the label-3 BE-09 and label-2 BE-03/BE-04) and for FE (FE-07, label 0, second); a keyword-list resume is in the semantic top 3 in BE, DA and ML. The recruiter, designer and product-manager resumes ("describes the role from outside") scored 0.53 to 0.69 in cosine, in the same range as relevant partial matches. The semantic score reflects topic, not whether the person does the work.
5. **Chunking:** BE-09 (295 tokens, 2 chunks) and DA-09 (282 tokens, 2 chunks), both label 3 with the relevant part at the end, scored 0.643 and 0.609 semantically, in the range of other relevant candidates. FS-09 (label 3) is slightly under the 254-token limit, so it is NOT chunked: its single embedding is diluted by the teaching history and got the lowest semantic score of its pool (0.350; the next lowest in that pool is 0.600). This shows a boundary effect of the "keep short documents whole" rule: a document just under the limit is embedded whole, so a relevant tail can be diluted. (Observation from one resume; not changed.)
6. **Skill coverage helps in some places and hurts in others.** Helps: it pushes down the adjacent-role and unrelated resumes (coverage 0.00 to 0.25 for BE-05, DA-08, DA-10) and lifted BE-09 and DA-09 into the top 3. Hurts: (a) it cannot see paraphrased evidence, so the label-2 "paraphrase" resumes (BE-03, FE-03, DA-03, ML-03, FS-03) get coverage 0.00 to 0.20 and drop to the bottom half of the combined ranking (for example FE-03 index 0.313 vs the keyword list FE-06 at 0.676); (b) it counts mentions, not use, so keyword lists, negation and "role from outside" resumes get high coverage (BE-08 recruiter 1.00, FS-07 "never used any of these" 1.00, FS-06 1.00); (c) the "PyTorch or TensorFlow" requirement is read as both, so even the strongest ML resumes get at most 0.67; (d) React was not detected in the JD-FS job description (see below), so its coverage is computed from 4 skills only.
7. **Job types:** C did well on BE (nDCG 0.746) and DA (0.746) and worst on ML (0.419, top 3 identical to TF-IDF: the keyword-list resume ML-06 ranked first, coverage 1.00). B did best on FE (0.813) and FS. No job type is solved; with one job per type these per-type statements are anecdotes.
8. **Exploratory ablations (not used to choose anything; looked at after the main result, so they are not unbiased):** computed from the stored raw components, macro P@3 / R@3 / MRR / nDCG@3: skills alone 0.600 / 0.390 / 0.800 / 0.623; semantic + TF-IDF without skills (weights renormalised) 0.667 / 0.440 / 1.000 / 0.635; semantic + skills without TF-IDF 0.667 / 0.430 / 0.900 / 0.694; equal weights 0.600 / 0.390 / 0.900 / 0.646. Reading, with caution: removing skill coverage or TF-IDF produced comparable or slightly higher precision, so the skill and TF-IDF terms of the initial formula are not clearly earning their weights on this data; and skill coverage alone ranks about as well as the semantic score on this set, which is plausibly because I wrote both the dictionary and the resumes and listed skills in dictionary vocabulary. None of these variants should be adopted on the basis of this table, and the initial weights remain 0.50 / 0.20 / 0.30.

### Skill extraction (17 hand-annotated resumes, mention-level, annotated before the extractor was run)
Micro-averaged over 17 resumes (89 annotated skill mentions): TP 88, FP 0, FN 1 -> **precision 1.000, recall 0.989, F1 0.994**. The only miss was BE-05: "Oracle databases" was not matched because the context rule for the ambiguous word "Oracle" looks for the exact word "database" and the text has the plural. These numbers are very optimistic and must not be read as real-world extractor accuracy: the task is mention-level detection of dictionary vocabulary on short, clean, invented text; the annotator also wrote the dictionary; negated mentions and mentions by recruiters count as correct detections, so the score says nothing about whether a skill is actually held; and skills outside the dictionary (Figma, OpenAPI, MERN...) were not annotated, so recall against the real vocabulary of resumes is unmeasured. Details: `artifacts/skill_extraction_eval.json`.

### Errors, unexpected results and limitations
- **Dictionary rule miss on a job description:** React is not detected in "React for the user interface" (JD-FS), because the ambiguous-word rule requires a context word in the same sentence. This was seen while checking the job descriptions before any ranking, and left unchanged because the dictionary was frozen. It lowers coverage accuracy for JD-FS (4 required skills instead of 5).
- **"A or B" requirements** cannot be represented (JD-ML requires PyTorch and TensorFlow both in coverage).
- **Unexpected:** the semantic score ranked a relevant candidate first in every job (MRR 1.000), better than I expected given the adversarial resumes; the combined index lost that because skill coverage rewards keyword lists. FS-09's low semantic score (above) was also unexpected.
- **Who wrote what:** the system, the resumes and the labels were all written by the project author with an AI assistant, who knew the system's weaknesses; adversarial profiles (negation, keyword lists, recruiters) were deliberately included. The dataset is therefore probably harder for keyword methods and the dictionary than typical data, and easy for the skill dictionary in one respect (resumes use dictionary words). Direction of the net bias is unknown.
- **Size and subjectivity:** 5 jobs, 46 judgments, one annotator, no agreement measured, graded labels are subjective (several 2-versus-3 and 1-versus-2 calls are debatable, see `labels.csv`). With K = 3 and 4 to 5 relevant candidates per job, Recall@3 cannot exceed 0.6 to 0.75.
- **Only 2 of the 46 resumes exercise chunking** (BE-09, DA-09).
- No result here says anything about real recruitment, hiring probability, or fairness.

### Reproduction
```
python -m src.evaluate            # reruns everything, rewrites artifacts/evaluation_*.csv|json and skill_extraction_eval.json
python -m pytest tests/test_evaluate.py -v
```
Semantic scores can differ in the 7th decimal between runs because of batching; no ranking changed between repeated runs in tests.

## Pre-registered evaluation rules (written before any evaluation was run)
- Method C uses fixed weights 0.50 / 0.20 / 0.30, run once on the evaluation set.
- Chunk size (200 tokens), overlap (about 40 tokens) and aggregation (JD-chunk-weighted mean of best resume-chunk match) are fixed.
- Weight sensitivity and the mean-pooling ablation, if run, go in a separate table labelled exploratory. They are not used to choose settings for the main result.
- Per-component mean and standard deviation of scores are reported as a diagnostic.

Status of these rules (checked in Phase 10): the weights, chunking and aggregation were not changed. The per-component mean / standard deviation diagnostic was **not** reported in Phase 8 (an omission); it is reported in the Phase 10 section below. The mean-pooling ablation was **not run**. The weight-sensitivity analysis was reported only through the four exploratory ablations in Phase 8. The placeholder results table that used to be here was superseded by "Phase 8 results" above and has been removed.

## Phase 10 validation (2026-10-24)
Nothing in this section changes any Phase 8 number.

### Evaluation reproducibility (measured)
`python -m src.evaluate --out <scratch folder>` was re-run after all Phase 9 changes. `evaluation_results.csv`, `evaluation_tie_bounds.csv` and `evaluation_exploratory.csv` are **byte-identical** to the stored artifacts; the maximum absolute difference over all 46 resumes x 4 scores in `evaluation_scores.csv` is 0.0; the SHA-256 hashes of the dataset, labels, rubric, annotations and skill dictionary equal the ones recorded at evaluation time. An automated test (`test_real_phase8_results_are_reproducible_in_process`) repeats the metric comparison on every run, another (`test_scoring_configuration_and_evaluation_protocol_are_unchanged`) guards the weights, chunk settings and data hashes, and another checks that the evaluation table quoted in the README (hard-coded in the test) still equals the stored results.

### Component score diagnostic (pre-registered in Phase 2, reported late)
Computed from `artifacts/evaluation_scores.csv` over all 46 judgments (no re-evaluation):

| Component | Mean | Std. dev. | Min | Max | Weight x std. dev. |
|---|---:|---:|---:|---:|---:|
| TF-IDF cosine | 0.089 | 0.051 | 0.017 | 0.223 | 0.20 x 0.051 = 0.010 |
| Semantic cosine (raw) | 0.652 | 0.089 | 0.350 | 0.872 | 0.50 x 0.089 = 0.045 |
| Required-skill coverage | 0.641 | 0.346 | 0.000 | 1.000 | 0.30 x 0.346 = 0.104 |
| Combined index | 0.536 | 0.130 | 0.239 | 0.778 | - |

**Correction to an earlier statement.** In Phase 7, on a 10-resume inspection set, I wrote that semantic scores spread about three times wider than TF-IDF scores and that TF-IDF's nominal 20% weight acted like a few percent. On the 46-resume evaluation set the picture is different: the largest spread is in **skill coverage** (std. dev. 0.346), then semantic (0.089), then TF-IDF (0.051). Taking weight x spread as a rough measure of influence on the ordering, skill coverage has about 2 times the influence of the semantic term and about 10 times that of TF-IDF on this data, so on this data the combined index is dominated by the skill dictionary. That is consistent with the Phase 8 findings (keyword-list and negation resumes with full coverage rising in the combined ranking). It is a descriptive statistic on one synthetic set, not a reason to change any weight.

### Why skill coverage dominates: additional measurements (Phase 11, from the stored artifacts, no re-evaluation)
Computed from `artifacts/evaluation_scores.csv` (46 judgments); the combined index equals 0.5 x max(0, semantic) + 0.2 x TF-IDF + 0.3 x coverage for every row (maximum difference 0.0).
| Measure | Skill coverage | Semantic | TF-IDF |
|---|---:|---:|---:|
| Pearson r with the combined index (all 46) | 0.932 | 0.589 | 0.667 |
| Mean within-job Spearman with the combined index (5 jobs) | 0.906 | 0.618 | 0.582 |
| Share of the variance of the combined index contributed by the term (cov(term, index) / var(index)) | 0.746 | 0.201 | 0.053 |

Interpretation (descriptive, one synthetic set): the term with the widest spread controls the ordering because the terms are added without rescaling. Not a reason to change a weight.

### Defects found in Phase 10 and what was done
1. **Resume text and file names were rendered as markdown (fixed).** File names, evidence snippets and some warnings were passed to `st.markdown`, expander labels and `st.warning`, so a resume or file name containing `[text](link)`, `**bold**`, `:emoji:` or `$math$` would be displayed as a link or formatting chosen by the resume's author. Reproduced with a file named `[evil](http://evil.example).pdf` (it showed as a clickable link in the expander label). Fixed with `escape_markdown` in `src/app_logic.py`, applied in `app.py`; a regression test shows the same input is displayed escaped, and failed when the fix was disabled in a check run.
2. **Unused dependency (fixed).** `plotly` was listed in `requirements.txt` but never imported; removed.
3. **No pinned versions (addressed).** Added `requirements-lock.txt` (exact tested versions) next to the unpinned `requirements.txt`.
4. **Stale documentation (fixed).** `EXPERIMENT_RESULTS.md` still said "No evaluation has been run yet" and contained an all-PENDING results table; `PROJECT_LOG.md` still listed Phase 2 placeholders as PENDING and an outdated folder structure; the empty `data/sample_*` folders had no purpose. Updated; the sample folders now hold synthetic demo files and edge-case PDFs.
5. **No README (created).**
6. **Omitted pre-registered diagnostic (reported above).**
Not defects, but limits found while probing and documented: non-Latin text gives TF-IDF no words; model loading contacts Hugging Face unless `HF_HUB_OFFLINE=1`; Streamlit's AppTest uses one fixed session id, so cross-session cache isolation can only be shown by switching the session id lookup (done in a test) and by reading Streamlit's code.

### End-to-end scenarios actually run
Each was run with real generated PDFs through the application logic with the REAL model (probe script, outputs inspected) and the same conditions are covered by automated tests (fake model, plus real-model tests where noted).

| Scenario | Probe with real model | Automated test |
|---|---|---|
| One valid resume | ranked (all 3 methods) | yes |
| Nine valid resumes | ranked (all 3 methods) | yes, incl. comparison with direct calls to the modules (real model) |
| Different names, identical bytes | second file listed as duplicate, not ranked | yes |
| Same name, changed content | both kept as `cv.pdf` and `cv (2).pdf` | yes (logic and cache tests) |
| Blank / textless PDF | excluded, "No usable text" | yes |
| Corrupted PDF, 0-byte file | excluded, reported | yes |
| Password-protected PDF (fixture built with PyMuPDF AES-256) | excluded, "Password-protected" | yes |
| Scanned / image-only PDF (a generated image-only page, not a real scan) | excluded, "OCR needed" | yes |
| Very long resume (37 pages, about 10,800 words) | ranked in about 20 s, no truncation warning | yes (fake model, and a shorter real-model case) |
| Mixture of valid and every kind of invalid file | valid ones ranked | yes |
| No resumes / missing job description | helpful error | yes (interface) |
| Job description with no detectable skills | warning, coverage blank | yes |
| Stop-word-only job description with TF-IDF | validation message; combined and semantic still work | yes |
| Required + preferred skills; unstructured (uncertain split) | correct sets / warning | yes |
| Long job description (multi-chunk) | ranked | yes (real model) |
| Non-English resumes | ranked, TF-IDF sees no words for Hindi | yes (fake model) |
| File names with unicode, commas, quotes, line breaks, `= + - @` | CSV parses; formulas neutralised | yes |

### Not tested (and why)
- **A real browser upload of PDFs** (status at Phase 10). The built-in browser tool cannot upload files, so uploading was then tested only through Streamlit's AppTest file uploader. **Update (Phase 13):** a real browser upload has since been exercised, by a Playwright script driving the installed Edge against the running app with the 15 synthetic files; see "Phase 13" below. A manual procedure is still in README.md.
- **A genuinely scanned paper resume**, and any real-world resume at all: the repository contains only synthetic PDFs. Real layouts (multi-column, tables, graphics) are untested.
- **Multiple simultaneous real users / real session isolation** (see above).
- **Absence of all disk writes by third-party libraries** (Streamlit, PyTorch, Hugging Face may write their own caches or temp files). The tests show this project's own code and the interface run create no files in the working folder and log no resume text; they do not prove anything about library internals.
- **Memory-only handling of the uploaded bytes by Streamlit itself** (it holds uploads in server memory by design).
- **Other operating systems and Python versions**: everything ran on Windows 11, Python 3.12.10, CPU only.

### Phase 11 verification (documentation phase; no code or test changed)
| Check | Result |
|---|---|
| Hashes of dataset, labels, rubric, annotations and skill dictionary vs `artifacts/evaluation_config.json` | all five equal |
| Paths written in README.md vs repository | all exist (non-files: two `.gitignore` entries and the model id) |
| Test counts per file (collect-only) | 28 + 23 + 23 + 25 + 20 + 64 + 60 + 57 + 50 + 39 = 389 |
| `HF_HUB_OFFLINE=1`, model cached | loads (about 76 s); the library prints that it could not reach the Hub and uses the cached copy |
| `HF_HUB_OFFLINE=1`, model never downloaded | raises `SemanticModelError` with a readable message |
| Sizes measured for the README prerequisites | virtual environment 1.48 GB; model cache 87 MB |
| Tests re-run in Phase 11? | No: nothing in `src/`, `app.py` or `tests/` changed after the Phase 10 run (389 passed, 0 failed, 0 skipped) |
References checked via web search in Phase 11: Reimers and Gurevych, Sentence-BERT (ACL Anthology D19-1410); Wang et al., MiniLM (arXiv 2002.10957); scikit-learn `TfidfTransformer` documentation (smoothed IDF and `sublinear_tf` formulas). The "about 1 billion sentence pairs" fine-tuning detail of `all-MiniLM-L6-v2` was found only on model-card mirrors, not on the official card.

### Test results for Phase 10
| Date | Command | Result |
|---|---|---|
| 2026-10-24 | `python -m pytest -q -rs` before any Phase 10 change | **350 passed, 0 failed, 0 skipped** in 175.9 s |
| 2026-10-24 | `python -m pytest tests/test_validation.py -q` (first run) | 34 passed, 4 failed, 1 error: all were mistakes in my new tests (a plugin flag disabled `caplog`; ranking-order assumptions about the crude fake embedding model; AppTest cannot show cross-session isolation; an impossible file name containing `/`), none was an application defect |
| 2026-10-24 | `python -m pytest tests/test_validation.py -q` (corrected) | 39 passed, 0 failed, 0 skipped |
| 2026-10-24 | regression check: same markdown test with `escape_markdown` disabled | failed as expected (the test detects the defect) |
| 2026-10-24 | `python -m pytest -q -rs -ra` (final, after all Phase 10 changes) | **389 passed, 0 failed, 0 skipped** in 207.0 s (350 earlier + 39 in `tests/test_validation.py`; real-model tests included and run) |

Environment: Windows 11, Python 3.12.10, CPU only, project inside OneDrive (slow file access; suite runtime 2 to 3.5 minutes). The only edits after the last full run were none: the README, requirements and documentation changes were made before it.

## Phase 13 presentation evidence (2026-10-24; no code, test, weight, dataset or evaluation result changed)

### Fresh full test run
`python -m pytest -q` on 2026-10-24 (started 11:54): **389 passed, 0 failed, 0 skipped in 191.69 s** (Phase 10: 207.04 s). Output saved in `docs/screenshots/evidence/pytest_run.txt`. As before, this shows behaviour on the cases tested, not ranking accuracy.

### Real browser run used for the screenshots
A Playwright script (`presentation/tools/capture_screenshots.js`) drove the installed Edge against the locally run app (localhost, `HF_HUB_OFFLINE=1`), pasted the synthetic Backend Software Developer job description, uploaded the 9 files of `data/sample_resumes/backend_demo/` and then the 6 files of `edge_cases/`, ranked with Combined, TF-IDF and Semantic, expanded candidate details, sorted by coverage and downloaded the CSV.

| Observation | Result |
|---|---|
| Files uploaded | 15 (9 backend + 6 edge cases) |
| Ranked / excluded | **10 ranked, 5 excluded** (blank, corrupted, empty, password-protected, image-only) |
| 37-page resume | readable and ranked (combined rank 8, coverage 5/5, semantic 0.345, TF-IDF 0.021) |
| TF-IDF ranking order | 09, 05, 06, 07, 04/08 (tied 0.065), 03, 01, 02, then the 37-page file (0.021) |
| Console noise | 315 benign torchvision file-watcher tracebacks (7,245 stderr lines left out of the saved evidence file; stated in its header); no effect on results |
| CSV | `docs/screenshots/09_downloaded_ranking_combined.csv`, the file the app produced; contains no resume text |

These rankings are of one synthetic demo set and are illustrations of the interface, not evaluation results.

### Deck verification (`presentation/tools/verify_deck.py`, read-only)
12 slides; speaker notes on every slide (each over 200 characters); slide-10 table equals the MACRO rows of `artifacts/evaluation_results.csv` at 3 decimals; default weights in `src/ranker.py` are 0.5/0.2/0.3 and equal the deck and diagram; model facts (384, 256, 128-token, 200 tokens, Apache-2.0) present and the 128-token vs 200-token effect stated as not tested; diagram says "dictionary of 114 skills" and the dictionary has 114; the 12 macro bars and 15 per-job bars in the SVG charts equal the CSV; variance shares 0.746/0.201/0.053 recomputed from `evaluation_scores.csv`; every cited author appears in `docs/RELATED_WORK.md`; an overclaim-phrase scan found the words only in negated or caveat contexts. All checks passed. Slides were also rendered through PowerPoint and inspected by eye.
