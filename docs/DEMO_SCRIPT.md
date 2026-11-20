# Demonstration script (about 10 minutes)

Audience: examiners at an internal evaluation. Use **only synthetic data**. Say that the data are synthetic at the start
and again when you show results. Nothing below has been timed with a live audience; the timings are estimates.

## Before you start (5 minutes, not part of the 10)
1. Open PowerShell in the project folder, activate the environment (`.\.venv\Scripts\Activate.ps1`), run `streamlit run app.py`.
2. In the sidebar pick *Backend Software Developer*, press *Fill in this job description*, upload the resume PDFs (the sidebar's "Download this scenario's synthetic resumes (ZIP of PDFs)" button gives the nine of this scenario; the folder
   `data/sample_resumes/realistic_backend_set/` holds ten fuller resumes with their own `job_description.txt`; the earlier `backend_demo/` and `edge_cases/` folders were deleted at the user's request), press *Rank candidates* once (Combined) so the model is loaded. Then reload the page.
   The first semantic run takes about a minute on the development machine; do not let that happen live.
3. (The edge-case PDFs were deleted; the app's file-validation behaviour is still covered by `tests/test_validation.py`.) Have a second browser tab with `README.md` rendered (or VS Code), and a terminal ready for `python -m pytest -q` (3 to 4 minutes: run it beforehand and show the saved result, or start it at the beginning of the demo).
4. Close notifications and personal tabs. Zoom the browser so the table is readable.

## Script

| Time | Do | Say |
|---|---|---|
| 0:00 | Home screen. | "This is a decision-support tool that ranks PDF resumes against a job description. It uses TF-IDF, a pretrained sentence-embedding model and a skill dictionary. Everything I show today is synthetic: invented resumes and invented job descriptions. Nobody is rejected by the tool; a person reviews every candidate." Point to the blue banner. |
| 1:00 | Click *Backend Software Developer* -> *Fill in this job description*. | "The job description has Requirements and Nice-to-have sections. The app detects the required skills Python, SQL, PostgreSQL, Docker and REST API and the preferred ones Kubernetes, AWS and CI/CD. This is a hand-written dictionary, not a trained model, so it only finds skills it knows." |
| 2:00 | Upload the nine backend resumes. | "Each file is read with PyMuPDF and gets a status. They are all readable. The file names are neutral on purpose." |
| 3:00 | Add the six edge-case files. | "Now bad inputs: a blank page, an image-only file, a password-protected file, a corrupted file, an empty file, and a 37-page resume. Each is reported with a reason; one bad file never stops the rest. The image-only one says OCR is needed. I did not implement OCR." |
| 4:00 | Method *TF-IDF baseline* -> *Rank candidates*. | "The traditional baseline: word-overlap cosine similarity. It is blind to meaning." Open a candidate: "These are the shared terms that produced the score." |
| 5:00 | Method *Semantic matching* -> *Rank candidates*. | "A pretrained model, all-MiniLM-L6-v2, used as published; I did not train it. It turns text into 384 numbers and compares them with cosine similarity. Long resumes are split into overlapping chunks because the model reads only 256 tokens. These values are raw cosines, not percentages." Open a candidate: "This is the passage that matched best." |
| 6:00 | Method *Combined ranking* -> *Rank candidates*. | "The combined index adds three parts: 0.50 times semantic, 0.20 times TF-IDF, 0.30 times required-skill coverage. These weights were chosen in advance and never tuned. It orders candidates; it is not a probability." Click the *Required-skill coverage* header to sort. |
| 7:00 | Open the top candidate, then a candidate with missing skills or a warning. | "Every number is explainable: matched and missing required skills, the evidence sentence, the warnings. The details come from the same results as the table." |
| 8:00 | Click *Download ranking as CSV*; open it. | "One row per candidate, best first, then excluded files with their reasons and no scores. Unavailable values are blank, never zero. File names that start with an equals sign or similar are neutralised against spreadsheet formula injection." |
| 8:30 | Show the README evaluation table. | "I evaluated on 5 synthetic jobs and 46 synthetic resumes with labels written from a rubric before running anything. The combined index beat TF-IDF but did not clearly beat semantic similarity alone: equal Precision@3 and Recall@3, a 0.003 nDCG difference, and a lower MRR. With five jobs I do not claim any difference is significant, and this does not show real-world hiring effectiveness." |
| 9:15 | Show the test summary (or `docs/VIVA_GUIDE.md`, section 9). | "389 automated tests pass. That shows the code behaves as specified; it does not show the rankings are accurate on real resumes. Known limitations: no OCR, English only, dictionary-based skills with no negation handling, synthetic evaluation, no real-world testing." |
| 10:00 | Stop. | "Questions?" |

## If something goes wrong
- **The first semantic or combined run is slow because the model is not loaded yet:** say so and wait (about a minute here). If time is short, show the TF-IDF method first (it does not need the model) and come back to the others.
- **A red "language model could not be loaded" message:** the machine is offline and the model was never downloaded; explain `HF_HUB_OFFLINE` and use screenshots instead.
- **Port in use:** `streamlit run app.py --server.port 8502`.
- **A result looks wrong (for example a keyword-list resume ranks high):** that is expected and documented; say "this is the dictionary limitation described in the evaluation" instead of hiding it.

## What to be ready to show or say if asked
- The formula and where it is in the code: `src/ranker.py`, function `combine_scores`.
- Why skill coverage dominates the combined score on this data (viva guide, Q34).
- The markdown-rendering vulnerability and its fix (viva guide, Q55).
- Offline mode: `$env:HF_HUB_OFFLINE = "1"; streamlit run app.py` (only after the first download).
- That the evaluation can be re-run: `python -m src.evaluate`.

## Do not say
"It is accurate", "it is ready for production", "the combined method is best", "the score is a probability", "it reads scanned resumes", "it is fully private" (say what was tested instead).
