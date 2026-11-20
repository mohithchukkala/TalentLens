# Screenshot capture checklist

## Status (updated in Phase 13, 2026-10-24)
Files that exist are in `docs/screenshots/`. Three kinds of evidence are kept apart on purpose:

- **Captured** = an unedited screenshot of the running app (a locally run Streamlit server on localhost, driven by a
  browser automation script, `presentation/tools/capture_screenshots.js`, using only synthetic demo files). Only the page
  content was captured, never the desktop, a file dialog or a user path. The Streamlit "Deploy" toolbar button was hidden
  by a style rule so the pictures show the application only; nothing in the results was altered.
- **Rendered from real text** = a picture drawn by `presentation/tools/render_text_images.js` from text that was really
  produced (the downloaded CSV, the real server console, the real pytest output, the real folder listing). It is NOT a
  screenshot of a terminal or editor window, and each picture says so in its footer.
- **Manual** = still to be captured by you if you want a real window screenshot.

| # | Screenshot | File | Status |
|---|---|---|---|
| 1 | Application home screen | `01_home_screen.png` | captured |
| 2 | Job description and required / preferred skills | `02_job_description_and_skills.png` | captured |
| 3 | Multiple resume uploads | `03_multiple_resume_uploads.png` | captured |
| 4 | File-validation and edge-case status table | `04_file_validation_and_edge_cases.png`, `04b_file_validation_edge_cases_scrolled.png` | captured |
| 5 | Ranking sorted by skill coverage | `05_combined_ranking_sorted_by_coverage.png` | captured |
| 6 | TF-IDF ranking | `06_tfidf_ranking.png` | captured |
| 7 | Semantic ranking | `07_semantic_ranking.png` | captured |
| 8 | Combined ranking, formula and candidate details | `08_candidate_details_and_score_explanation.png` | captured |
| 9 | CSV export and sample contents | `09a_csv_download_button.png` (captured); `09b_csv_contents_rendered.png` (rendered from `09_downloaded_ranking_combined.csv`, the file the app really produced) | captured + rendered |
| 10 | Offline model execution | `10_offline_model_execution_console_rendered.png` from `evidence/offline_server_console.txt` | rendered from real text; real PowerShell screenshot is **manual** |
| 11 | Automated test result (389 passed) | `11_automated_tests_389_passed_rendered.png` from `evidence/pytest_run.txt` | rendered from real text; real terminal screenshot is **manual** |
| 12 | Project folder structure and documentation | `12_project_structure_rendered.png` (listing read from the file system) | rendered from real text; real VS Code Explorer screenshot is **manual** |

Facts about the captured run (so that nobody over-reads the pictures):
- The uploads were the 9 files of `data/sample_resumes/backend_demo/` plus the 6 files of `data/sample_resumes/edge_cases/`
  (15 files). **Ten were ranked and five were excluded** (blank, corrupted, empty, password-protected, image-only). The
  37-page file `long_resume_37_pages.pdf` is readable and **is ranked** (rank 8 in the combined table), so older wording
  in this project that said "only the nine resumes are ranked" was wrong and has been corrected.
- `evidence/pytest_run.txt` is a fresh run on 2026-10-24: **389 passed in 191.69 s** (the earlier Phase 10 run was 207.04 s).
- `evidence/offline_server_console.txt` is the console of a server started with `HF_HUB_OFFLINE=1`. 7,245 stderr lines
  were left out of the saved file (315 benign torchvision file-watcher tracebacks that Streamlit prints when it scans
  optional modules); the file header says so. They did not affect any result.
- Screenshots 01 to 09a show synthetic data only. Nothing personal appears in any of them.
- The screenshots were taken at a viewport of about 1720 px wide, not at the 1400 px suggested below; text is therefore small
  in full-page pictures, which is why the slides use crops of items 4b and 8 (`presentation/images/crop_*.png`).

**Still manual (only if you want real window screenshots):** items 10, 11 and 12 as real PowerShell / VS Code windows,
following the instructions below. The rendered pictures are honest substitutes for slides, not replacements for them.

**One point about item 5.** The app has three selectable methods (TF-IDF, semantic, combined). Skill coverage is a
component of the combined index, not a separate method. For screenshot 5, run *Combined ranking* and click the
"Required-skill coverage" column header so the table is sorted by coverage. Say so in the caption. Do not call it a
fourth ranking method. (Ranking by coverage alone appears only as an exploratory ablation in `EXPERIMENT_RESULTS.md`.)

## General rules (privacy and honesty)
- **Use only synthetic data:** the sidebar's *Synthetic demo* and the files in `data/sample_resumes/`. Never screenshot a real resume, a real person's name, e-mail address, phone number or employer.
- Close other browser tabs and bookmarks bars, hide notifications, and use a private window so personal tabs, profile pictures and e-mail previews are not captured.
- In a terminal, shorten the prompt before capturing so your Windows user name and folders are not shown: `function prompt { "PS> " }`. Avoid screenshots that show full paths such as `C:\Users\<your name>\...`; if a path is unavoidable, crop it or use the project folder name only.
- Capture with **Win + Shift + S** (Snipping Tool), PNG, at 100% browser zoom, window about 1400 px wide. Save as `docs/screenshots/NN_name_manual.png` (so your own captures never overwrite the ones already in the folder).
- Take every screenshot from a real run of the app. Never edit numbers, hide a warning you don't like, or crop out the caveats. If a result looks bad (for example a keyword-list resume ranked high), that is a legitimate screenshot to explain, not to hide.
- Add a one-line caption stating that the data are synthetic.

## Preparation (do once)
1. `streamlit run app.py` (or the offline variant for item 10).
2. Press *Rank candidates* once with the combined method before the demo so the model is already loaded (the first run takes about a minute).
3. **Note:** `data/sample_resumes/backend_demo/` (9 files) and `edge_cases/` (6 files), which the Phase 13 screenshots used, were deleted later at the user's request. Use `data/sample_resumes/realistic_backend_set/` (10 files) instead, or recreate bad files by hand for the edge-case items.

---

### 1. Application home screen
- **Must be visible:** title "Resume Screening and Job Matching Assistant", the blue "Decision support only" banner, the collapsed "About, privacy and limits" panel, the sidebar with the three methods (Combined ranking selected) and the Synthetic demo section.
- **Demonstrates:** purpose, the decision-support framing, the default method, that demo data are synthetic.
- **How:** open http://localhost:8501 on a fresh page; before typing anything. Capture the full window.
- **Privacy:** nothing personal appears; check the browser tab list is hidden.

### 2. Job description input and required / preferred skills
- **Must be visible:** the synthetic backend job description in the text box, and the panels "Required skills detected" (Python, SQL, PostgreSQL, Docker, REST API) and "Preferred skills detected" (Kubernetes, AWS, CI/CD), with the note that skills come from a hand-written dictionary.
- **Demonstrates:** skill extraction and the required/preferred split.
- **How:** sidebar *Demo scenario: Backend Software Developer* -> *Fill in this job description*; scroll so the text box and both skill panels fit.
- **Optional second capture:** replace the text with `We need Python, SQL and Docker.` to show the yellow "split is uncertain" warning, or with a sentence containing no skills to show the "No skills from the dictionary" warning.

### 3. Multiple resume uploads
- **Must be visible:** the file uploader after selecting the nine files from `data/sample_resumes/backend_demo/`, the file names `synthetic_resume_01.pdf` ... `_09.pdf`, and the status table below it showing every file as "Readable".
- **Demonstrates:** batch upload and extraction status.
- **How:** drag the nine files into the uploader. Do not include the edge-case files yet.
- **Privacy:** the file names are neutral and synthetic; make sure the file dialog or Explorer window with your folders is not in the shot.

### 4. File-validation and edge-case status table
- **Must be visible:** the status table with the nine good files plus the six edge-case files: "No usable text" (`blank_page.pdf`), "Probably scanned: OCR needed" (`scanned_image_only.pdf`), "Password-protected", "Unreadable or corrupted", "Empty file (0 bytes)", and the 37-page file as "Readable"; the "Used in ranking" column and the Note column; the orange "N of M file(s) will not be ranked" warning.
- **Demonstrates:** robust handling of bad inputs, honest OCR message, nothing silently dropped.
- **How:** add the files from `edge_cases/` to the same upload. Capture the whole table (zoom out the browser if necessary).
- **Caption:** "The scanned file is a generated image-only PDF, not a real scan. OCR is not implemented."

### 5. Ranking sorted by skill coverage
- **Must be visible:** the results table from *Combined ranking* sorted by the "Required-skill coverage" column (click its header; the sort arrow shows), the "Required skills matched n/m" column, and the caption that scores are not probabilities.
- **Demonstrates:** that skill coverage is a visible component, and (honestly) that keyword-heavy resumes can have full coverage.
- **How:** method = Combined ranking -> *Rank candidates* -> click the coverage header.
- **Caption:** "Skill coverage is a component of the combined index, shown here by sorting the combined table; it is not a separate ranking method."

### 6. TF-IDF ranking
- **Must be visible:** "Method: TF-IDF baseline", the table with only Rank, Candidate and "TF-IDF similarity (cosine)", the sidebar note about the traditional baseline.
- **Demonstrates:** the baseline method and that only applicable columns are shown.
- **How:** choose *TF-IDF baseline* in the sidebar -> *Rank candidates*. Optionally open one candidate to show "Top shared terms".

### 7. Semantic ranking
- **Must be visible:** "Method: Semantic matching", the table with "Semantic similarity (raw cosine)" only, the sidebar line naming the pretrained model.
- **Demonstrates:** the embedding-based method and raw cosine values (not percentages).
- **How:** choose *Semantic matching* -> *Rank candidates* (about a minute the first time). Optionally open a candidate to show the "Best-matching passage".

### 8. Combined-score explanation and candidate details
- **Must be visible:** the line "Combined matching index = 0.50 x max(0, semantic cosine) + 0.20 x TF-IDF cosine + 0.30 x required-skill coverage", the four columns (Combined matching index, Semantic, TF-IDF, Required-skill coverage), and one **expanded** candidate showing matched / missing required skills, preferred skills, evidence snippets and any warnings.
- **Demonstrates:** transparency and explainability of the final score.
- **How:** Combined ranking -> open the first candidate's expander (and, in a second shot, a candidate with missing skills or a warning).
- **Tip:** a second screenshot of the "Files not ranked" table and the stale-results banner (change the job description after ranking) shows additional safeguards.

### 9. CSV export and sample contents
- **Must be visible:** the *Download ranking as CSV* button and, in a spreadsheet or text editor, the downloaded `ranking_combined.csv` with its columns (`method, rank, candidate_id, status, combined_matching_index, semantic_cosine_raw, tfidf_cosine, required_skill_coverage, ...`), including at least one `excluded_...` row with blank scores.
- **Demonstrates:** the export contents, blank (not zero) for unavailable values, excluded files listed with a reason.
- **How:** click the download button; open the file in Excel/Notepad.
- **Privacy:** the CSV contains no resume text; check the spreadsheet window title and folder path are not showing your name. Do not show a CSV from real resumes.

### 10. Offline model execution (where appropriate)
- **Must be visible:** a PowerShell window showing `$env:HF_HUB_OFFLINE = "1"; streamlit run app.py` and the server start line, next to (or followed by) the app's results from a semantic or combined run.
- **Demonstrates:** the model runs from the local cache with no Hugging Face contact (the model had to be downloaded once beforehand).
- **How:** run the command above after at least one earlier successful semantic run. A second optional capture is a Python session printing the library's "Could not reach the Hub ... offline mode is enabled ... Using cached commit hash" message.
- **Caveat:** this shows the model loading offline; it is not a proof that no other process uses the network.

### 11. Automated test results showing 389 passed
- **Must be visible:** the command `python -m pytest -q` and the final summary line `389 passed in ... s` (the Phase 10 result was `389 passed in 207.04s` on the development machine; your timing will differ).
- **Demonstrates:** the automated test suite state.
- **How:** run it in the project folder with the virtual environment active; it takes 3 to 4 minutes. Capture only if the line really reads 389 passed; if the number differs, report the number you actually see.
- **Caption:** "Tests verify behaviour, not ranking accuracy."

### 12. Project folder structure and documentation
- **Must be visible:** the VS Code Explorer (or `tree`-style listing) showing `app.py`, `README.md`, `PROJECT_LOG.md`, `EXPERIMENT_RESULTS.md`, `requirements.txt`, `requirements-lock.txt`, `src/`, `tests/`, `data/evaluation/`, `data/sample_resumes/`, `docs/`, `artifacts/`; optionally the README evaluation table.
- **Demonstrates:** organisation, reproducibility and documentation.
- **How:** collapse `.venv` first (do not show it); crop out the window title and user path.
