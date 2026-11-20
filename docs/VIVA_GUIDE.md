# Viva guide

**Project:** AI-Powered Resume Screening and Job Matching System Using Natural Language Processing
**How to use this guide:** read each section, then try to answer the questions out loud before reading the answer.
The *follow-up* questions test whether you understand the implementation, not only the definition. Every number
below was measured on this project and is in `EXPERIMENT_RESULTS.md`; do not quote any number you cannot find there.

**Three sentences to be able to say without notes**
1. "It ranks PDF resumes against a job description using TF-IDF, a pretrained sentence-embedding model, and a combined index that adds required-skill coverage from a dictionary."
2. "On a small synthetic evaluation (5 invented jobs, 46 invented resumes) the combined index beat TF-IDF but did not clearly beat semantic similarity alone, so it does not establish real-world hiring effectiveness."
3. "It is decision support: a person reviews every candidate, nobody is rejected automatically, and the scores are not probabilities."

---

## 1. Problem statement and objectives

**Q1. What problem does the project solve?**
A recruiter receives many resumes for one job. Reading them all is slow and inconsistent. The system extracts the text of
each PDF, finds skills, scores every resume against the job description in three ways, and shows a ranked list with the
evidence, so the recruiter knows where to look first and why.

**Q2. What were the objectives?**
(1) Extract text from PDFs safely. (2) Extract skills transparently. (3) Score with a traditional method (TF-IDF) and a
pretrained semantic model, and combine them with skill coverage. (4) Compare the three methods on labelled data with
proper ranking metrics. (5) Provide a Streamlit interface with CSV export. (6) Document honestly what was and was not shown.

**Q3. Is this a classification problem?**
No. It is a ranking (retrieval) problem: the output is an ordering of candidates for one job. That is why the metrics are
Precision@K, Recall@K, MRR and nDCG, and why I do not report accuracy.

**Q4. Why is resume screening an NLP application?**
Resumes and job descriptions are unstructured natural language. Matching them needs text extraction, normalisation,
tokenisation, vocabulary statistics (TF-IDF), semantic representation (embeddings) and phrase matching (skills).

**Q5. Which parts are traditional NLP and which use deep learning?**
Traditional: preprocessing, TF-IDF, cosine similarity, dictionary skill matching, the weighted formula. Deep learning:
only the pretrained sentence-embedding model (a transformer), used as published. I did not train or fine-tune anything.
*Follow-up:* "So did you train a model?" No. TF-IDF "fitting" counts words and document frequencies; there are no
gradients, epochs or labels. The embedding model was trained by others.

## 2. Motivation and applications

**Q6. Who would use it?** HR teams or recruiters shortlisting for one vacancy; college placement cells; a student
project showing how text-matching methods behave. It is not validated for production hiring.
**Q7. Why not just search for keywords?** Keyword search misses synonyms and rewards keyword stuffing. That is exactly
what the evaluation showed for TF-IDF (keyword-list resumes ranked above genuinely strong candidates in three of five jobs).
**Q8. Why not use an LLM?** Out of scope by design: cost, privacy (resume text would leave the machine), reproducibility
and explainability. The project keeps every score traceable to a formula or a visible match. Trade-off: an LLM might
understand negation and context better; it would need separate validation and a privacy decision.

## 3. Architecture and workflow

**Q9. Walk me through the pipeline.**
PDF bytes -> `pdf_parser` (PyMuPDF; status per file) -> `preprocessing` (three views) -> `skill_extractor` (dictionary,
required/preferred from the job description) -> `tfidf_matcher` and `semantic_matcher` -> `ranker` (combined index,
coverage, ranking) -> `app_logic` (table, details, CSV) -> `app.py` (Streamlit).
**Q10. Why is `app_logic.py` separate from `app.py`?** So the behaviour is testable without a browser, and so the
interface contains no matching algorithm. A test checks `app.py` and `app_logic.py` import no sklearn, numpy or model libraries.
**Q11. What does the parser do when a PDF is bad?** It returns a status instead of raising: ok, ok_with_warnings,
scanned_likely, no_text, encrypted, corrupted, empty_file, too_large, extraction_failed. Only usable files are ranked; the
rest are listed with the reason. One bad file never stops the others.
**Q12. How do you decide a PDF is scanned?** Heuristic: fewer than 100 non-whitespace characters and a page that has an
image and under 20 characters. It is a guess, not proof, and OCR is not implemented.
**Q13. How are duplicates detected?** SHA-256 of the file bytes. Identical bytes under different names: the later file is
marked duplicate and not ranked twice. The same resume re-exported from Word has different bytes and would *not* be caught.
**Q14. Same file name uploaded twice?** Unique ids: `cv.pdf`, `cv (2).pdf`; both are kept.

## 4. Technology stack

| Technology | Why | Trade-off |
|---|---|---|
| Python 3.12 | stable, supported by every package used | other versions untested |
| PyMuPDF | fast text extraction with block/reading order, handles encrypted/corrupt files by exception | no OCR; multi-column layouts can come out in the wrong order |
| scikit-learn | `TfidfVectorizer`, `cosine_similarity` | none relevant here |
| sentence-transformers + `all-MiniLM-L6-v2` | small (87 MB), CPU-friendly, standard sentence-similarity model | English-oriented, 256-token limit, not trained on recruitment data |
| Streamlit | quick local interface, built-in testing (`AppTest`) | holds uploads in server memory; not a production framework |
| pandas / NumPy | tables and arrays | none |
| pytest | 389 tests | tests prove behaviour, not accuracy |

**Q15. Why `all-MiniLM-L6-v2`?** Small enough for a laptop CPU, widely used for sentence similarity, 384-dimensional
embeddings. I measured its limit: `max_seq_length` = 256 tokens (254 usable). Limitations: English-centred, general-purpose,
reflects topic not skill level, ignores negation.
**Q16. Where does the model come from, and did you train it?** Published through the sentence-transformers project; used as
published. Background reading I verified: Reimers and Gurevych, *Sentence-BERT*, EMNLP-IJCNLP 2019 (DOI 10.18653/v1/D19-1410),
and the MiniLM paper (arXiv:2002.10957). The fine-tuning details (about 1B sentence pairs) I only found on model-card mirrors,
so I say "reported" rather than "verified".

## 5. The ranking methods and their mathematics

### 5.1 TF-IDF with cosine similarity
- **Term frequency** (sublinear): `tf = 1 + ln(count)` (0 if absent). Repeating a keyword 20 times does not count 20 times.
- **Inverse document frequency** (smoothed): `idf = ln((1 + n) / (1 + df)) + 1`, with `n` documents used to fit and `df`
  documents containing the term. Rare terms weigh more.
- **Weight** = tf x idf; each document vector is scaled to unit length (L2). **Cosine** = dot product of unit vectors, in [0, 1].
- **Tokens:** a custom tokenizer keeps `c++`, `c#`, `.net`, `node.js`, `ci/cd`, `scikit-learn` as single tokens; the vectorizer uses
  `lowercase=False`, `token_pattern=\S+`, `ngram_range=(1, 2)`, no document-frequency pruning.

**Q17. Why a custom tokenizer?** sklearn's default pattern would cut `c++` to nothing and `node.js` into `node` and `js`.
A test shows the default destroys these terms.
**Q18. What is "pairwise" TF-IDF and why?** The IDF is fitted on just two documents: the job description and that one resume.
Reason: a score must not change when an unrelated candidate is added. In batch mode I measured one resume scoring 0.2553 in a
batch of 10 and 0.2134 alone. Cost: with two documents, IDF is nearly flat (shared terms 1.0, unique terms 1.405), so
this TF-IDF behaves close to plain term-frequency cosine. *Follow-up:* "Is that still TF-IDF?" Yes mechanically, but the
IDF contributes little; a fixed reference corpus would give a more meaningful IDF and is a possible improvement.
**Q19. Hand calculation.** Documents "python sql" and "python docker": idf(python)=1, idf(sql)=idf(docker)=ln(3/2)+1=1.405.
Resume vector over (python, docker) = (1, 1.405)/1.725; job vector over (python, sql) = (1, 1.405)/1.725. Shared term python:
cosine = 1/(1 + 1.405^2) = 0.336. A test checks this value.
**Q20. Why cosine and not Euclidean distance?** Cosine ignores document length and compares direction; a long resume is not
penalised for having larger raw counts.
**Q21. What are the weaknesses?** No meaning: "sklearn" and "scikit-learn" are different tokens; keyword lists and negation score
high ("has not worked with Python" still contains "python").

### 5.2 Semantic similarity
- An **embedding** is a vector of numbers (384 here) produced by a neural network so that texts with similar meaning point in
  similar directions. Similarity = cosine of the two vectors, in [-1, 1].
- **Chunking** (because of the 256-token limit): documents of at most 254 tokens stay whole. Longer ones are split into units
  (lines, then sentences, then words, then halved characters) packed into chunks of at most 200 tokens, with up to 40 tokens of
  overlap made of whole lines or sentences. Each chunk is a slice of the original text, so it can be shown as evidence.
- **Aggregation:** for each job-description chunk take the best cosine against any resume chunk (`top_k=1`), then average over job
  chunks weighted by their token counts.

**Q22. Why chunk at all?** The model silently ignores everything after 256 tokens. In one constructed case a resume with the skills
on the last page scored 0.034 as one truncated embedding and 0.302 chunked.
**Q23. Why 200 and not 254?** A safety margin; token counts are exact only because this tokenizer splits on whitespace, so I did not
run at the limit. The values were fixed before the evaluation and not tuned.
**Q24. What is the weakness of taking the maximum?** A single resume chunk can decide the score when the job description fits in one
chunk, and a keyword-dense chunk can score high without the person doing the work. The best-matching passage is stored so a human can read it.
**Q24b. The model card says the training sequence length was 128 tokens, and your chunks are up to 200. Why?** The 200 was fixed in Phase 2 as a safety margin under the 254 usable tokens of the 256-token inference limit; I learned about the 128-token training length only when I read the official model card in Phase 12. The model accepts 200 tokens, but it was trained on shorter sequences, and I did not evaluate whether 128-token chunks would rank better. I did not change it, because the evaluation settings were frozen. It is a legitimate open question and a possible future experiment (smaller chunks, compared on new data).
**Q25. What does a cosine of 0.7 mean?** Only that the two texts are close in the model's space. It is not 70% of anything.
In the evaluation, resumes labelled "not relevant" scored between 0.47 and 0.75, nowhere near 0, while their TF-IDF scores were 0.02 to 0.14 (10 resumes); the two kinds of score live on very different scales, so a value from one cannot be read as if it came from the other.
**Q26. Can the semantic score be negative?** Yes, for very short unrelated strings (I saw -0.026). The combined index uses `max(0, cosine)`
and flags it; the raw value is always kept and displayed.

### 5.3 Skill coverage
- A **hand-written dictionary** (`data/skill_dictionary.json`, 114 skills) with aliases and rules. Each alias becomes a regular
  expression that may not touch a letter or digit on either side (so SQL is not found inside MySQL or NoSQL, and Java is not found inside JavaScript).
  Overlaps are resolved by longest match. Ambiguous words (Go, R, C, React, Express...) need exact capitalisation and/or a context word
  in the same sentence.
- **Required vs preferred** come from the job description: headings ("Requirements:", "Nice to have:") and cue phrases ("a plus", "bonus")
  that apply to their own sentence; if there is no structure everything counts as required and the app warns that the split is uncertain.
- **Coverage** = number of matched required skills / number of required skills, using unique canonical names (aliases map to one name; repeated
  mentions count once).

**Q27. Is this NER?** No. It is a rule-based phrase matcher over a fixed vocabulary, not a trained named-entity-recognition model.
It cannot find skills that are not in the dictionary, cannot understand negation ("I have not used Python" still matches Python) and
cannot tell who performed the work (a recruiter hiring Python developers matches Python).
**Q28. How do you avoid wrong substring matches?** Custom boundaries plus overlap resolution, with tests for MySQL/NoSQL/SQL, JavaScript/Java, GitHub/Git, ASP.NET/.NET, Node.js/js, C++/C#/C and R&D/R.
**Q29. What if the job description has no required skills?** Coverage is unavailable (blank, never 1.0). The index is computed from the
other two components with their weights renormalised (0.50/0.70 and 0.20/0.70), and a warning says that such an index is not comparable
with one from a job description that had required skills.
**Q30. Known skill-extraction errors from my own evaluation?** "React for the user interface" did not detect React (the rule needs a
context word in the same sentence); "Oracle databases" missed the plural; "PyTorch or TensorFlow" is read as both required.
*Follow-up:* "Why didn't you fix them?" The dictionary and extractor were frozen for the evaluation so I could not tune on the test data; fixes would be a separate, labelled experiment.

## 6. The combined ranking score

`index = 0.50 x max(0, semantic cosine) + 0.20 x TF-IDF cosine + 0.30 x required-skill coverage`

**Q31. Where do 0.50 / 0.20 / 0.30 come from?** They are initial design choices, written down before any evaluation. They are not optimal and were
never tuned on the evaluation data. Tuning on the same 46 resumes and then reporting those numbers would be a form of data leakage.
**Q32. Why not min-max normalise each component?** Min-max over the uploaded batch makes a candidate's score depend on who else was uploaded and is
undefined for one resume. Every component here is computed from one resume and the job description only, and a test shows adding an unrelated candidate changes nothing.
**Q33. Are the three components comparable?** No. A semantic 0.5 and a TF-IDF 0.5 mean different things and have different spreads. The weights are applied to fixed-scale
quantities; a component's real influence is roughly weight x spread.
**Q34. Why does skill coverage dominate the combined score on the evaluation set?** Measured on the 46 judgments: standard deviations 0.346
(coverage), 0.089 (semantic), 0.051 (TF-IDF). Coverage ranges from 0 to 1; semantic cosines sit in a narrow band (0.35 to 0.87). Of the variance of the combined
index about 75% comes from the coverage term (semantic 20%, TF-IDF 5%); the index correlates r = 0.93 with coverage and 0.59 with semantic, and within a job the Spearman
correlation of the ranking with coverage is 0.91. Because the terms are added without rescaling, the term with the widest spread controls the ordering. Consequence: keyword lists, negation and "role from
outside" resumes that mention every skill get coverage 1.0 and rise in the ranking. This is a property of this data and these fixed weights, not a bug, and I did not change anything.
*Follow-up:* "Does this mean the 0.30 weight is wrong?" Not necessarily: it shows the weight alone does not describe influence. Changing it would need new labelled data, a development/test split and an honest report.
*Follow-up:* "What would you do?" Calibrate each component on a separate calibration set, or learn weights with cross-validation on real labelled data, and report on held-out data.
**Q35. What is the index good for?** Ordering candidates for human review. It is not a probability of being hired or qualified, and not a percentage.
**Q36. What does the app do with ties?** Equal scores (12 decimals) share a rank and keep upload order; resumes with no usable text are ranked last and flagged.

## 7. Evaluation methodology, data and findings

**Q37. How did you evaluate?** Five synthetic job descriptions (backend, frontend, data analyst, machine learning, full-stack), 9 or 10 synthetic resumes each, 46 in total, graded 0 to 3
by a written rubric (`data/evaluation/RUBRIC.md`). The protocol was written before the data existed; labels were written from the rubric and the text alone, reviewed once and frozen
(SHA-256 hashes recorded), and not changed after seeing results. The three methods were read from one `CandidateRanker` run per job so all saw identical inputs.
**Q38. Why synthetic data?** I did not verify any suitable public labelled resume/job dataset, and real resumes are personal data. Synthetic data lets me design hard cases
(negation, keyword lists, paraphrase, "recruiter describes the role", long resumes) with known labels. Cost: it says nothing about real hiring.
**Q39. Define the metrics.** Relevant = label 2 or 3, K = 3.
- Precision@3 = relevant in the top 3 / 3.
- Recall@3 = relevant in the top 3 / all relevant in that job's pool (cannot reach 1 when a pool has more than 3 relevant candidates; here the cap is 0.6 to 0.75).
- MRR = 1 / rank of the first relevant candidate.
- nDCG@3 = DCG/IDCG with gain `2^label - 1` and discount `1/log2(position+1)`; uses the graded labels.
- Macro average: each job counts equally; undefined jobs are excluded and flagged.
**Q40. Why not accuracy?** There is no classification task with a defined negative class; accuracy would be meaningless for ranking.
**Q41. Results?**

| Method | Precision@3 | Recall@3 | MRR | nDCG@3 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.467 | 0.300 | 0.567 | 0.461 |
| Semantic | 0.600 | 0.390 | 1.000 | 0.651 |
| Combined | 0.600 | 0.390 | 0.900 | 0.654 |

**Q42. Which method won?** Combined beat TF-IDF on all four macro metrics, but did **not clearly beat semantic alone**: Precision@3 and Recall@3 are equal, nDCG differs by 0.003, MRR is lower.
Per job on nDCG, combined was better on backend and data analyst and worse on frontend, machine learning and full-stack. With five jobs, these differences are not evidence of significance.
**Q43. Why is semantic MRR exactly 1.000?** In every job the top-ranked candidate was a relevant one (usually the strong resume that also used the job's wording). It does not mean the semantic method is perfect: its Precision@3 is 0.600 and negation resumes ranked second in two jobs.
**Q44. Where did each method fail?** TF-IDF: keyword-list and negation resumes (FE-06, DA-06, ML-06 ranked first). Semantic: negation and "describes the role from outside" resumes scored in the same range as relevant partial matches. Skill coverage: cannot see paraphrased evidence (label-2 paraphrase resumes got coverage 0 to 0.2), counts mentions rather than use (a resume saying "never used any of these" got 1.00).
**Q45. Why did the evaluation use ties from a hashed order?** So candidate ids such as BE-01 (a strong match) do not win ties by sorting first. Best-case/worst-case tie bounds were identical to the main result, so ties did not matter here.
**Q46. What are the threats to validity?** Same author wrote system, data and labels; one annotator, no agreement measured; subjective labels; 5 jobs; only 2 of 46 resumes exceed the token limit; short clean invented resumes; adversarial profiles chosen by someone who knew the system's weaknesses.
**Q47. Did you tune anything on the evaluation set?** No. Weights, chunk size, overlap, `top_k`, the dictionary and the extractor were fixed beforehand. I did look at exploratory ablations afterwards (skills only 0.600/0.390/0.800/0.623; semantic + TF-IDF 0.667/0.440/1.000/0.635; semantic + skills 0.667/0.430/0.900/0.694; equal weights 0.600/0.390/0.900/0.646) and reported them as exploratory because they were chosen after seeing the main result.
**Q48. Skill-extraction precision 1.000 and recall 0.989: is that good?** It is not evidence of real-world accuracy. It is mention-level detection of dictionary words on 17 short synthetic resumes, annotated by the same person who wrote the dictionary, and negated or recruiter mentions count as correct detections.
**Q49. Was the evaluation reproducible?** Yes: re-running `python -m src.evaluate` after all later changes gave byte-identical results CSVs and a maximum score difference of 0.0; the data hashes matched.

## 8. Security, privacy, caching and robustness

**Q50. Are uploaded resumes stored?** Not by this project's code: tests show processing and a full interface run create no files and log no resume text. Streamlit holds uploads in server memory and caches extraction/matching results in memory (per browser session, at most 8 entries, one hour); "Clear cached data now" empties them. I cannot guarantee what third-party libraries do internally.
**Q51. Is any data sent to a service?** The project code imports no networking library and sends no resume text anywhere. The sentence-transformers library contacts Hugging Face when the model loads (download the first time, update check later); no resume text is involved. `HF_HUB_OFFLINE=1` after the first download stops that (verified by hand; a never-downloaded model then raises a clear error).
**Q52. How does the cache avoid returning a wrong result?** Cache keys contain the actual content: file names and bytes, job description text, the method and a configuration signature (weights, model name, chunk settings). The ranker object is excluded from the key. Tests show a changed job description, a changed PDF with the same file name, and a different method each recompute, and that the cache is cleared by the button.
**Q53. Can one user see another user's results?** Data caches are session-scoped (Streamlit keys them by session id; AppTest always uses one fixed id, so I demonstrated the keying by switching the session-id lookup in a test and by reading Streamlit's code). The loaded model is shared, but matcher objects hold no resume text (tested). I did not test real simultaneous users.
**Q54. What is CSV formula injection and how is it prevented?** A cell beginning with `=`, `+`, `-` or `@` can be executed as a formula by a spreadsheet, so a file named `=HYPERLINK(...).pdf` could run code. Cells that start with those characters (or tab/carriage return) get a leading apostrophe. Tested for all four prefixes, for every method.
**Q55. What was the markdown-rendering vulnerability?** In Phase 10 I probed the app with a file named `[click me](evil.example).pdf` and resume text containing links and formatting. File names, evidence snippets and some warnings were passed to `st.markdown`, expander labels and `st.warning`, so they rendered as clickable links or formatting chosen by the resume's author, which could mislead a reviewer. Fix: `escape_markdown` escapes `\ ` * _ [ ] ( ) < > ~ $ : |` in untrusted text before display, applied in four places in `app.py`. A regression test checks the escaped output and I confirmed it fails when the fix is disabled.
*Follow-up:* "Why not `unsafe_allow_html`?" It is off; the issue was markdown, not HTML. *Follow-up:* "Why are dataframes safe?" They display cell text as plain text.
**Q56. Why bind the server to localhost?** On its first launch Streamlit also listened on the network and printed a network URL. `.streamlit/config.toml` restricts it to `localhost`, with usage statistics off and a 25 MB upload cap (the parser separately refuses files over 10 MB).
**Q57. What about personal data and fairness?** The system removes e-mail addresses, phone numbers and URLs before matching but **not names** or other personal details, which stay in the text and are seen by the embedding model. No fairness audit was done. Scores can reflect wording and formatting rather than ability, and the dictionary favours people who write skills the dictionary knows. That is why a human must review every candidate and nobody is rejected automatically.

## 9. Known limitations (say these before the examiner does)
- **No OCR:** scanned PDFs are detected and reported, not read. Fixture used in tests is a generated image-only PDF, not a real scan.
- **English only:** a Hindi resume gives TF-IDF no words; the embedding model is English-oriented.
- **Dictionary-based skills:** unlisted skills, paraphrases, negation and "who did the work" are invisible; "A or B" is read as both.
- **Synthetic, small, single-author evaluation:** 46 invented resumes, 5 jobs, one annotator.
- **Limited real-world testing:** no real resumes; no real scanned resume; browser upload was tested through Streamlit's test harness, not by clicking in a browser.
- **Semantic limits:** one chunk can decide the score; a document just under 254 tokens is embedded whole and a relevant tail can be diluted (FS-09 scored 0.350, lowest of its pool).
- **Duplicate detection** by identical bytes only.
- **Not deployed, not production-ready.**

**Q58. Why does passing 389 automated tests not prove the rankings are accurate?**
Tests check that the code does what I specified on the cases I wrote: parsing, boundaries, arithmetic, CSV, caching, error handling. They would pass even if the ranking were poor, because they do not
contain real recruiters' judgements. Accuracy is a different question, measured separately by the evaluation, which is synthetic and small. Also: I wrote both code and tests, so shared misunderstandings would
pass; some tests use a fake embedding model (the real-model tests are a minority); and no test uses a real resume.
*Follow-up:* "Which tests are most meaningful for correctness?" The hand-calculated metric and TF-IDF tests, the direct-comparison test with the real model, and the reproducibility test of the Phase 8 results.
*Follow-up:* "Give a bug the tests would not catch." A dictionary that misses a skill used by real candidates, or a PDF layout (columns) that makes the text read in the wrong order.

## 10. Future improvements and trade-offs

| Improvement | Benefit | Trade-off / risk |
|---|---|---|
| OCR (for example Tesseract) | read scanned resumes | extra dependency, errors in OCR text, more processing of personal data |
| Evidence-aware skills (negation, role detection, trained NER) | fixes the biggest weakness found | needs labelled data and its own validation; harder to explain |
| Evaluate on real, consented, labelled data | real-world evidence | privacy, consent, labelling cost, fairness analysis required |
| Calibrate or learn the weights | influence matches intent | needs train/validation/test split and enough data; leakage risk on tiny sets |
| Fixed reference corpus for IDF | meaningful IDF, batch independent | need a representative corpus; terms outside it are ignored |
| Multilingual model | non-English resumes | larger model, slower, needs multilingual evaluation |
| Semantic top-k or learned aggregation | less dependence on one chunk | more complexity; must be validated on new data |
| Fairness audit | detect bias | needs demographic labels and careful ethics |
| Deployment | wider access | privacy, authentication, security hardening, hosting costs (all out of scope) |

## 11. Follow-up questions that test understanding
1. "In the combined formula, what happens if a resume has a negative semantic cosine?" (Counts as 0, flagged; raw value displayed.)
2. "You add a tenth resume to the upload. Do the other nine scores change? Why not?" (No: every component uses only the job description and that resume; pairwise IDF; measured change 1e-8 from floating-point batching.)
3. "Why do the TF-IDF and semantic columns show raw values rather than percentages?" (They are cosines on different scales; a percentage would be a false claim.)
4. "What does `top_k=1` do and what are you giving up?" (Best chunk only; one chunk can decide.)
5. "A job description says 'Python or Java'. What does coverage do?" (Treats both as required: a candidate with only one gets 0.5; limitation.)
6. "Show me where a keyword-stuffed resume would beat a strong one." (Combined: coverage 1.0 gets 0.30 even with no experience; evaluation: BE-06, FE-06, DA-06, ML-06 in top-3 lists.)
7. "Why is Recall@3 at most 0.6 for the backend job?" (5 relevant candidates; only 3 slots.)
8. "Why did you rank blank resumes last rather than at 0?" (A blank scoring 0.0 would outrank a genuinely dissimilar resume with a negative cosine; a test covers this.)
9. "What exactly does the cache key contain?" (See Q52.)
10. "If the dictionary adds a new skill, what must you re-do?" (Reload only; but the Phase 8 results become stale for that dictionary, since the hash changes; re-run the evaluation and report it as a new experiment.)
11. "Why is `data/evaluation/` protected by hashes in a test?" (So the reported results cannot silently drift from the data.)
12. "How would you show the combined score is better than semantic alone?" (More jobs, real labelled data, a development/test split, confidence intervals or a paired test; here I cannot claim it.)

## 12. Things not to say
- "The system is accurate / 90% accurate / ready for production." (No such measurement.)
- "The combined method is the best." (Not shown.)
- "The score is the probability the candidate is suitable." (It is not.)
- "It uses AI to understand resumes." (Only the embedding model is neural; the skill part is a dictionary.)
- "It is secure / private" without qualification. (Say what was tested and what was not.)
- "It reads scanned resumes." (No OCR.)
