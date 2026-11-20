// Builds presentation/Resume_Screening_Project.pptx (editable, 12 slides, speaker notes on every slide).
// Tooling only (not part of the application, not in requirements.txt). Setup outside the project folder:  npm install pptxgenjs
// Run from the project folder:   node presentation/build_deck.js      (NODE_PATH must contain pptxgenjs; APPLY_THEME may point to apply_theme.js)
// Images come from presentation/images (diagrams and charts rendered from docs/figures) and docs/screenshots (real captures of the running app).
// All numbers on the slides are copied from README.md / EXPERIMENT_RESULTS.md and the stored artifacts; nothing is invented here.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const root = path.resolve(__dirname, "..");
const img = (f) => path.join(root, "presentation", "images", f);
const shot = (f) => path.join(root, "docs", "screenshots", f);
const pngSize = (p) => { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };
function fit(p, maxW, maxH) { const s = pngSize(p); const r = Math.min(maxW / s.w, maxH / s.h); return { w: s.w * r, h: s.h * r }; }

const THEME = {
  name: "Resume Screening Academic",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: { dk1: "14213D", lt1: "FFFFFF", dk2: "3B4A6B", lt2: "EEF1F6", accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A",
            accent4: "6250D6", accent5: "D9A400", accent6: "5B6475", hlink: "2A78D6", folHlink: "6250D6" },
};
const hex = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "AI-Powered Resume Screening and Job Matching System Using NLP";
pres.subject = "NLP capstone project presentation";
pres.company = "";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: hex.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.5, w: 7.2, h: 2.4, fontFace: "+mj-lt", fontSize: 38, bold: true, color: C.background1, align: "left", valign: "top" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 4.1, w: 7.2, h: 1.9, fontFace: "+mn-lt", fontSize: 18, color: C.background2, align: "left", valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: hex.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 12.3, h: 0.85, fontFace: "+mj-lt", fontSize: 30, bold: true, color: C.text1, align: "left", valign: "middle" }, text: "" } },
    { text: { text: "Resume Screening and Job Matching | synthetic data only | decision support, not hiring decisions", options: { x: 0.5, y: 7.02, w: 10.5, h: 0.3, fontFace: "+mn-lt", fontSize: 10.5, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.02, w: 0.6, h: 0.3, fontFace: "+mn-lt", fontSize: 10.5, color: C.text2, align: "right" },
});

const bullets = (items, opts = {}) => items.map((t, i) => {
  const isObj = typeof t === "object";
  return { text: isObj ? t.text : t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: opts.space ?? 8, ...(isObj ? t.options : {}) } };
});
const body = (slide, items, x, y, w, h, extra = {}) => slide.addText(bullets(items, extra), {
  x, y, w, h, fontFace: "Calibri", fontSize: extra.size ?? 16, color: C.text1, valign: "top", margin: 0.05, isTextBox: true, objectName: extra.name ?? "Body bullets" });
function card(slide, x, y, w, h, head, text, colorHex, opts = {}) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: opts.fill ?? hex.lt2 }, line: { color: colorHex, width: 1.5 }, rectRadius: 0.08, objectName: `Card ${head}` });
  slide.addText([{ text: head, options: { bold: true, fontSize: opts.headSize ?? 16, color: colorHex === hex.accent5 ? "8A6A00" : colorHex, breakLine: true } },
                 { text, options: { fontSize: opts.size ?? 14, color: hex.dk1 } }],
    { x: x + 0.12, y: y + 0.08, w: w - 0.24, h: h - 0.16, fontFace: "Calibri", valign: "top", margin: 0.03, isTextBox: true, objectName: `Card text ${head}` });
}
const caption = (slide, text, x, y, w, h = 0.3) => slide.addText(text, { x, y, w, h, fontFace: "Calibri", fontSize: 11, color: C.text2, italic: true, margin: 0, valign: "top", isTextBox: true, objectName: "Caption" });
function picture(slide, p, x, y, maxW, maxH, alt, align = "left") {
  const d = fit(p, maxW, maxH);
  const xx = align === "center" ? x + (maxW - d.w) / 2 : x;
  slide.addImage({ path: p, x: xx, y, w: d.w, h: d.h, altText: alt });
  return d;
}
let section = (t) => pres.addSection({ title: t });

// ---------------------------------------------------------------- 1
section("Introduction");
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("AI-Powered Resume Screening and Job Matching System Using NLP", { placeholder: "title" });
s.addText([{ text: "TF-IDF, pretrained sentence embeddings and a skill dictionary, compared side by side", options: { breakLine: true } },
           { text: "Decision support, not hiring decisions. All demo and evaluation data are synthetic.", options: { breakLine: true } },
           { text: "Individual NLP capstone project (internal assessment)" }], { placeholder: "body" });
{ const d = fit(shot("01_home_screen.png"), 4.9, 5.6); s.addImage({ path: shot("01_home_screen.png"), x: 8.1, y: 0.9, w: d.w, h: d.h, altText: "Home screen of the Streamlit application" }); }
s.addNotes("Introduce the project in one sentence: it ranks PDF resumes against a job description using TF-IDF, a pretrained sentence-embedding model and a combined index that adds required-skill coverage from a dictionary. State the framing early: this is decision support, a person reviews every candidate, and every resume and job description shown today is synthetic (invented). The picture is a real screenshot of the running app (docs/screenshots/01_home_screen.png). Add your name, guide and institution when you present; they are intentionally not on the slide. Likely questions: Is this fully automated hiring? No. What is new here? Nothing is claimed as new; the contribution is an integrated, tested and honestly evaluated comparison (docs/RELATED_WORK.md, section 7).");

// ---------------------------------------------------------------- 2
section("Problem and objectives");
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem and objectives" });
s.addText("Problem and motivation", { placeholder: "title" });
body(s, ["One vacancy, many PDF resumes: reading them all is slow and inconsistent",
         "Keyword search misses synonyms (sklearn vs scikit-learn) and rewards keyword stuffing",
         "Embedding similarity relates meanings, but can rate unrelated or negated text as similar",
         "A skill checklist counts mentions, not real experience",
         "Recruiters need to see why a candidate ranks where they do"], 0.5, 1.4, 6.2, 5.3, { size: 19, space: 14 });
card(s, 7.1, 1.4, 2.8, 2.45, "Keyword overlap", "Transparent, but blind to meaning; long keyword lists fool it.", hex.accent1, { size: 16, headSize: 18 });
card(s, 10.1, 1.4, 2.7, 2.45, "Semantic similarity", "Finds related wording; reflects topic, not whether the person did the work.", hex.accent2, { size: 16, headSize: 18 });
card(s, 7.1, 4.05, 2.8, 2.45, "Skill coverage", "Easy to read; only as good as the dictionary; ignores negation.", hex.accent4, { size: 16, headSize: 18 });
card(s, 10.1, 4.05, 2.7, 2.45, "Human review", "Needed for every candidate. Nobody is rejected automatically.", hex.accent3, { size: 16, headSize: 18 });
s.addNotes("Motivation, not a claimed research gap. Each approach alone has a known weakness, and this project measures those weaknesses on synthetic data (slide 10). Why not an LLM? Privacy (resume text would leave the machine), cost, explainability and reproducibility; the trade-off is weaker understanding of context. Who would use it? Recruiters or placement cells; it is not validated for real use. Likely question: why compare several methods? Because each fails on different inputs and a single number hides that.");

// ---------------------------------------------------------------- 3
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem and objectives" });
s.addText("Objectives and scope", { placeholder: "title" });
s.addText("In scope", { x: 0.5, y: 1.35, w: 6, h: 0.4, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true });
body(s, ["Extract text from PDF resumes (PyMuPDF), with a status for every file",
         "Find skills transparently with an editable dictionary",
         "Score with TF-IDF, a pretrained sentence model and a combined index",
         "Evaluate with ranking metrics on labelled synthetic data",
         "Streamlit interface with details and CSV export, all local"], 0.5, 1.85, 6.0, 4.9, { size: 18, space: 12 });
s.addText("Out of scope", { x: 6.9, y: 1.35, w: 6, h: 0.4, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true });
body(s, ["OCR for scanned resumes (detected and reported only)", "LLMs, RAG, fine-tuning, a database, login, cloud deployment",
         "Evaluation on real resumes", "Fairness audit"], 6.9, 1.85, 5.9, 2.8, { size: 18, space: 12 });
card(s, 6.9, 4.6, 5.9, 1.9, "Nothing is trained here", "TF-IDF fitting only counts words. The embedding model is pretrained by others and used as published, not fine-tuned.", hex.accent5, { fill: "FFF6DD", size: 17, headSize: 19 });
s.addNotes("Say the exclusions out loud: they are decisions, not omissions. Did you train a model? No: TF-IDF fitting counts words and document frequencies (no gradients, no labels); the embedding model all-MiniLM-L6-v2 was trained by others and is used as published. Why no OCR? Scope and risk; scanned files are detected with a heuristic and reported as needing OCR, they are not read. The OCR fixture used in testing is a generated image-only PDF, not a real scan.");

// ---------------------------------------------------------------- 4
section("Foundations");
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Foundations" });
s.addText("Related work and research foundation", { placeholder: "title" });
const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: hex.dk1 }, fontSize: 14 } });
const cell = (t, o = {}) => ({ text: t, options: { fontSize: 13, color: hex.dk1, valign: "top", ...o } });
s.addTable([
  [hdr("Work (reviewed, not implemented)"), hdr("What it addresses"), hdr("Relation to this project")],
  [cell("Spärck Jones 1972; Salton and Buckley 1988", { bold: true }), cell("Term weighting: rare shared terms count more (IDF); single-term baselines"), cell("TF-IDF baseline (method A), scikit-learn variant; pairwise IDF is this project's own trade-off")],
  [cell("Reimers and Gurevych 2019 (SBERT); Wang et al. 2020 (MiniLM); model card", { bold: true }), cell("Sentence embeddings comparable by cosine; compact transformer base"), cell("Model used as published: all-MiniLM-L6-v2, 384 dimensions, 256-token input limit")],
  [cell("Karpukhin et al. 2020 (DPR); Thakur et al. 2021 (BEIR)", { bold: true }), cell("Dense retrieval; hybrid BM25 + dense score; BM25 as robust baseline"), cell("Precedent for a weighted lexical + semantic sum; hybrids help on some datasets, not all")],
  [cell("Gugnani and Misra 2020 (AAAI)", { bold: true }), cell("Resume/job matching with extracted and implicit skills plus embeddings"), cell("Explicit dictionary skills only here; implicit skills not inferred")],
  [cell("Raghavan et al. 2020; Bolukbasi et al. 2016; EU AI Act Service Desk", { bold: true }), cell("Bias in hiring tools and embeddings; recruitment ranking can be high-risk"), cell("Why a human reviews every candidate; no fairness audit or compliance claim")],
], { x: 0.5, y: 1.35, w: 12.3, colW: [3.7, 4.3, 4.3], border: { type: "solid", color: "C9CFDB", pt: 0.75 }, fill: { color: "FFFFFF" }, rowH: [0.4, 0.95, 0.95, 0.95, 0.8, 0.95], margin: 0.06, fontFace: "Calibri", objectName: "Related work table" });
caption(s, "Small, non-systematic review: no novelty or research-gap claim. Full references and verification notes: docs/RELATED_WORK.md. Paper claims are quoted at abstract level.", 0.5, 6.55, 12.3, 0.4);
s.addNotes("Every paper here was reviewed, none was implemented; the project uses the standard TF-IDF idea through scikit-learn and a published embedding model through sentence-transformers. References: Spärck Jones, J. Documentation 28(1), 1972, doi 10.1108/eb026526; Salton and Buckley, Information Processing and Management 24(5), 1988, doi 10.1016/0306-4573(88)90021-0; Reimers and Gurevych, EMNLP-IJCNLP 2019, aclanthology.org/D19-1410; Wang et al., NeurIPS 2020, arXiv 2002.10957; model card huggingface.co/sentence-transformers/all-MiniLM-L6-v2; Karpukhin et al., EMNLP 2020, arXiv 2004.04906 (their hybrid is BM25 + lambda times dense score with lambda 1.1 tuned on a development set; this project fixed its weights in advance and tuned nothing); Thakur et al., NeurIPS 2021 Datasets and Benchmarks, arXiv 2104.08663; Gugnani and Misra, AAAI-20, doi 10.1609/aaai.v34i08.7038; Raghavan et al., FAT* 2020, arXiv 1906.09208; Bolukbasi et al., NeurIPS 2016; European Commission AI Act Service Desk, page Employment (Annex III point 4(a)). Likely questions: Which paper did you implement? None. Is the combination novel? I did not find the exact combination in a small search; that is not evidence it is new.");

// ---------------------------------------------------------------- 5
section("System and methods");
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "System and methods" });
s.addText("System architecture", { placeholder: "title" });
{ const d = picture(s, img("diagram_architecture.png"), 0.4, 1.2, 9.0, 5.7, "System architecture diagram"); }
body(s, ["Three selectable methods: A TF-IDF, B semantic, C combined (default)",
         "Skill coverage is a component of C, not a mode",
         "Every file gets a status; bad files are listed, not ranked",
         "Local and in memory; the interface contains no matching algorithm"], 9.6, 1.5, 3.3, 4.8, { size: 15, space: 10 });
s.addNotes("Walk through the diagram top to bottom: inputs in the Streamlit interface, validation and PDF extraction, three text views, three independent scores, the combined index, then ranking and outputs. Point out the violet dashed box: skill coverage is a component of the combined index and cannot be selected on its own. The dashed arrows show that choosing method A or B ranks by that score only. A test checks that app.py and app_logic.py import no sklearn, numpy or model libraries, so the interface contains no matching algorithm. Stack: Python 3.12, PyMuPDF, scikit-learn, sentence-transformers (sentence-transformers/all-MiniLM-L6-v2, CPU), Streamlit, pytest. Likely questions: why separate app_logic.py from app.py? Testability without a browser.");

// ---------------------------------------------------------------- 6
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "System and methods" });
s.addText("Workflow and PDF processing", { placeholder: "title" });
picture(s, img("diagram_pipeline.png"), 0.4, 1.2, 8.3, 5.7, "End-to-end processing pipeline diagram");
body(s, ["Statuses: readable, no usable text, probably scanned (OCR needed), password-protected, corrupted, empty, too large, duplicate",
         "Duplicates: SHA-256 of the file bytes",
         "Usable file: at least 100 readable characters",
         "Three text views: skills, TF-IDF, embeddings",
         "E-mail, phone numbers and URLs removed"], 8.9, 1.4, 4.0, 5.3, { size: 15, space: 12 });
s.addNotes("Sequential stages go left to right, the three scoring paths run in parallel for each resume and merge only in the combined score. A bad file never stops the others and is never treated as an empty resume. Thresholds: fewer than 100 readable characters means unusable; the scanned heuristic is an image page with under 20 characters; the parser refuses files over 10 MB and Streamlit over 25 MB. The scanned fixture is a generated image-only PDF. Duplicate detection only catches identical bytes, not the same resume saved again from Word. Names and other personal details stay in the text; only e-mail, phone and URLs are removed. Likely question: how do you know a file is scanned? A heuristic, not proof.");

// ---------------------------------------------------------------- 7
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "System and methods" });
s.addText("TF-IDF and cosine similarity", { placeholder: "title" });
const eq = (slide, y, label, formula, color) => {
  slide.addShape(pres.ShapeType.roundRect, { x: 0.5, y, w: 6.9, h: 1.05, fill: { color: hex.lt2 }, line: { color, width: 1.5 }, rectRadius: 0.08, objectName: "Equation card " + label });
  slide.addText([{ text: label, options: { fontSize: 13, color: "5B6475", breakLine: true } }, { text: formula, options: { fontFace: "Cambria", fontSize: 20, bold: true, color: hex.dk1 } }],
    { x: 0.65, y: y + 0.05, w: 6.6, h: 0.95, valign: "middle", margin: 0.03, isTextBox: true, objectName: "Equation " + label });
};
eq(s, 1.4, "term frequency (sublinear)", "tf = 1 + ln(count)", hex.accent1);
eq(s, 2.6, "inverse document frequency (smoothed)", "idf = ln((1 + n) / (1 + df)) + 1", hex.accent1);
eq(s, 3.8, "cosine similarity of unit-length vectors", "cos(a, b) = a · b / (|a| |b|)  =  a · b", hex.accent1);
card(s, 0.5, 5.05, 6.9, 1.6, "Worked example (checked by a unit test)", "Documents \"python sql\" and \"python docker\": idf(python) = 1, idf(sql) = idf(docker) = 1.405, so cosine = 1 / (1 + 1.405²) = 0.336 (single-word version; the app also uses word pairs).", hex.accent1, { size: 13 });
body(s, ["Score range 0 to 1: word overlap, not meaning",
         "Tokens keep c++, c#, .net, node.js, ci/cd as single terms; sklearn's default would split or drop them",
         "IDF is fitted per job-description / resume pair, so one score never depends on the other uploads (0.2553 in a batch of 10 vs 0.2134 alone)",
         "Cost: with two documents IDF is almost flat (1.0 or 1.405)",
         "Weakness: sklearn and scikit-learn are different tokens; keyword lists score high"], 7.8, 1.4, 5.0, 5.3, { size: 15, space: 9 });
s.addNotes("TF-IDF is the traditional baseline (method A). The formulas are scikit-learn's documented smoothed IDF and sublinear tf (scikit-learn TfidfTransformer documentation, version 1.9.1). Why pairwise IDF? A score must not change when an unrelated candidate is added; in the earlier batch mode I measured one resume at 0.2553 in a batch of 10 and 0.2134 alone. The price is a weak IDF. A fixed reference corpus would give a more meaningful IDF and is future work. Vectorizer: lowercase false, token pattern non-whitespace, word pairs allowed, no document-frequency pruning. Likely questions: why cosine rather than Euclidean distance? Cosine ignores document length. Why not a reference corpus? None was available.");

// ---------------------------------------------------------------- 8
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "System and methods" });
s.addText("Transformer-based semantic similarity", { placeholder: "title" });
const flow = [["Resume text", "chunks of at most 200 tokens, up to 40 tokens overlap"], ["Embedding", "all-MiniLM-L6-v2: one 384-number vector per chunk"], ["Cosine", "best resume chunk for each job-description part"], ["Score S", "weighted mean, raw cosine, not a percentage"]];
flow.forEach((f, i) => {
  const x = 0.5 + i * 1.62;
  card(s, x, 1.4, 1.5, 2.05, f[0], f[1], i === 3 ? hex.accent2 : hex.accent5, { size: 13, headSize: 15, fill: i === 3 ? "FDEBE3" : "FFF6DD" });
});
card(s, 0.5, 3.65, 3.2, 1.35, "Why chunk?", "The model reads only 256 tokens. Skills on the last page: 0.034 as one truncated embedding, 0.302 chunked (one constructed case).", hex.accent2, { size: 12.5, headSize: 14 });
card(s, 3.85, 3.65, 3.1, 1.35, "Limits", "No negation, no sense of who did the work; one chunk can decide the score.", hex.accent2, { size: 12.5, headSize: 14 });
card(s, 0.5, 5.2, 6.45, 1.55, "Open question (not tested)", "The model card says it was trained with 128-token sequences; chunks here go up to 200 tokens (within its 256-token input limit). The effect of this choice was not evaluated and the setting was not changed.", hex.accent5, { size: 13, headSize: 14, fill: "FFF6DD" });
s.addText("Model card facts", { x: 7.4, y: 1.35, w: 5.4, h: 0.4, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true });
body(s, ["sentence-transformers/all-MiniLM-L6-v2, used as published, never trained by us",
         "384-dimensional vectors; input truncated beyond 256 word pieces",
         "Base nreimers/MiniLM-L6-H384-uncased; contrastive training on more than 1 billion sentence pairs",
         "Apache-2.0; \"sentence and short paragraph encoder\"",
         "Raw cosine, range -1 to 1; clipped to 0 only inside the combined index"], 7.4, 1.85, 5.4, 4.9, { size: 16, space: 12 });
s.addNotes("The model turns each text chunk into 384 numbers so that texts with similar meaning point in similar directions; similarity is the cosine of the two vectors. Sentence-BERT (Reimers and Gurevych, EMNLP-IJCNLP 2019) explains why embedding each text once and comparing with cosine scales, and MiniLM (Wang et al., NeurIPS 2020) is the compression method behind the small base model; the model itself is used exactly as published. Facts on the slide come from the official model card, and the 384 dimensions and 256-token limit were also measured on this machine. Aggregation: for each job-description chunk take the best cosine against any resume chunk, then average weighted by job-chunk length; top_k is 1. Weaknesses: the model ignores negation (in a small check it scored the sentence about having no experience in Python or SQL at 0.376 against a backend job) and one resume chunk can decide the score when the job description fits in one chunk. The 128-token versus 200-token point is a finding from reading the model card in Phase 12; it is untested. Likely questions: Is 0.7 a 70 percent match? No. Why chunk size 200? A margin under the 254 usable tokens, fixed before the evaluation.");

// ---------------------------------------------------------------- 9
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "System and methods" });
s.addText("Combined ranking and score interpretation", { placeholder: "title" });
s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 1.35, w: 6.4, h: 1.0, fill: { color: "DFF5EC" }, line: { color: hex.accent3, width: 2 }, rectRadius: 0.08, objectName: "Formula card" });
s.addText("index = 0.50·max(0, S) + 0.20·T + 0.30·K", { x: 0.55, y: 1.4, w: 6.3, h: 0.9, fontFace: "Cambria", fontSize: 22, bold: true, color: hex.dk1, align: "center", valign: "middle", margin: 0.03, isTextBox: true, objectName: "Formula" });
card(s, 0.5, 2.55, 2.05, 1.9, "S  semantic", "raw cosine of the sentence embeddings, clipped at 0", hex.accent2, { size: 14, headSize: 16 });
card(s, 2.72, 2.55, 2.05, 1.9, "T  TF-IDF", "cosine of the pairwise TF-IDF vectors, 0 to 1", hex.accent1, { size: 14, headSize: 16 });
card(s, 4.95, 2.55, 1.95, 1.9, "K  coverage", "matched required skills / required skills", hex.accent4, { size: 14, headSize: 16 });
body(s, ["Weights: initial design choices, fixed before any evaluation, never tuned",
         "No required skills in the job description: K is blank, the other weights are renormalised (0.50/0.70, 0.20/0.70), warning shown",
         "The index orders candidates; it is not a probability, a percentage or verified competence"], 0.5, 4.6, 6.4, 2.3, { size: 15, space: 8 });
picture(s, img("crop_candidate_details.png"), 7.2, 1.4, 5.7, 3.1, "Candidate details: matched and missing required skills with evidence sentences");
caption(s, "Real screenshot of the app, cropped (synthetic demo data): matched skills, evidence sentences and the best-matching passage", 7.2, 4.6, 5.7, 0.55);
card(s, 7.2, 5.35, 5.7, 1.45, "Read it as", "an ordering aid with visible parts, never as the chance that someone is suitable or will be hired.", hex.accent3, { size: 14, headSize: 15 });
s.addNotes("Each component is computed from one resume and the job description only, so adding another candidate never changes a score (tested; measured differences are floating-point noise of about 1e-8). Skill coverage uses unique canonical skills: aliases such as sklearn and scikit-learn map to one name, and repeated mentions count once. The formula is in src/ranker.py, function combine_scores. The three components are on different scales; a component's real influence is roughly weight times spread, which is why skill coverage dominates on the evaluation set (next slide). The picture is a crop of a real screenshot (docs/screenshots/08_candidate_details_and_score_explanation.png): matched required skills with the evidence sentence for each. Likely questions: where do the weights come from? Initial design choices, not optimal. What if the job description lists no skills? Coverage is blank, never 1.0, and the app warns.");

// ---------------------------------------------------------------- 10
section("Evaluation");
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("Evaluation methodology and results (synthetic data)", { placeholder: "title" });
const th = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: hex.dk1 }, fontSize: 14, align: "center" } });
const tc = (t, o = {}) => ({ text: t, options: { fontSize: 14, align: "center", color: hex.dk1, ...o } });
s.addTable([
  [th("Method"), th("Precision@3"), th("Recall@3"), th("MRR"), th("nDCG@3")],
  [tc("TF-IDF", { align: "left", bold: true }), tc("0.467"), tc("0.300"), tc("0.567"), tc("0.461")],
  [tc("Semantic", { align: "left", bold: true }), tc("0.600"), tc("0.390"), tc("1.000"), tc("0.651")],
  [tc("Combined", { align: "left", bold: true }), tc("0.600"), tc("0.390"), tc("0.900"), tc("0.654")],
], { x: 0.5, y: 1.35, w: 6.6, colW: [1.6, 1.25, 1.25, 1.2, 1.3], rowH: 0.42, border: { type: "solid", color: "C9CFDB", pt: 0.75 }, fill: { color: "FFFFFF" }, objectName: "Evaluation results table" });
body(s, ["5 invented jobs, 46 invented resumes; rubric labels frozen before any run; relevant = 2 or 3; K = 3",
         "Combined beat TF-IDF but not clearly semantic alone: equal Precision@3 and Recall@3, nDCG +0.003, MRR lower",
         "Per job on nDCG: combined won 2, lost 3",
         "Skill coverage drives about 75% of the index variance (r = 0.93): descriptive, not proof of better ranking",
         "Keyword lists fooled TF-IDF; negation and role-from-outside fooled semantic"], 0.5, 3.2, 6.7, 3.7, { size: 14, space: 7 });
picture(s, img("fig_ndcg_per_job.png"), 7.5, 1.3, 5.4, 3.35, "nDCG@3 per job for the three methods");
picture(s, img("fig_component_influence.png"), 7.5, 4.75, 5.4, 2.2, "Share of the combined index variance by term");
s.addNotes("Protocol: written before the data existed; five synthetic job descriptions (backend, frontend, data analyst, machine learning, full-stack) with 9 or 10 synthetic resumes each; labels written from the rubric in data/evaluation/RUBRIC.md and the text only, reviewed once and frozen with file hashes; the three methods were read from one ranker run per job so all saw identical inputs; ties used a label-independent hashed order. Metrics: Precision@3, Recall@3 (capped at 0.6 to 0.75 because each job has 4 or 5 relevant resumes), MRR, nDCG@3 with gain 2 to the power label minus 1. The values in the table are exactly those stored in artifacts/evaluation_results.csv. Critical findings: the combined index beat TF-IDF on all four macro metrics, but was not clearly better than semantic similarity alone; per job on nDCG it was better on backend and data analyst and worse on frontend, machine learning and full-stack. Component diagnostic: standard deviations 0.346 (coverage), 0.089 (semantic), 0.051 (TF-IDF); about 75 percent of the index variance comes from coverage, 20 percent semantic, 5 percent TF-IDF. This is descriptive on one synthetic set and is not a reason to change a weight. Threats to validity: the same author wrote the system, the resumes and the labels; one annotator; five jobs; no significance claim; not real-world performance. The charts were generated from the stored CSV artifacts by docs/figures/make_figures.py. Likely questions: So is combined better? Better than TF-IDF here, not shown better than semantic. Why is semantic MRR exactly 1.000? A relevant resume was first in every job; its Precision@3 is still 0.600. Why does coverage dominate? It has the widest spread and the terms are added without rescaling.");

// ---------------------------------------------------------------- 11
section("Responsible use");
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Responsible use" });
s.addText("Security, privacy, limitations and responsible use", { placeholder: "title" });
s.addText("Tested", { x: 0.5, y: 1.3, w: 3, h: 0.35, fontFace: "Cambria", fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true });
body(s, ["Uploads stay in memory; no files written, no resume text logged by project code",
         "Localhost only, telemetry off; no networking library in project code",
         "CSV formula injection blocked; markdown rendering vulnerability found and fixed",
         "Cache keys use file bytes, job text, method and configuration"], 0.5, 1.7, 6.1, 2.6, { size: 14, space: 6 });
s.addText("Limits and not tested", { x: 0.5, y: 4.2, w: 6, h: 0.35, fontFace: "Cambria", fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true });
body(s, ["No OCR; English only; dictionary skills ignore negation and role",
         "Synthetic, small, single-author evaluation; no real resumes tested",
         "Model load contacts Hugging Face unless HF_HUB_OFFLINE=1",
         "No fairness audit; names stay in the text; not deployed",
         "389 passed tests show behaviour, not ranking accuracy"], 0.5, 4.6, 6.1, 2.4, { size: 14, space: 5 });
picture(s, img("crop_status_edge_cases.png"), 6.9, 1.35, 6.0, 3.7, "File validation table showing blank, corrupted, empty, password-protected and scanned files");
caption(s, "Real screenshot, cropped (synthetic edge-case files): blank, corrupted, empty, password-protected and image-only files are each listed with a status and not ranked", 6.9, 4.95, 6.0, 0.7);
card(s, 6.9, 5.8, 6.0, 1.05, "Human oversight", "A person reviews every candidate; nobody is rejected automatically.", hex.accent3, { size: 13, headSize: 14 });
s.addNotes("Say what was tested and what was not; avoid the words secure or private without qualification. Tested: processing and a full interface run create no files and log no resume text; no networking library is imported by project code; the server is bound to localhost with telemetry off. The markdown vulnerability: file names and resume snippets were displayed as markdown, so a hostile file such as a name containing a link would show as a clickable link; fixed with an escape function and a regression test that fails when the fix is disabled. CSV: cells starting with equals, plus, minus or at get a leading apostrophe. Not tested: real browser upload (only Streamlit's test harness), real resumes, a real scanned resume, simultaneous users, library internals. The model load contacts Hugging Face to download or check the model; no resume text is involved; setting HF_HUB_OFFLINE to 1 after the first download prevents that. Names and other personal details stay in the text seen by the embedding model, and no fairness audit was done. Responsible-use context: Raghavan et al. 2020; Bolukbasi et al. 2016; the European Commission AI Act Service Desk page on employment says ranking tools used for recruitment may be high-risk; this project is an unreleased student project and makes no compliance claim. Why does a passing test suite not prove accuracy? Tests check behaviour on cases I wrote; they contain no real recruiter judgements and some use a fake embedding model. The full suite was re-run on 2026-10-24: 389 passed, 0 failed, 0 skipped.");

// ---------------------------------------------------------------- 12
section("Conclusion");
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
s.addText("Conclusion, future work and references", { placeholder: "title" });
s.addText("Delivered and found", { x: 0.5, y: 1.3, w: 4.0, h: 0.35, fontFace: "Cambria", fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true });
body(s, ["Parser, three ranking methods, interface, CSV export, 389 passing tests",
         "Reproducible synthetic evaluation and honest documentation",
         "Combined > TF-IDF, but not clearly > semantic alone",
         "The skill dictionary dominates the combined score on this data"], 0.5, 1.7, 4.2, 3.2, { size: 14, space: 6 });
s.addText("Future work (not done)", { x: 4.9, y: 1.3, w: 4.2, h: 0.35, fontFace: "Cambria", fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true });
body(s, ["Evaluation on real, consented data: needs privacy and fairness care",
         "Evidence-aware skills (negation, role): needs labelled data",
         "Calibrated or learned weights: needs a proper split",
         "OCR and multilingual models: extra dependencies and checks",
         "Smaller chunks vs the 128-token training length: untested"], 4.9, 1.7, 4.4, 3.4, { size: 14, space: 6 });
card(s, 9.6, 1.3, 3.3, 3.5, "Claims not made", "No novelty, no real-world validation, no statistical significance, no production readiness, no deployment, no OCR.", hex.accent5, { size: 17, headSize: 19, fill: "FFF6DD" });
s.addText("References", { x: 0.5, y: 5.0, w: 3, h: 0.3, fontFace: "Cambria", fontSize: 14, bold: true, color: C.text1, margin: 0, isTextBox: true });
s.addText("Spärck Jones (1972) J. Documentation 28(1); Salton and Buckley (1988) Inf. Proc. & Management 24(5); Reimers and Gurevych (2019) EMNLP-IJCNLP, ACL D19-1410; Wang et al. (2020) NeurIPS, arXiv 2002.10957; model card sentence-transformers/all-MiniLM-L6-v2 (Hugging Face); Karpukhin et al. (2020) EMNLP, arXiv 2004.04906; Thakur et al. (2021) NeurIPS Datasets and Benchmarks, arXiv 2104.08663; Gugnani and Misra (2020) AAAI-20, doi 10.1609/aaai.v34i08.7038; Raghavan et al. (2020) FAT*, arXiv 1906.09208; Bolukbasi et al. (2016) NeurIPS; Järvelin and Kekäläinen (2002) ACM TOIS 20(4); scikit-learn and PyMuPDF documentation; European Commission AI Act Service Desk, Employment. Links: docs/RELATED_WORK.md.",
  { x: 0.5, y: 5.3, w: 12.3, h: 1.6, fontFace: "Calibri", fontSize: 11, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "References" });
s.addNotes("End with the three-sentence summary: it ranks PDF resumes against a job description with TF-IDF, a pretrained sentence-embedding model and a combined index that adds required-skill coverage; on a small synthetic evaluation the combined index beat TF-IDF but did not clearly beat semantic similarity alone, so it does not establish real-world hiring effectiveness; it is decision support, a person reviews every candidate and nobody is rejected automatically. Future work, each with its trade-off: real labelled data (privacy, consent, fairness), evidence-aware skills (needs labels and its own validation), calibrated weights (needs a train, validation and test split), OCR and multilingual models (dependencies and extra checks), and a test of smaller chunks given the model's 128-token training length. What is the main weakness? The skill dictionary and the synthetic single-author evaluation. What would you do next? Real labelled data and a fair evaluation, then evidence-aware skills. Add presenter name, guide and institution before presenting.");

(async () => {
  const out = path.join(root, "presentation", "Resume_Screening_Project.pptx");
  await pres.writeFile({ fileName: out });
  const themeModule = process.env.APPLY_THEME;
  if (themeModule) { const { applyTheme } = require(themeModule); await applyTheme(out, THEME); console.log("theme applied"); }
  console.log("wrote", out);
})();
