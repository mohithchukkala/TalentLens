// Builds presentation/Resume_Screening_NLP_10_slides.pptx: a 10-slide, plain-language version of the project presentation.
// Tooling only (not part of the application, not in requirements.txt). Run from the project folder:
//   node presentation/build_deck_short.js      (NODE_PATH must contain pptxgenjs; APPLY_THEME may point to apply_theme.js)
// Every number is copied from README.md / EXPERIMENT_RESULTS.md and the stored artifacts; nothing is invented here.
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
pres.author = "Ch. Mohith (124AD0028)";
pres.subject = "Natural Language Processing";
pres.company = "";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: hex.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.0, w: 7.3, h: 2.3, fontFace: "+mj-lt", fontSize: 36, bold: true, color: C.background1, align: "left", valign: "top" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.5, w: 7.3, h: 3.2, fontFace: "+mn-lt", fontSize: 20, color: C.background2, align: "left", valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: hex.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 12.3, h: 0.85, fontFace: "+mj-lt", fontSize: 30, bold: true, color: C.text1, align: "left", valign: "middle" }, text: "" } },
    { text: { text: "Ch. Mohith (124AD0028) | AI&DS | Natural Language Processing | demo data are synthetic", options: { x: 0.5, y: 7.02, w: 10.5, h: 0.3, fontFace: "+mn-lt", fontSize: 10.5, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.02, w: 0.6, h: 0.3, fontFace: "+mn-lt", fontSize: 10.5, color: C.text2, align: "right" },
});

const bullets = (items, opts = {}) => items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: opts.space ?? 8 } }));
const body = (slide, items, x, y, w, h, extra = {}) => slide.addText(bullets(items, extra), {
  x, y, w, h, fontFace: "Calibri", fontSize: extra.size ?? 18, color: C.text1, valign: "top", margin: 0.05, isTextBox: true, objectName: extra.name ?? "Body bullets" });
function card(slide, x, y, w, h, head, text, colorHex, opts = {}) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: opts.fill ?? hex.lt2 }, line: { color: colorHex, width: 1.5 }, rectRadius: 0.08, objectName: `Card ${head}` });
  slide.addText([{ text: head, options: { bold: true, fontSize: opts.headSize ?? 17, color: colorHex === hex.accent5 ? "8A6A00" : colorHex, breakLine: true } },
                 { text, options: { fontSize: opts.size ?? 15, color: hex.dk1 } }],
    { x: x + 0.12, y: y + 0.08, w: w - 0.24, h: h - 0.16, fontFace: "Calibri", valign: "top", margin: 0.03, isTextBox: true, objectName: `Card text ${head}` });
}
const caption = (slide, text, x, y, w, h = 0.3) => slide.addText(text, { x, y, w, h, fontFace: "Calibri", fontSize: 11, color: C.text2, italic: true, margin: 0, valign: "top", isTextBox: true, objectName: "Caption" });
function picture(slide, p, x, y, maxW, maxH, alt) { const d = fit(p, maxW, maxH); slide.addImage({ path: p, x, y, w: d.w, h: d.h, altText: alt }); return d; }
const label = (slide, text, x, y, w) => slide.addText(text, { x, y, w, h: 0.4, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true });

// ---------------------------------------------------------------- 1 title
let s = pres.addSlide({ masterName: "TITLE_DARK" });
s.addText("AI-Powered Resume Screening and Job Matching System Using NLP", { placeholder: "title" });
s.addText([{ text: "Ch. Mohith (124AD0028)", options: { bold: true, color: "FFFFFF", fontSize: 24, breakLine: true } },
           { text: "Branch: AI&DS", options: { breakLine: true } },
           { text: "Subject: Natural Language Processing", options: { breakLine: true } },
           { text: " ", options: { breakLine: true, fontSize: 8 } },
           { text: "A helper that ranks resumes for a job. A person still makes the decision. All demo data are invented.", options: { fontSize: 16 } }], { placeholder: "body" });
{ const d = fit(shot("01_home_screen.png"), 4.6, 5.8); s.addImage({ path: shot("01_home_screen.png"), x: 8.3, y: 0.8, w: d.w, h: d.h, altText: "Home screen of the Streamlit application" }); }
s.addNotes("Introduce yourself: Ch. Mohith, roll number 124AD0028, branch AI&DS, subject Natural Language Processing. One sentence about the project: it reads PDF resumes, compares each one with a job description in three different ways and shows a ranked list. A person still reviews every candidate, and every resume used in the demo is invented. The picture is a real screenshot of the app that was built.");

// ---------------------------------------------------------------- 2 problem
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("The problem: too many resumes, too little time", { placeholder: "title" });
body(s, ["One job opening can get hundreds of resumes",
         "Reading them all by hand is slow, and different readers judge differently",
         "Matching only exact words misses the same idea written differently (for example sklearn and scikit-learn)",
         "Matching only on meaning can wrongly treat unrelated or negative text as similar",
         "So the idea: compare several methods and show clearly why someone ranks high"], 0.5, 1.4, 6.3, 5.4, { size: 20, space: 14 });
card(s, 7.2, 1.4, 5.6, 1.6, "Word matching (TF-IDF)", "Counts shared words. Easy to explain, but it does not understand meaning.", hex.accent1);
card(s, 7.2, 3.15, 5.6, 1.6, "Meaning matching (AI model)", "Compares what the texts mean. Finds related wording, but can be fooled.", hex.accent2);
card(s, 7.2, 4.9, 5.6, 1.6, "Combined", "Mixes both and adds how many required skills the resume has.", hex.accent3);
s.addNotes("Explain the problem in everyday words. A recruiter gets many PDF resumes for one job. Word matching is simple but blind to meaning. Meaning matching understands related words but can be fooled, for example by a sentence saying the person has no experience in a skill. So the project builds all three and compares them. No claim is made that this is a new idea.");

// ---------------------------------------------------------------- 3 what it does
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("What the project does", { placeholder: "title" });
label(s, "It can", 0.5, 1.35, 5);
body(s, ["Read text from PDF resumes and report a status for every file",
         "Find skills (Python, SQL, Docker ...) using a word list of 114 skills",
         "Score each resume with three selectable methods: TF-IDF, semantic, combined",
         "Show a ranked table, the reasons for each score, and a CSV download"], 0.5, 1.85, 6.1, 4.9, { size: 18, space: 12 });
label(s, "It does not", 6.9, 1.35, 5);
body(s, ["Read scanned (image-only) resumes: no OCR",
         "Use ChatGPT-style models, a database or a login",
         "Run online: it runs on the local computer only",
         "Decide who gets hired"], 6.9, 1.85, 5.9, 2.9, { size: 18, space: 12 });
card(s, 6.9, 4.7, 5.9, 1.9, "Nothing is trained here", "The AI model (all-MiniLM-L6-v2) was already trained by others and is used as it is. TF-IDF only counts words.", hex.accent5, { fill: "FFF6DD", size: 16, headSize: 18 });
s.addNotes("Keep it plain. The three methods are TF-IDF, semantic similarity and a combined score. Skill coverage, meaning how many of the required skills are found, is not a fourth method; it is one part of the combined score. Said out loud: no OCR, no online deployment, no training of a model. If asked: did you train a model? No, the sentence model was pretrained and is used as published.");

// ---------------------------------------------------------------- 4 architecture
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("System architecture", { placeholder: "title" });
picture(s, img("diagram_architecture.png"), 0.4, 1.2, 9.0, 5.7, "System architecture diagram");
body(s, ["Top: you upload resumes and paste the job description",
         "Middle: the three methods score each resume",
         "Skill coverage is only a part of the combined score",
         "Bottom: ranked table, reasons, CSV file",
         "Everything stays on your computer"], 9.6, 1.5, 3.3, 5.0, { size: 16, space: 12 });
s.addNotes("Read the diagram from top to bottom. Inputs come in through the Streamlit page. Each PDF is checked and its text extracted. Then three scores are computed independently: TF-IDF, semantic, and skill coverage. The combined score mixes the three. The user picks one method and gets a ranking and a CSV. The violet dashed box shows that skill coverage cannot be chosen alone. All of it runs locally and nothing is uploaded to a server.");

// ---------------------------------------------------------------- 5 pipeline
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Step by step: from PDF to ranking", { placeholder: "title" });
picture(s, img("diagram_pipeline.png"), 0.4, 1.2, 8.3, 5.7, "End-to-end processing pipeline diagram");
body(s, ["Each PDF gets a status: readable, blank, scanned, password-protected, corrupted, empty or duplicate",
         "Bad files are listed with a reason, never silently dropped",
         "E-mail, phone numbers and links are removed before matching",
         "Good files are scored by all methods"], 8.9, 1.4, 4.0, 5.3, { size: 16, space: 14 });
s.addNotes("This is the same system as a step-by-step flow. First the file check: a file with fewer than 100 readable characters is treated as unusable, scanned files are detected but not read, and a duplicate is a file with identical bytes. Contact details are removed before the text is compared. A bad file never stops the others. The three scoring paths run for each good resume and join in the combined score.");

// ---------------------------------------------------------------- 6 TF-IDF
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Method A: TF-IDF (matching words)", { placeholder: "title" });
body(s, ["Idea: words that appear in both the job and the resume raise the score",
         "Rare words count more than common words",
         "Each text becomes a list of numbers; the angle between the two lists (cosine) is the score, 0 to 1",
         "Good: simple and transparent. Weak: does not know that sklearn and scikit-learn mean the same"], 0.5, 1.4, 6.5, 3.8, { size: 18, space: 12 });
[["term frequency", "tf = 1 + ln(count)"], ["inverse document frequency", "idf = ln((1 + n) / (1 + df)) + 1"], ["similarity", "cosine = a · b / (|a| |b|)"]].forEach((e, i) => {
  const y = 1.4 + i * 1.2;
  s.addShape(pres.ShapeType.roundRect, { x: 7.4, y, w: 5.4, h: 1.05, fill: { color: hex.lt2 }, line: { color: hex.accent1, width: 1.5 }, rectRadius: 0.08 });
  s.addText([{ text: e[0], options: { fontSize: 13, color: "5B6475", breakLine: true } }, { text: e[1], options: { fontFace: "Cambria", fontSize: 19, bold: true, color: hex.dk1 } }],
    { x: 7.55, y: y + 0.05, w: 5.1, h: 0.95, valign: "middle", margin: 0.03, isTextBox: true });
});
card(s, 0.5, 5.3, 12.3, 1.45, "Small example (checked by a unit test)", "\"python sql\" and \"python docker\" share only the word python, so the score is 0.336. In the app, the word-pair version is used, so real numbers differ.", hex.accent1, { size: 16 });
s.addNotes("TF-IDF in plain words: count the words, give rare words more weight, turn each text into a list of numbers and measure how close the two lists are. The formulas come from the scikit-learn documentation. The example is hand-calculated and also checked by a unit test. A design choice worth mentioning: the word-rarity part is computed for each job and resume pair, so adding another resume never changes someone else's score.");

// ---------------------------------------------------------------- 7 semantic
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Method B: semantic similarity (matching meaning)", { placeholder: "title" });
[["1. Split", "Long resume text is cut into pieces of up to 200 tokens (word pieces)"], ["2. Embed", "A pretrained AI model turns each piece into 384 numbers"], ["3. Compare", "For each part of the job, find the closest resume piece"], ["4. Score", "Average these best matches: the semantic score"]].forEach((f, i) => {
  card(s, 0.5 + i * 3.1, 1.4, 2.95, 1.9, f[0], f[1], i === 3 ? hex.accent2 : hex.accent5, { fill: i === 3 ? "FDEBE3" : "FFF6DD", size: 15, headSize: 18 });
});
label(s, "Good", 0.5, 3.6, 4);
body(s, ["Understands related wording", "Works even when the exact words differ"], 0.5, 4.05, 5.8, 1.4, { size: 18, space: 8 });
label(s, "Careful", 6.9, 3.6, 4);
body(s, ["Does not understand \"no experience in Python\"", "A value like 0.7 is not a 70% match"], 6.9, 4.05, 5.9, 1.4, { size: 18, space: 8 });
card(s, 0.5, 5.55, 12.3, 1.25, "Open question (not tested)", "The model was trained on text of about 128 tokens, but this project uses pieces of up to 200. Whether that matters was not tested, so the setting was left unchanged.", hex.accent5, { fill: "FFF6DD", size: 15, headSize: 16 });
s.addNotes("The model all-MiniLM-L6-v2 turns a piece of text into 384 numbers so that texts with similar meaning give similar numbers. It reads at most 256 tokens, so long resumes are split into pieces. In one constructed example, skills on the last page scored 0.034 when truncated and 0.302 when split. The score is a raw similarity, not a percentage. It does not understand negation. The 128 versus 200 token point comes from reading the model card; it is untested and openly stated as a limitation.");

// ---------------------------------------------------------------- 8 combined
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Method C: combined score (default)", { placeholder: "title" });
s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 1.35, w: 6.4, h: 1.0, fill: { color: "DFF5EC" }, line: { color: hex.accent3, width: 2 }, rectRadius: 0.08 });
s.addText("score = 0.50·max(0, S) + 0.20·T + 0.30·K", { x: 0.55, y: 1.4, w: 6.3, h: 0.9, fontFace: "Cambria", fontSize: 22, bold: true, color: hex.dk1, align: "center", valign: "middle", margin: 0.03, isTextBox: true });
card(s, 0.5, 2.55, 2.05, 1.9, "S  semantic", "meaning similarity from the AI model", hex.accent2, { size: 14, headSize: 17 });
card(s, 2.72, 2.55, 2.05, 1.9, "T  TF-IDF", "shared-word similarity", hex.accent1, { size: 14, headSize: 17 });
card(s, 4.95, 2.55, 1.95, 1.9, "K  coverage", "required skills found / required skills", hex.accent4, { size: 14, headSize: 17 });
body(s, ["The weights 0.50, 0.20, 0.30 were chosen at the start and never tuned",
         "It is a score for ordering, not a probability of being hired"], 0.5, 4.65, 6.4, 2.2, { size: 17, space: 10 });
picture(s, img("crop_candidate_details.png"), 7.2, 1.4, 5.7, 3.1, "Candidate details: matched and missing skills with evidence sentences");
caption(s, "Real screenshot of the app (invented demo data): matched skills with the sentence that proves each one", 7.2, 4.6, 5.7, 0.55);
card(s, 7.2, 5.35, 5.7, 1.4, "Why it helps", "You can open any candidate and see the three parts, the matched and missing skills and the evidence.", hex.accent3, { size: 15, headSize: 17 });
s.addNotes("The combined score is a weighted sum of the three parts. The weights were fixed before any evaluation and were never changed to improve results. If the job description lists no required skills, coverage is left blank and the other two weights are rescaled, with a warning. The screenshot is a crop of the real app showing matched skills and the sentence found in the resume as evidence. The score only orders candidates.");

// ---------------------------------------------------------------- 9 results
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Results on invented test data", { placeholder: "title" });
const th = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: hex.dk1 }, fontSize: 14, align: "center" } });
const tc = (t, o = {}) => ({ text: t, options: { fontSize: 15, align: "center", color: hex.dk1, ...o } });
s.addTable([
  [th("Method"), th("Precision@3"), th("Recall@3"), th("MRR"), th("nDCG@3")],
  [tc("TF-IDF", { align: "left", bold: true }), tc("0.467"), tc("0.300"), tc("0.567"), tc("0.461")],
  [tc("Semantic", { align: "left", bold: true }), tc("0.600"), tc("0.390"), tc("1.000"), tc("0.651")],
  [tc("Combined", { align: "left", bold: true }), tc("0.600"), tc("0.390"), tc("0.900"), tc("0.654")],
], { x: 0.5, y: 1.35, w: 6.6, colW: [1.6, 1.25, 1.25, 1.2, 1.3], rowH: 0.45, border: { type: "solid", color: "C9CFDB", pt: 0.75 }, fill: { color: "FFFFFF" }, objectName: "Evaluation results table" });
caption(s, "Higher is better. 5 invented jobs, 46 invented resumes, labels written before testing.", 0.5, 3.25, 6.6, 0.4);
body(s, ["Combined did better than TF-IDF",
         "Combined was NOT clearly better than semantic alone",
         "About 75% of the combined score's variation comes from the skill-coverage part",
         "Keyword lists fooled TF-IDF; negative sentences fooled the semantic method"], 0.5, 3.75, 6.7, 3.2, { size: 17, space: 10 });
picture(s, img("fig_evaluation_macro.png"), 7.5, 1.3, 5.4, 3.3, "Macro evaluation metrics per method");
picture(s, img("fig_component_influence.png"), 7.5, 4.7, 5.4, 2.25, "Share of the combined score variance by part");
s.addNotes("Metrics in simple words: Precision at 3 is how many of the top three are really relevant; Recall at 3 is how many of all the relevant resumes appear in the top three; MRR rewards a relevant resume appearing first; nDCG rewards putting more relevant resumes higher. The numbers are exactly those stored in artifacts/evaluation_results.csv. The test data are small and invented: 5 jobs and 46 resumes, written and labelled by the same person, so this says nothing about real hiring. Honest finding: combined beat TF-IDF, but is not clearly better than semantic alone; per job on nDCG it won 2 and lost 3. Skill coverage has the widest spread, so it drives about 75 percent of the combined score variation; this describes the data and does not prove it is better or worse.");

// ---------------------------------------------------------------- 10 limits and conclusion
s = pres.addSlide({ masterName: "CONTENT" });
s.addText("Limits, conclusion and references", { placeholder: "title" });
label(s, "Limits", 0.5, 1.3, 4);
body(s, ["No OCR; English only", "Skills come from a word list; negation is ignored", "Tested only on invented, small data", "Names stay in the text; no fairness audit; not deployed", "389 passing tests show the code works, not that rankings are accurate"], 0.5, 1.75, 6.2, 3.3, { size: 16, space: 8 });
label(s, "Conclusion", 6.9, 1.3, 4);
body(s, ["A working, tested prototype with three ways of ranking", "Combined beat TF-IDF but not clearly semantic alone", "A person reviews every candidate; nobody is rejected automatically", "Next: real consented data, evidence-aware skills, a fairness check"], 6.9, 1.75, 5.9, 3.3, { size: 16, space: 8 });
card(s, 0.5, 4.6, 12.3, 0.75, "Not claimed", "no novelty, no real-world validation, no production readiness", hex.accent5, { fill: "FFF6DD", size: 14, headSize: 15 });
s.addText("References: Spärck Jones (1972) J. Documentation 28(1); Salton and Buckley (1988) Inf. Proc. & Management 24(5); Reimers and Gurevych (2019) EMNLP-IJCNLP, aclanthology.org/D19-1410; Wang et al. (2020) MiniLM, arXiv 2002.10957; model card sentence-transformers/all-MiniLM-L6-v2 (Hugging Face); Järvelin and Kekäläinen (2002) ACM TOIS 20(4); scikit-learn documentation.",
  { x: 0.5, y: 5.6, w: 12.3, h: 1.1, fontFace: "Calibri", fontSize: 12.5, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "References" });
s.addNotes("Close with three sentences: the project ranks PDF resumes against a job description with TF-IDF, a pretrained sentence model and a combined score; on a small invented test set the combined score beat TF-IDF but did not clearly beat the semantic method alone, so it does not prove real-world hiring performance; it is decision support and a person reviews every candidate. Limits to say aloud: no OCR, small invented data, dictionary skills, no fairness audit. Future work: real consented data, smarter skill detection, a fairness check. Thank the audience and invite questions.");

(async () => {
  const out = path.join(root, "presentation", "Resume_Screening_NLP_10_slides.pptx");
  await pres.writeFile({ fileName: out });
  const themeModule = process.env.APPLY_THEME;
  if (themeModule) { const { applyTheme } = require(themeModule); await applyTheme(out, THEME); console.log("theme applied"); }
  console.log("wrote", out);
})();
