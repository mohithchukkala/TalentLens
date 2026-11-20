"""Streamlit interface for the NLP resume screening and job matching project.

Run from the project folder:   streamlit run app.py

The app is a thin layer over the existing modules (see src/app_logic.py). Uploaded resumes are held
in memory only for the running session: nothing is written to disk, nothing is logged, and no
external service is called (the sentence-transformer model runs locally).
"""

import hashlib

import pandas as pd
import streamlit as st

from src import app_logic as logic
from src.semantic_matcher import SemanticModelError

st.set_page_config(page_title="Resume Screening Assistant", page_icon=None, layout="wide")

SYNTHETIC_NOTE = ("The built-in demo scenarios are SYNTHETIC (invented) job descriptions and resumes written for this "
                  "project; they are not real people or real hiring data.")


# ---- caching ---------------------------------------------------------------
# Resources: one ranker (and so one loaded model) per server process.
# Data: keyed by the CONTENT of the inputs (file bytes, job description text, method, config
# signature), so a changed resume or job description can never be served a stale result.
# In-memory only, per browser session, at most 8 entries, expiring after one hour.
@st.cache_resource(show_spinner=False)
def get_ranker():
    return logic.build_ranker()


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600, scope="session")
def cached_extract(files: tuple):
    return logic.extract_files(files)


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600, scope="session")
def cached_match(method: str, job_description: str, files: tuple, config_signature: str, _ranker):
    return logic.match(method, job_description, cached_extract(files), _ranker)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_demo_zip(job_id: str):
    return logic.demo_resumes_zip(job_id)


def input_signature(method, job_description, files, config_signature) -> str:
    digest = hashlib.sha256()
    for part in (method, job_description, config_signature):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    for name, data in files:
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def load_demo_job_description():
    job_id = st.session_state.get("demo_scenario")
    if job_id:
        st.session_state["jd_text"] = logic.demo_job_description(job_id)


def show_skill_list(title: str, skills):
    st.markdown(f"**{title}**")
    if skills:
        st.markdown(" ".join(f"`{s}`" for s in skills))
    else:
        st.caption("none detected")


# ---- page ------------------------------------------------------------------
st.title("Resume Screening and Job Matching Assistant")
st.caption("An NLP project: TF-IDF, pretrained sentence embeddings and a skill dictionary, shown side by side.")
st.info("**Decision support only.** The scores come from text matching and a skill dictionary. They do not verify "
        "competence, they are not probabilities of being hired, and nobody is rejected automatically: a person must "
        "review every candidate.")
with st.expander("About, privacy and limits"):
    st.markdown(
        "- **Decision support, not decisions.** Rankings help a human reviewer decide where to look first.\n"
        "- **What the scores are.** TF-IDF and semantic similarity are cosine similarities between texts; skill coverage "
        "is the share of the job's required skills found by a **dictionary** (not a trained model, no understanding of "
        "negation or of who did the work). The combined index is a number for ordering candidates, not a probability "
        "or a percentage.\n"
        "- **Evidence so far.** A small SYNTHETIC evaluation (5 invented jobs, 46 invented resumes) found the combined "
        "method better than TF-IDF alone but **not clearly better than semantic matching alone**. It does not establish "
        "real-world recruitment performance.\n"
        "- **Your data.** Only upload resumes you have permission to process. Files are kept in memory for this session "
        "(for at most an hour, up to 8 cached inputs), never written to disk or logged, and no external service is used. "
        "Use the button in the sidebar to clear them now.\n"
        f"- {SYNTHETIC_NOTE}")

ranker = get_ranker()
config_signature = logic.config_signature(ranker)

# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Matching method")
    method = st.radio("Choose how candidates are ranked", options=list(logic.METHOD_LABELS),
                      format_func=logic.METHOD_LABELS.get, index=0, key="method")
    if method == logic.METHOD_COMBINED:
        w = ranker.config.as_dict()
        st.caption(f"index = {w['semantic']:.2f} x semantic + {w['tfidf']:.2f} x TF-IDF + {w['skills']:.2f} x "
                   "required-skill coverage. Initial design weights, not optimised.")
    elif method == logic.METHOD_TFIDF:
        st.caption("Traditional baseline: TF-IDF cosine similarity between the job description and each resume.")
    else:
        st.caption(f"Pretrained model `{ranker.semantic.model_name}` run locally; resumes longer than the model limit "
                   "are split into overlapping chunks.")
    st.divider()
    scenarios = logic.demo_scenarios()
    if scenarios:
        st.header("Synthetic demo")
        st.caption(SYNTHETIC_NOTE)
        st.selectbox("Demo scenario", options=[s[0] for s in scenarios],
                     format_func=dict(scenarios).get, key="demo_scenario")
        st.button("Fill in this job description", on_click=load_demo_job_description)
        st.download_button("Download this scenario's synthetic resumes (ZIP of PDFs)",
                           data=cached_demo_zip(st.session_state.get("demo_scenario", scenarios[0][0])),
                           file_name="synthetic_resumes.zip", mime="application/zip")
        st.divider()
    if st.button("Clear cached data now"):
        st.cache_data.clear()
        st.session_state.pop("result", None)
        st.session_state.pop("result_signature", None)
        st.success("Cached inputs and results were cleared.")

# ---- 1. job description ------------------------------------------------------
st.header("1. Job description")
job_description = st.text_area("Paste the job description", key="jd_text", height=260,
                               placeholder="Paste the job description here. Headings such as 'Requirements:' and "
                                           "'Nice to have:' help the app tell required from preferred skills.")
job = None
if job_description.strip():
    job, _ = ranker.analyse_job(job_description)
    left, right = st.columns(2)
    with left:
        show_skill_list("Required skills detected", job.required_skills)
    with right:
        show_skill_list("Preferred skills detected", job.preferred_skills)
    for message in job.warnings:
        st.warning(message)
    st.caption("Skills are found with a hand-written dictionary, so a skill that is not in it, or is worded "
               "differently, will not be detected.")
else:
    st.info("Enter a job description to see the skills the app detects in it.")

# ---- 2. resumes ---------------------------------------------------------------
st.header("2. Resumes")
uploads = st.file_uploader("Upload one or more PDF resumes", type=["pdf"], accept_multiple_files=True,
                           key="resume_files",
                           help="Text PDFs only. Scanned (image) PDFs need OCR, which this version does not do.")
files = tuple((f.name, f.getvalue()) for f in (uploads or []))
extractions = []
if files:
    with st.spinner("Reading the PDFs..."):
        extractions = cached_extract(files)
    st.dataframe(pd.DataFrame(logic.status_rows(extractions)), hide_index=True)
    excluded_count = sum(1 for r in extractions if not r.usable or r.duplicate_of is not None)
    if excluded_count:
        st.warning(f"{excluded_count} of {len(extractions)} file(s) will not be ranked (see the table). "
                   "They are listed, not rejected: someone should look at them manually.")
    with st.expander("Check the extracted text (first 600 characters of each readable file)"):
        for r in extractions:
            if r.usable:
                st.markdown(f"**{logic.escape_markdown(r.file_id)}**")
                st.text(r.text[:600])
else:
    st.info("Upload at least one PDF resume.")

# ---- 3. run -------------------------------------------------------------------
st.header("3. Rank")
if st.button("Rank candidates", type="primary"):
    problems = logic.validate_inputs(job_description, len(files), method)
    if problems:
        for message in problems:
            st.error(message)
    else:
        try:
            with st.spinner("Matching... the first semantic run loads the language model and can take a minute."):
                result = cached_match(method, job_description, files, config_signature, ranker)
            st.session_state["result"] = result
            st.session_state["result_signature"] = input_signature(method, job_description, files, config_signature)
        except SemanticModelError as exc:
            st.error(f"The language model could not be loaded: {exc}")

# ---- 4. results ---------------------------------------------------------------
result = st.session_state.get("result")
if result is not None:
    st.header("4. Results")
    current = input_signature(method, job_description, files, config_signature)
    if st.session_state.get("result_signature") != current:
        st.warning("The job description, resumes or method changed since these results were produced. "
                   "Press 'Rank candidates' to update them.")
    st.subheader(f"Method: {logic.METHOD_LABELS[result.method]}")
    for message in result.warnings:
        st.warning(logic.escape_markdown(message))

    if result.rows:
        st.caption("Scores are text-matching values for ordering candidates, not probabilities of being hired. "
                   "Equal scores share a rank. Click a column header to sort.")
        if result.method == logic.METHOD_COMBINED:
            w = result.weights
            st.caption(f"Combined matching index = {w['semantic']:.2f} x max(0, semantic cosine) + {w['tfidf']:.2f} x "
                       f"TF-IDF cosine + {w['skills']:.2f} x required-skill coverage. Coverage is blank when the job "
                       "description has no identifiable required skills (the other weights are then renormalised).")
        table = pd.DataFrame(logic.results_table(result))
        score_columns = {logic.SCORE_LABELS[k]: st.column_config.NumberColumn(format="%.3f") for k in result.score_keys}
        st.dataframe(table, hide_index=True, column_config=score_columns)
        st.download_button("Download ranking as CSV", data=logic.result_to_csv(result),
                           file_name=f"ranking_{result.method}.csv", mime="text/csv")

        st.subheader("Candidate details")
        for row in result.rows:
            headline = " | ".join(f"{logic.SCORE_LABELS[k]}: {row.scores[k]:.3f}" for k in result.score_keys
                                  if row.scores.get(k) is not None)
            label = f"#{row.rank}  {logic.escape_markdown(row.candidate_id)}"
            with st.expander(label + (f"  ({headline})" if headline else "")):
                if row.empty:
                    st.warning("This resume has no usable text; its scores are placeholders.")
                if row.matched_required is not None:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        show_skill_list("Required skills matched", row.matched_required)
                    with c2:
                        show_skill_list("Required skills missing", row.missing_required)
                    with c3:
                        show_skill_list("Preferred skills matched", row.matched_preferred)
                    for skill, snippets in row.skill_evidence.items():
                        st.markdown(f"Evidence for `{skill}`: " + " / ".join(f"*{logic.escape_markdown(s)}*" for s in snippets))
                for label, passage in row.text_evidence:
                    st.markdown(f"**{label}**")
                    st.text(passage)
                for message in row.warnings:
                    st.warning(message)
    elif not result.warnings:
        st.info("No candidates were ranked.")

    if result.excluded:
        st.subheader("Files not ranked")
        st.dataframe(pd.DataFrame([{"File": e.file_id, "Reason": e.reason.replace("_", " "), "Details": e.message}
                                   for e in result.excluded]), hide_index=True)
    for note in result.notes:
        st.caption(note)
