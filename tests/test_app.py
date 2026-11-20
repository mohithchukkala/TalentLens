"""Interface tests for app.py using Streamlit's AppTest.

Most tests use a ranker built on a FAKE embedding model (fast, content-dependent) injected through
`app_logic.build_ranker`. Tests at the end run the whole interface with the REAL pretrained model and
are skipped, visibly, if it cannot be loaded.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import app_logic as logic
from src.ranker import CandidateRanker, combine_scores
from src.semantic_matcher import SemanticMatcher, SemanticModelError
from tests.test_app_logic import (JD, PARTIAL, STRONG, UNRELATED, _blank_pdf, _image_only_pdf)
from tests.test_semantic_matcher import BagOfWordsModel, words

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
PDF = "application/pdf"


def pdf(text):
    return logic.text_to_pdf_bytes(text)


class CountingModel(BagOfWordsModel):
    """Fake model that counts how many times the app asks it to embed something."""


@pytest.fixture()
def env(monkeypatch):
    """Fresh caches and an injected fake-model ranker for every test."""
    st.cache_data.clear()
    st.cache_resource.clear()
    model = CountingModel()
    built = []

    def build():
        ranker = CandidateRanker(semantic_matcher=SemanticMatcher(model=model, token_counter=words))
        built.append(ranker)
        return ranker

    monkeypatch.setattr(logic, "build_ranker", build)
    yield type("Env", (), {"model": model, "built": built})
    st.cache_data.clear()
    st.cache_resource.clear()


def new_app():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    return at


def rank_button(at):
    return next(b for b in at.button if b.label == "Rank candidates")


def run_ranking(at, jd=JD, files=(), method=None):
    at.text_area(key="jd_text").set_value(jd)
    at.file_uploader(key="resume_files").set_value([(n, d, PDF) for n, d in files] if files else None)
    if method:
        at.radio(key="method").set_value(method)
    rank_button(at).click()
    at.run()
    return at


def result_of(at):
    return at.session_state["result"]


def error_texts(at):
    return [e.value for e in at.error]


STANDARD = [("strong.pdf", pdf(STRONG)), ("partial.pdf", pdf(PARTIAL)), ("unrelated.pdf", pdf(UNRELATED))]


# =============================================================================
# Start-up and validation
# =============================================================================
def test_app_imports_and_renders_without_errors(env):
    at = new_app()
    assert not at.exception
    assert at.title[0].value == "Resume Screening and Job Matching Assistant"
    assert any("Decision support only" in i.value for i in at.info)
    assert [h.value for h in at.header][:3] == ["1. Job description", "2. Resumes", "3. Rank"]
    assert [h.value for h in at.sidebar.header] == ["Matching method", "Synthetic demo"]
    assert at.radio(key="method").value == logic.DEFAULT_METHOD == "combined"
    assert at.radio(key="method").options == list(logic.METHOD_LABELS.values())


def test_privacy_and_limits_notice_is_shown(env):
    at = new_app()
    text = " ".join(m.value for e in at.expander if "privacy" in e.label.lower() for m in e.markdown)
    for phrase in ("Decision support, not decisions", "not a probability", "SYNTHETIC evaluation",
                   "not clearly better than semantic matching alone", "permission", "never written to disk"):
        assert phrase in text


def test_empty_job_description_and_no_resumes_show_helpful_errors(env):
    at = new_app()
    rank_button(at).click()
    at.run()
    assert not at.exception
    assert error_texts(at) == ["Please enter a job description.", "Please upload at least one PDF resume."]
    assert "result" not in at.session_state


def test_job_description_without_resumes(env):
    at = new_app()
    at.text_area(key="jd_text").set_value(JD)
    rank_button(at).click()
    at.run()
    assert error_texts(at) == ["Please upload at least one PDF resume."]


def test_resumes_without_job_description(env):
    at = run_ranking(new_app(), jd="   ", files=STANDARD)
    assert error_texts(at) == ["Please enter a job description."]
    assert not at.exception


def test_job_description_panel_shows_required_and_preferred_skills(env):
    at = new_app()
    at.text_area(key="jd_text").set_value(JD)
    at.run()
    text = " ".join(m.value for m in at.markdown)
    assert "`Python`" in text and "`SQL`" in text and "`Docker`" in text and "`Kubernetes`" in text
    assert not at.warning


def test_uncertain_split_and_no_skills_warnings(env):
    at = new_app()
    at.text_area(key="jd_text").set_value("We need Python, SQL and Docker.")
    at.run()
    assert any("uncertain" in w.value for w in at.warning)
    at.text_area(key="jd_text").set_value("We want a friendly and motivated colleague for our growing team.")
    at.run()
    assert any("No skills from the dictionary" in w.value for w in at.warning)


# =============================================================================
# Uploading and ranking
# =============================================================================
def test_multiple_pdfs_are_processed_and_ranked_with_correct_association(env):
    at = run_ranking(new_app(), files=STANDARD)
    assert not at.exception and not error_texts(at)
    result = result_of(at)
    assert result.method == "combined"
    assert [r.candidate_id for r in result.rows] == ["strong.pdf", "partial.pdf", "unrelated.pdf"]
    by = {r.candidate_id: r for r in result.rows}
    assert by["strong.pdf"].matched_required == ("Python", "SQL", "Docker")
    assert by["partial.pdf"].missing_required == ("Docker",)
    assert by["unrelated.pdf"].matched_required == ()
    # status table lists every file, results table lists every ranked candidate
    status = at.dataframe[0].value
    assert list(status["File"]) == ["strong.pdf", "partial.pdf", "unrelated.pdf"]
    assert set(status["Used in ranking"]) == {"Yes"}
    table = at.dataframe[1].value
    assert list(table["Candidate"]) == ["strong.pdf", "partial.pdf", "unrelated.pdf"]
    assert list(table["Rank"]) == [1, 2, 3]
    for column in ("Combined matching index", "Semantic similarity (raw cosine)", "TF-IDF similarity (cosine)",
                   "Required-skill coverage"):
        assert column in table.columns
    for i, row in enumerate(result.rows):                       # the table shows each candidate's own numbers
        assert table["Combined matching index"][i] == pytest.approx(row.scores["combined_index"])
        assert table["TF-IDF similarity (cosine)"][i] == pytest.approx(row.scores["tfidf_cosine"])


def test_combined_ranking_in_the_interface_preserves_the_phase7_formula(env):
    result = result_of(run_ranking(new_app(), files=STANDARD))
    assert result.weights == {"semantic": 0.5, "tfidf": 0.2, "skills": 0.3}
    for r in result.rows:
        s, t, k = r.scores["semantic_cosine"], r.scores["tfidf_cosine"], r.scores["required_skill_coverage"]
        assert r.scores["combined_index"] == pytest.approx(0.5 * max(0, s) + 0.2 * t + 0.3 * k)
        assert r.scores["combined_index"] == pytest.approx(combine_scores(s, t, k).index)


def test_one_invalid_pdf_does_not_break_the_valid_ones(env):
    files = [("good.pdf", pdf(STRONG)), ("bad.pdf", b"this is not a pdf"), ("also_good.pdf", pdf(PARTIAL))]
    at = run_ranking(new_app(), files=files)
    assert not at.exception and not error_texts(at)
    result = result_of(at)
    assert [r.candidate_id for r in result.rows] == ["good.pdf", "also_good.pdf"]
    assert [(e.file_id, e.reason) for e in result.excluded] == [("bad.pdf", "not_scored")]
    status = at.dataframe[0].value.set_index("File")
    assert status.loc["bad.pdf", "Status"] == "Unreadable or corrupted"
    assert status.loc["bad.pdf", "Used in ranking"] == "No"
    assert any("will not be ranked" in w.value for w in at.warning)


def test_every_kind_of_excluded_file_is_explained(env):
    good = pdf(STRONG)
    files = [("good.pdf", good), ("same_content.pdf", good), ("bad.pdf", b"nope"), ("empty.pdf", b""),
             ("scan.pdf", _image_only_pdf()), ("blank.pdf", _blank_pdf())]
    at = run_ranking(new_app(), files=files)
    assert not at.exception
    status = at.dataframe[0].value.set_index("File")
    assert status.loc["same_content.pdf", "Used in ranking"] == "No (duplicate)"
    assert status.loc["empty.pdf", "Status"] == "Empty file (0 bytes)"
    assert status.loc["scan.pdf", "Status"] == "Probably scanned: OCR needed" and "OCR" in status.loc["scan.pdf", "Note"]
    assert status.loc["blank.pdf", "Status"] == "No usable text"
    assert [r.candidate_id for r in result_of(at).rows] == ["good.pdf"]
    excluded_table = at.dataframe[-1].value
    assert set(excluded_table["File"]) == {"same_content.pdf", "bad.pdf", "empty.pdf", "scan.pdf", "blank.pdf"}


def test_duplicate_file_names_get_unique_identifiers(env):
    files = [("cv.pdf", pdf(STRONG)), ("cv.pdf", pdf(PARTIAL))]
    result = result_of(run_ranking(new_app(), files=files))
    assert sorted(r.candidate_id for r in result.rows) == ["cv (2).pdf", "cv.pdf"]
    by = {r.candidate_id: r for r in result.rows}
    assert by["cv.pdf"].matched_required == ("Python", "SQL", "Docker")      # the first upload
    assert by["cv (2).pdf"].matched_required == ("Python", "SQL")            # the second upload


def test_nothing_usable_shows_a_warning_not_a_crash(env):
    at = run_ranking(new_app(), files=[("bad.pdf", b"nope"), ("blank.pdf", _blank_pdf())])
    assert not at.exception
    assert any("None of the uploaded files had usable text" in w.value for w in at.warning)
    assert result_of(at).rows == [] and len(result_of(at).excluded) == 2


def test_candidate_detail_sections_show_skills_evidence_and_warnings(env):
    at = run_ranking(new_app(), files=STANDARD)
    labels = [e.label for e in at.expander if e.label.startswith("#")]
    assert labels[0].startswith("#1  strong.pdf") and labels[2].startswith("#3  unrelated.pdf")
    strong_panel = next(e for e in at.expander if e.label.startswith("#1  strong.pdf"))
    text = " ".join(m.value for m in strong_panel.markdown)
    assert "Required skills matched" in text and "`Python`" in text and "Evidence for `Python`" in text
    assert "Preferred skills matched" in text and "`Kubernetes`" in text
    partial_panel = next(e for e in at.expander if e.label.startswith("#2  partial.pdf"))
    assert "`Docker`" in " ".join(m.value for m in partial_panel.markdown)       # shown under 'missing'
    unrelated_panel = next(e for e in at.expander if e.label.startswith("#3  unrelated.pdf"))
    assert any("No dictionary skills were found" in w.value for w in unrelated_panel.warning)


def test_results_are_not_called_probabilities(env):
    at = run_ranking(new_app(), files=STANDARD)
    captions = " ".join(c.value for c in at.caption)
    assert "not probabilities" in captions and "Combined matching index" in captions


# =============================================================================
# Methods
# =============================================================================
def test_each_method_returns_correctly_associated_results_through_the_interface(env):
    expected_columns = {
        "combined": ["Rank", "Candidate", "Combined matching index", "Semantic similarity (raw cosine)",
                     "TF-IDF similarity (cosine)", "Required-skill coverage", "Required skills matched"],
        "tfidf": ["Rank", "Candidate", "TF-IDF similarity (cosine)"],
        "semantic": ["Rank", "Candidate", "Semantic similarity (raw cosine)"],
    }
    at = new_app()
    for method in ("tfidf", "semantic", "combined"):
        at = run_ranking(at, files=STANDARD, method=method)
        assert not at.exception, method
        result = result_of(at)
        assert result.method == method
        table = at.dataframe[1].value
        assert list(table.columns) == expected_columns[method]
        assert list(table["Candidate"]) == [r.candidate_id for r in result.rows]
        # The fake embedding model is a crude bag of words, so only TF-IDF and combined must put the
        # strong resume first; the real model's ordering is checked in the real-model test below.
        assert result.rows[-1].candidate_id == "unrelated.pdf"
        if method != "semantic":
            assert result.rows[0].candidate_id == "strong.pdf"
        key = logic.METHOD_SCORES[method][0]
        label = logic.SCORE_LABELS[key]
        for i, row in enumerate(result.rows):
            assert table[label][i] == pytest.approx(row.scores[key])


def test_csv_download_matches_the_selected_method(env):
    for method, header in (("tfidf", "tfidf_cosine"), ("semantic", "semantic_cosine_raw"),
                           ("combined", "combined_matching_index")):
        at = run_ranking(new_app(), files=STANDARD, method=method)
        csv_text = logic.result_to_csv(result_of(at))
        assert csv_text.splitlines()[0].split(",")[4] == header
        assert len(at.get("download_button")) >= 2                      # demo ZIP + ranking CSV


def test_blank_text_resume_is_flagged_and_ranked_last(env):
    filler = "the and of to in a is it " * 8
    at = run_ranking(new_app(), files=[("filler.pdf", pdf(filler)), ("strong.pdf", pdf(STRONG))], method="tfidf")
    result = result_of(at)
    assert [r.candidate_id for r in result.rows] == ["strong.pdf", "filler.pdf"]
    assert result.rows[-1].empty
    panel = next(e for e in at.expander if "filler.pdf" in e.label)
    assert any("no usable text" in w.value.lower() for w in panel.warning)
    assert any("No usable words" in w.value for w in panel.warning)


# =============================================================================
# Caching: effective, and never wrong
# =============================================================================
def test_model_and_ranker_are_created_once_across_reruns(env):
    at = new_app()
    at.run()
    run_ranking(at, files=STANDARD)
    at.run()
    assert len(env.built) == 1


def test_same_inputs_are_not_recomputed_but_changed_inputs_are(env):
    at = run_ranking(new_app(), files=STANDARD)
    first_calls = len(env.model.calls)
    assert first_calls >= 1
    rank_button(at).click()
    at.run()
    assert len(env.model.calls) == first_calls                           # identical inputs: served from cache
    # different job description: recomputed, and the result reflects the new text
    old = {r.candidate_id: r.scores["tfidf_cosine"] for r in result_of(at).rows}
    run_ranking(at, jd="Requirements:\n- Python\n- Kubernetes", files=STANDARD)
    assert len(env.model.calls) == first_calls + 1
    new = {r.candidate_id: r.scores["tfidf_cosine"] for r in result_of(at).rows}
    assert new != old
    assert result_of(at).job.required_skills == ("Python", "Kubernetes")
    # one resume changed (same file name, different content): recomputed with the new content
    changed = [("strong.pdf", pdf(UNRELATED)), STANDARD[1], STANDARD[2]]
    run_ranking(at, jd="Requirements:\n- Python\n- Kubernetes", files=changed)
    assert len(env.model.calls) == first_calls + 2
    by = {r.candidate_id: r for r in result_of(at).rows}
    assert by["strong.pdf"].matched_required == ()                       # the new content, not the cached old one


def test_different_methods_do_not_share_cached_results(env):
    at = run_ranking(new_app(), files=STANDARD, method="tfidf")
    tfidf_result = result_of(at)
    run_ranking(at, files=STANDARD, method="semantic")
    assert result_of(at).method == "semantic" and tfidf_result.method == "tfidf"
    assert result_of(at).score_keys != tfidf_result.score_keys


def test_stale_results_are_flagged_when_inputs_change(env):
    at = run_ranking(new_app(), files=STANDARD)
    assert not any("changed since" in w.value for w in at.warning)
    at.text_area(key="jd_text").set_value(JD + "\n- Terraform")
    at.run()
    assert any("changed since" in w.value for w in at.warning)


def test_clear_cache_button_removes_results(env):
    at = run_ranking(new_app(), files=STANDARD)
    assert "result" in at.session_state
    next(b for b in at.button if b.label == "Clear cached data now").click()
    at.run()
    assert "result" not in at.session_state


# =============================================================================
# Model failure, demo
# =============================================================================
def test_model_loading_failure_is_a_friendly_error_not_a_traceback(monkeypatch):
    st.cache_data.clear()
    st.cache_resource.clear()

    class Broken:
        model_name = "broken"

        def rank(self, *a, **k):
            raise SemanticModelError("simulated: no network and no local copy")

    monkeypatch.setattr(logic, "build_ranker", lambda: CandidateRanker(semantic_matcher=Broken()))
    at = run_ranking(new_app(), files=STANDARD)
    assert not at.exception
    assert any("language model could not be loaded" in e for e in error_texts(at))
    st.cache_data.clear()
    st.cache_resource.clear()


def test_programming_errors_are_not_hidden(monkeypatch):
    st.cache_data.clear()
    st.cache_resource.clear()

    class Buggy:
        model_name = "buggy"

        def rank(self, *a, **k):
            raise KeyError("a genuine bug")

    monkeypatch.setattr(logic, "build_ranker", lambda: CandidateRanker(semantic_matcher=Buggy()))
    at = run_ranking(new_app(), files=STANDARD)
    assert at.exception                                                   # shown, not swallowed
    st.cache_data.clear()
    st.cache_resource.clear()


def test_demo_scenario_fills_the_job_description(env):
    at = new_app()
    at.selectbox(key="demo_scenario").set_value("JD-DA")
    next(b for b in at.button if b.label == "Fill in this job description").click()
    at.run()
    assert at.text_area(key="jd_text").value == logic.demo_job_description("JD-DA")
    assert not at.exception


# =============================================================================
# Real pretrained model through the whole interface (skipped visibly if unavailable)
# =============================================================================
@pytest.fixture()
def real_env():
    st.cache_data.clear()
    st.cache_resource.clear()
    try:
        SemanticMatcher().model
    except SemanticModelError as exc:     # pragma: no cover
        pytest.skip(f"pretrained model not available: {exc}")
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def test_real_model_all_three_methods_through_the_interface(real_env):
    at = new_app()
    seen = {}
    for method in ("combined", "tfidf", "semantic"):
        at = run_ranking(at, files=STANDARD, method=method)
        assert not at.exception, method
        result = result_of(at)
        seen[method] = result
        assert result.method == method
        assert [r.candidate_id for r in result.rows][0] == "strong.pdf", method
        assert result.rows[-1].candidate_id == "unrelated.pdf", method
    combined = {r.candidate_id: r for r in seen["combined"].rows}
    for r in combined.values():                                        # formula preserved with the real model
        assert r.scores["combined_index"] == pytest.approx(
            combine_scores(r.scores["semantic_cosine"], r.scores["tfidf_cosine"],
                           r.scores["required_skill_coverage"]).index)
    # the single-method runs reproduce the components of the combined run
    for r in seen["tfidf"].rows:
        assert r.scores["tfidf_cosine"] == pytest.approx(combined[r.candidate_id].scores["tfidf_cosine"])
    for r in seen["semantic"].rows:
        assert r.scores["semantic_cosine"] == pytest.approx(combined[r.candidate_id].scores["semantic_cosine"], abs=1e-6)


def test_real_model_demo_scenario_end_to_end(real_env):
    import io
    import zipfile
    job_id = "JD-BE"
    with zipfile.ZipFile(io.BytesIO(logic.demo_resumes_zip(job_id))) as z:
        files = [(n, z.read(n)) for n in z.namelist()]
    at = new_app()
    at = run_ranking(at, jd=logic.demo_job_description(job_id), files=files, method="combined")
    assert not at.exception
    result = result_of(at)
    assert len(result.rows) == 9 and not result.excluded
    assert result.job.required_skills == ("Python", "SQL", "PostgreSQL", "Docker", "REST API")
    indexes = [r.scores["combined_index"] for r in result.rows]
    assert indexes == sorted(indexes, reverse=True)
