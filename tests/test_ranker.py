"""Tests for src/ranker.py (combined matching index and ranking).

Logic tests use a STUB semantic matcher with hand-set cosines (no model needed) together with the
real preprocessing, skill extractor and TF-IDF matcher. Integration tests at the end use the REAL
pretrained model and are skipped, visibly, if it cannot be loaded. All texts are synthetic.
"""

import io
import itertools
import math
import textwrap

import pandas as pd
import pytest

from src.pdf_parser import extract_many
from src.ranker import (CandidateRanker, ScoringConfig, combine_scores)
from src.semantic_matcher import SemanticMatcher, SemanticModelError, SemanticRanking, SemanticScore
from src.tfidf_matcher import TfidfMatcher, rank_resumes
from tests.test_pdf_parser import make_pdf

JD = "Requirements:\n- Python\n- SQL\n- Docker\nNice to have:\n- Kubernetes"


class StubSemantic:
    """Returns hand-set cosines per candidate id. Independent of the other candidates by design."""
    model_name = "stub-model"

    def __init__(self, scores=None, empty_ids=(), default=0.5, fail=False):
        self.scores, self.empty_ids, self.default, self.fail = scores or {}, set(empty_ids), default, fail
        self.calls = 0

    def rank(self, jd, pairs):
        self.calls += 1
        if self.fail:
            raise AssertionError("semantic matcher must not be called")
        rows = [SemanticScore(cid, 0.0 if cid in self.empty_ids else self.scores.get(cid, self.default),
                              0, cid in self.empty_ids) for cid, _ in pairs]
        return SemanticRanking(scores=rows, model_name=self.model_name, max_seq_length=256,
                               chunk_tokens=200, overlap_tokens=40, top_k=1)


def ranker(**stub_kwargs):
    return CandidateRanker(semantic_matcher=StubSemantic(**stub_kwargs))


# =============================================================================
# combine_scores: pure arithmetic
# =============================================================================
def test_formula_arithmetic_with_default_weights():
    c = combine_scores(0.8, 0.4, 0.5)
    assert c.index == pytest.approx(0.50 * 0.8 + 0.20 * 0.4 + 0.30 * 0.5)   # 0.63
    assert c.index == pytest.approx(0.63)
    assert c.weights_used == pytest.approx({"semantic": 0.5, "tfidf": 0.2, "skills": 0.3})
    assert (c.semantic_component, c.tfidf_component, c.skill_component) == (0.8, 0.4, 0.5)


def test_extremes():
    assert combine_scores(1.0, 1.0, 1.0).index == pytest.approx(1.0)
    assert combine_scores(0.0, 0.0, 0.0).index == 0.0


def test_negative_semantic_cosine_counts_as_zero_and_is_flagged():
    c = combine_scores(-0.3, 0.5, 1.0)
    assert c.semantic_component == 0.0 and c.semantic_clipped is True
    assert c.index == pytest.approx(0.20 * 0.5 + 0.30 * 1.0)
    assert combine_scores(0.0, 0.5, 1.0).semantic_clipped is False       # zero is not "clipped"
    assert combine_scores(0.3, 0.5, 1.0).semantic_clipped is False


def test_unavailable_skill_coverage_renormalises_weights_and_is_not_a_perfect_score():
    c = combine_scores(0.8, 0.4, None)
    assert c.skill_component is None
    assert c.index == pytest.approx((0.50 * 0.8 + 0.20 * 0.4) / 0.70)
    assert c.weights_used == pytest.approx({"semantic": 5 / 7, "tfidf": 2 / 7})
    assert sum(c.weights_used.values()) == pytest.approx(1.0)
    # treating "unknown" as 1.0 would give a different (higher) number: that is what we avoid
    assert c.index < combine_scores(0.8, 0.4, 1.0).index


def test_unavailable_tfidf_renormalises_over_the_rest():
    c = combine_scores(0.6, None, 0.5)
    assert c.index == pytest.approx((0.50 * 0.6 + 0.30 * 0.5) / 0.80)
    c2 = combine_scores(0.6, None, None)
    assert c2.index == pytest.approx(0.6) and c2.weights_used == {"semantic": 1.0}


def test_custom_weights():
    cfg = ScoringConfig(w_semantic=0.2, w_tfidf=0.3, w_skills=0.5)
    assert combine_scores(1.0, 0.0, 0.0, cfg).index == pytest.approx(0.2)
    assert combine_scores(0.0, 0.0, 1.0, cfg).index == pytest.approx(0.5)


@pytest.mark.parametrize("kwargs,error", [
    ({"w_semantic": 0.6}, ValueError),                                  # sum != 1
    ({"w_semantic": -0.1, "w_tfidf": 0.8, "w_skills": 0.3}, ValueError),
    ({"w_semantic": "0.5"}, TypeError), ({"w_semantic": float("nan")}, TypeError),
    ({"w_semantic": True, "w_tfidf": 0.0, "w_skills": 0.0}, TypeError),
])
def test_invalid_weights_rejected(kwargs, error):
    with pytest.raises(error):
        ScoringConfig(**kwargs)


@pytest.mark.parametrize("args", [(1.5, 0.1, 0.1), (-1.5, 0.1, 0.1), (0.5, -0.1, 0.1), (0.5, 1.1, 0.1),
                                  (0.5, 0.1, 1.2), (0.5, 0.1, -0.2), (float("nan"), 0.1, 0.1),
                                  (0.5, float("inf"), 0.1), ("0.5", 0.1, 0.1), (True, 0.1, 0.1)])
def test_out_of_range_or_invalid_components_are_rejected_not_silently_clipped(args):
    with pytest.raises(ValueError):
        combine_scores(*args)


def test_index_always_between_zero_and_one_on_a_grid():
    grid = [-1.0, -0.2, 0.0, 0.3, 0.7, 1.0]
    for s, t, k in itertools.product(grid, [0.0, 0.4, 1.0, None], [0.0, 0.5, 1.0, None]):
        assert 0.0 <= combine_scores(s, t, k).index <= 1.0


def test_monotonic_in_every_component():
    base = combine_scores(0.4, 0.2, 0.5).index
    assert combine_scores(0.5, 0.2, 0.5).index > base
    assert combine_scores(0.4, 0.3, 0.5).index > base
    assert combine_scores(0.4, 0.2, 0.6).index > base


# =============================================================================
# Required-skill coverage
# =============================================================================
def test_coverage_counts_matched_required_over_total_required():
    r = ranker().rank(JD, {"cv": "Python and SQL developer"})
    c = r.candidates[0]
    assert c.required_skills == ("Python", "SQL", "Docker")
    assert c.matched_required == ("Python", "SQL") and c.missing_required == ("Docker",)
    assert c.required_skill_coverage == pytest.approx(2 / 3)


def test_all_and_none_matched():
    r = ranker().rank(JD, {"all": "Python, SQL and Docker", "none": "Baking bread"})
    by = r.by_id()
    assert by["all"].required_skill_coverage == 1.0 and by["all"].missing_required == ()
    assert by["none"].required_skill_coverage == 0.0 and by["none"].matched_required == ()


@pytest.mark.parametrize("jd_wording,resume_wording,canonical", [
    ("scikit-learn", "sklearn", "Scikit-learn"), ("sklearn", "Scikit-Learn", "Scikit-learn"),
    ("PostgreSQL", "Postgres", "PostgreSQL"), ("Node.js", "NodeJS", "Node.js"),
    ("Microsoft SQL Server", "MSSQL", "Microsoft SQL Server"), ("Kubernetes", "K8s", "Kubernetes"),
])
def test_alias_normalisation_between_job_and_resume(jd_wording, resume_wording, canonical):
    r = ranker().rank(f"Requirements:\n- {jd_wording}", {"cv": f"Hands-on {resume_wording} experience"})
    c = r.candidates[0]
    assert c.required_skills == (canonical,)
    assert c.matched_required == (canonical,) and c.required_skill_coverage == 1.0


def test_repeated_mentions_in_the_resume_count_once():
    text = "Python Python python PYTHON. Used Python daily. SQL."
    c = ranker().rank("Requirements:\n- Python\n- SQL\n- Docker\n- Git", {"cv": text}).candidates[0]
    assert c.matched_required == ("Python", "SQL")
    assert c.required_skill_coverage == pytest.approx(2 / 4)       # not 6/4, not capped at 1


def test_repeated_mentions_in_the_job_description_make_one_required_skill():
    c = ranker().rank("Requirements:\n- Python\n- Python 3\n- python scripting\n- SQL", {"cv": "Python"}).candidates[0]
    assert c.required_skills == ("Python", "SQL")
    assert c.required_skill_coverage == pytest.approx(1 / 2)


def test_preferred_skills_are_not_counted_as_required():
    c = ranker().rank(JD, {"cv": "Docker and Kubernetes"}).candidates[0]
    assert c.required_skills == ("Python", "SQL", "Docker")
    assert c.matched_required == ("Docker",)
    assert c.required_skill_coverage == pytest.approx(1 / 3)
    assert c.matched_preferred == ("Kubernetes",) and c.missing_preferred == ()
    r2 = ranker().rank(JD, {"cv": "Python SQL Docker"}).candidates[0]
    assert r2.required_skill_coverage == 1.0 and r2.matched_preferred == () and r2.missing_preferred == ("Kubernetes",)


def test_skill_evidence_is_kept_for_matched_skills_only():
    c = ranker().rank(JD, {"cv": "Education: BTech\nBuilt services in Python with Docker.\nHobby: chess"}).candidates[0]
    assert set(c.skill_evidence) == {"Python", "Docker"}
    assert c.skill_evidence["Python"] == ("Built services in Python with Docker.",)


# =============================================================================
# Missing / ambiguous job-description skill information
# =============================================================================
def test_no_required_skills_means_unavailable_coverage_not_a_perfect_score():
    r = ranker(default=0.5).rank("Nice to have:\n- Docker\n- Kubernetes", {"cv": "Docker expert"})
    c = r.candidates[0]
    assert c.required_skills == () and c.required_skill_coverage is None
    assert any("only in preferred" in w for w in r.job.warnings)
    assert any("coverage unavailable" in w.lower() for w in c.warnings)
    expected = combine_scores(0.5, c.tfidf_similarity, None).index
    assert c.combined_index == pytest.approx(expected)
    assert set(c.weights_used) == {"semantic", "tfidf"}
    assert c.matched_preferred == ("Docker",)


def test_job_description_without_any_identifiable_skill():
    r = ranker().rank("We are looking for a friendly and motivated colleague for our team.", {"cv": "Python SQL"})
    c = r.candidates[0]
    assert r.job.required_skills == () and r.job.preferred_skills == ()
    assert c.required_skill_coverage is None
    assert any("No skills from the dictionary" in w for w in r.job.warnings)
    assert c.combined_index > 0                      # still ranked, from semantic + TF-IDF


def test_unstructured_job_description_marks_the_split_as_uncertain():
    r = ranker().rank("We need Python, SQL and Docker.", {"cv": "Python"})
    assert r.job.split_uncertain is True
    assert any("uncertain" in w for w in r.job.warnings)
    assert any("uncertain" in w for w in r.candidates[0].warnings)
    assert r.candidates[0].required_skill_coverage == pytest.approx(1 / 3)   # all treated as required


def test_structured_job_description_is_not_flagged_uncertain():
    r = ranker().rank(JD, {"cv": "Python"})
    assert r.job.split_uncertain is False and r.job.warnings == ()
    assert not any("uncertain" in w for w in r.candidates[0].warnings)


def test_resume_with_no_dictionary_skills_gets_a_cautious_warning():
    c = ranker().rank(JD, {"cv": "I am a pastry chef with many years of experience."}).candidates[0]
    assert any("No dictionary skills were found" in w for w in c.warnings)
    assert c.required_skill_coverage == 0.0


def test_job_description_with_only_stop_words_has_no_tfidf_but_still_works():
    r = ranker().rank("the and of to in", {"cv": "Python developer"})
    c = r.candidates[0]
    assert c.tfidf_similarity is None
    assert any("TF-IDF is not part" in w for w in c.warnings)
    assert set(c.weights_used) == {"semantic"}                 # no skills, no tfidf: semantic only
    assert c.combined_index == pytest.approx(0.5)               # stub cosine 0.5


# =============================================================================
# Components, empties, negatives, zeros
# =============================================================================
def test_ranker_arithmetic_matches_returned_components():
    r = ranker(scores={"cv": 0.6}).rank(JD, {"cv": "Python developer using SQL"})
    c = r.candidates[0]
    expected = 0.5 * c.semantic_component + 0.2 * c.tfidf_similarity + 0.3 * c.required_skill_coverage
    assert c.combined_index == pytest.approx(expected)
    assert c.semantic_similarity == 0.6 and c.required_skill_coverage == pytest.approx(2 / 3)


def test_zero_tfidf_similarity_is_kept_visible_and_used():
    r = ranker(scores={"cv": 0.4}).rank(JD, {"cv": "Pastry chef baking bread"})
    c = r.candidates[0]
    assert c.tfidf_similarity == 0.0
    assert c.combined_index == pytest.approx(0.5 * 0.4 + 0.2 * 0.0 + 0.3 * 0.0)


def test_negative_semantic_cosine_is_retained_raw_and_clipped_in_the_index():
    r = ranker(scores={"cv": -0.4}).rank(JD, {"cv": "Python and SQL and Docker"})
    c = r.candidates[0]
    assert c.semantic_similarity == -0.4 and c.semantic_component == 0.0 and c.semantic_clipped
    assert any("negative" in w for w in c.warnings)
    assert c.combined_index == pytest.approx(0.2 * c.tfidf_similarity + 0.3 * 1.0)


def test_empty_resume_is_flagged_and_ranked_last_even_when_scores_tie_at_zero():
    # No skills in the JD and no shared words -> both candidates have index exactly 0.0.
    r = ranker(scores={"real": -0.2}, empty_ids={"blank"}).rank(
        "friendly motivated colleague wanted", [("blank", ""), ("real", "completely different words here")])
    assert [c.candidate_id for c in r.candidates] == ["real", "blank"]
    by = r.by_id()
    assert by["real"].combined_index == by["blank"].combined_index == 0.0
    assert by["blank"].empty and not by["real"].empty
    assert by["real"].rank == 1 and by["blank"].rank == 2
    assert any("No usable text" in w for w in by["blank"].warnings)


def test_empty_resume_does_not_outrank_a_weak_nonempty_resume():
    r = ranker(scores={"weak": 0.01}, empty_ids={"blank"}).rank(JD, [("blank", ""), ("weak", "Baking bread")])
    assert [c.candidate_id for c in r.candidates] == ["weak", "blank"]


# =============================================================================
# Ranking order, ties, identifiers, independence from the batch
# =============================================================================
def test_ranked_best_first_and_ids_stay_attached_to_their_results():
    scores = {"low.pdf": 0.1, "high.pdf": 0.9, "mid.pdf": 0.5}
    texts = {"low.pdf": "Baking bread", "high.pdf": "Python SQL Docker developer", "mid.pdf": "Python only"}
    r = ranker(scores=scores).rank(JD, texts)
    assert [c.candidate_id for c in r.candidates] == ["high.pdf", "mid.pdf", "low.pdf"]
    assert [c.rank for c in r.candidates] == [1, 2, 3]
    by = r.by_id()
    assert by["high.pdf"].matched_required == ("Python", "SQL", "Docker")
    assert by["mid.pdf"].matched_required == ("Python",)
    assert by["low.pdf"].matched_required == ()
    assert by["high.pdf"].semantic_similarity == 0.9


def test_ties_share_a_rank_and_keep_input_order_deterministically():
    cands = [("first", "Python developer"), ("second", "Python developer"), ("other", "Baking bread")]
    r = ranker(scores={"first": 0.5, "second": 0.5, "other": 0.1}).rank(JD, cands)
    assert [c.candidate_id for c in r.candidates] == ["first", "second", "other"]
    assert [c.rank for c in r.candidates] == [1, 1, 3]
    assert r.candidates[0].combined_index == r.candidates[1].combined_index
    rev = ranker(scores={"first": 0.5, "second": 0.5, "other": 0.1}).rank(JD, list(reversed(cands)))
    assert [c.candidate_id for c in rev.candidates] == ["second", "first", "other"]


def test_repeated_runs_are_identical():
    cands = [("a", "Python SQL"), ("b", "Docker"), ("c", "Baking")]
    rk = ranker(scores={"a": 0.7, "b": 0.3, "c": 0.1})
    first = rk.rank(JD, cands)
    for _ in range(3):
        assert rk.rank(JD, cands).candidates == first.candidates


def test_adding_an_unrelated_candidate_does_not_change_existing_scores():
    base = {"A": "Python and SQL developer who ships Docker images", "B": "Python scripts"}
    scores = {"A": 0.7, "B": 0.4, "Z": 0.1}
    alone = ranker(scores=scores).rank(JD, base)
    crowded = ranker(scores=scores).rank(JD, {**base, "Z": "Pastry chef baking sourdough bread and cakes"})
    for cid in base:
        a, b = alone.by_id()[cid], crowded.by_id()[cid]
        assert a.combined_index == b.combined_index                     # exactly equal
        assert a.tfidf_similarity == b.tfidf_similarity
        assert a.required_skill_coverage == b.required_skill_coverage
    assert alone.tfidf_scope == "pairwise"


def test_batch_mode_tfidf_would_have_been_batch_dependent_which_is_why_pairwise_is_used():
    jd = "Python SQL Docker developer"
    a_text = "Python developer with Docker"
    alone = rank_resumes(jd, {"A": a_text}).score_by_id()["A"]
    crowded = rank_resumes(jd, {"A": a_text, "B": "SQL SQL", "C": "baking bread"}).score_by_id()["A"]
    assert alone != pytest.approx(crowded)         # the Phase 5 'batch' mode, not used by the ranker


def test_reference_scope_is_supported_and_batch_independent():
    corpus = ["Python SQL Docker developer", "Pastry chef baking", "Frontend React CSS", "Data analyst Tableau SQL"]
    rk = CandidateRanker(semantic_matcher=StubSemantic({"A": 0.7, "Z": 0.1}),
                         tfidf_matcher=TfidfMatcher().fit(corpus))
    alone = rk.rank(JD, {"A": "Python SQL Docker"})
    crowded = rk.rank(JD, {"A": "Python SQL Docker", "Z": "Baking bread"})
    assert alone.tfidf_scope == crowded.tfidf_scope == "reference"
    assert alone.by_id()["A"].combined_index == pytest.approx(crowded.by_id()["A"].combined_index)


def test_custom_weights_flow_through_the_ranker():
    rk = CandidateRanker(semantic_matcher=StubSemantic({"cv": 0.6}),
                         config=ScoringConfig(w_semantic=1.0, w_tfidf=0.0, w_skills=0.0))
    assert rk.rank(JD, {"cv": "Python"}).candidates[0].combined_index == pytest.approx(0.6)


# =============================================================================
# Input handling
# =============================================================================
def test_empty_candidate_collection_returns_empty_report_without_calling_the_model():
    stub = StubSemantic(fail=True)
    r = CandidateRanker(semantic_matcher=stub).rank(JD, {})
    assert r.candidates == [] and r.job.required_skills == ("Python", "SQL", "Docker") and stub.calls == 0


@pytest.mark.parametrize("jd", ["", "  \n "])
def test_blank_job_description_is_an_error(jd):
    with pytest.raises(ValueError, match="empty"):
        ranker().rank(jd, {"cv": "Python"})


def test_invalid_types_and_duplicate_ids():
    with pytest.raises(TypeError, match="job_description"):
        ranker().rank(None, {"cv": "Python"})
    with pytest.raises(TypeError):
        ranker().rank(JD, "not candidates")
    with pytest.raises(TypeError):
        ranker().rank(JD, {"cv": None})
    with pytest.raises(ValueError, match="duplicate candidate id"):
        ranker().rank(JD, [("a", "x"), ("a", "y")])


def test_report_documents_configuration():
    r = ranker().rank(JD, {"cv": "Python"})
    assert r.weights == {"semantic": 0.5, "tfidf": 0.2, "skills": 0.3}
    assert r.semantic_model == "stub-model" and "not a probability" in r.notes[0]


# =============================================================================
# Records / CSV, parser integration
# =============================================================================
def test_to_records_and_csv_round_trip():
    r = ranker(scores={"a.pdf": 0.8, "b.pdf": 0.2}).rank(JD, {"b.pdf": "Docker", "a.pdf": "Python, SQL, Docker"})
    records = r.to_records()
    assert [row["candidate_id"] for row in records] == ["a.pdf", "b.pdf"]
    assert records[0]["matched_required"] == "Python; SQL; Docker" and records[0]["missing_required"] == ""
    assert records[1]["missing_required"] == "Python; SQL"
    buffer = io.StringIO()
    pd.DataFrame(records).to_csv(buffer, index=False)
    back = pd.read_csv(io.StringIO(buffer.getvalue()), keep_default_na=False)
    assert list(back["candidate_id"]) == ["a.pdf", "b.pdf"]
    assert list(back["rank"]) == [1, 2]
    assert back.loc[0, "combined_index"] == pytest.approx(records[0]["combined_index"])


def test_unavailable_coverage_is_exported_as_blank_not_one():
    r = ranker().rank("friendly colleague wanted", {"cv": "Python"})
    assert r.to_records()[0]["required_skill_coverage"] is None


def pdf_of(text):
    """A one-page PDF with the text wrapped into short lines (the parser needs >= 100 characters)."""
    return make_pdf([textwrap.wrap(text, 80)])


def test_rank_extractions_scores_usable_files_and_lists_the_rest_with_reasons():
    good = pdf_of("Asha - Python developer. Skills: Python, SQL, Docker, Git, Machine Learning, Flask. "
                  "Built internal tools and reporting dashboards for a logistics team.")
    blank = make_pdf()
    results = extract_many([("good.pdf", good), ("copy.pdf", good), ("blank.pdf", blank), ("bad.pdf", b"nope")])
    r = ranker().rank_extractions(JD, results)
    assert [c.candidate_id for c in r.candidates] == ["good.pdf"]
    excluded = {e.file_id: e for e in r.excluded}
    assert set(excluded) == {"copy.pdf", "blank.pdf", "bad.pdf"}
    assert excluded["copy.pdf"].reason == "duplicate" and "good.pdf" in excluded["copy.pdf"].message
    assert excluded["blank.pdf"].reason == "not_scored" and excluded["bad.pdf"].reason == "not_scored"
    assert all("manual" in e.message for e in r.excluded if e.reason == "not_scored")


# =============================================================================
# Integration with the REAL pretrained model (skipped visibly if unavailable)
# =============================================================================
@pytest.fixture(scope="module")
def real_ranker():
    matcher = SemanticMatcher()
    try:
        matcher.model
    except SemanticModelError as exc:     # pragma: no cover
        pytest.skip(f"pretrained model not available: {exc}")
    return CandidateRanker(semantic_matcher=matcher)


REAL_JD = ("Backend engineer.\nRequirements:\n- Python\n- SQL and PostgreSQL\n- Docker\n- REST APIs\n- CI/CD\n"
           "Nice to have:\n- Kubernetes\n")
REAL = {
    "A.pdf": "Python developer. REST APIs with Flask, PostgreSQL and SQL tuning. Docker images and CI/CD pipelines.",
    "B.pdf": "Python scripts and SQL reports for finance; some Docker usage. Used sklearn.",
    "C.pdf": "Frontend developer. React, JavaScript, CSS, Figma design systems.",
    "D.pdf": "Pastry chef. Baked bread, croissants and wedding cakes.",
    "E.pdf": "",
}


def test_real_end_to_end_ranking(real_ranker):
    r = real_ranker.rank(REAL_JD, REAL)
    assert [c.candidate_id for c in r.candidates][:2] == ["A.pdf", "B.pdf"]
    assert r.candidates[-1].candidate_id == "E.pdf" and r.candidates[-1].empty
    by = r.by_id()
    assert by["A.pdf"].required_skill_coverage == 1.0 and by["A.pdf"].missing_required == ()
    assert by["B.pdf"].matched_required == ("Python", "SQL", "Docker")
    assert set(by["B.pdf"].missing_required) == {"PostgreSQL", "REST API", "CI/CD"}
    assert by["D.pdf"].required_skill_coverage == 0.0 and by["D.pdf"].tfidf_similarity == 0.0
    assert by["A.pdf"].semantic_evidence
    for c in r.candidates:
        assert 0.0 <= c.combined_index <= 1.0 and math.isfinite(c.semantic_similarity)
        expected = combine_scores(c.semantic_similarity, c.tfidf_similarity, c.required_skill_coverage).index
        assert c.combined_index == pytest.approx(expected)


def test_real_adding_an_unrelated_candidate_changes_nothing(real_ranker):
    base = {k: REAL[k] for k in ("A.pdf", "B.pdf")}
    alone = real_ranker.rank(REAL_JD, base)
    crowded = real_ranker.rank(REAL_JD, {**base, "D.pdf": REAL["D.pdf"], "C.pdf": REAL["C.pdf"]})
    for cid in base:
        a, b = alone.by_id()[cid], crowded.by_id()[cid]
        assert a.tfidf_similarity == b.tfidf_similarity                       # pairwise TF-IDF: exact
        assert a.semantic_similarity == pytest.approx(b.semantic_similarity, abs=1e-6)
        assert a.required_skill_coverage == b.required_skill_coverage
        assert a.combined_index == pytest.approx(b.combined_index, abs=1e-6)


def test_real_pipeline_from_pdfs(real_ranker):
    a_text = REAL["A.pdf"] + " Worked in a small team on internal tooling."
    a_bytes = pdf_of(a_text)     # the same bytes twice: duplicates are detected by file hash, and two
    pdfs = [("a.pdf", a_bytes), ("dup.pdf", a_bytes),   # separately generated PDFs differ in their bytes
            ("blank.pdf", make_pdf()),
            ("chef.pdf", pdf_of(REAL["D.pdf"] + " Ten years in bakeries and catering for large events. "
                                                 "Trained apprentices and designed seasonal dessert menus."))]
    r = real_ranker.rank_extractions(REAL_JD, extract_many(pdfs))
    assert [c.candidate_id for c in r.candidates] == ["a.pdf", "chef.pdf"]
    assert {e.file_id for e in r.excluded} == {"dup.pdf", "blank.pdf"}
