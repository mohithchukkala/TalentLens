"""Tests for src/tfidf_matcher.py (Method A, the TF-IDF baseline).

All texts are short synthetic strings written for these tests.
"""

import math

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import preprocess_for_tfidf
from src.tfidf_matcher import TfidfMatcher, make_vectorizer, rank_resumes

JD = "Backend engineer with Python, SQL, PostgreSQL and Docker. Build REST APIs."
GOOD = "Python developer. Built REST APIs with PostgreSQL and SQL, deployed with Docker."
PARTIAL = "Python scripting and data cleaning with SQL reports."
UNRELATED = "Pastry chef. Baked bread, croissants and wedding cakes for a busy bakery."


# ---- ranking behaviour ------------------------------------------------------
def test_relevant_resume_ranks_above_unrelated():
    r = rank_resumes(JD, {"good": GOOD, "unrelated": UNRELATED})
    assert [s.candidate_id for s in r.scores] == ["good", "unrelated"]
    assert r.scores[0].score > r.scores[1].score


def test_more_overlap_scores_higher_than_less_overlap():
    r = rank_resumes(JD, {"unrelated": UNRELATED, "partial": PARTIAL, "good": GOOD})
    assert [s.candidate_id for s in r.scores] == ["good", "partial", "unrelated"]


def test_zero_overlap_scores_exactly_zero():
    r = rank_resumes(JD, {"good": GOOD, "unrelated": UNRELATED})
    assert r.score_by_id()["unrelated"] == 0.0
    assert r.scores[-1].top_terms == ()


def test_results_sorted_descending_and_cover_every_resume():
    cands = {f"c{i}": t for i, t in enumerate([UNRELATED, PARTIAL, GOOD, "SQL only", "Docker Python"])}
    r = rank_resumes(JD, cands)
    scores = [s.score for s in r.scores]
    assert scores == sorted(scores, reverse=True)
    assert {s.candidate_id for s in r.scores} == set(cands)  # zero scorers are included
    assert [s.rank for s in r.scores][0] == 1


def test_identifiers_stay_attached_to_their_own_scores():
    cands = [("zeta.pdf", UNRELATED), ("alpha.pdf", GOOD), ("mid.pdf", PARTIAL)]
    r = rank_resumes(JD, cands)
    by_id = r.score_by_id()
    # Recompute each candidate alone (different input order, other candidates removed):
    solo_good = rank_resumes(JD, [("alpha.pdf", GOOD)]).score_by_id()["alpha.pdf"]
    assert by_id["alpha.pdf"] > by_id["mid.pdf"] > by_id["zeta.pdf"] == 0.0
    assert solo_good > 0.0
    assert r.scores[0].candidate_id == "alpha.pdf"


def test_accepts_mapping_and_pair_list_equally():
    a = rank_resumes(JD, {"x": GOOD, "y": UNRELATED})
    b = rank_resumes(JD, [("x", GOOD), ("y", UNRELATED)])
    assert a.score_by_id() == b.score_by_id()


def test_score_is_cosine_in_unit_interval_and_identical_text_scores_one():
    r = rank_resumes(JD, {"same": JD, "good": GOOD, "unrelated": UNRELATED})
    for s in r.scores:
        assert 0.0 <= s.score <= 1.0
    assert r.score_by_id()["same"] == pytest.approx(1.0)


def test_top_term_contributions_add_up_to_the_score():
    # Only the top 5 terms are stored, so use a resume sharing <= 5 terms with the JD.
    r = rank_resumes("python sql", {"a": "python sql"})
    s = r.scores[0]
    assert sum(w for _, w in s.top_terms) == pytest.approx(s.score)
    assert {t for t, _ in s.top_terms} == {"python", "sql", "python sql"}


# ---- duplicates -------------------------------------------------------------
def test_duplicate_text_gets_equal_score_and_shared_rank():
    r = rank_resumes(JD, [("first", GOOD), ("copy", GOOD), ("other", UNRELATED)])
    by_id = {s.candidate_id: s for s in r.scores}
    assert by_id["first"].score == by_id["copy"].score
    assert by_id["first"].rank == by_id["copy"].rank == 1
    assert by_id["other"].rank == 3                      # competition ranking: 1, 1, 3
    assert [s.candidate_id for s in r.scores][:2] == ["first", "copy"]  # ties keep input order


def test_duplicate_resume_does_not_change_other_scores():
    without = rank_resumes(JD, {"good": GOOD, "partial": PARTIAL}).score_by_id()
    with_dupes = rank_resumes(JD, [("good", GOOD), ("partial", PARTIAL),
                                   ("good2", GOOD), ("good3", GOOD)]).score_by_id()
    assert with_dupes["good"] == pytest.approx(without["good"])
    assert with_dupes["partial"] == pytest.approx(without["partial"])


def test_duplicate_candidate_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate candidate id"):
        rank_resumes(JD, [("a", GOOD), ("a", PARTIAL)])


# ---- empty and degenerate inputs --------------------------------------------
@pytest.mark.parametrize("jd", ["", "   \n\t", "the and of to in", "!!! ... ---", "2019 2023 2024"])
def test_unusable_job_description_raises_clear_error(jd):
    with pytest.raises(ValueError, match="no usable terms"):
        rank_resumes(jd, {"a": GOOD})


def test_empty_resume_collection_returns_empty_ranking():
    for empty in ({}, [], ()):
        r = rank_resumes(JD, empty)
        assert r.scores == [] and r.warnings == []


def test_empty_resume_text_scores_zero_with_warning():
    r = rank_resumes(JD, {"blank": "", "good": GOOD})
    blank = next(s for s in r.scores if s.candidate_id == "blank")
    assert blank.score == 0.0 and blank.token_count == 0
    assert any("blank" in w for w in r.warnings)
    assert r.scores[0].candidate_id == "good"


@pytest.mark.parametrize("text", ["the and of to in a", "!!! ??? ...", "  \n  ", "2019 - 2023 12345"])
def test_stop_word_punctuation_or_number_only_resume_scores_zero(text):
    r = rank_resumes(JD, {"junk": text, "good": GOOD})
    junk = next(s for s in r.scores if s.candidate_id == "junk")
    assert junk.score == 0.0 and junk.token_count == 0


def test_all_resumes_empty_does_not_crash():
    r = rank_resumes(JD, {"a": "", "b": "the and of"})
    assert [s.score for s in r.scores] == [0.0, 0.0]
    assert [s.rank for s in r.scores] == [1, 1]
    assert len(r.vocabulary) > 0  # vocabulary comes from the job description


def test_single_resume_works():
    r = rank_resumes(JD, {"only": GOOD})
    assert len(r.scores) == 1 and 0 < r.scores[0].score < 1


# ---- technical terms --------------------------------------------------------
TECH = "C++ C# .NET Node.js CI/CD scikit-learn"


def test_technical_terms_become_single_vocabulary_entries():
    r = rank_resumes(TECH, {"a": TECH})
    vocab = set(r.vocabulary)
    for term in ("c++", "c#", ".net", "node.js", "ci/cd", "scikit-learn"):
        assert term in vocab, sorted(vocab)
    for broken in ("c", "js", "node", "cd", "scikit", "learn", "net"):
        assert broken not in vocab


def test_default_sklearn_settings_would_have_destroyed_these_terms():
    # Evidence for the configuration choice: same text, sklearn defaults on raw text.
    default_vocab = set(TfidfVectorizer().fit([TECH]).get_feature_names_out())
    assert "c++" not in default_vocab and ".net" not in default_vocab
    assert "node" in default_vocab and "js" in default_vocab  # split apart


def test_each_technical_term_matches_only_its_own_kind():
    # 'c' language must not match 'c++' or 'c#'
    r = rank_resumes("C++ developer", {"cpp": "Built C++ services", "csharp": "Built C# services",
                                       "dotnet": "ASP.NET engineer"})
    by_id = r.score_by_id()
    assert by_id["cpp"] > 0
    assert by_id["csharp"] == 0.0 and by_id["dotnet"] == 0.0
    r2 = rank_resumes("Node.js and CI/CD", {"a": "node.js pipelines with CI/CD", "b": "plain JS and node"})
    assert r2.scores[0].candidate_id == "a" and r2.score_by_id()["b"] == 0.0


def test_front_back_full_stack_survive():
    r = rank_resumes("full stack, front end and back end", {"a": "full stack engineer"})
    assert {"full", "stack", "front", "end", "back", "full stack"} <= set(r.vocabulary)


def test_aliases_are_not_merged_in_the_baseline():
    # Known weakness of keyword matching, deliberately kept: sklearn != scikit-learn.
    r = rank_resumes("scikit-learn models", {"alias": "sklearn models", "exact": "scikit-learn models"})
    by_id = r.score_by_id()
    assert by_id["exact"] > by_id["alias"] > 0  # alias only shares the word 'models'


def test_uses_phase4_preprocessing_not_a_copy():
    # Contact details must be stripped by src.preprocessing before vectorising.
    r = rank_resumes("Python developer asha@example.com", {"x": "python developer"})
    assert "asha@example.com" not in r.vocabulary
    assert r.score_by_id()["x"] == pytest.approx(1.0)
    expected = preprocess_for_tfidf("Python developer asha@example.com")
    assert expected == "python developer"


# ---- vectorizer configuration ----------------------------------------------
def test_vectorizer_configuration_is_as_documented():
    v = make_vectorizer()
    assert v.lowercase is False
    assert v.token_pattern == r"\S+"
    assert v.ngram_range == (1, 2)
    assert v.sublinear_tf is True and v.use_idf is True and v.smooth_idf is True
    assert v.norm == "l2" and v.min_df == 1 and v.max_df == 1.0


def test_idf_and_tfidf_values_match_the_documented_formulas():
    # Hand calculation. Fit docs (batch mode): JD "python sql", resume "python docker".
    # n = 2.  df(python)=2, df(sql)=1, df(docker)=1, ngrams are unique to one document.
    import numpy as np
    v = make_vectorizer(ngram_range=(1, 1))
    v.fit(["python sql", "python docker"])
    idf = dict(zip(v.get_feature_names_out(), v.idf_))
    assert idf["python"] == pytest.approx(math.log((1 + 2) / (1 + 2)) + 1)   # = 1.0
    assert idf["sql"] == pytest.approx(math.log((1 + 2) / (1 + 1)) + 1)      # ~ 1.405465
    jd = v.transform(["python sql"]).toarray()[0]
    names = list(v.get_feature_names_out())
    raw = np.array([1.0 * 1.0, 1.0 * idf["sql"]])  # tf=1 for both terms
    expected = raw / np.linalg.norm(raw)
    assert jd[names.index("python")] == pytest.approx(expected[0])
    assert jd[names.index("sql")] == pytest.approx(expected[1])
    assert np.linalg.norm(jd) == pytest.approx(1.0)
    # The resume vector is [python 1.0, docker idf_sql] / norm (docker has the same idf as sql),
    # so the only shared term is 'python': cosine = 1 / (1 + idf_sql^2).
    r = TfidfMatcher(ngram_range=(1, 1)).rank("python sql", {"a": "python docker"})
    assert r.scores[0].score == pytest.approx(1.0 / (1.0 + idf["sql"] ** 2))
    assert r.scores[0].score == pytest.approx(0.336089, abs=1e-5)  # hand-computed value


def test_sublinear_tf_dampens_keyword_stuffing():
    stuffed = "python " * 50 + "baking bread"
    r = rank_resumes("python sql", {"stuffed": stuffed, "plain": "python sql"})
    assert r.score_by_id()["plain"] > r.score_by_id()["stuffed"]


# ---- determinism and order independence -------------------------------------
def test_repeated_runs_give_identical_results():
    cands = [("a", GOOD), ("b", PARTIAL), ("c", UNRELATED)]
    first = rank_resumes(JD, cands)
    for _ in range(5):
        again = rank_resumes(JD, cands)
        assert again == first


def test_scores_do_not_depend_on_input_order():
    cands = [("a", GOOD), ("b", PARTIAL), ("c", UNRELATED), ("d", "SQL Docker")]
    forward = rank_resumes(JD, cands).score_by_id()
    backward = rank_resumes(JD, list(reversed(cands))).score_by_id()
    for key in forward:
        assert forward[key] == pytest.approx(backward[key])


def test_matcher_keeps_no_state_between_batch_calls():
    m = TfidfMatcher()
    a = m.rank(JD, {"good": GOOD})
    m.rank(JD, {"x": UNRELATED, "y": PARTIAL})
    b = m.rank(JD, {"good": GOOD})
    assert a == b


# ---- batch vs reference mode ------------------------------------------------
def test_fit_scope_is_reported():
    assert rank_resumes(JD, {"a": GOOD}).fit_scope == "batch"
    assert rank_resumes(JD, {"a": GOOD}, reference_texts=[GOOD, PARTIAL, UNRELATED]).fit_scope == "reference"


def test_reference_mode_scores_do_not_depend_on_the_batch():
    ref = [GOOD, PARTIAL, UNRELATED, "Java Spring developer", "Python Django SQL"]
    m = TfidfMatcher().fit(ref)
    alone = m.rank(JD, {"good": GOOD}).score_by_id()["good"]
    in_batch = m.rank(JD, {"good": GOOD, "p": PARTIAL, "u": UNRELATED}).score_by_id()["good"]
    assert alone == pytest.approx(in_batch)


def test_batch_mode_idf_depends_on_the_batch_documented_limitation():
    # Same resume, same JD, different companions: score may differ in batch mode.
    alone = rank_resumes(JD, {"good": GOOD}).score_by_id()["good"]
    crowd = rank_resumes(JD, {"good": GOOD, "p": PARTIAL, "u": UNRELATED, "q": "SQL SQL Docker"}).score_by_id()["good"]
    assert alone != pytest.approx(crowd)


def test_reference_mode_ignores_terms_outside_reference_vocabulary_with_warning():
    m = TfidfMatcher().fit(["bread cakes pastry"])
    r = m.rank(JD, {"a": GOOD})
    assert r.scores[0].score == 0.0
    assert any("reference vocabulary" in w for w in r.warnings)


def test_unusable_reference_corpus_is_rejected():
    with pytest.raises(ValueError, match="no usable text"):
        TfidfMatcher().fit(["", "the and of"])
    with pytest.raises(TypeError):
        TfidfMatcher().fit("not a list of texts")
    with pytest.raises(TypeError):
        TfidfMatcher().fit([123])


# ---- invalid input types ----------------------------------------------------
@pytest.mark.parametrize("jd", [None, 123, ["python"], b"python"])
def test_non_string_job_description_raises_type_error(jd):
    with pytest.raises(TypeError, match="job_description"):
        rank_resumes(jd, {"a": GOOD})


@pytest.mark.parametrize("bad", ["a plain string", b"bytes", 42, None])
def test_non_collection_candidates_raise_type_error(bad):
    with pytest.raises(TypeError):
        rank_resumes(JD, bad)


def test_bad_candidate_items_raise_type_error():
    with pytest.raises(TypeError, match="pair"):
        rank_resumes(JD, ["just a string"])
    with pytest.raises(TypeError, match="pair"):
        rank_resumes(JD, [("a", "b", "c")])
    with pytest.raises(TypeError, match="id must be a string"):
        rank_resumes(JD, [(1, GOOD)])
    with pytest.raises(TypeError, match="must be a string"):
        rank_resumes(JD, {"a": None})
    with pytest.raises(TypeError, match="must be a string"):
        rank_resumes(JD, {"a": 5})
