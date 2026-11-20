"""Tests for src/app_logic.py (everything behind the Streamlit app that does not need Streamlit).

A fake embedding model with content-dependent vectors stands in for the pretrained model so the
three methods can be checked quickly and exactly. The real model is used in tests/test_app.py.
"""

import csv
import io
import json
import zipfile

import pytest

from src import app_logic as logic
from src.pdf_parser import Status, extract_many, extract_text_from_bytes
from src.ranker import CandidateRanker, ScoringConfig, combine_scores
from src.semantic_matcher import SemanticMatcher
from tests.test_semantic_matcher import BagOfWordsModel, words

JD = "Requirements:\n- Python\n- SQL\n- Docker\nNice to have:\n- Kubernetes"
STRONG = ("Python developer. Built services with Python and SQL, shipped with Docker and Kubernetes on the cloud. "
          "Wrote reliable tests and mentored juniors across the team over four years of professional work.")
PARTIAL = ("Analyst who writes SQL reports daily and some Python scripts for finance dashboards. Worked with business "
           "teams on weekly reporting and data quality checks for three years.")
UNRELATED = ("Pastry chef with ten years in bakeries and catering. Baked bread, croissants and wedding cakes, trained "
             "apprentices and designed seasonal dessert menus for large events.")


def fake_ranker(config=None):
    return CandidateRanker(semantic_matcher=SemanticMatcher(model=BagOfWordsModel(), token_counter=words),
                           config=config)


def pdfs(**named_texts):
    return [(name, logic.text_to_pdf_bytes(text)) for name, text in named_texts.items()]


def extractions_of(**named_texts):
    return logic.extract_files(pdfs(**named_texts))


def by_id(result):
    return {r.candidate_id: r for r in result.rows}


# =============================================================================
# Input validation
# =============================================================================
def test_validate_inputs_messages():
    assert logic.validate_inputs(JD, 2) == []
    assert logic.validate_inputs("", 2) == ["Please enter a job description."]
    assert logic.validate_inputs("   \n", 2) == ["Please enter a job description."]
    assert logic.validate_inputs(JD, 0) == ["Please upload at least one PDF resume."]
    both = logic.validate_inputs("", 0)
    assert len(both) == 2
    assert logic.validate_inputs(JD, 1, "nonsense")[0].startswith("Unknown matching method")


def test_tfidf_needs_usable_words_in_the_job_description():
    msg = logic.validate_inputs("the and of 2024", 1, logic.METHOD_TFIDF)
    assert msg and "usable words" in msg[0]
    assert logic.validate_inputs("the and of 2024", 1, logic.METHOD_SEMANTIC) == []   # semantic can still try


def test_match_refuses_invalid_inputs_instead_of_crashing_later():
    with pytest.raises(ValueError, match="job description"):
        logic.match(logic.METHOD_COMBINED, "", extractions_of(a=STRONG), fake_ranker())
    with pytest.raises(ValueError, match="upload"):
        logic.match(logic.METHOD_COMBINED, JD, [], fake_ranker())


# =============================================================================
# Extraction status table
# =============================================================================
def test_status_rows_explain_every_kind_of_file():
    good = logic.text_to_pdf_bytes(STRONG)
    scanned = _image_only_pdf()
    ex = logic.extract_files([("good.pdf", good), ("same.pdf", good), ("bad.pdf", b"not a pdf"),
                              ("empty.pdf", b""), ("scan.pdf", scanned), ("blank.pdf", _blank_pdf())])
    rows = {r["File"]: r for r in logic.status_rows(ex)}
    assert rows["good.pdf"]["Used in ranking"] == "Yes"
    assert rows["same.pdf"]["Used in ranking"] == "No (duplicate)" and "good.pdf" in rows["same.pdf"]["Note"]
    assert rows["bad.pdf"]["Status"] == "Unreadable or corrupted" and rows["bad.pdf"]["Used in ranking"] == "No"
    assert rows["empty.pdf"]["Status"] == "Empty file (0 bytes)"
    assert rows["scan.pdf"]["Status"] == "Probably scanned: OCR needed" and "OCR" in rows["scan.pdf"]["Note"]
    assert rows["blank.pdf"]["Status"] == "No usable text"
    assert all("manual" in r["Note"] for r in rows.values() if r["Used in ranking"] == "No")


def test_oversized_file_uses_the_parsers_limit(monkeypatch):
    from src import pdf_parser
    monkeypatch.setattr(pdf_parser, "MAX_FILE_BYTES", 50)
    ex = logic.extract_files([("big.pdf", logic.text_to_pdf_bytes(STRONG))])
    assert ex[0].status is Status.TOO_LARGE
    assert logic.status_rows(ex)[0]["Status"] == "Too large"


def _blank_pdf():
    import pymupdf
    doc = pymupdf.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def _image_only_pdf():
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60), False)
    pix.clear_with(200)
    page.insert_image(pymupdf.Rect(100, 100, 300, 300), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


# =============================================================================
# The three methods return correctly associated results
# =============================================================================
def test_combined_matches_the_phase7_formula_and_keeps_candidates_apart():
    ex = extractions_of(strong=STRONG, partial=PARTIAL, unrelated=UNRELATED)
    result = logic.match(logic.METHOD_COMBINED, JD, ex, fake_ranker())
    assert result.method == logic.METHOD_COMBINED and result.weights == {"semantic": 0.5, "tfidf": 0.2, "skills": 0.3}
    rows = by_id(result)
    assert set(rows) == {"strong", "partial", "unrelated"}
    for r in rows.values():
        s, t, k = r.scores["semantic_cosine"], r.scores["tfidf_cosine"], r.scores["required_skill_coverage"]
        assert r.scores["combined_index"] == pytest.approx(0.50 * max(0, s) + 0.20 * t + 0.30 * k)
        assert r.scores["combined_index"] == pytest.approx(combine_scores(s, t, k).index)
    assert rows["strong"].matched_required == ("Python", "SQL", "Docker") and rows["strong"].missing_required == ()
    assert rows["partial"].matched_required == ("Python", "SQL") and rows["partial"].missing_required == ("Docker",)
    assert rows["unrelated"].matched_required == () and rows["strong"].matched_preferred == ("Kubernetes",)
    assert [r.candidate_id for r in result.rows] == ["strong", "partial", "unrelated"]
    assert [r.rank for r in result.rows] == [1, 2, 3]
    assert result.score_keys == logic.METHOD_SCORES[logic.METHOD_COMBINED]


def test_combined_keeps_a_custom_configuration_untouched_by_the_app_layer():
    ex = extractions_of(a=STRONG)
    r = logic.match(logic.METHOD_COMBINED, JD, ex, fake_ranker(ScoringConfig(1.0, 0.0, 0.0)))
    row = r.rows[0]
    assert row.scores["combined_index"] == pytest.approx(max(0, row.scores["semantic_cosine"]))


def test_tfidf_method_equals_the_matcher_pairwise_and_calculates_nothing_else():
    ex = extractions_of(strong=STRONG, partial=PARTIAL, unrelated=UNRELATED)
    ranker = fake_ranker()
    result = logic.match(logic.METHOD_TFIDF, JD, ex, ranker)
    assert result.score_keys == ("tfidf_cosine",)
    for row in result.rows:
        assert set(row.scores) == {"tfidf_cosine"}                      # no invented semantic / coverage values
        assert row.matched_required is None and row.required_total is None
        direct = ranker.tfidf.rank(JD, [(row.candidate_id, next(e.text for e in ex if e.file_id == row.candidate_id))])
        assert row.scores["tfidf_cosine"] == direct.scores[0].score
    assert [r.candidate_id for r in result.rows][0] == "strong"
    assert by_id(result)["unrelated"].scores["tfidf_cosine"] == 0.0


def test_tfidf_scores_do_not_depend_on_the_other_uploads():
    alone = logic.match(logic.METHOD_TFIDF, JD, extractions_of(strong=STRONG), fake_ranker())
    crowd = logic.match(logic.METHOD_TFIDF, JD, extractions_of(strong=STRONG, partial=PARTIAL, unrelated=UNRELATED),
                        fake_ranker())
    assert by_id(alone)["strong"].scores["tfidf_cosine"] == by_id(crowd)["strong"].scores["tfidf_cosine"]


def test_semantic_method_returns_only_the_raw_cosine():
    ex = extractions_of(strong=STRONG, unrelated=UNRELATED)
    result = logic.match(logic.METHOD_SEMANTIC, JD, ex, fake_ranker())
    assert result.score_keys == ("semantic_cosine",)
    rows = by_id(result)
    assert set(rows["strong"].scores) == {"semantic_cosine"} and rows["strong"].matched_required is None
    assert rows["strong"].scores["semantic_cosine"] > rows["unrelated"].scores["semantic_cosine"]
    assert [r.candidate_id for r in result.rows] == ["strong", "unrelated"]
    assert rows["strong"].text_evidence and rows["strong"].text_evidence[0][0] == "Best-matching passage"


def test_every_method_keeps_the_candidate_to_score_association_when_input_order_changes():
    for method in logic.METHOD_LABELS:
        a = logic.match(method, JD, extractions_of(strong=STRONG, partial=PARTIAL, unrelated=UNRELATED), fake_ranker())
        b = logic.match(method, JD, extractions_of(unrelated=UNRELATED, strong=STRONG, partial=PARTIAL), fake_ranker())
        key = logic.METHOD_SCORES[method][0]
        for cid in ("strong", "partial", "unrelated"):
            assert by_id(a)[cid].scores[key] == pytest.approx(by_id(b)[cid].scores[key])


def test_tfidf_ties_share_a_rank_and_keep_input_order():
    ex = extractions_of(first=STRONG, second=STRONG + " ", other=UNRELATED)
    # identical text under different bytes: not byte-duplicates, but identical scores
    result = logic.match(logic.METHOD_TFIDF, JD, ex, fake_ranker())
    rows = result.rows
    assert [r.candidate_id for r in rows][:2] == ["first", "second"]
    assert rows[0].scores == rows[1].scores and rows[0].rank == rows[1].rank == 1 and rows[2].rank == 3


# =============================================================================
# Blank resumes, excluded files, nothing usable
# =============================================================================
def test_excluded_files_are_listed_and_not_scored_for_every_method():
    good = logic.text_to_pdf_bytes(STRONG)
    ex = logic.extract_files([("good.pdf", good), ("copy.pdf", good), ("bad.pdf", b"nope"), ("blank.pdf", _blank_pdf())])
    for method in logic.METHOD_LABELS:
        result = logic.match(method, JD, ex, fake_ranker())
        assert [r.candidate_id for r in result.rows] == ["good.pdf"]
        reasons = {e.file_id: e.reason for e in result.excluded}
        assert reasons == {"copy.pdf": "duplicate", "bad.pdf": "not_scored", "blank.pdf": "not_scored"}


def test_nothing_usable_returns_an_empty_ranking_with_a_warning_and_no_model_call():
    class Exploding:
        model_name = "none"

        def rank(self, *a, **k):
            raise AssertionError("the model must not be used when nothing is usable")

    ranker = CandidateRanker(semantic_matcher=Exploding())
    ex = logic.extract_files([("bad.pdf", b"nope"), ("empty.pdf", b"")])
    for method in logic.METHOD_LABELS:
        result = logic.match(method, JD, ex, ranker)
        assert result.rows == [] and len(result.excluded) == 2
        assert "None of the uploaded files had usable text" in result.warnings[0]
        assert logic.results_table(result) == []


def test_resume_with_only_stop_words_is_flagged_and_ranked_last_in_tfidf_and_semantic():
    # extraction found text (>= 100 characters) but nothing usable remains after preprocessing
    filler = "the and of to in a is it " * 8
    ex = extractions_of(real=STRONG, filler=filler)
    tf = logic.match(logic.METHOD_TFIDF, JD, ex, fake_ranker())
    assert [r.candidate_id for r in tf.rows] == ["real", "filler"]
    assert tf.rows[-1].empty and tf.rows[-1].scores["tfidf_cosine"] == 0.0
    assert any("No usable words" in w for w in tf.rows[-1].warnings)
    combined = logic.match(logic.METHOD_COMBINED, JD, ex, fake_ranker())
    assert [r.candidate_id for r in combined.rows][0] == "real"


# =============================================================================
# Table and CSV
# =============================================================================
def read_csv(text):
    return list(csv.DictReader(io.StringIO(text)))


def test_results_table_has_only_the_columns_each_method_calculated():
    ex = extractions_of(strong=STRONG, partial=PARTIAL)
    cols = {m: list(logic.results_table(logic.match(m, JD, ex, fake_ranker()))[0]) for m in logic.METHOD_LABELS}
    assert cols[logic.METHOD_COMBINED] == ["Rank", "Candidate", "Combined matching index",
                                           "Semantic similarity (raw cosine)", "TF-IDF similarity (cosine)",
                                           "Required-skill coverage", "Required skills matched"]
    assert cols[logic.METHOD_TFIDF] == ["Rank", "Candidate", "TF-IDF similarity (cosine)"]
    assert cols[logic.METHOD_SEMANTIC] == ["Rank", "Candidate", "Semantic similarity (raw cosine)"]
    table = logic.results_table(logic.match(logic.METHOD_COMBINED, JD, ex, fake_ranker()))
    assert table[0]["Candidate"] == "strong" and table[0]["Required skills matched"] == "3/3"
    assert table[1]["Required skills matched"] == "2/3"


def test_combined_csv_has_the_expected_fields_and_values():
    ex = logic.extract_files(pdfs(strong=STRONG, partial=PARTIAL) + [("bad.pdf", b"nope")])
    result = logic.match(logic.METHOD_COMBINED, JD, ex, fake_ranker())
    rows = read_csv(logic.result_to_csv(result))
    assert list(rows[0]) == logic.CSV_COLUMNS[logic.METHOD_COMBINED]
    ranked = [r for r in rows if r["status"] == "ranked"]
    assert [r["candidate_id"] for r in ranked] == ["strong", "partial"] and [r["rank"] for r in ranked] == ["1", "2"]
    strong = ranked[0]
    ui = by_id(result)["strong"]
    assert float(strong["combined_matching_index"]) == pytest.approx(ui.scores["combined_index"], abs=1e-6)
    assert float(strong["semantic_cosine_raw"]) == pytest.approx(ui.scores["semantic_cosine"], abs=1e-6)
    assert float(strong["tfidf_cosine"]) == pytest.approx(ui.scores["tfidf_cosine"], abs=1e-6)
    assert float(strong["required_skill_coverage"]) == 1.0 and strong["required_skills_total"] == "3"
    assert strong["matched_required_skills"] == "Python; SQL; Docker" and strong["missing_required_skills"] == ""
    assert strong["matched_preferred_skills"] == "Kubernetes" and strong["method"] == "combined"
    assert ranked[1]["missing_required_skills"] == "Docker"
    excluded = [r for r in rows if r["status"].startswith("excluded")]
    assert [(r["candidate_id"], r["status"]) for r in excluded] == [("bad.pdf", "excluded_not_scored")]
    assert excluded[0]["rank"] == "" and excluded[0]["combined_matching_index"] == "" and excluded[0]["warnings"]


def test_tfidf_and_semantic_csv_do_not_contain_scores_that_were_not_calculated():
    ex = extractions_of(strong=STRONG, unrelated=UNRELATED)
    tf = read_csv(logic.result_to_csv(logic.match(logic.METHOD_TFIDF, JD, ex, fake_ranker())))
    assert list(tf[0]) == logic.CSV_COLUMNS[logic.METHOD_TFIDF]
    assert "semantic_cosine_raw" not in tf[0] and "combined_matching_index" not in tf[0]
    assert tf[0]["candidate_id"] == "strong" and float(tf[0]["tfidf_cosine"]) > 0
    assert "python" in tf[0]["top_shared_terms"]
    assert float(tf[1]["tfidf_cosine"]) == 0.0                                    # a real zero, not a missing value
    sem = read_csv(logic.result_to_csv(logic.match(logic.METHOD_SEMANTIC, JD, ex, fake_ranker())))
    assert list(sem[0]) == logic.CSV_COLUMNS[logic.METHOD_SEMANTIC]
    assert "tfidf_cosine" not in sem[0] and "required_skill_coverage" not in sem[0]


def test_unavailable_coverage_is_blank_in_the_csv_not_zero_or_one():
    ex = extractions_of(a=STRONG)
    result = logic.match(logic.METHOD_COMBINED, "We want a friendly and motivated colleague for our growing team.",
                         ex, fake_ranker())
    row = read_csv(logic.result_to_csv(result))[0]
    assert row["required_skill_coverage"] == "" and row["combined_matching_index"] != ""
    assert "unavailable" in row["warnings"].lower()


def test_csv_neutralises_spreadsheet_formulas_in_file_names():
    good = logic.text_to_pdf_bytes(STRONG)
    ex = logic.extract_files([("=HYPERLINK(1).pdf", good), ("@cmd.pdf", logic.text_to_pdf_bytes(PARTIAL))])
    rows = read_csv(logic.result_to_csv(logic.match(logic.METHOD_TFIDF, JD, ex, fake_ranker())))
    assert all(r["candidate_id"][0] == "'" for r in rows)
    assert {r["candidate_id"] for r in rows} == {"'=HYPERLINK(1).pdf", "'@cmd.pdf"}


# =============================================================================
# Cache key / signature, demo material
# =============================================================================
def test_config_signature_changes_with_the_configuration():
    a = logic.config_signature(fake_ranker())
    assert a == logic.config_signature(fake_ranker())
    assert a != logic.config_signature(fake_ranker(ScoringConfig(0.4, 0.3, 0.3)))
    assert json.loads(a)["weights"] == {"semantic": 0.5, "tfidf": 0.2, "skills": 0.3}


def test_text_to_pdf_round_trips_through_the_parser():
    text = "Python developer.\n" + "Built services with SQL and Docker. " * 6
    r = extract_text_from_bytes(logic.text_to_pdf_bytes(text), "x.pdf")
    assert r.usable and "Python developer." in r.text and "Docker" in r.text


def test_demo_material_is_synthetic_neutral_and_free_of_labels():
    scenarios = logic.demo_scenarios()
    assert [s[0] for s in scenarios] == ["JD-BE", "JD-FE", "JD-DA", "JD-ML", "JD-FS"]
    assert "Requirements:" in logic.demo_job_description("JD-BE")
    with pytest.raises(ValueError):
        logic.demo_job_description("nope")
    with zipfile.ZipFile(io.BytesIO(logic.demo_resumes_zip("JD-BE"))) as z:
        names = z.namelist()
        assert len(names) == 9 and names[0] == "synthetic_resume_01.pdf"
        assert not any(c in n for n in names for c in ("BE-", "label"))
        first = extract_text_from_bytes(z.read(names[0]), names[0])
        assert first.usable
    with pytest.raises(ValueError):
        logic.demo_resumes_zip("nope")
    assert logic.demo_scenarios(path="does-not-exist.json") == []
