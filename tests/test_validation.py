"""Phase 10: end-to-end validation, privacy, caching, CSV and configuration-guard tests.

Fast tests use a fake embedding model (hand-controlled, content dependent). Tests marked "real"
use the pretrained all-MiniLM-L6-v2 and are skipped, visibly, if it cannot be loaded.
All PDFs are generated here; all text is synthetic.
"""

import ast
import builtins
import copy
import csv
import hashlib
import io
import json
import pickle
import tomllib
from pathlib import Path
from unittest import mock

import pymupdf
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import app_logic as logic
from src import evaluate as ev
from src.pdf_parser import Status
from src.ranker import CandidateRanker, ScoringConfig
from src.semantic_matcher import SemanticMatcher, SemanticModelError
from src.tfidf_matcher import TfidfMatcher
from tests.test_app import (APP_PATH, CountingModel, STANDARD, env, error_texts, new_app, rank_button,  # noqa: F401
                            result_of, run_ranking)
from tests.test_app_logic import (JD, PARTIAL, STRONG, UNRELATED, _blank_pdf, _image_only_pdf, extractions_of,
                                  fake_ranker, read_csv)
from tests.test_semantic_matcher import words

ROOT = Path(__file__).resolve().parent.parent
MARKER = "ZXQMARKER7731"


def pdf(text):
    return logic.text_to_pdf_bytes(text)


def locked_pdf():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Python developer with SQL and Docker experience. " * 3)
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    doc.close()
    return data


# =============================================================================
# End-to-end scenarios (parser -> preprocessing -> skills -> matching -> ranking -> details -> CSV)
# =============================================================================
@pytest.mark.parametrize("method", list(logic.METHOD_LABELS))
def test_one_valid_resume_end_to_end(method):
    result = logic.match(method, JD, extractions_of(only=STRONG), fake_ranker())
    assert [r.candidate_id for r in result.rows] == ["only"] and result.rows[0].rank == 1
    rows = read_csv(logic.result_to_csv(result))
    assert [r["candidate_id"] for r in rows] == ["only"] and rows[0]["status"] == "ranked"


def test_mixture_of_every_kind_of_invalid_pdf_still_ranks_the_valid_ones():
    files = [("blank.pdf", _blank_pdf()), ("bad.pdf", b"%PDF-1.4 junk"), ("locked.pdf", locked_pdf()),
             ("scan.pdf", _image_only_pdf()), ("empty.pdf", b""), ("good1.pdf", pdf(STRONG)),
             ("good2.pdf", pdf(PARTIAL))]
    extractions = logic.extract_files(files)
    statuses = {e.file_id: e.status for e in extractions}
    assert statuses == {"blank.pdf": Status.NO_TEXT, "bad.pdf": Status.CORRUPTED, "locked.pdf": Status.ENCRYPTED,
                        "scan.pdf": Status.SCANNED_LIKELY, "empty.pdf": Status.EMPTY_FILE,
                        "good1.pdf": Status.OK, "good2.pdf": Status.OK}
    notes = {r["File"]: r["Note"] for r in logic.status_rows(extractions)}
    assert "password" in notes["locked.pdf"].lower() and "OCR" in notes["scan.pdf"]
    for method in logic.METHOD_LABELS:
        result = logic.match(method, JD, extractions, fake_ranker())
        assert {r.candidate_id for r in result.rows} == {"good1.pdf", "good2.pdf"}        # both valid files ranked
        assert [r.rank for r in result.rows] == [1, 2]
        assert {e.file_id for e in result.excluded} == {"blank.pdf", "bad.pdf", "locked.pdf", "scan.pdf", "empty.pdf"}


def test_identical_bytes_under_different_names_are_ranked_once():
    b = pdf(STRONG)
    ex = logic.extract_files([("a.pdf", b), ("b.pdf", b), ("c.pdf", pdf(PARTIAL))])
    for method in logic.METHOD_LABELS:
        result = logic.match(method, JD, ex, fake_ranker())
        assert {r.candidate_id for r in result.rows} == {"a.pdf", "c.pdf"} and len(result.rows) == 2
        assert [(e.file_id, e.reason) for e in result.excluded] == [("b.pdf", "duplicate")]


def test_same_file_name_with_different_content_keeps_both_with_their_own_scores():
    ex = logic.extract_files([("cv.pdf", pdf(STRONG)), ("cv.pdf", pdf(UNRELATED))])
    result = logic.match(logic.METHOD_COMBINED, JD, ex, fake_ranker())
    by = {r.candidate_id: r for r in result.rows}
    assert set(by) == {"cv.pdf", "cv (2).pdf"}
    assert by["cv.pdf"].matched_required == ("Python", "SQL", "Docker") and by["cv (2).pdf"].matched_required == ()


def test_unusually_long_resume_is_processed_in_full_without_truncation_warnings():
    body = "\n".join(f"Project {i}: coordinated releases, wrote documentation and trained colleagues on the process."
                     for i in range(900))
    long_pdf = pdf(body + "\nSkills: Python, SQL and Docker used in production services.")
    ex = logic.extract_files([("long.pdf", long_pdf), ("short.pdf", pdf(STRONG))])
    assert ex[0].usable and ex[0].page_count > 20
    for method in logic.METHOD_LABELS:
        result = logic.match(method, JD, ex, fake_ranker())
        assert {r.candidate_id for r in result.rows} == {"long.pdf", "short.pdf"}
        assert not [w for w in result.warnings if "truncated" in w]
    combined = {r.candidate_id: r for r in logic.match(logic.METHOD_COMBINED, JD, ex, fake_ranker()).rows}
    assert combined["long.pdf"].matched_required == ("Python", "SQL", "Docker")   # the skills line on the last page


def test_job_description_variants_through_the_logic_layer():
    ex = extractions_of(a=STRONG, b=PARTIAL)
    none = logic.match(logic.METHOD_COMBINED, "We want a friendly and motivated colleague for our growing team.", ex,
                       fake_ranker())
    assert none.job.required_skills == () and any("No skills from the dictionary" in w for w in none.job.warnings)
    assert all(r.scores["required_skill_coverage"] is None for r in none.rows)            # unavailable, not 0 or 1
    assert all(any("unavailable" in w.lower() for w in r.warnings) for r in none.rows)
    uncertain = logic.match(logic.METHOD_COMBINED, "We need Python, SQL and Docker.", ex, fake_ranker())
    assert uncertain.job.split_uncertain and any("uncertain" in w for w in uncertain.job.warnings)
    structured = logic.match(logic.METHOD_COMBINED, JD, ex, fake_ranker())
    assert structured.job.required_skills == ("Python", "SQL", "Docker") and structured.job.preferred_skills == ("Kubernetes",)
    assert not structured.job.split_uncertain and structured.job.warnings == ()
    stop_only = "the and of to in 2024"
    assert logic.validate_inputs(stop_only, 1, logic.METHOD_TFIDF)
    with pytest.raises(ValueError, match="usable words"):
        logic.match(logic.METHOD_TFIDF, stop_only, ex, fake_ranker())
    assert logic.match(logic.METHOD_COMBINED, stop_only, ex, fake_ranker()).rows        # other methods still work


def test_non_english_resume_is_scored_without_crashing_but_tfidf_sees_no_words():
    hindi = pdf("मैं एक सॉफ्टवेयर इंजीनियर हूँ। पायथन और एसक्यूएल में काम किया है। " * 12)
    ex = logic.extract_files([("hindi.pdf", hindi), ("ok.pdf", pdf(STRONG))])
    tfidf = logic.match(logic.METHOD_TFIDF, JD, ex, fake_ranker())
    assert [r.candidate_id for r in tfidf.rows] == ["ok.pdf", "hindi.pdf"] and tfidf.rows[-1].empty
    assert logic.match(logic.METHOD_COMBINED, JD, ex, fake_ranker()).rows[0].candidate_id == "ok.pdf"


# =============================================================================
# Methods call the intended existing modules, and nothing else
# =============================================================================
def test_each_method_invokes_exactly_its_own_existing_module():
    ex = extractions_of(a=STRONG, b=PARTIAL, c=UNRELATED)
    # TF-IDF: TfidfMatcher.rank once per resume with ONE candidate (pairwise IDF), no semantic model use
    r = fake_ranker()
    with mock.patch.object(r.tfidf, "rank", wraps=r.tfidf.rank) as tf, mock.patch.object(
            r.semantic, "rank", side_effect=AssertionError("semantic must not run")), mock.patch.object(
            r, "rank", side_effect=AssertionError("combined must not run")):
        logic.match(logic.METHOD_TFIDF, JD, ex, r)
    assert tf.call_count == 3 and all(len(c.args[1]) == 1 for c in tf.call_args_list)
    # semantic: SemanticMatcher.rank once with all candidates, TF-IDF never used
    r = fake_ranker()
    with mock.patch.object(r.semantic, "rank", wraps=r.semantic.rank) as sem, mock.patch.object(
            r.tfidf, "rank", side_effect=AssertionError("tfidf must not run")), mock.patch.object(
            r, "rank", side_effect=AssertionError("combined must not run")):
        logic.match(logic.METHOD_SEMANTIC, JD, ex, r)
    assert sem.call_count == 1 and len(sem.call_args.args[1]) == 3
    # combined: CandidateRanker.rank once
    r = fake_ranker()
    with mock.patch.object(r, "rank", wraps=r.rank) as comb:
        logic.match(logic.METHOD_COMBINED, JD, ex, r)
    assert comb.call_count == 1


def test_ui_layer_contains_no_matching_algorithm():
    forbidden = {"sklearn", "numpy", "scipy", "sentence_transformers", "torch", "transformers"}
    for path in (ROOT / "app.py", ROOT / "src" / "app_logic.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & forbidden, (path.name, imported & forbidden)


def test_scoring_configuration_and_evaluation_protocol_are_unchanged():
    cfg = ScoringConfig()
    assert (cfg.w_semantic, cfg.w_tfidf, cfg.w_skills) == (0.50, 0.20, 0.30)
    assert (ev.K, ev.RELEVANT_MIN_LABEL, ev.METHODS) == (3, 2, ("A_tfidf", "B_semantic", "C_combined"))
    sem = SemanticMatcher(model=mock.Mock(max_seq_length=256), token_counter=words)
    assert (sem.chunk_tokens, sem.overlap_tokens, sem.top_k, sem.model_name) == (
        200, 40, 1, "sentence-transformers/all-MiniLM-L6-v2")
    recorded = json.loads((ROOT / "artifacts" / "evaluation_config.json").read_text())["data"]["sha256"]
    files = {"dataset.json": ROOT / "data/evaluation/dataset.json", "labels.csv": ROOT / "data/evaluation/labels.csv",
             "RUBRIC.md": ROOT / "data/evaluation/RUBRIC.md",
             "skill_annotations.json": ROOT / "data/evaluation/skill_annotations.json",
             "skill_dictionary.json": ROOT / "data/skill_dictionary.json"}
    for name, path in files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == recorded[name], f"{name} changed since the evaluation"


def test_documented_evaluation_table_matches_the_stored_results():
    macro = {}
    with open(ROOT / "artifacts" / "evaluation_results.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["scope"] == "MACRO":
                macro[(row["method"], row["metric"])] = float(row["value"])
    documented = {"A_tfidf": (0.467, 0.300, 0.567, 0.461), "B_semantic": (0.600, 0.390, 1.000, 0.651),
                  "C_combined": (0.600, 0.390, 0.900, 0.654)}
    for method, values in documented.items():
        measured = tuple(round(macro[(method, m)], 3) for m in ("precision@3", "recall@3", "mrr", "ndcg@3"))
        assert measured == values, method


# =============================================================================
# CSV export
# =============================================================================
@pytest.mark.parametrize("method", list(logic.METHOD_LABELS))
def test_csv_agrees_with_the_displayed_results_for_every_method(method):
    good = pdf(STRONG)
    ex = logic.extract_files([("a.pdf", good), ("dup.pdf", good), ("b.pdf", pdf(PARTIAL)), ("bad.pdf", b"nope"),
                              ("scan.pdf", _image_only_pdf())])
    result = logic.match(method, JD, ex, fake_ranker())
    rows = read_csv(logic.result_to_csv(result))
    assert list(rows[0]) == logic.CSV_COLUMNS[method]
    ranked = [r for r in rows if r["status"] in ("ranked", "no_usable_text")]
    assert [r["candidate_id"] for r in ranked] == [r.candidate_id for r in result.rows]
    assert [int(r["rank"]) for r in ranked] == [r.rank for r in result.rows]
    column = {"combined": "combined_matching_index", "tfidf": "tfidf_cosine", "semantic": "semantic_cosine_raw"}[method]
    key = logic.METHOD_SCORES[method][0]
    for csv_row, ui_row in zip(ranked, result.rows):
        assert float(csv_row[column]) == pytest.approx(ui_row.scores[key], abs=1e-6)
    excluded = {r["candidate_id"]: r for r in rows if r["status"].startswith("excluded")}
    assert set(excluded) == {"dup.pdf", "bad.pdf", "scan.pdf"}
    assert excluded["dup.pdf"]["status"] == "excluded_duplicate"
    for r in excluded.values():                       # exclusion rows carry a reason and no fabricated numbers
        assert r["rank"] == "" and r["warnings"]
        assert all(r[c] == "" for c in logic.CSV_COLUMNS[method] if c.endswith(("cosine", "cosine_raw", "index", "coverage")))
    assert excluded["scan.pdf"]["warnings"].count("OCR") >= 1


def test_csv_only_has_applicable_columns_and_blank_means_unavailable():
    ex = extractions_of(a=STRONG)
    tf = logic.result_to_csv(logic.match(logic.METHOD_TFIDF, JD, ex, fake_ranker())).splitlines()[0]
    assert "semantic" not in tf and "coverage" not in tf and "combined" not in tf
    sem = logic.result_to_csv(logic.match(logic.METHOD_SEMANTIC, JD, ex, fake_ranker())).splitlines()[0]
    assert "tfidf" not in sem and "coverage" not in sem and "combined" not in sem
    no_skills = logic.match(logic.METHOD_COMBINED, "We want a friendly and motivated colleague for our team.", ex,
                            fake_ranker())
    row = read_csv(logic.result_to_csv(no_skills))[0]
    assert row["required_skill_coverage"] == "" and row["combined_matching_index"] not in ("", "0.000000")


@pytest.mark.parametrize("name", ["=HYPERLINK(1).pdf", "+cmd.pdf", "-2+3.pdf", "@SUM(A1).pdf"])
def test_formula_prefixes_in_file_names_are_neutralised(name):
    ex = logic.extract_files([(name, pdf(STRONG)), ("normal.pdf", pdf(PARTIAL))])
    for method in logic.METHOD_LABELS:
        rows = read_csv(logic.result_to_csv(logic.match(method, JD, ex, fake_ranker())))
        ids = {r["candidate_id"] for r in rows}
        assert "'" + name in ids and name not in ids and "normal.pdf" in ids
        assert all(not r["candidate_id"].startswith(("=", "+", "-", "@")) for r in rows)


def test_csv_round_trips_unicode_commas_quotes_and_line_breaks_in_names():
    names = ["résumé 日本語.pdf", 'we"ird,name.pdf', "semi;colon.pdf", "new\nline.pdf"]
    ex = logic.extract_files([(n, pdf(STRONG + f" Variant {i}.")) for i, n in enumerate(names)])
    result = logic.match(logic.METHOD_TFIDF, JD, ex, fake_ranker())
    text = logic.result_to_csv(result)
    parsed = list(csv.DictReader(io.StringIO(text, newline="")))
    assert {r["candidate_id"] for r in parsed} == set(names)
    # and it is readable by pandas too
    import pandas as pd
    frame = pd.read_csv(io.StringIO(text))
    assert set(frame["candidate_id"]) == set(names) and len(frame) == 4


def test_exporting_does_not_modify_the_ranking_result():
    result = logic.match(logic.METHOD_COMBINED, JD, extractions_of(a=STRONG, b=PARTIAL, c=UNRELATED), fake_ranker())
    before = pickle.dumps(result)
    first = logic.result_to_csv(result)
    logic.results_table(result)
    assert pickle.dumps(result) == before
    assert logic.result_to_csv(result) == first                                # deterministic


# =============================================================================
# Privacy: nothing persisted, nothing logged, no network libraries, no per-user state in shared objects
# =============================================================================
def test_no_network_or_messaging_library_is_imported_by_the_application_code():
    banned = {"requests", "urllib", "urllib3", "http", "socket", "httpx", "aiohttp", "smtplib", "ftplib",
              "websocket", "websockets", "logging"}
    for path in [ROOT / "app.py"] + sorted((ROOT / "src").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            assert not set(names) & banned, (path.name, names)


def test_processing_resumes_writes_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    writes = []
    real_open = builtins.open

    def spy(file, mode="r", *args, **kwargs):
        if any(c in str(mode) for c in "wax+"):
            writes.append((str(file), mode))
        return real_open(file, mode, *args, **kwargs)

    with mock.patch.object(builtins, "open", spy):
        ex = logic.extract_files([("a.pdf", pdf(STRONG + " " + MARKER)), ("b.pdf", pdf(PARTIAL))])
        for method in logic.METHOD_LABELS:
            logic.result_to_csv(logic.match(method, JD, ex, fake_ranker()))
    assert writes == []
    assert list(tmp_path.iterdir()) == []


def test_the_full_interface_run_writes_no_files_and_logs_no_resume_text(env, tmp_path, monkeypatch, caplog, capfd):
    import logging
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)
    at = run_ranking(new_app(), files=[("a.pdf", pdf(STRONG + " " + MARKER)), ("b.pdf", pdf(PARTIAL))])
    assert not at.exception and result_of(at).rows
    out, err = capfd.readouterr()
    assert MARKER not in caplog.text and MARKER not in out and MARKER not in err
    assert list(tmp_path.iterdir()) == []


def test_shared_matcher_objects_keep_no_resume_text():
    r = fake_ranker()

    def snapshot():
        return {name: repr(value) for obj in (r, r.semantic, r.tfidf, r.extractor)
                for name, value in vars(obj).items() if name not in ("_model",)}

    before = snapshot()
    logic.match(logic.METHOD_COMBINED, JD, extractions_of(a=STRONG + " " + MARKER), r)
    logic.match(logic.METHOD_TFIDF, JD, extractions_of(a=STRONG + " " + MARKER), r)
    logic.match(logic.METHOD_SEMANTIC, JD, extractions_of(a=STRONG + " " + MARKER), r)
    after = snapshot()
    assert before == after
    assert all(MARKER not in v for v in after.values())


def test_streamlit_server_configuration_stays_local_and_private():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert config["server"]["address"] == "localhost"
    assert config["browser"]["gatherUsageStats"] is False
    assert config["server"]["maxUploadSize"] == 25
    assert (ROOT / ".gitignore").read_text().count("data/private/") == 1


# =============================================================================
# Caching, beyond tests/test_app.py
# =============================================================================
def test_cache_clear_button_forces_recomputation_of_identical_inputs(env):
    at = run_ranking(new_app(), files=STANDARD)
    calls = len(env.model.calls)
    rank_button(at).click()
    at.run()
    assert len(env.model.calls) == calls                                      # cached
    next(b for b in at.button if b.label == "Clear cached data now").click()
    at.run()
    assert "result" not in at.session_state
    rank_button(at).click()
    at.run()
    assert len(env.model.calls) == calls + 1 and result_of(at).rows           # recomputed after clearing


def test_switching_method_recomputes_instead_of_reusing_another_methods_cache(env):
    at = run_ranking(new_app(), files=STANDARD, method="combined")
    calls = len(env.model.calls)
    run_ranking(at, files=STANDARD, method="semantic")
    assert len(env.model.calls) == calls + 1 and result_of(at).method == "semantic"
    run_ranking(at, files=STANDARD, method="combined")                        # back to combined: cached
    assert len(env.model.calls) == calls + 1 and result_of(at).method == "combined"


def test_data_caches_are_keyed_per_browser_session_but_the_model_is_shared(env, monkeypatch):
    # AppTest always runs with the fixed session id "test session id", so true multi-session isolation
    # cannot be observed with it. Streamlit keeps session-scoped caches in a dict keyed by the session id
    # it gets from get_session_id_or_throw(); here that lookup is switched between two ids, which shows
    # the app's decorators really use scope="session" (a global cache would serve session B from A's cache).
    from streamlit.runtime.caching import cache_data_api
    current = {"id": "session-A"}
    monkeypatch.setattr(cache_data_api, "get_session_id_or_throw", lambda: current["id"])
    at_a = run_ranking(new_app(), files=STANDARD)
    calls = len(env.model.calls)
    rank_button(at_a).click()
    at_a.run()
    assert len(env.model.calls) == calls                                      # same session: cache hit
    current["id"] = "session-B"
    other = run_ranking(new_app(), files=STANDARD)                            # same inputs, different session
    assert len(env.model.calls) == calls + 1                                  # computed again: not shared
    assert len(env.built) == 1                                                # one ranker / model for the process
    assert result_of(other).rows


# =============================================================================
# Interface behaviour, beyond tests/test_app.py
# =============================================================================
def test_stop_word_only_job_description_with_tfidf_shows_the_validation_message(env):
    at = run_ranking(new_app(), jd="the and of to in 2024", files=STANDARD, method="tfidf")
    assert not at.exception and any("no usable words for TF-IDF" in e for e in error_texts(at))
    assert "result" not in at.session_state


def test_textless_scanned_locked_and_corrupted_pdfs_through_the_interface(env):
    files = [("blank.pdf", _blank_pdf()), ("scan.pdf", _image_only_pdf()), ("locked.pdf", locked_pdf()),
             ("bad.pdf", b"nope"), ("ok.pdf", pdf(STRONG))]
    at = run_ranking(new_app(), files=files)
    assert not at.exception
    status = at.dataframe[0].value.set_index("File")
    assert status.loc["locked.pdf", "Status"] == "Password-protected"
    assert status.loc["scan.pdf", "Status"] == "Probably scanned: OCR needed"
    assert [r.candidate_id for r in result_of(at).rows] == ["ok.pdf"]


def test_interface_never_shows_decisions_or_probabilities(env):
    at = run_ranking(new_app(), files=STANDARD)
    table = at.dataframe[1].value
    for banned in ("hire", "reject", "decision", "recommend", "probability", "chance", "%"):
        assert not any(banned in str(c).lower() for c in table.columns)
    visible = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
                       + [w.value for w in at.warning]).lower()
    assert "you should hire" not in visible and "automatically rejected" not in visible
    assert "not probabilities" in visible and "nobody is rejected automatically" in visible
    for e in at.expander:                                                       # details never carry a verdict
        assert "reject" not in " ".join(m.value for m in e.markdown).lower().replace("rejected automatically", "")


def test_markdown_in_file_names_and_resume_text_is_displayed_as_plain_text(env):
    hostile = "[click me](evil.example).pdf"         # '/' and ':' cannot occur in real file names
    text = "Python developer. SQL and Docker services with [Click here](x) **bold** :smile: $x$ " * 3
    at = run_ranking(new_app(), files=[(hostile, pdf(text)), ("ok.pdf", pdf(STRONG))])
    assert not at.exception
    labels = [e.label for e in at.expander if e.label.startswith("#")]
    hostile_label = next(l for l in labels if "evil" in l)
    assert "\\[click me\\]\\(evil.example\\)" in hostile_label                  # brackets and parentheses escaped
    assert "](" not in hostile_label.replace("\\]\\(", "")
    evidence = [m.value for e in at.expander for m in e.markdown if m.value.startswith("Evidence for")]
    assert evidence and all("[Click here](x)" not in v and "**bold**" not in v for v in evidence)
    assert any("\\[Click here\\]\\(x\\)" in v for v in evidence)


def test_escape_markdown_neutralises_markup():
    nasty = "[a](http://x) ![i](y) **b** _c_ `d` :red[e] :smile: $m$ <b>h</b> ~~s~~ | t"
    out = logic.escape_markdown(nasty)
    for raw in ("](", "**", "`d`", ":red[", ":smile:", "$m$", "<b>", "~~"):
        assert raw not in out
    assert logic.escape_markdown("plain text 123.pdf") == "plain text 123.pdf"


# =============================================================================
# REAL model: direct comparison against the underlying modules, long inputs, reproducibility
# =============================================================================
@pytest.fixture(scope="module")
def real():
    st.cache_data.clear()
    st.cache_resource.clear()
    ranker = logic.build_ranker()
    try:
        ranker.semantic.model
    except SemanticModelError as exc:     # pragma: no cover
        pytest.skip(f"pretrained model not available: {exc}")
    return ranker


def demo_extractions(job_id="JD-BE"):
    import zipfile
    with zipfile.ZipFile(io.BytesIO(logic.demo_resumes_zip(job_id))) as z:
        return logic.extract_files([(n, z.read(n)) for n in z.namelist()]), logic.demo_job_description(job_id)


def test_real_app_logic_equals_direct_calls_to_the_underlying_modules(real):
    ex, jd = demo_extractions()
    pairs = [(e.file_id, e.text) for e in ex if e.usable]
    assert len(pairs) == 9

    direct = real.rank(jd, pairs)
    combined = logic.match(logic.METHOD_COMBINED, jd, ex, real)
    assert [r.candidate_id for r in combined.rows] == [c.candidate_id for c in direct.candidates]
    for ui, d in zip(combined.rows, direct.candidates):
        assert ui.scores["combined_index"] == pytest.approx(d.combined_index, abs=1e-9)
        assert ui.scores["semantic_cosine"] == pytest.approx(d.semantic_similarity, abs=1e-9)
        assert ui.scores["tfidf_cosine"] == d.tfidf_similarity
        assert ui.scores["required_skill_coverage"] == d.required_skill_coverage
        assert ui.matched_required == d.matched_required and ui.missing_required == d.missing_required
        assert ui.matched_preferred == d.matched_preferred and ui.rank == d.rank

    tfidf = logic.match(logic.METHOD_TFIDF, jd, ex, real)
    for row in tfidf.rows:
        text = dict(pairs)[row.candidate_id]
        expected = TfidfMatcher().rank(jd, [(row.candidate_id, text)]).scores[0].score
        assert row.scores["tfidf_cosine"] == expected                          # same pairwise configuration
    assert [r.scores["tfidf_cosine"] for r in tfidf.rows] == sorted((r.scores["tfidf_cosine"] for r in tfidf.rows),
                                                                    reverse=True)

    semantic = logic.match(logic.METHOD_SEMANTIC, jd, ex, real)
    expected = SemanticMatcher().rank(jd, pairs)
    assert [r.candidate_id for r in semantic.rows] == [s.candidate_id for s in expected.scores]
    for row, s in zip(semantic.rows, expected.scores):
        assert row.scores["semantic_cosine"] == pytest.approx(s.score, abs=1e-6)

    # the displayed combined index is the Phase 7 formula applied to the displayed components
    for row in combined.rows:
        s, t, k = row.scores["semantic_cosine"], row.scores["tfidf_cosine"], row.scores["required_skill_coverage"]
        assert row.scores["combined_index"] == pytest.approx(0.5 * max(0.0, s) + 0.2 * t + 0.3 * k, abs=1e-9)


def test_real_long_job_description_and_long_resume(real):
    jd = (logic.demo_job_description("JD-BE") + "\n"
          + "\n".join(f"Responsibility {i}: work with stakeholders, plan releases, review code and mentor colleagues."
                     for i in range(40)))
    filler = "\n".join(f"Project {i}: coordinated releases, wrote documentation and trained colleagues."
                       for i in range(150))
    long_pdf = pdf(filler + "\nSkills: Python, SQL, PostgreSQL, Docker and REST APIs in production services.")
    ex = logic.extract_files([("long.pdf", long_pdf), ("short.pdf", pdf(STRONG))])
    assert len(real.semantic.chunk_document(jd, "jd")) > 1
    assert len(real.semantic.chunk_document(ex[0].text, "long.pdf")) > 3
    for method in logic.METHOD_LABELS:
        result = logic.match(method, jd, ex, real)
        assert {r.candidate_id for r in result.rows} == {"long.pdf", "short.pdf"}
        assert not [w for w in result.warnings if "truncated" in w]
    combined = {r.candidate_id: r for r in logic.match(logic.METHOD_COMBINED, jd, ex, real).rows}
    assert combined["long.pdf"].matched_required == ("Python", "SQL", "PostgreSQL", "Docker", "REST API")


def test_real_phase8_results_are_reproducible_in_process(real):
    ds = ev.load_dataset()
    rows = ev.run_methods(ds, real)
    results = ev.evaluate_scores(rows, ev.METHOD_SCORE_COLUMN, "input")
    with open(ROOT / "artifacts" / "evaluation_results.csv", newline="", encoding="utf-8") as fh:
        stored = {(r["method"], r["scope"], r["metric"]): float(r["value"]) for r in csv.DictReader(fh) if r["value"]}
    for method, per_job in results.items():
        for scope, metrics in per_job.items():
            if scope == "MACRO_defined_jobs":
                continue
            for metric, value in metrics.items():
                assert value == pytest.approx(stored[(method, scope, metric)], abs=1e-6), (method, scope, metric)
