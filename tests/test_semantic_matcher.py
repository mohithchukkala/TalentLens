"""Tests for src/semantic_matcher.py (Method B).

Two groups:
  * Logic tests use a FAKE model with hand-controlled vectors, so ranking, chunking and
    aggregation can be checked against exact hand-computed values, without loading anything.
  * Integration tests use the REAL pretrained all-MiniLM-L6-v2. They are skipped (and the skip
    is reported by pytest) if the model cannot be loaded, e.g. no network and no local cache.
All texts are short synthetic strings written for these tests.
"""

import math
import zlib

import numpy as np
import pytest

from src import semantic_matcher as sm
from src.semantic_matcher import (Chunk, SemanticMatcher, SemanticModelError, chunk_text,
                                  semantic_rank)

words = lambda texts: [len(t.split()) for t in texts]  # fake tokenizer: 1 word = 1 token


# =============================================================================
# Fake models
# =============================================================================
class BagOfWordsModel:
    """Deterministic fake: hashes each word into one of 64 dimensions and counts."""
    max_seq_length = 256

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), 64))
        for i, text in enumerate(texts):
            for w in text.lower().replace(",", " ").replace(".", " ").split():
                out[i, zlib.crc32(w.encode()) % 64] += 1.0
        return out


class LookupModel:
    """Fake whose vector depends only on the FIRST word of the text (so tests control cosines)."""

    def __init__(self, vectors, max_seq_length=12):
        self.vectors = vectors
        self.max_seq_length = max_seq_length
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.array([self.vectors[t.split()[0]] for t in texts], dtype=float)


def bow_matcher(**kwargs):
    return SemanticMatcher(model=BagOfWordsModel(), token_counter=words, **kwargs)


JD = "python sql docker rest apis backend"
GOOD = "python developer building rest apis with sql and docker for the backend"
UNRELATED = "pastry chef baking bread croissants and wedding cakes"


# =============================================================================
# Chunking (fake tokenizer: one word = one token)
# =============================================================================
def lines_doc(n_lines, words_per_line, prefix="line"):
    return "\n".join(f"{prefix}{i} " + " ".join(["w"] * (words_per_line - 1)) for i in range(n_lines))


def test_short_document_is_one_unsplit_chunk_with_offsets():
    text = "Python developer.\nBuilt REST APIs."
    chunks = chunk_text(text, "cv1", words, chunk_tokens=50, overlap_tokens=10)
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.doc_id, c.index, c.start, c.end, c.text) == ("cv1", 0, 0, len(text), text)
    assert c.token_count == 5


def test_blank_text_gives_no_chunks():
    assert chunk_text("", "x", words) == []
    assert chunk_text("  \n \n ", "x", words) == []


def test_document_that_fits_the_whole_doc_limit_is_not_split_even_above_chunk_size():
    text = lines_doc(4, 10)                       # 40 tokens
    assert len(chunk_text(text, "x", words, chunk_tokens=20, overlap_tokens=5)) > 1
    one = chunk_text(text, "x", words, chunk_tokens=20, overlap_tokens=5, whole_doc_tokens=40)
    assert len(one) == 1 and one[0].token_count == 40


def test_long_document_is_split_within_budget_and_covers_everything():
    text = lines_doc(30, 10)                      # 300 tokens
    chunks = chunk_text(text, "cv", words, chunk_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.token_count <= 50 for c in chunks)
    assert all(text[c.start:c.end] == c.text for c in chunks)          # exact slices: traceable
    assert all(c.doc_id == "cv" for c in chunks)
    covered = np.zeros(len(text), dtype=bool)
    for c in chunks:
        covered[c.start:c.end] = True
    assert all(covered[i] for i, ch in enumerate(text) if not ch.isspace())   # nothing lost
    assert all(chunks[i + 1].end > chunks[i].end for i in range(len(chunks) - 1))  # always progresses


def test_adjacent_chunks_overlap_by_whole_lines_within_the_overlap_budget():
    text = lines_doc(12, 10)                      # 12 lines of 10 tokens
    chunks = chunk_text(text, "cv", words, chunk_tokens=50, overlap_tokens=20)
    first, second = chunks[0], chunks[1]
    shared = text[second.start:first.end]
    assert words([shared]) == [20]                # exactly two lines shared
    assert shared.count("\n") == 1
    for a, b in zip(chunks, chunks[1:]):
        assert 0 < words([text[b.start:a.end]])[0] <= 20


def test_zero_overlap_means_no_shared_text():
    text = lines_doc(12, 10)
    chunks = chunk_text(text, "cv", words, chunk_tokens=50, overlap_tokens=0)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start >= a.end


def test_overlap_larger_than_the_last_unit_is_skipped_not_exceeded():
    text = lines_doc(8, 30)                       # each line (30 tokens) > overlap budget (10)
    chunks = chunk_text(text, "cv", words, chunk_tokens=60, overlap_tokens=10)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start >= a.end                   # nothing carried, nothing exceeded


def test_boundaries_fall_between_sentences_when_sentences_fit():
    sentences = [f"Sentence number {i} is here." for i in range(20)]   # 5 tokens each
    text = " ".join(sentences)                    # one long line, no newlines
    chunks = chunk_text(text, "cv", words, chunk_tokens=20, overlap_tokens=0)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith("Sentence") and c.text.endswith(".")
        assert c.token_count <= 20


def test_line_without_punctuation_is_split_into_words():
    text = " ".join(f"w{i}" for i in range(500))        # one 500-token line
    chunks = chunk_text(text, "cv", words, chunk_tokens=100, overlap_tokens=20)
    assert len(chunks) > 5 and all(c.token_count <= 100 for c in chunks)
    assert chunks[0].text.startswith("w0") and chunks[-1].text.endswith("w499")


def test_unbroken_giant_word_is_halved_until_it_fits():
    chars = lambda texts: [math.ceil(len(t) / 4) for t in texts]     # 4 characters = 1 token
    text = "x" * 400                                                   # 100 tokens, no spaces
    chunks = chunk_text(text, "cv", chars, chunk_tokens=10, overlap_tokens=0)
    assert all(c.token_count <= 10 for c in chunks)
    assert "".join(c.text for c in chunks) == text


def test_chunking_is_deterministic():
    text = lines_doc(30, 10)
    assert chunk_text(text, "a", words, 50, 10) == chunk_text(text, "a", words, 50, 10)


# =============================================================================
# Ranking with the bag-of-words fake model
# =============================================================================
def test_relevant_resume_ranks_above_unrelated_and_ids_are_kept():
    r = bow_matcher().rank(JD, {"zeta.pdf": UNRELATED, "alpha.pdf": GOOD})
    assert [s.candidate_id for s in r.scores] == ["alpha.pdf", "zeta.pdf"]
    assert r.scores[0].score > r.scores[1].score
    assert [s.rank for s in r.scores] == [1, 2]


def test_scores_are_finite_and_within_cosine_bounds():
    r = bow_matcher().rank(JD, [("a", GOOD), ("b", UNRELATED), ("c", JD), ("d", "")])
    for s in r.scores:
        assert math.isfinite(s.score) and -1.0 <= s.score <= 1.0
    assert r.score_by_id()["c"] == pytest.approx(1.0)       # identical text


def test_empty_candidate_collection_returns_empty_ranking_with_config():
    r = bow_matcher().rank(JD, {})
    assert r.scores == [] and r.model_name == sm.MODEL_NAME and r.chunk_tokens == 200
    assert r.jd_chunks and r.jd_chunks[0].doc_id == "job_description"


def test_blank_resume_is_flagged_and_ranked_last_even_below_a_negative_score():
    vectors = {"jd": (1, 0), "neg": (-1, 0), "pos": (1, 0)}
    m = SemanticMatcher(model=LookupModel(vectors), token_counter=words, chunk_tokens=10, overlap_tokens=0)
    r = m.rank("jd a b", [("blank", "   "), ("neg", "neg a b"), ("pos", "pos a b")])
    assert [s.candidate_id for s in r.scores] == ["pos", "neg", "blank"]
    blank = r.scores[-1]
    assert blank.empty and blank.score == 0.0 and blank.chunks == ()
    assert r.scores[1].score == pytest.approx(-1.0) and not r.scores[1].empty
    assert any("blank" in w for w in r.warnings)


def test_all_resumes_blank_does_not_crash():
    r = bow_matcher().rank(JD, {"a": "", "b": "  \n "})
    assert [s.empty for s in r.scores] == [True, True]
    assert [s.rank for s in r.scores] == [1, 1]


def test_duplicate_text_gives_identical_scores_shared_rank_and_input_order_ties():
    cands = [("first", GOOD), ("second", GOOD), ("other", UNRELATED)]
    r = bow_matcher().rank(JD, cands)
    by_id = {s.candidate_id: s for s in r.scores}
    assert by_id["first"].score == by_id["second"].score          # exactly equal
    assert by_id["first"].rank == by_id["second"].rank == 1 and by_id["other"].rank == 3
    assert [s.candidate_id for s in r.scores][:2] == ["first", "second"]
    reversed_run = bow_matcher().rank(JD, list(reversed(cands)))
    assert [s.candidate_id for s in reversed_run.scores][:2] == ["second", "first"]


def test_repeated_runs_give_identical_results():
    cands = [("a", GOOD), ("b", UNRELATED), ("c", "sql only")]
    m = bow_matcher()
    first = m.rank(JD, cands)
    for _ in range(3):
        again = m.rank(JD, cands)
        assert [(s.candidate_id, s.score, s.rank) for s in again.scores] == \
               [(s.candidate_id, s.score, s.rank) for s in first.scores]


def test_identical_texts_are_embedded_only_once_in_one_batch():
    model = BagOfWordsModel()
    m = SemanticMatcher(model=model, token_counter=words)
    m.rank(JD, [("a", GOOD), ("b", GOOD), ("c", GOOD), ("d", UNRELATED)])
    assert len(model.calls) == 1                                   # one batched call
    assert len(model.calls[0]) == len(set(model.calls[0])) == 3    # JD, GOOD, UNRELATED


def test_model_is_loaded_once_and_shared(monkeypatch):
    loads = []

    def fake_get_model(name, device):
        loads.append((name, device))
        return BagOfWordsModel()

    monkeypatch.setattr(sm, "get_model", fake_get_model)
    m = SemanticMatcher(token_counter=words)
    assert loads == []                                             # lazy: nothing loaded yet
    for _ in range(3):
        m.rank(JD, {"a": GOOD})
    assert loads == [(sm.MODEL_NAME, "cpu")]


def test_chunks_and_best_matches_trace_back_to_the_document():
    r = bow_matcher().rank(JD, {"cv": GOOD})
    s = r.scores[0]
    assert s.chunks and all(c.doc_id == "cv" for c in s.chunks)
    assert s.best_matches[0].jd_chunk == 0 and 0 <= s.best_matches[0].resume_chunk < len(s.chunks)


# =============================================================================
# Aggregation, checked against hand-computed values
# =============================================================================
# max_seq_length=12 -> usable 10 tokens. JD = a 5-token line "alpha ..." + a 10-token line "beta ..."
# (15 tokens > 10, so two JD chunks with weights 5 and 10). Resume lines have 10 tokens each,
# so a two-line resume is split into two chunks.
VECTORS = {"alpha": (1, 0), "beta": (0, 1), "rx": (1, 0), "ry": (0, 1), "rz": (0.6, 0.8), "rw": (0, 0.1)}
JD_TWO = "alpha a b c d\nbeta a b c d e f g h i"
R = lambda name: f"{name} " + " ".join(["t"] * 9)       # a 10-token line


def lookup_matcher(**kwargs):
    return SemanticMatcher(model=LookupModel(VECTORS), token_counter=words,
                           chunk_tokens=10, overlap_tokens=0, **kwargs)


def test_fixture_produces_the_intended_chunks():
    m = lookup_matcher()
    jd_chunks = m.chunk_document(JD_TWO, "job_description")
    assert [c.token_count for c in jd_chunks] == [5, 10]
    assert len(m.chunk_document(R("rx") + "\n" + R("ry"), "cv")) == 2


def test_weighted_average_of_best_match_per_jd_chunk():
    m = lookup_matcher()
    only_x = m.rank(JD_TWO, {"cv": R("rx") + "\n" + R("rx")}).scores[0].score
    # J1=(1,0) w=5 -> best 1.0 ; J2=(0,1) w=10 -> best 0.0 : (5*1 + 10*0) / 15
    assert only_x == pytest.approx(1 / 3)
    both = m.rank(JD_TWO, {"cv": R("rx") + "\n" + R("ry")}).scores[0].score
    assert both == pytest.approx(1.0)                      # each JD part has a perfect partner
    z = m.rank(JD_TWO, {"cv": R("rz") + "\n" + R("rz")}).scores[0].score
    assert z == pytest.approx((5 * 0.6 + 10 * 0.8) / 15)   # 0.7333...


def test_longer_jd_chunks_weigh_more():
    m = lookup_matcher()
    covers_long_part = m.rank(JD_TWO, {"cv": R("ry") + "\n" + R("ry")}).scores[0].score   # matches beta (w=10)
    covers_short_part = m.rank(JD_TWO, {"cv": R("rx") + "\n" + R("rx")}).scores[0].score  # matches alpha (w=5)
    assert covers_long_part == pytest.approx(2 / 3) and covers_short_part == pytest.approx(1 / 3)


def test_best_match_uses_the_best_resume_chunk_not_the_average():
    # Single-chunk JD vs a resume with one good and one useless chunk: the score is the good one.
    m = SemanticMatcher(model=LookupModel(VECTORS), token_counter=words, chunk_tokens=10, overlap_tokens=0)
    r = m.rank("alpha a b c d", {"cv": R("rx") + "\n" + R("rw")})
    assert r.scores[0].score == pytest.approx(1.0)
    assert r.scores[0].best_matches[0].resume_chunk == 0


def test_top_k_two_averages_the_two_best_chunks_documented_tradeoff():
    # One chunk matches perfectly (1.0), the other not at all (0.0).
    resume = R("rx") + "\n" + R("ry")
    m1 = SemanticMatcher(model=LookupModel(VECTORS), token_counter=words, chunk_tokens=10, overlap_tokens=0, top_k=1)
    m2 = SemanticMatcher(model=LookupModel(VECTORS), token_counter=words, chunk_tokens=10, overlap_tokens=0, top_k=2)
    assert m1.rank("alpha a b c d", {"cv": resume}).scores[0].score == pytest.approx(1.0)
    assert m2.rank("alpha a b c d", {"cv": resume}).scores[0].score == pytest.approx(0.5)


def test_top_k_larger_than_the_number_of_chunks_is_clamped():
    m = SemanticMatcher(model=LookupModel(VECTORS), token_counter=words, chunk_tokens=10, overlap_tokens=0, top_k=5)
    assert m.rank("alpha a b", {"cv": "rx a b"}).scores[0].score == pytest.approx(1.0)


# =============================================================================
# Invalid input and configuration
# =============================================================================
@pytest.mark.parametrize("jd", [None, 5, ["python"], b"python"])
def test_non_string_job_description_raises_type_error(jd):
    with pytest.raises(TypeError, match="job_description"):
        bow_matcher().rank(jd, {"a": GOOD})


@pytest.mark.parametrize("jd", ["", "   \n\t "])
def test_blank_job_description_raises_value_error(jd):
    with pytest.raises(ValueError, match="empty"):
        bow_matcher().rank(jd, {"a": GOOD})


@pytest.mark.parametrize("bad", ["a plain string", b"bytes", 42, None])
def test_non_collection_candidates_raise_type_error(bad):
    with pytest.raises(TypeError):
        bow_matcher().rank(JD, bad)


def test_bad_candidate_items_and_duplicate_ids():
    m = bow_matcher()
    with pytest.raises(TypeError, match="pair"):
        m.rank(JD, ["just a string"])
    with pytest.raises(TypeError, match="id must be a string"):
        m.rank(JD, [(1, GOOD)])
    with pytest.raises(TypeError, match="must be a string"):
        m.rank(JD, {"a": None})
    with pytest.raises(ValueError, match="duplicate candidate id"):
        m.rank(JD, [("a", GOOD), ("a", UNRELATED)])


def test_chunk_document_rejects_non_strings():
    with pytest.raises(TypeError):
        bow_matcher().chunk_document(None)


@pytest.mark.parametrize("kwargs,error", [
    ({"chunk_tokens": 0}, ValueError), ({"overlap_tokens": -1}, ValueError),
    ({"chunk_tokens": 20, "overlap_tokens": 20}, ValueError), ({"top_k": 0}, ValueError),
    ({"batch_size": 0}, ValueError), ({"chunk_tokens": 2.5}, TypeError), ({"top_k": True}, TypeError),
    ({"chunk_tokens": 255}, ValueError),     # larger than 256 - 2 usable tokens
])
def test_invalid_configuration_is_rejected(kwargs, error):
    with pytest.raises(error):
        SemanticMatcher(model=BagOfWordsModel(), token_counter=words, **kwargs)


# =============================================================================
# Model failures
# =============================================================================
def test_model_load_failure_is_reported_clearly(monkeypatch):
    import sentence_transformers

    def broken(*args, **kwargs):
        raise OSError("simulated: no network")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", broken)
    with pytest.raises(SemanticModelError, match="Could not load the model"):
        sm.get_model("fake/model-for-this-test", "cpu")


def test_encode_failure_and_invalid_embeddings_are_reported():
    class Exploding(BagOfWordsModel):
        def encode(self, texts, **kw):
            raise RuntimeError("boom")

    class NotANumber(BagOfWordsModel):
        def encode(self, texts, **kw):
            return np.full((len(texts), 4), np.nan)

    with pytest.raises(SemanticModelError, match="failed while encoding"):
        SemanticMatcher(model=Exploding(), token_counter=words).rank(JD, {"a": GOOD})
    with pytest.raises(SemanticModelError, match="invalid embeddings"):
        SemanticMatcher(model=NotANumber(), token_counter=words).rank(JD, {"a": GOOD})


def test_semantic_rank_helper_accepts_a_shared_matcher():
    m = bow_matcher()
    assert semantic_rank(JD, {"a": GOOD}, matcher=m).scores[0].candidate_id == "a"


# =============================================================================
# Integration: the REAL pretrained model (skipped if it cannot be loaded)
# =============================================================================
@pytest.fixture(scope="module")
def real():
    try:
        return SemanticMatcher()          # CPU, default model
    except Exception as exc:              # pragma: no cover
        pytest.skip(f"cannot create matcher: {exc}")


@pytest.fixture(scope="module")
def real_model(real):
    try:
        return real.model
    except SemanticModelError as exc:     # pragma: no cover
        pytest.skip(f"pretrained model not available: {exc}")


def test_real_model_configuration(real, real_model):
    assert real.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert real.max_seq_length == 256 and real.usable_tokens == 254
    assert real_model.get_embedding_dimension() == 384
    assert str(real_model.device) == "cpu"


def test_real_model_is_loaded_once_and_shared(real_model):
    # Regression: with a bare lru_cache on get_model, these three spellings were three cache keys.
    assert sm.get_model() is real_model
    assert sm.get_model(sm.MODEL_NAME) is real_model
    assert sm.get_model(sm.MODEL_NAME, "cpu") is real_model
    assert SemanticMatcher().model is real_model


def test_real_similar_texts_score_higher_than_unrelated(real, real_model):
    jd = "Software developer who writes Python code"
    r = real.rank(jd, {"similar": "Programmer with experience in Python scripting",
                       "unrelated": "Pastry chef who bakes wedding cakes"})
    by_id = r.score_by_id()
    assert by_id["similar"] > by_id["unrelated"]
    assert by_id["similar"] - by_id["unrelated"] > 0.2          # a clear gap, not a rounding accident


def test_real_synonyms_without_shared_words_are_ranked_correctly_where_tfidf_scores_zero(real, real_model):
    from src.tfidf_matcher import rank_resumes
    jd = "Machine learning engineer"
    cands = {"relevant": "Trained neural networks for image classification and speech recognition",
             "unrelated": "Managed restaurant staff schedules and food inventory"}
    semantic = real.rank(jd, cands)
    assert semantic.scores[0].candidate_id == "relevant"
    tfidf = rank_resumes(jd, cands).score_by_id()
    assert tfidf["relevant"] == 0.0 and tfidf["unrelated"] == 0.0   # no shared words at all


def test_real_small_synthetic_ranking_with_valid_bounds(real, real_model):
    jd = ("Backend engineer: Python, SQL, PostgreSQL, Docker and REST APIs. "
          "Build and deploy web services.")
    cands = {
        "backend.pdf": "Python developer. Built REST APIs with Flask and PostgreSQL, containerised with Docker.",
        "frontend.pdf": "Frontend developer using React, CSS and responsive layouts.",
        "chef.pdf": "Head pastry chef with ten years in bakeries and catering.",
    }
    r = real.rank(jd, cands)
    assert r.scores[0].candidate_id == "backend.pdf"
    for s in r.scores:
        assert math.isfinite(s.score) and -1.0 <= s.score <= 1.0


def test_real_repeated_runs_agree(real, real_model):
    cands = {"a": GOOD, "b": UNRELATED}
    first, second = real.rank(JD, cands), real.rank(JD, cands)
    for cid in cands:
        assert first.score_by_id()[cid] == pytest.approx(second.score_by_id()[cid], abs=1e-6)


def test_real_text_at_the_token_limit_stays_whole_and_one_over_is_split(real, real_model):
    count = real.count_tokens
    assert count(["python"]) == [1]                       # precondition for building exact lengths
    at_limit = " ".join(["python"] * real.usable_tokens)  # 254 tokens + 2 special = 256
    over = " ".join(["python"] * (real.usable_tokens + 1))
    assert count([at_limit]) == [254] and count([over]) == [255]
    assert len(real.chunk_document(at_limit, "cv")) == 1
    over_chunks = real.chunk_document(over, "cv")
    assert len(over_chunks) > 1 and all(c.token_count <= 200 for c in over_chunks)


def long_resume():
    sections = []
    for i in range(40):
        sections.append(f"Project {i}: Designed and delivered internal tooling for the logistics team, "
                        f"coordinating with warehouse managers and reporting weekly progress.")
    return "\n".join(sections)


def test_real_long_resume_chunks_fit_the_model_and_cover_the_document(real, real_model):
    text = long_resume()
    chunks = real.chunk_document(text, "cv")
    assert len(chunks) > 3
    pre = chunks[0].text  # sanity: a slice of the preprocessed text
    recount = real.count_tokens([c.text for c in chunks])
    assert all(n <= 200 for n in recount)
    assert all(n + sm.SPECIAL_TOKENS <= real.max_seq_length for n in recount)
    assert recount == [c.token_count for c in chunks]     # additive counting matches a real recount
    assert chunks[0].start == 0
    assert chunks[-1].end == len(sm.preprocess_for_embeddings(text))
    assert real.rank("logistics tooling", {"cv": text}).warnings == []


def test_real_chunking_finds_skills_that_a_single_truncated_embedding_misses(real, real_model):
    filler = "\n".join(["Enjoys baking sourdough bread, hiking in the mountains and gardening on weekends."] * 30)
    tail = "Skills: Python programming, machine learning, training neural networks."
    resume = filler + "\n" + tail
    jd = "Machine learning engineer with Python and neural network experience"
    assert real.count_tokens([resume])[0] > real.usable_tokens           # it really is too long
    truncated = float(np.dot(real_model.encode(jd, normalize_embeddings=True),
                             real_model.encode(resume, normalize_embeddings=True)))  # model reads only the start
    chunked = real.rank(jd, {"cv": resume}).scores[0].score
    assert chunked > truncated + 0.2
    best = real.rank(jd, {"cv": resume}).scores[0]
    assert "neural networks" in best.chunks[best.best_matches[0].resume_chunk].text


def test_real_whole_pipeline_from_pdf_text(real, real_model):
    from src.pdf_parser import extract_text_from_bytes
    from tests.test_pdf_parser import make_pdf
    pdf = make_pdf([["Asha Rao - Software Engineer",
                     "Skills: Python, SQL, Docker, Git, Machine Learning",
                     "Built REST APIs with Flask and deployed them on AWS."]])
    result = extract_text_from_bytes(pdf, "asha.pdf")
    r = real.rank("Python backend engineer with SQL and Docker", {result.file_id: result.text,
                                                                  "chef.pdf": UNRELATED})
    assert r.scores[0].candidate_id == "asha.pdf"
