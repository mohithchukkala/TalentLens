"""Tests for src/preprocessing.py."""

from src.preprocessing import (TFIDF_STOP_WORDS, normalize_text, preprocess_for_embeddings,
                               preprocess_for_skills, preprocess_for_tfidf,
                               remove_contact_info, tfidf_tokens)


# ---- shared normalisation ---------------------------------------------------
def test_normalize_empty_and_none_like():
    assert normalize_text("") == ""
    assert normalize_text("   \n \t ") == ""


def test_normalize_collapses_whitespace_but_keeps_line_breaks_and_case():
    out = normalize_text("Python   and\tSQL \n\n\n\nDocker")
    assert out == "Python and SQL\n\nDocker"


def test_normalize_unicode_ligatures_quotes_dashes_and_invisible():
    assert normalize_text("ﬁnal") == "final"
    assert normalize_text("it’s “ok”") == "it's \"ok\""
    assert normalize_text("2019 – 2023") == "2019 - 2023"
    assert normalize_text("Py​thon") == "Python"


def test_normalize_bullets_become_dashes():
    assert normalize_text("• Python\n SQL").split("\n") == ["- Python", "- SQL"]
    assert normalize_text("Python · SQL") == "Python | SQL"


def test_normalize_joins_hyphen_at_line_break_but_keeps_hyphen():
    assert normalize_text("scikit-\nlearn") == "scikit-learn"
    # an ordinary list dash before a capital letter is not joined
    assert normalize_text("Skills -\nPython") == "Skills -\nPython"


# ---- contact information ----------------------------------------------------
def test_remove_contact_info():
    text = "Mail asha@example.com, phone +91 98765 43210, see https://github.com/asha and www.site.io"
    out = remove_contact_info(text)
    for gone in ("@example.com", "98765", "github.com", "www.site.io"):
        assert gone not in out
    assert "Mail" in out and "phone" in out


def test_year_ranges_are_not_mistaken_for_phone_numbers():
    assert "2019 - 2023" in remove_contact_info("Intern, 2019 - 2023")


# ---- skills view ------------------------------------------------------------
def test_skills_view_preserves_technical_terms_case_and_stop_words():
    text = "Experience with C++, C#, .NET, Node.js and CI/CD in the cloud."
    out = preprocess_for_skills(text)
    for term in ("C++", "C#", ".NET", "Node.js", "CI/CD", "in the cloud", "Experience"):
        assert term in out


def test_skills_view_empty():
    assert preprocess_for_skills("") == ""
    assert preprocess_for_skills(None or "") == ""


# ---- TF-IDF view ------------------------------------------------------------
def test_tfidf_keeps_technical_tokens_whole():
    tokens = tfidf_tokens("Built with C++, C#, .NET, Node.js, CI/CD, scikit-learn and ASP.NET.")
    for expected in ("c++", "c#", ".net", "node.js", "ci/cd", "scikit-learn", "asp.net"):
        assert expected in tokens, tokens
    assert "c" not in tokens  # "c++" must not be cut down to "c"


def test_tfidf_lowercases_and_drops_stop_words():
    tokens = tfidf_tokens("The Python developer is in the team")
    assert "python" in tokens and "developer" in tokens and "team" in tokens
    assert "the" not in tokens and "is" not in tokens and "in" not in tokens


def test_tfidf_does_not_drop_front_back_full_system():
    # sklearn's default English stop list contains these; they matter in tech text.
    assert {"front", "back", "full", "system"} & set(__import__("sklearn.feature_extraction.text", fromlist=["x"]).ENGLISH_STOP_WORDS)
    tokens = tfidf_tokens("front end, back end, full stack, system design")
    assert tokens == ["front", "end", "back", "end", "full", "stack", "system", "design"]
    assert "front" not in TFIDF_STOP_WORDS


def test_tfidf_keeps_go_language_regression():
    # Regression (found in Phase 5): "go" is in sklearn's stop list, so Go vanished.
    assert "go" in tfidf_tokens("Backend services in Go and Python")
    assert "go" not in TFIDF_STOP_WORDS


def test_tfidf_drops_numbers_but_keeps_alphanumeric_tech_terms():
    tokens = tfidf_tokens("2019 2023 3.12 python3 s3 ec2 k8s")
    assert tokens == ["python3", "s3", "ec2", "k8s"]


def test_tfidf_keeps_single_letter_languages_only():
    tokens = tfidf_tokens("I know C and R and a bit of B")
    assert "c" in tokens and "r" in tokens
    assert "b" not in tokens and "i" not in tokens


def test_tfidf_removes_contact_details_and_handles_empty():
    tokens = tfidf_tokens("mail asha@example.com python")
    assert "asha@example.com" not in tokens and "example.com" not in tokens
    assert "python" in tokens
    assert tfidf_tokens("") == []
    assert preprocess_for_tfidf("   ") == ""


def test_tfidf_aliases_are_not_merged():
    tokens = tfidf_tokens("sklearn scikit-learn postgres postgresql")
    assert tokens == ["sklearn", "scikit-learn", "postgres", "postgresql"]


def test_tfidf_string_form_is_whitespace_tokenisable():
    s = preprocess_for_tfidf("C++ and Node.js engineer")
    assert s == "c++ node.js engineer"
    assert s.split() == tfidf_tokens("C++ and Node.js engineer")


# ---- embeddings view --------------------------------------------------------
def test_embedding_view_is_near_original():
    text = "I have built REST APIs in Python, and I led a team of 4 engineers."
    assert preprocess_for_embeddings(text) == text


def test_embedding_view_keeps_case_stop_words_and_lines_removes_contact():
    text = "Skills:\n• Python, C++\nEmail: a@b.com"
    out = preprocess_for_embeddings(text)
    assert out.split("\n")[0] == "Skills:"
    assert "- Python, C++" in out
    assert "a@b.com" not in out
    assert preprocess_for_embeddings("") == ""
