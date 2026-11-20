"""Tests for src/skill_extractor.py (and its integration with the PDF parser)."""

import json

import pytest

from src.pdf_parser import extract_text_from_bytes
from src.preprocessing import preprocess_for_skills
from src.skill_extractor import SkillExtractor, get_default_extractor
from tests.test_pdf_parser import make_pdf

EX = get_default_extractor()


def names(text):
    return set(EX.extract(text))


# ---- punctuation-sensitive skills ------------------------------------------
def test_cpp_csharp_dotnet_nodejs_cicd():
    found = names("Languages: C++, C# and .NET. Backend in Node.js with CI/CD.")
    assert {"C++", "C#", ".NET", "Node.js", "CI/CD"} <= found


def test_c_is_not_matched_inside_cpp_or_csharp():
    found = names("I use C++ and C# daily")
    assert "C" not in found


def test_c_language_needs_context_and_exact_case():
    assert "C" in names("Programming languages: C, Python")
    assert "C" not in names("Grade C in the final exam")
    assert "C" not in names("languages: c++ and python")  # lowercase c++ only


def test_dotnet_not_inside_aspnet():
    found = names("Built sites with ASP.NET Core")
    assert "ASP.NET" in found
    assert ".NET" not in found


# ---- substring errors -------------------------------------------------------
def test_sql_not_found_inside_mysql_postgresql_nosql():
    assert names("MySQL") == {"MySQL"}
    assert names("PostgreSQL") == {"PostgreSQL"}
    assert names("NoSQL databases") == {"NoSQL"}


def test_sql_found_when_standalone_or_in_lists():
    assert "SQL" in names("Python/SQL/Docker")
    assert "SQL" in names("SQL-based reporting")
    assert "SQL" in names("wrote sql queries")


def test_java_is_not_found_inside_javascript_and_vice_versa():
    assert names("JavaScript developer") == {"JavaScript"}
    assert names("Java developer") == {"Java"}
    assert names("Java and JavaScript") == {"Java", "JavaScript"}


def test_git_not_inside_github_digit_legitimate_or_digital():
    assert names("GitHub") == {"GitHub"}
    assert "Git" not in names("digital digit legitimate")
    assert "Git" in names("Used Git and GitHub")


def test_unrelated_words_produce_no_skills():
    text = "I managed a team, organised events, and enjoy hiking, painting and cooking."
    assert names(text) == set()
    assert "Go" not in names("Let's go to the market")
    assert "R" not in names("R&D department")
    assert "Spark" not in names("A spark of creativity")


def test_pandas_numpy_inside_words_not_matched():
    assert "NumPy" not in names("numpyramid")
    assert "Pandas" not in names("expandas")


def test_context_must_be_in_the_same_sentence():
    # Regression found by probing a one-line text: "React" and "JS" were each "vouched for"
    # by a context word that belonged to a neighbouring sentence.
    assert "React" not in names("Frontend work done. React quickly to incidents.")
    assert "JavaScript" not in names("Used React. Visited the JS Bach exhibition.")
    assert "React" in names("Frontend work done. Built UIs in React with hooks.")


# ---- aliases ----------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("sklearn", "Scikit-learn"),
    ("scikit-learn", "Scikit-learn"),
    ("Scikit Learn", "Scikit-learn"),
    ("Postgres", "PostgreSQL"),
    ("NodeJS", "Node.js"),
    ("Node JS", "Node.js"),
    ("MSSQL", "Microsoft SQL Server"),
    ("SQL Server", "Microsoft SQL Server"),
    ("MS SQL Server", "Microsoft SQL Server"),
    ("K8s", "Kubernetes"),
    ("Amazon Web Services", "AWS"),
    ("Natural Language Processing", "NLP"),
    ("machine-learning", "Machine Learning"),
    ("Golang", "Go"),
    ("ReactJS", "React"),
    ("Expressjs", "Express.js"),
])
def test_aliases_map_to_canonical_names(text, expected):
    assert expected in names(f"Experience with {text}.")


def test_words_not_in_the_dictionary_are_not_found():
    # The extractor only knows what is in the dictionary; unknown tools are invisible.
    assert names("Experience with Zigfoo and Quuxlang.") == set()


def test_microsoft_sql_server_beats_sql():
    found = names("Administered Microsoft SQL Server")
    assert found == {"Microsoft SQL Server"}


def test_javascript_alias_js_only_with_context_and_exact_case():
    assert "JavaScript" in names("Frontend developer skilled in JS and HTML")
    assert "JavaScript" not in names("Visited the JS Bach exhibition")
    assert "JavaScript" not in names("Built with Node.js")  # js inside Node.js is not JavaScript
    assert names("Node.js") == {"Node.js"}


def test_react_js_is_one_skill_not_react_plus_javascript():
    assert names("React JS developer") == {"React"}


def test_react_as_a_verb_is_not_a_skill():
    assert "React" not in names("React quickly to production incidents")
    assert "React" in names("Frontend work in React with hooks")


# ---- capitalisation, punctuation, line breaks -------------------------------
def test_capitalisation_variants():
    for variant in ("python", "PYTHON", "Python", "pYtHoN"):
        assert "Python" in names(f"Skilled in {variant}")


def test_punctuation_around_skills():
    found = names("(Python), [Docker]; 'Git'. SQL! Flask?")
    assert {"Python", "Docker", "Git", "SQL", "Flask"} <= found


def test_skill_split_across_a_line_break():
    assert "Machine Learning" in names("Applied machine\nlearning to text")
    assert "Scikit-learn" in names("used scikit-\nlearn for models")
    assert "Scikit-learn" in names(preprocess_for_skills("used scikit-\nlearn for models"))


# ---- original wording and evidence ------------------------------------------
def test_original_wording_and_evidence_are_preserved():
    text = "Education: BTech\nBuilt classifiers with sklearn and tuned them.\nOther: hiking"
    hit = EX.extract(text)["Scikit-learn"]
    assert hit.mentions == ["sklearn"]
    assert hit.count == 1
    assert hit.evidence == ["Built classifiers with sklearn and tuned them."]


def test_multiple_mentions_are_counted_and_wordings_listed():
    text = "scikit-learn models.\nLater used sklearn again.\nAnd sklearn once more."
    hit = EX.extract(text)["Scikit-learn"]
    assert hit.count == 3
    assert hit.mentions == ["scikit-learn", "sklearn"]
    assert len(hit.evidence) == 3


def test_long_line_evidence_is_trimmed_around_the_match():
    text = "x " * 200 + "Docker" + " y" * 200
    ev = EX.extract(text)["Docker"].evidence[0]
    assert "Docker" in ev and len(ev) < 220 and ev.startswith("...")


def test_match_positions_point_at_the_original_text():
    text = "We use Node.js here"
    m = EX.find_matches(text)[0]
    assert text[m.start:m.end] == m.matched_text == "Node.js"


# ---- empty input ------------------------------------------------------------
@pytest.mark.parametrize("text", ["", "   ", "\n\n", None])
def test_empty_input_returns_nothing(text):
    assert EX.extract(text or "") == {}
    assert EX.find_matches(text or "") == []


# ---- job descriptions: required vs preferred --------------------------------
JD = """Senior Backend Engineer

Requirements:
- 3+ years of Python
- Strong SQL and PostgreSQL
- Docker

Nice to have:
- Kubernetes
- Strong communication skills
- AWS
"""


def test_job_skills_split_required_and_preferred():
    js = EX.extract_job_skills(JD)
    assert set(js.required) == {"Python", "SQL", "PostgreSQL", "Docker"}
    assert set(js.preferred) == {"Kubernetes", "AWS"}


def test_bullet_after_heading_does_not_flip_the_section():
    # "Strong communication skills" mentions 'skills' but is a bullet, not a heading
    js = EX.extract_job_skills(JD)
    assert "AWS" in js.preferred


def test_unstructured_jd_defaults_to_required_and_cue_affects_only_its_sentence():
    # Regression: a single-paragraph JD with one "a plus" used to demote EVERY skill.
    js = EX.extract_job_skills("We need Python, SQL and Docker. Kubernetes is a plus.")
    assert set(js.required) == {"Python", "SQL", "Docker"}
    assert set(js.preferred) == {"Kubernetes"}


def test_jd_without_any_cue_is_all_required():
    js = EX.extract_job_skills("We need Python, SQL and Docker.")
    assert set(js.required) == {"Python", "SQL", "Docker"} and js.preferred == {}


def test_inline_preferred_phrase_marks_that_line_only():
    js = EX.extract_job_skills("Python required\nDocker is a plus\nSQL")
    assert set(js.required) == {"Python", "SQL"}
    assert set(js.preferred) == {"Docker"}


def test_skill_in_both_places_counts_as_required():
    js = EX.extract_job_skills("Requirements:\nPython\nNice to have:\nPython, Docker")
    assert "Python" in js.required and "Python" not in js.preferred
    assert set(js.preferred) == {"Docker"}


def test_structured_flag_tells_whether_the_split_rests_on_headings_or_cues():
    assert EX.extract_job_skills(JD).structured is True                              # headings
    assert EX.extract_job_skills("Python required\nDocker is a plus").structured is True   # inline cue
    assert EX.extract_job_skills("Requirements:\nPython and SQL").structured is True       # explicit heading
    assert EX.extract_job_skills("We need Python, SQL and Docker.").structured is False    # default only
    assert EX.extract_job_skills("Python required, SQL required").structured is False      # 'required' on a skill line is not a heading
    assert EX.extract_job_skills("").structured is False


def test_jd_without_skills_gives_empty_sets():
    js = EX.extract_job_skills("We are a friendly company in a great location.")
    assert js.required == {} and js.preferred == {}
    assert EX.extract_job_skills("").all == {}


# ---- configurable dictionary ------------------------------------------------
def test_custom_dictionary_and_adding_a_skill():
    custom = SkillExtractor.from_dict({"skills": [
        {"name": "Rustlang Toolkit", "category": "Tools", "aliases": ["rtk"]},
    ]})
    assert set(custom.extract("we use RTK daily")) == {"Rustlang Toolkit"}
    assert custom.extract("Python") == {}


def test_dictionary_from_json_file(tmp_path):
    path = tmp_path / "skills.json"
    path.write_text(json.dumps({"skills": [{"name": "Foo", "aliases": ["foo-lang"]}]}), encoding="utf-8")
    assert set(SkillExtractor.from_json(path).extract("Using FOO-LANG now")) == {"Foo"}


def test_duplicate_alias_across_skills_is_rejected():
    with pytest.raises(ValueError, match="used by both"):
        SkillExtractor([{"name": "A", "aliases": ["x"]}, {"name": "B", "aliases": ["X"]}])


def test_duplicate_skill_name_and_bad_rule_key_are_rejected():
    with pytest.raises(ValueError):
        SkillExtractor([{"name": "A"}, {"name": "A"}])
    with pytest.raises(ValueError):
        SkillExtractor([{"name": "A", "rules": {"Nope": {"case_sensitive": True}}}])


def test_shipped_dictionary_is_valid_and_reasonably_sized():
    assert len(EX.skill_names) >= 100
    for required in ("Python", "Java", "C++", "JavaScript", "SQL", "MySQL", "PostgreSQL", "MongoDB",
                     "React", "Node.js", "Express.js", "FastAPI", "Flask", "Django", "TensorFlow",
                     "PyTorch", "Scikit-learn", "NLP", "Machine Learning", "Deep Learning", "Git",
                     "Docker", "AWS"):
        assert required in EX.skill_names


# ---- integration with the PDF parser ----------------------------------------
def test_pdf_to_skills_pipeline():
    pdf = make_pdf([[
        "Asha Rao - Software Engineer",
        "Skills: Python, SQL, Docker, Git, Machine Learning, Node.js",
        "Built REST APIs with Flask on AWS using Postgres and sklearn.",
    ]])
    result = extract_text_from_bytes(pdf, "asha.pdf")
    assert result.usable
    skills = EX.extract(preprocess_for_skills(result.text))
    assert {"Python", "SQL", "Docker", "Git", "Machine Learning", "Node.js", "Flask",
            "AWS", "PostgreSQL", "Scikit-learn", "REST API"} <= set(skills)
    assert skills["PostgreSQL"].mentions == ["Postgres"]
