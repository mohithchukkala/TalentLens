"""Writes expected_profiles.csv and README.md from the candidate data plus the qualitative annotations below."""
import csv
from pathlib import Path

import pymupdf

from resume_data import CANDIDATES

folder = Path(__file__).resolve().parent.parent
A = {
    "resume_01_strong_backend.pdf": ("Strong backend candidate",
        "Python, SQL, PostgreSQL, FastAPI, REST API, Git, data structures, Docker, Pytest, GitHub Actions CI, Redis",
        "Django, AWS, MongoDB", "Fresher with internship",
        "Covers the required skills and most preferred skills, with evidence in an internship and two projects."),
    "resume_02_backend_fresher.pdf": ("Strong skills, limited experience",
        "Python, SQL, PostgreSQL, FastAPI, REST API, Git, algorithms (interval clash checker), Pytest",
        "Docker, AWS, CI/CD, Redis, MongoDB, Django, any work or internship experience", "Fresher, projects only",
        "Tests whether project descriptions alone (no employment section) are recognised as relevant evidence."),
    "resume_03_java_backend.pdf": ("Java backend developer",
        "Java, Spring Boot, MySQL, SQL, REST API, Git, JUnit, some Python, Docker for local use",
        "PostgreSQL, FastAPI, Django, AWS, CI/CD, Redis, MongoDB, strong Python", "About 3 years",
        "Related technologies instead of the exact stack: real backend experience that exact-keyword matching may undervalue (MySQL for PostgreSQL, Spring for FastAPI or Django)."),
    "resume_04_data_analyst.pdf": ("Data analyst moving to backend",
        "SQL, PostgreSQL, Python, Pandas, Git, basic Flask REST endpoints, data pipelines",
        "Docker (limited), AWS, CI/CD, FastAPI, Django, Redis, MongoDB, data structures and algorithms", "About 2 years (analyst)",
        "Strong SQL and Python but analysis rather than service-building evidence; two-column layout tests text extraction order."),
    "resume_05_fullstack.pdf": ("Full-stack developer",
        "React, JavaScript, Node.js, Express.js, REST API, MongoDB, MySQL and SQL, Git, Jest, basic Python",
        "PostgreSQL, FastAPI, Django, Docker, AWS, CI/CD, Redis, data structures and algorithms", "About 1.5 years",
        "Different stack with partial overlap: Python and PostgreSQL are weak or absent while MongoDB, SQL, REST and Git are present."),
    "resume_06_experienced_backend.pdf": ("Experienced backend developer",
        "Python, Django, Django REST Framework, FastAPI, PostgreSQL, Redis, Docker, AWS, GitHub Actions, Pytest, Git, REST API",
        "MongoDB; data structures and algorithms not stated explicitly", "About 2 years",
        "Detailed employment with responsibilities separated from measurable achievements; two-page layout."),
    "resume_07_ml_developer.pdf": ("Machine learning candidate",
        "Python, NumPy, Pandas, Scikit-learn, FastAPI inference APIs, basic SQL, Git, Pytest (basic)",
        "PostgreSQL, Django, Docker, AWS, CI/CD, Redis, MongoDB, production backend experience, data structures and algorithms", "Fresher with internship",
        "Strong Python and FastAPI evidence in an ML context but limited backend and database depth; two-column layout."),
    "resume_08_devops_developer.pdf": ("DevOps-oriented developer",
        "Linux, Git, Docker, GitHub Actions, Jenkins, AWS, Python, Flask REST endpoint, SQL and PostgreSQL queries",
        "FastAPI, Django, Redis, MongoDB, unit testing evidence, data structures and algorithms", "About 2 years",
        "Strong on the preferred deployment skills (Docker, AWS, CI/CD) with moderate evidence for the required ones."),
    "resume_09_keyword_heavy.pdf": ("Keyword overlap, weak evidence",
        "Skills section lists nearly every required and preferred technology; the evidence is a Django tutorial project, a short API-call script and a web-page internship",
        "Demonstrated use of PostgreSQL, FastAPI, Docker, AWS, CI/CD, Redis, MongoDB, Kubernetes, unit testing", "Student, short internship",
        "Tests whether skill coverage overestimates a candidate who lists skills without supporting evidence."),
    "resume_10_semantic_match.pdf": ("Strong experience, different wording",
        "Python, relational database (Postgres) tuning, HTTP web services, automated checks, source control with pull requests, container images, algorithm vocabulary",
        "Few exact keywords: REST API, SQL, PostgreSQL, Git, Docker, unit testing and CI/CD are described in other words or only once", "About 2.5 years",
        "Tests whether semantic similarity recognises relevant experience when exact keyword overlap is low."),
}
layout = {"single_sans": "single column, sans-serif", "single_serif": "single column, serif",
          "traditional": "traditional corporate (centred header, serif)", "two_col": "two columns (sidebar and main)"}

with open(folder / "expected_profiles.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["Filename", "Candidate name", "Profile category", "Key relevant skills", "Important missing skills", "Experience level", "Testing purpose"])
    for c in CANDIDATES:
        cat, skills, missing, lvl, purpose = A[c["file"]]
        w.writerow([c["file"], c["name"], cat, skills, missing, lvl, purpose])

pages = {c["file"]: len(pymupdf.open(folder / c["file"])) for c in CANDIDATES if (folder / c["file"]).exists()}
md = [
    "# realistic_backend_set: 10 fictional developer resumes for the Backend Software Developer job",
    "",
    "All candidates, organisations, e-mail addresses (`example.com`), telephone numbers and certifications are **fictional**. Course names are invented sample",
    "courses, not real credentials, and no certificate IDs are given. Each page footer says the document is fictional. The documents are meant for manual",
    "testing of PDF extraction, ranking behaviour and the candidate-detail display.",
    "",
    "`job_description.txt` is the target job (Backend Software Developer; required: Python, SQL, PostgreSQL, REST API development, Git, data structures and",
    "algorithms; preferred: FastAPI, Django, Docker, AWS, CI/CD, Redis, MongoDB, unit testing).",
    "",
    "**The notes below are qualitative test annotations, not ground-truth relevance labels, and no ranking is predicted or claimed.** The set was not",
    "tuned to favour any ranking method. This folder is separate from `backend_demo/` and `edge_cases/` and is not part of the evaluation dataset.",
    "",
    "| File | Pages | Layout |", "|---|---:|---|",
]
for c in CANDIDATES:
    md.append(f"| `{c['file']}` | {pages.get(c['file'], '?')} | {layout[c['layout']]} |")
md += ["", "## Candidate profiles"]
for c in CANDIDATES:
    cat, skills, missing, lvl, purpose = A[c["file"]]
    md += ["", f"### {c['file']}: {c['name']}",
           f"- **Profile:** {cat} ({lvl}).",
           f"- **Strengths / relevant skills:** {skills}.",
           f"- **Missing or weak requirements:** {missing}.",
           f"- **Purpose of this sample:** {purpose}"]
md += ["", "## How to use",
       "1. Paste `job_description.txt` into the app.",
       "2. Upload all ten PDFs and run each of the three methods.",
       "3. Compare what each method does with the profile notes above; they describe the documents, not a correct answer.",
       "",
       "## Regenerating and validating",
       "```powershell",
       "cd data\\sample_resumes\\realistic_backend_set\\generator",
       "python build_resumes.py ..",
       "python make_annotations.py",
       "cd ..\\..\\..\\..",
       "python data/sample_resumes/realistic_backend_set/generator/validate_resumes.py",
       "```",
       "The builder and the validator need only PyMuPDF (already installed). Delete any `*.pdf.raw` files the builder leaves behind."]
(folder / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("wrote CSV and README")
