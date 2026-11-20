# realistic_backend_set: 10 fictional developer resumes for the Backend Software Developer job

All candidates, organisations, e-mail addresses (`example.com`), telephone numbers and certifications are **fictional**. Course names are invented sample
courses, not real credentials, and no certificate IDs are given. Each page footer says the document is fictional. The documents are meant for manual
testing of PDF extraction, ranking behaviour and the candidate-detail display.

`job_description.txt` is the target job (Backend Software Developer; required: Python, SQL, PostgreSQL, REST API development, Git, data structures and
algorithms; preferred: FastAPI, Django, Docker, AWS, CI/CD, Redis, MongoDB, unit testing).

**The notes below are qualitative test annotations, not ground-truth relevance labels, and no ranking is predicted or claimed.** The set was not
tuned to favour any ranking method. This folder is separate from `backend_demo/` and `edge_cases/` and is not part of the evaluation dataset.

| File | Pages | Layout |
|---|---:|---|
| `resume_01_strong_backend.pdf` | 2 | single column, sans-serif |
| `resume_02_backend_fresher.pdf` | 2 | single column, sans-serif |
| `resume_03_java_backend.pdf` | 1 | traditional corporate (centred header, serif) |
| `resume_04_data_analyst.pdf` | 1 | two columns (sidebar and main) |
| `resume_05_fullstack.pdf` | 1 | single column, sans-serif |
| `resume_06_experienced_backend.pdf` | 2 | single column, serif |
| `resume_07_ml_developer.pdf` | 1 | two columns (sidebar and main) |
| `resume_08_devops_developer.pdf` | 1 | single column, sans-serif |
| `resume_09_keyword_heavy.pdf` | 1 | traditional corporate (centred header, serif) |
| `resume_10_semantic_match.pdf` | 1 | single column, serif |

## Candidate profiles

### resume_01_strong_backend.pdf: Rohan Deshmukh
- **Profile:** Strong backend candidate (Fresher with internship).
- **Strengths / relevant skills:** Python, SQL, PostgreSQL, FastAPI, REST API, Git, data structures, Docker, Pytest, GitHub Actions CI, Redis.
- **Missing or weak requirements:** Django, AWS, MongoDB.
- **Purpose of this sample:** Covers the required skills and most preferred skills, with evidence in an internship and two projects.

### resume_02_backend_fresher.pdf: Ananya Iyer
- **Profile:** Strong skills, limited experience (Fresher, projects only).
- **Strengths / relevant skills:** Python, SQL, PostgreSQL, FastAPI, REST API, Git, algorithms (interval clash checker), Pytest.
- **Missing or weak requirements:** Docker, AWS, CI/CD, Redis, MongoDB, Django, any work or internship experience.
- **Purpose of this sample:** Tests whether project descriptions alone (no employment section) are recognised as relevant evidence.

### resume_03_java_backend.pdf: Vikram Nair
- **Profile:** Java backend developer (About 3 years).
- **Strengths / relevant skills:** Java, Spring Boot, MySQL, SQL, REST API, Git, JUnit, some Python, Docker for local use.
- **Missing or weak requirements:** PostgreSQL, FastAPI, Django, AWS, CI/CD, Redis, MongoDB, strong Python.
- **Purpose of this sample:** Related technologies instead of the exact stack: real backend experience that exact-keyword matching may undervalue (MySQL for PostgreSQL, Spring for FastAPI or Django).

### resume_04_data_analyst.pdf: Sneha Kulkarni
- **Profile:** Data analyst moving to backend (About 2 years (analyst)).
- **Strengths / relevant skills:** SQL, PostgreSQL, Python, Pandas, Git, basic Flask REST endpoints, data pipelines.
- **Missing or weak requirements:** Docker (limited), AWS, CI/CD, FastAPI, Django, Redis, MongoDB, data structures and algorithms.
- **Purpose of this sample:** Strong SQL and Python but analysis rather than service-building evidence; two-column layout tests text extraction order.

### resume_05_fullstack.pdf: Karthik Reddy
- **Profile:** Full-stack developer (About 1.5 years).
- **Strengths / relevant skills:** React, JavaScript, Node.js, Express.js, REST API, MongoDB, MySQL and SQL, Git, Jest, basic Python.
- **Missing or weak requirements:** PostgreSQL, FastAPI, Django, Docker, AWS, CI/CD, Redis, data structures and algorithms.
- **Purpose of this sample:** Different stack with partial overlap: Python and PostgreSQL are weak or absent while MongoDB, SQL, REST and Git are present.

### resume_06_experienced_backend.pdf: Meera Joshi
- **Profile:** Experienced backend developer (About 2 years).
- **Strengths / relevant skills:** Python, Django, Django REST Framework, FastAPI, PostgreSQL, Redis, Docker, AWS, GitHub Actions, Pytest, Git, REST API.
- **Missing or weak requirements:** MongoDB; data structures and algorithms not stated explicitly.
- **Purpose of this sample:** Detailed employment with responsibilities separated from measurable achievements; two-page layout.

### resume_07_ml_developer.pdf: Arjun Malhotra
- **Profile:** Machine learning candidate (Fresher with internship).
- **Strengths / relevant skills:** Python, NumPy, Pandas, Scikit-learn, FastAPI inference APIs, basic SQL, Git, Pytest (basic).
- **Missing or weak requirements:** PostgreSQL, Django, Docker, AWS, CI/CD, Redis, MongoDB, production backend experience, data structures and algorithms.
- **Purpose of this sample:** Strong Python and FastAPI evidence in an ML context but limited backend and database depth; two-column layout.

### resume_08_devops_developer.pdf: Nikhil Banerjee
- **Profile:** DevOps-oriented developer (About 2 years).
- **Strengths / relevant skills:** Linux, Git, Docker, GitHub Actions, Jenkins, AWS, Python, Flask REST endpoint, SQL and PostgreSQL queries.
- **Missing or weak requirements:** FastAPI, Django, Redis, MongoDB, unit testing evidence, data structures and algorithms.
- **Purpose of this sample:** Strong on the preferred deployment skills (Docker, AWS, CI/CD) with moderate evidence for the required ones.

### resume_09_keyword_heavy.pdf: Priya Sharma
- **Profile:** Keyword overlap, weak evidence (Student, short internship).
- **Strengths / relevant skills:** Skills section lists nearly every required and preferred technology; the evidence is a Django tutorial project, a short API-call script and a web-page internship.
- **Missing or weak requirements:** Demonstrated use of PostgreSQL, FastAPI, Docker, AWS, CI/CD, Redis, MongoDB, Kubernetes, unit testing.
- **Purpose of this sample:** Tests whether skill coverage overestimates a candidate who lists skills without supporting evidence.

### resume_10_semantic_match.pdf: Farhan Qureshi
- **Profile:** Strong experience, different wording (About 2.5 years).
- **Strengths / relevant skills:** Python, relational database (Postgres) tuning, HTTP web services, automated checks, source control with pull requests, container images, algorithm vocabulary.
- **Missing or weak requirements:** Few exact keywords: REST API, SQL, PostgreSQL, Git, Docker, unit testing and CI/CD are described in other words or only once.
- **Purpose of this sample:** Tests whether semantic similarity recognises relevant experience when exact keyword overlap is low.

## How to use
1. Paste `job_description.txt` into the app.
2. Upload all ten PDFs and run each of the three methods.
3. Compare what each method does with the profile notes above; they describe the documents, not a correct answer.

## Regenerating and validating
```powershell
cd data\sample_resumes\realistic_backend_set\generator
python build_resumes.py ..
python make_annotations.py
cd ..\..\..\..
python data/sample_resumes/realistic_backend_set/generator/validate_resumes.py
```
The builder and the validator need only PyMuPDF (already installed). Delete any `*.pdf.raw` files the builder leaves behind.
