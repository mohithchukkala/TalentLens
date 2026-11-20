# Relevance rubric (SYNTHETIC evaluation data)

All job descriptions and resumes in `dataset.json` are **invented** for this project. Labels
(`labels.csv`) were written from this rubric and the resume text alone, **before any matching
method was run**, then reviewed once and frozen. They are one person's judgments and are
subjective. They say nothing about real-world hiring.

## Scale (relevance of one resume to its job description)

| Label | Meaning | Guide |
|---|---|---|
| 3 | Highly relevant | Demonstrates, in the context of real work or projects, essentially **all** required items (about 90% or more). Equivalent wording and aliases are accepted (Postgres = PostgreSQL, ES6 = JavaScript, T-SQL on SQL Server = advanced SQL). |
| 2 | Relevant, with gaps | Demonstrates roughly **half to most** of the required items, or all of them but only implied. Typical: one named tool is missing or not confirmed (for example "relational database" instead of PostgreSQL), or one required item is explicitly absent. |
| 1 | Weakly relevant | Some evidence (**under half** of the requirements), or an adjacent role, or claims without supporting experience (a keyword list with no work that uses it), or too little detail to confirm anything. |
| 0 | Not relevant | Different field, **or** explicitly states the required skills are absent, **or** supports/hires/documents/manages the role without performing it. |

## Rules applied to every judgment

1. **Evidence, not words.** A skill counts only if the resume shows it being *used*. A bare
   keyword list is a claim, not evidence.
2. **Negation.** "I have not worked with X" removes X as evidence, and the whole mention is not
   counted toward relevance.
3. **Naming.** Generic skills may be satisfied by equivalent wording (containers for Docker is
   *not* enough when Docker is named as a required tool: it leaves a gap, giving label 2 at best).
4. **"A or B" requirements** are satisfied by either one (PyTorch or TensorFlow).
5. **Seniority.** Years of experience are checked against stated minimums only (e.g. 2+ years).
6. **Role from outside.** Recruiters, designers, product managers and technical writers who talk
   about the skills but do not do the work are labelled 0.
7. **Position in text.** A long resume is judged on its whole content, wherever the relevant
   section appears.
8. Labels are **not** changed because a method ranks a resume unexpectedly.

## Relevance used by the metrics

Precision@3, Recall@3 and MRR treat labels 2 and 3 as relevant. nDCG@3 uses the graded labels
with gain `2^label - 1`.

## Limitations

- One annotator (the project author, assisted by an AI assistant); no inter-annotator agreement.
- The same author wrote the system, so the data may be easier or harder for it than independent data.
- 5 job descriptions and about 46 judgments: far too few for statistical claims.
- Resumes are short, clean and invented; real resumes are longer, messier and more varied.
