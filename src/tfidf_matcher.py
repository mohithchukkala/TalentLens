"""Method A: TF-IDF + cosine similarity (the traditional NLP baseline).

What "fitting" means here
-------------------------
TfidfVectorizer.fit() reads a set of documents and *counts*: it builds the vocabulary
(every distinct token / token pair) and, for each term, the number of documents that
contain it (document frequency, df). From df it computes the inverse document
frequency. There are no weights being trained by gradient descent, no epochs and no
labels: this is not neural-network training.

Formulas (scikit-learn, smooth_idf=True, sublinear_tf=True, norm="l2")
    tf(t, d)     = 1 + ln(count of t in d)           (0 if t is absent)
    idf(t)       = ln((1 + n) / (1 + df(t))) + 1      n = number of documents used to fit
    weight(t, d) = tf(t, d) * idf(t)
    vector(d)    = weights scaled to unit length (L2 norm)
    cosine(a, b) = a . b / (|a| |b|)  = a . b when both are unit vectors

All weights are >= 0, so the cosine lies in [0, 1]: 0 = no shared terms, 1 = same
direction. The value is a text-overlap score. It is NOT a probability, NOT a percentage
of "job match", and a score of 0.8 does not mean "80% suitable".

Two fitting modes (the result records which one was used in `fit_scope`)
    "batch"      (default) fit on the job description + the unique resumes of this call.
                 Needs no setup, but IDF values depend on which resumes were supplied,
                 so the same resume can score slightly differently in another batch.
    "reference"  call fit(reference_texts) once; IDF then comes from that fixed corpus and
                 does not change between calls. Terms absent from the reference
                 vocabulary are ignored.
Which mode the evaluation and the app use is decided in later phases and recorded in
PROJECT_LOG.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.inputs import Candidates, normalise_candidates as _normalise_candidates
from src.preprocessing import preprocess_for_tfidf

TOP_TERMS = 5
_TIE_DECIMALS = 12  # scores equal to 12 decimals share a rank


@dataclass(frozen=True)
class TfidfScore:
    candidate_id: str
    score: float                 # raw cosine similarity in [0, 1]
    rank: int                    # 1 = best; equal scores share a rank (1, 1, 3, ...)
    token_count: int             # tokens left after preprocessing (0 = nothing to match)
    top_terms: Tuple[Tuple[str, float], ...] = ()  # (term, contribution to the score)


@dataclass(frozen=True)
class TfidfRanking:
    scores: List[TfidfScore]                 # sorted best to worst; ties keep input order
    fit_scope: str                           # "batch" or "reference"
    vocabulary: Tuple[str, ...] = ()
    jd_token_count: int = 0
    warnings: List[str] = field(default_factory=list)

    def score_by_id(self) -> Dict[str, float]:
        return {s.candidate_id: s.score for s in self.scores}


def make_vectorizer(ngram_range: Tuple[int, int] = (1, 2), sublinear_tf: bool = True,
                    min_df: int = 1, max_df: float = 1.0) -> TfidfVectorizer:
    """The vectorizer configuration used by the baseline.

    lowercase=False      preprocess_for_tfidf already lowercased; avoids a second pass.
    token_pattern=\\S+   the preprocessing output is whitespace-separated tokens. sklearn's
                         default pattern (\\b\\w\\w+\\b) would cut "c++" to nothing, "node.js"
                         into "node" + "js" and drop one-letter tokens like "c" and "r".
    ngram_range=(1, 2)   single tokens plus adjacent pairs, so "machine learning" is a term.
    sublinear_tf=True    1 + ln(tf): repeating a keyword 20 times does not count 20 times.
    min_df=1, max_df=1.0 no pruning: with a handful of documents any pruning removes real terms.
    smooth_idf / norm    sklearn defaults (smooth idf, L2-normalised rows).
    These are fixed, documented choices, not values tuned on any evaluation data.
    """
    return TfidfVectorizer(
        lowercase=False, token_pattern=r"\S+", ngram_range=ngram_range,
        sublinear_tf=sublinear_tf, min_df=min_df, max_df=max_df,
        use_idf=True, smooth_idf=True, norm="l2", dtype=np.float64)


class TfidfMatcher:
    """Rank resumes against one job description with TF-IDF cosine similarity."""

    def __init__(self, **vectorizer_options):
        self._options = vectorizer_options
        self._reference_vectorizer: Optional[TfidfVectorizer] = None

    # -- optional fixed-corpus mode ------------------------------------------
    def fit(self, reference_texts: Sequence[str]) -> "TfidfMatcher":
        """Learn vocabulary and IDF from a fixed reference corpus ("reference" mode)."""
        if isinstance(reference_texts, (str, bytes)) or not isinstance(reference_texts, Iterable):
            raise TypeError("reference_texts must be an iterable of strings")
        processed = []
        for text in reference_texts:
            if not isinstance(text, str):
                raise TypeError("every reference text must be a string")
            processed.append(preprocess_for_tfidf(text))
        unique = [t for t in dict.fromkeys(processed) if t.strip()]
        if not unique:
            raise ValueError("the reference corpus has no usable text after preprocessing")
        vectorizer = make_vectorizer(**self._options)
        vectorizer.fit(unique)
        self._reference_vectorizer = vectorizer
        return self

    @property
    def fit_scope(self) -> str:
        return "reference" if self._reference_vectorizer is not None else "batch"

    # -- ranking --------------------------------------------------------------
    def rank(self, job_description: str, candidates: Candidates) -> TfidfRanking:
        """Score and rank candidates.

        Raises TypeError for wrong input types and ValueError for a job description
        without usable terms or duplicate candidate ids. An empty candidate collection
        returns an empty ranking. A resume with no usable text scores 0.0.
        """
        if not isinstance(job_description, str):
            raise TypeError(f"job_description must be a string, got {type(job_description).__name__}")
        pairs = _normalise_candidates(candidates)
        jd_proc = preprocess_for_tfidf(job_description)
        if not jd_proc.strip():
            raise ValueError("the job description is empty or has no usable terms "
                             "(only stop words, punctuation or numbers)")

        warnings: List[str] = []
        scope = self.fit_scope
        if not pairs:
            return TfidfRanking(scores=[], fit_scope=scope, jd_token_count=len(jd_proc.split()))

        processed = [preprocess_for_tfidf(text) for _, text in pairs]

        if self._reference_vectorizer is not None:
            vectorizer = self._reference_vectorizer
        else:
            # Fresh vectorizer per call (no state leaks between calls). Identical resumes
            # are counted once when learning IDF so a duplicate cannot shift other scores.
            vectorizer = make_vectorizer(**self._options)
            fit_docs = [jd_proc] + [t for t in dict.fromkeys(processed) if t.strip() and t != jd_proc]
            vectorizer.fit(fit_docs)

        matrix = vectorizer.transform([jd_proc] + processed)
        jd_vec, resume_vecs = matrix[0], matrix[1:]
        if jd_vec.nnz == 0:
            warnings.append("None of the job description's terms are in the reference vocabulary; "
                            "every score is 0.")
        raw = cosine_similarity(jd_vec, resume_vecs)[0]
        raw = np.clip(raw, 0.0, 1.0)  # remove floating-point noise such as 1.0000000000000002

        terms = vectorizer.get_feature_names_out()
        scored = []
        for i, ((cid, _), proc) in enumerate(zip(pairs, processed)):
            scored.append((float(raw[i]), i, cid, len(proc.split()), _top_terms(jd_vec, resume_vecs[i], terms)))
        empty_ids = [cid for _, _, cid, n, _ in scored if n == 0]
        if empty_ids:
            warnings.append(f"No usable text after preprocessing for: {', '.join(empty_ids)} (score 0).")

        scored.sort(key=lambda r: (-r[0], r[1]))  # best first, ties keep input order
        results: List[TfidfScore] = []
        previous_key, previous_rank = None, 0
        for position, (score, _, cid, n_tokens, top) in enumerate(scored, start=1):
            key = round(score, _TIE_DECIMALS)
            rank = previous_rank if key == previous_key else position
            previous_key, previous_rank = key, rank
            results.append(TfidfScore(cid, score, rank, n_tokens, top))

        return TfidfRanking(scores=results, fit_scope=scope, vocabulary=tuple(terms),
                            jd_token_count=len(jd_proc.split()), warnings=warnings)


def _top_terms(jd_vec, resume_vec, terms) -> Tuple[Tuple[str, float], ...]:
    """Terms that contribute most to the cosine. Both vectors are unit length, so the
    contributions of all shared terms add up exactly to the similarity score."""
    shared = jd_vec.multiply(resume_vec).tocoo()
    if shared.nnz == 0:
        return ()
    order = np.argsort(-shared.data, kind="stable")[:TOP_TERMS]
    return tuple((str(terms[shared.col[j]]), float(shared.data[j])) for j in order)


def rank_resumes(job_description: str, candidates: Candidates,
                 reference_texts: Optional[Sequence[str]] = None) -> TfidfRanking:
    """One-call helper. With reference_texts the IDF comes from that fixed corpus."""
    matcher = TfidfMatcher()
    if reference_texts is not None:
        matcher.fit(reference_texts)
    return matcher.rank(job_description, candidates)


if __name__ == "__main__":
    # SYNTHETIC demonstration data written for this project. Not real resumes.
    jd = ("Backend engineer. Required: Python, SQL, PostgreSQL, Docker, REST APIs, "
          "CI/CD. Nice to have: Kubernetes.")
    demo = {
        "synthetic_A.pdf": "Python developer. Built REST APIs with Flask and PostgreSQL, "
                           "containerised with Docker, CI/CD on GitHub Actions. SQL tuning.",
        "synthetic_B.pdf": "Frontend developer. React, JavaScript, CSS, Figma design systems, "
                           "accessibility, responsive web layouts.",
        "synthetic_C.pdf": "Software engineer using C++ and C# on .NET. Embedded firmware, "
                           "unit tests, Windows services.",
        "synthetic_D.pdf": "Data analyst: Python, pandas, SQL dashboards, Tableau reporting.",
    }
    ranking = rank_resumes(jd, demo)
    print(f"SYNTHETIC EXAMPLE | fit_scope={ranking.fit_scope} | vocabulary={len(ranking.vocabulary)} terms")
    print("Scores are raw cosine similarities (text overlap), not percentages or probabilities.")
    for s in ranking.scores:
        print(f"  rank {s.rank}  {s.candidate_id:<16} cosine={s.score:.4f}  top terms: "
              f"{[(t, round(w, 3)) for t, w in s.top_terms[:3]]}")
