"""Method C: combined matching index and candidate ranking.

    index = w_semantic * S + w_tfidf * T + w_skills * K        (initial weights 0.50 / 0.20 / 0.30)

    S  semantic component   = max(0, raw semantic cosine)               in [0, 1]
    T  TF-IDF component     = raw TF-IDF cosine                         in [0, 1]
    K  required-skill coverage = |matched required| / |required|        in [0, 1]

The weights are initial design choices, not proven optimal, and are never tuned on evaluation
labels. The index is a number for ORDERING candidates. It is not a probability of being hired or
qualified, and not a validated percentage match.

Why these transformations (and no others)
-----------------------------------------
* Every component is computed from ONE resume and the job description only. Nothing depends on
  the other candidates in the batch, so adding or removing an unrelated candidate cannot change
  a candidate's score. (Batch min-max/z-score normalisation was rejected for exactly this reason.)
* TF-IDF: its IDF values would depend on the batch if fitted on all uploads. So by default the
  TF-IDF vectorizer is fitted per (job description, one resume) pair (`tfidf_scope="pairwise"`),
  or on a fixed reference corpus if a fitted TfidfMatcher is supplied (`"reference"`). The raw
  TF-IDF cosine is already in [0, 1] because TF-IDF weights are non-negative.
* Semantic: the MiniLM cosine is in [-1, 1]. A negative value means "dissimilar", not "negatively
  qualified", so the component is max(0, cosine). This is a documented clip, applied only when the
  raw value is negative, flagged on the candidate (`semantic_clipped`), and the raw cosine is always
  kept in the result.
* Not rescaled: a TF-IDF cosine of 0.2 and a semantic cosine of 0.2 do NOT mean the same thing.
  Observed on synthetic data, semantic cosines of unrelated resumes sit well above 0 while TF-IDF
  cosines of unrelated resumes are exactly 0, and the semantic scores spread about three times wider.
  So a component's real influence on the ordering is roughly weight x spread, not the weight alone.
  No calibration constants are applied; a calibrated variant would need a separate calibration set
  and is not part of this baseline.

Missing information
-------------------
* No required skills identified in the job description: K is UNAVAILABLE (None), never 1.0. The index
  is then computed from the available components with their weights renormalised to sum to 1
  (0.50/0.70 and 0.20/0.70). All candidates of one job description are treated alike, so their ranking
  stays consistent, but such an index is not comparable with the index of a job description that had
  required skills. A warning says so.
* If the job description shows no heading or cue phrase, every identified skill is treated as required
  by default and the result says the split is uncertain (`split_uncertain`).
* Skills come from a dictionary matcher, not a trained model: missing a skill in the resume may mean
  it is not in the dictionary or is worded differently. Skill coverage is therefore evidence for a
  human reviewer, not a verdict.
* Nobody is rejected automatically. Files that could not be read (scanned, corrupt, ...) or that are
  duplicates are listed in `excluded` with the reason, and need a human decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.inputs import Candidates, normalise_candidates
from src.pdf_parser import ExtractionResult
from src.preprocessing import preprocess_for_skills, preprocess_for_tfidf
from src.semantic_matcher import SemanticMatcher
from src.skill_extractor import SkillExtractor, get_default_extractor
from src.tfidf_matcher import TfidfMatcher

_TIE_DECIMALS = 12
_EVIDENCE_CHARS = 300

LIMITATION_NOTE = ("Skills are found by a dictionary matcher and may be missed or over-matched; "
                   "the index orders candidates for human review and is not a probability or percentage.")


# ---- configuration and arithmetic -------------------------------------------
@dataclass(frozen=True)
class ScoringConfig:
    """Weights of the combined index. They must be non-negative and sum to 1."""
    w_semantic: float = 0.50
    w_tfidf: float = 0.20
    w_skills: float = 0.30

    def __post_init__(self):
        values = (self.w_semantic, self.w_tfidf, self.w_skills)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise TypeError("weights must be finite numbers")
        if any(v < 0 for v in values):
            raise ValueError("weights must be non-negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError(f"weights must sum to 1, got {sum(values)}")

    def as_dict(self) -> Dict[str, float]:
        return {"semantic": self.w_semantic, "tfidf": self.w_tfidf, "skills": self.w_skills}


@dataclass(frozen=True)
class CombinedScore:
    index: float
    semantic_component: float
    tfidf_component: Optional[float]
    skill_component: Optional[float]
    weights_used: Dict[str, float]      # effective weights after renormalising over available components
    semantic_clipped: bool              # True if a negative semantic cosine was set to 0


def combine_scores(semantic_cosine: float, tfidf_cosine: Optional[float], coverage: Optional[float],
                   config: ScoringConfig = ScoringConfig()) -> CombinedScore:
    """The combined index for one candidate. Pure arithmetic, no models.

    `tfidf_cosine` or `coverage` may be None (unavailable): that component is left out and the
    remaining weights are divided by their sum so they again add up to 1.
    """
    def check(name, value, lo, hi):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
        if not (lo - 1e-9 <= value <= hi + 1e-9):
            raise ValueError(f"{name}={value} is outside [{lo}, {hi}]")
        return min(max(float(value), lo), hi)

    semantic_raw = check("semantic_cosine", semantic_cosine, -1.0, 1.0)
    semantic_component = max(0.0, semantic_raw)
    tfidf_component = None if tfidf_cosine is None else check("tfidf_cosine", tfidf_cosine, 0.0, 1.0)
    skill_component = None if coverage is None else check("coverage", coverage, 0.0, 1.0)

    parts = {"semantic": (config.w_semantic, semantic_component)}
    if tfidf_component is not None:
        parts["tfidf"] = (config.w_tfidf, tfidf_component)
    if skill_component is not None:
        parts["skills"] = (config.w_skills, skill_component)
    total_weight = sum(w for w, _ in parts.values())
    if total_weight <= 0:
        raise ValueError("no component with a positive weight is available")
    weights_used = {name: w / total_weight for name, (w, _) in parts.items()}
    index = sum(weights_used[name] * value for name, (_, value) in parts.items())
    return CombinedScore(min(max(index, 0.0), 1.0), semantic_component, tfidf_component,
                         skill_component, weights_used, semantic_raw < 0)


# ---- result objects ---------------------------------------------------------
@dataclass(frozen=True)
class JobAnalysis:
    required_skills: Tuple[str, ...]        # canonical names, unique, in order of appearance
    preferred_skills: Tuple[str, ...]       # appear only in preferred parts
    split_uncertain: bool                   # True: no headings/cues, everything defaulted to required
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    rank: int                               # 1 = best; ties share a rank; empty resumes rank last
    combined_index: float
    tfidf_similarity: Optional[float]       # raw TF-IDF cosine (None if the job description has no usable terms)
    semantic_similarity: float              # raw MiniLM cosine, may be negative
    semantic_component: float               # max(0, semantic_similarity)
    semantic_clipped: bool
    required_skill_coverage: Optional[float]  # None = unavailable (no required skills in the job description)
    required_skills: Tuple[str, ...]
    matched_required: Tuple[str, ...]
    missing_required: Tuple[str, ...]
    matched_preferred: Tuple[str, ...]
    missing_preferred: Tuple[str, ...]
    skill_evidence: Dict[str, Tuple[str, ...]]   # matched skill -> snippets from the resume
    weights_used: Dict[str, float]
    empty: bool                             # no usable text; component scores are placeholders
    semantic_evidence: str = ""             # the resume chunk that best matched the job description
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ExcludedFile:
    file_id: str
    filename: str
    reason: str                              # "not_scored" or "duplicate"
    status: str
    message: str


@dataclass(frozen=True)
class RankingReport:
    candidates: List[CandidateResult]        # best first
    excluded: List[ExcludedFile]
    job: JobAnalysis
    weights: Dict[str, float]                # configured weights
    tfidf_scope: str                         # "pairwise" or "reference"
    semantic_model: str
    notes: Tuple[str, ...] = (LIMITATION_NOTE,)

    def by_id(self) -> Dict[str, CandidateResult]:
        return {c.candidate_id: c for c in self.candidates}

    def to_records(self) -> List[dict]:
        """Flat rows (one per candidate, best first) for a table or CSV."""
        join = "; ".join
        rows = []
        for c in self.candidates:
            rows.append({
                "rank": c.rank,
                "candidate_id": c.candidate_id,
                "combined_index": c.combined_index,
                "semantic_similarity": c.semantic_similarity,
                "tfidf_similarity": c.tfidf_similarity,
                "required_skill_coverage": c.required_skill_coverage,
                "required_skills_total": len(c.required_skills),
                "matched_required": join(c.matched_required),
                "missing_required": join(c.missing_required),
                "matched_preferred": join(c.matched_preferred),
                "no_usable_text": c.empty,
                "warnings": " | ".join(c.warnings),
            })
        return rows


def partition_extractions(extractions: Sequence[ExtractionResult]) -> Tuple[List[Tuple[str, str]], List[ExcludedFile]]:
    """Split parser output into ([(file_id, text)] to score, [ExcludedFile] with reasons).

    Unusable files (scanned, corrupt, blank, too large, ...) and duplicates are not scored but are
    always listed, never dropped silently.
    """
    usable: List[Tuple[str, str]] = []
    excluded: List[ExcludedFile] = []
    for r in extractions:
        if not r.usable:
            excluded.append(ExcludedFile(r.file_id, r.filename, "not_scored", r.status.value,
                                         f"{r.message} Needs a manual look; the file was not scored."))
        elif r.duplicate_of is not None:
            excluded.append(ExcludedFile(r.file_id, r.filename, "duplicate", r.status.value,
                                         f"Identical content to '{r.duplicate_of}'; not ranked a second time."))
        else:
            usable.append((r.file_id, r.text))
    return usable, excluded


# ---- the ranker -------------------------------------------------------------
class CandidateRanker:
    """Integrates parser output, skill extraction, TF-IDF and semantic matching.

    Create once and reuse (the semantic model is loaded once, lazily, by its matcher).
    `tfidf_matcher`: pass a TfidfMatcher fitted on a reference corpus to use "reference" scope;
    by default TF-IDF is fitted per (job description, resume) pair.
    """

    def __init__(self, semantic_matcher: Optional[SemanticMatcher] = None,
                 tfidf_matcher: Optional[TfidfMatcher] = None,
                 skill_extractor: Optional[SkillExtractor] = None,
                 config: Optional[ScoringConfig] = None):
        self.semantic = semantic_matcher or SemanticMatcher()
        self.tfidf = tfidf_matcher or TfidfMatcher()
        self.extractor = skill_extractor or get_default_extractor()
        self.config = config or ScoringConfig()

    # -- job description ------------------------------------------------------
    def analyse_job(self, job_description: str) -> Tuple[JobAnalysis, "object"]:
        """Required / preferred skills of the job description, with uncertainty warnings."""
        jobskills = self.extractor.extract_job_skills(preprocess_for_skills(job_description))
        required, preferred = tuple(jobskills.required), tuple(jobskills.preferred)
        warnings: List[str] = []
        uncertain = False
        if not required and not preferred:
            warnings.append("No skills from the dictionary were found in the job description, so required-skill "
                            "coverage is unavailable and the index uses only semantic and TF-IDF similarity "
                            "(weights renormalised). Such an index is not comparable with one computed for a "
                            "job description that has required skills.")
        elif not required:
            warnings.append("All skills identified in the job description appear only in preferred / nice-to-have "
                            "parts, so there are no required skills: coverage is unavailable and the index uses "
                            "only semantic and TF-IDF similarity (weights renormalised).")
        elif not jobskills.structured:
            uncertain = True
            warnings.append(f"The job description has no headings or cue phrases (e.g. 'Requirements:', 'nice to "
                            f"have'), so all {len(required)} identified skills are treated as required by default; "
                            "the required/preferred split is uncertain.")
        return JobAnalysis(required, preferred, uncertain, tuple(warnings)), jobskills

    # -- scoring --------------------------------------------------------------
    def _tfidf_scores(self, job_description: str, pairs: Sequence[Tuple[str, str]]):
        """({id: raw cosine}, scope). Scope 'pairwise' fits on (JD, that resume) only, so a score never
        depends on the other candidates; 'reference' uses the fixed IDF of the supplied matcher."""
        if not preprocess_for_tfidf(job_description).strip():
            return None, ("reference" if self.tfidf.fit_scope == "reference" else "pairwise")
        if self.tfidf.fit_scope == "reference":
            return self.tfidf.rank(job_description, pairs).score_by_id(), "reference"
        scores = {}
        for cid, text in pairs:
            scores[cid] = self.tfidf.rank(job_description, [(cid, text)]).scores[0].score
        return scores, "pairwise"

    def rank(self, job_description: str, candidates: Candidates,
             excluded: Sequence[ExcludedFile] = ()) -> RankingReport:
        """Score and rank candidates given as {id: text} or [(id, text), ...]."""
        if not isinstance(job_description, str):
            raise TypeError(f"job_description must be a string, got {type(job_description).__name__}")
        pairs = normalise_candidates(candidates)
        if not job_description.strip():
            raise ValueError("the job description is empty")

        job, jobskills = self.analyse_job(job_description)
        scope = "reference" if self.tfidf.fit_scope == "reference" else "pairwise"
        model_name = self.semantic.model_name
        if not pairs:
            return RankingReport([], list(excluded), job, self.config.as_dict(), scope, model_name)

        tfidf_scores, scope = self._tfidf_scores(job_description, pairs)
        semantic = self.semantic.rank(job_description, pairs)
        sem_by_id = {s.candidate_id: s for s in semantic.scores}

        required = job.required_skills
        coverage_available = bool(required)
        rows = []
        for position, (cid, text) in enumerate(pairs):
            sem = sem_by_id[cid]
            resume_hits = self.extractor.extract(preprocess_for_skills(text))
            found = set(resume_hits)
            matched_req = tuple(s for s in required if s in found)
            missing_req = tuple(s for s in required if s not in found)
            coverage = len(matched_req) / len(required) if coverage_available else None
            pref_matched = tuple(s for s in job.preferred_skills if s in found)
            pref_missing = tuple(s for s in job.preferred_skills if s not in found)

            tfidf_value = None if tfidf_scores is None else tfidf_scores[cid]
            combined = combine_scores(sem.score, tfidf_value, coverage, self.config)

            warnings: List[str] = []
            if sem.empty:
                warnings.append("No usable text in this resume: component scores are placeholders (0) and the "
                                "candidate is ranked last. Check the file manually.")
            if combined.semantic_clipped:
                warnings.append(f"Semantic cosine was negative ({sem.score:.3f}); it counts as 0 in the index.")
            if not coverage_available:
                warnings.append("Required-skill coverage unavailable (no required skills in the job description); "
                                "the index uses semantic and TF-IDF similarity only.")
            elif job.split_uncertain:
                warnings.append("Required/preferred split of the job description is uncertain; coverage treats "
                                "every identified skill as required.")
            if tfidf_scores is None:
                warnings.append("The job description has no usable terms for TF-IDF; TF-IDF is not part of the index.")
            if not sem.empty and not resume_hits:
                warnings.append("No dictionary skills were found in this resume. This can reflect the resume's "
                                "layout or wording, or skills missing from the dictionary, not a lack of skills.")

            evidence = ""
            if sem.best_matches and sem.chunks:
                best = max(sem.best_matches, key=lambda m: m.cosine)
                evidence = sem.chunks[best.resume_chunk].text[:_EVIDENCE_CHARS]
            rows.append((sem.empty, combined.index, position, CandidateResult(
                candidate_id=cid, rank=0, combined_index=combined.index,
                tfidf_similarity=tfidf_value, semantic_similarity=sem.score,
                semantic_component=combined.semantic_component, semantic_clipped=combined.semantic_clipped,
                required_skill_coverage=coverage, required_skills=required,
                matched_required=matched_req, missing_required=missing_req,
                matched_preferred=pref_matched, missing_preferred=pref_missing,
                skill_evidence={s: tuple(resume_hits[s].evidence) for s in matched_req},
                weights_used=combined.weights_used, empty=sem.empty,
                semantic_evidence=evidence, warnings=tuple(warnings))))

        # Non-empty resumes first; then descending index; ties keep input order (deterministic).
        rows.sort(key=lambda r: (r[0], -round(r[1], _TIE_DECIMALS), r[2]))
        ranked: List[CandidateResult] = []
        previous_key, previous_rank = None, 0
        for pos, (empty, index, _, result) in enumerate(rows, start=1):
            key = (empty, round(index, _TIE_DECIMALS))
            rank = previous_rank if key == previous_key else pos
            previous_key, previous_rank = key, rank
            ranked.append(CandidateResult(**{**result.__dict__, "rank": rank}))
        return RankingReport(ranked, list(excluded), job, self.config.as_dict(), scope, model_name)

    # -- integration with the PDF parser -------------------------------------
    def rank_extractions(self, job_description: str,
                         extractions: Sequence[ExtractionResult]) -> RankingReport:
        """Rank the usable, non-duplicate results of `extract_many`. Everything else is listed in
        `excluded` with its reason; nothing is silently dropped and nobody is rejected automatically."""
        usable, excluded = partition_extractions(extractions)
        return self.rank(job_description, usable, excluded)


if __name__ == "__main__":
    # SYNTHETIC demonstration data written for this project. Not real resumes.
    jd = ("Backend engineer.\nRequirements:\n- Python\n- SQL and PostgreSQL\n- Docker\n- REST APIs\n- CI/CD\n"
          "Nice to have:\n- Kubernetes\n")
    demo = {
        "synthetic_A.pdf": "Python developer. REST APIs with Flask, PostgreSQL and SQL tuning. Docker images and "
                           "CI/CD pipelines on GitHub Actions.",
        "synthetic_B.pdf": "Python scripts and SQL reports for finance; some Docker usage. Used sklearn.",
        "synthetic_C.pdf": "Frontend developer. React, JavaScript, CSS, Figma design systems.",
        "synthetic_D.pdf": "Pastry chef. Baked bread, croissants and wedding cakes.",
        "synthetic_E.pdf": "",
    }
    report = CandidateRanker().rank(jd, demo)
    print(f"SYNTHETIC EXAMPLE | weights={report.weights} tfidf_scope={report.tfidf_scope}")
    print(f"required={list(report.job.required_skills)} preferred={list(report.job.preferred_skills)}")
    print("index = matching index for ordering, not a probability or percentage")
    for c in report.candidates:
        cov = "n/a" if c.required_skill_coverage is None else f"{c.required_skill_coverage:.2f}"
        print(f"  rank {c.rank} {c.candidate_id:<16} index={c.combined_index:.4f} sem={c.semantic_similarity:+.4f} "
              f"tfidf={c.tfidf_similarity:.4f} coverage={cov} matched={list(c.matched_required)} "
              f"missing={list(c.missing_required)}")
