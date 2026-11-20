"""Logic behind the Streamlit app, kept free of Streamlit so it can be tested directly.

It only orchestrates the existing modules (parser, TF-IDF matcher, semantic matcher, combined
ranker) and turns their results into rows for a table and a CSV file. No matching algorithm lives
here, and nothing is written to disk.

Three methods, each calling the existing public API:
    combined   CandidateRanker.rank(...)                       (Phase 7 formula and weights, unchanged)
    tfidf      TfidfMatcher.rank(jd, [(id, text)]) per resume  (pairwise IDF, as in the evaluation)
    semantic   SemanticMatcher.rank(jd, pairs)                 (chunking and aggregation unchanged)
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pymupdf

from src.pdf_parser import ExtractionResult, Status, extract_many
from src.preprocessing import preprocess_for_tfidf
from src.ranker import (CandidateRanker, ExcludedFile, JobAnalysis, partition_extractions)

METHOD_COMBINED = "combined"
METHOD_TFIDF = "tfidf"
METHOD_SEMANTIC = "semantic"
DEFAULT_METHOD = METHOD_COMBINED
METHOD_LABELS = {
    METHOD_COMBINED: "Combined ranking (default)",
    METHOD_TFIDF: "TF-IDF baseline",
    METHOD_SEMANTIC: "Semantic matching",
}

SCORE_LABELS = {
    "combined_index": "Combined matching index",
    "semantic_cosine": "Semantic similarity (raw cosine)",
    "tfidf_cosine": "TF-IDF similarity (cosine)",
    "required_skill_coverage": "Required-skill coverage",
}
METHOD_SCORES = {
    METHOD_COMBINED: ("combined_index", "semantic_cosine", "tfidf_cosine", "required_skill_coverage"),
    METHOD_TFIDF: ("tfidf_cosine",),
    METHOD_SEMANTIC: ("semantic_cosine",),
}

STATUS_LABELS = {
    Status.OK: "Readable",
    Status.OK_WITH_WARNINGS: "Readable, with warnings",
    Status.SCANNED_LIKELY: "Probably scanned: OCR needed",
    Status.NO_TEXT: "No usable text",
    Status.ENCRYPTED: "Password-protected",
    Status.CORRUPTED: "Unreadable or corrupted",
    Status.EMPTY_FILE: "Empty file (0 bytes)",
    Status.TOO_LARGE: "Too large",
    Status.EXTRACTION_FAILED: "Text extraction failed",
}

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "dataset.json"
_TIE_DECIMALS = 12
_EVIDENCE_CHARS = 300
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


# =============================================================================
# Result objects
# =============================================================================
@dataclass(frozen=True)
class UiRow:
    candidate_id: str
    rank: int
    empty: bool                                   # no usable text; scores are placeholders
    scores: Dict[str, Optional[float]]            # only the scores this method calculated
    required_total: Optional[int] = None          # None: skills are not part of this method
    matched_required: Optional[Tuple[str, ...]] = None
    missing_required: Optional[Tuple[str, ...]] = None
    matched_preferred: Optional[Tuple[str, ...]] = None
    missing_preferred: Optional[Tuple[str, ...]] = None
    skill_evidence: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    text_evidence: Tuple[Tuple[str, str], ...] = ()      # (label, passage)
    extra: str = ""                                       # TF-IDF: top shared terms, for the CSV
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MatchResult:
    method: str
    rows: List[UiRow]                              # best first
    excluded: List[ExcludedFile]
    job: JobAnalysis
    score_keys: Tuple[str, ...]
    weights: Optional[Dict[str, float]] = None     # combined only
    notes: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()


# =============================================================================
# Displaying untrusted text
# =============================================================================
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]()<>~$:|])")


def escape_markdown(text: str) -> str:
    """Make text from an uploaded file (file names, resume snippets, ids) safe to show through
    st.markdown / labels: it must display as plain text, never as a link, image, emphasis, colour
    directive, emoji shortcode or formula chosen by whoever wrote the resume."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", str(text))


# =============================================================================
# Inputs
# =============================================================================
def validate_inputs(job_description: str, n_files: int, method: str = DEFAULT_METHOD) -> List[str]:
    """User-facing messages for expected input problems (empty list = fine)."""
    problems: List[str] = []
    if method not in METHOD_LABELS:
        problems.append(f"Unknown matching method '{method}'.")
    if not isinstance(job_description, str) or not job_description.strip():
        problems.append("Please enter a job description.")
    elif method == METHOD_TFIDF and not preprocess_for_tfidf(job_description).strip():
        problems.append("The job description has no usable words for TF-IDF (only stop words, punctuation or numbers). "
                        "Please add the role's requirements.")
    if n_files <= 0:
        problems.append("Please upload at least one PDF resume.")
    return problems


def extract_files(files: Sequence[Tuple[str, bytes]]) -> List[ExtractionResult]:
    """Existing parser on a batch of (filename, bytes). Handles bad files itself; never raises for them."""
    return extract_many(list(files))


def status_rows(extractions: Sequence[ExtractionResult]) -> List[dict]:
    """One row per uploaded file for the extraction-status table."""
    rows = []
    for r in extractions:
        if r.duplicate_of is not None:
            used, note = "No (duplicate)", f"Identical content to '{r.duplicate_of}'; not ranked a second time."
        elif r.usable:
            used, note = "Yes", " ".join(r.warnings)
        else:
            used, note = "No", f"{r.message} Needs a manual look; the file was not scored."
        rows.append({"File": r.file_id, "Status": STATUS_LABELS.get(r.status, r.status.value),
                     "Pages": r.page_count, "Characters": r.char_count, "Used in ranking": used, "Note": note})
    return rows


# =============================================================================
# Running the three methods
# =============================================================================
def build_ranker() -> CandidateRanker:
    """The long-lived objects (semantic model loads lazily on first use). The app caches this."""
    return CandidateRanker()


def config_signature(ranker: CandidateRanker) -> str:
    """A string that changes whenever the scoring configuration changes (part of the cache key)."""
    sem = ranker.semantic
    return json.dumps({"weights": ranker.config.as_dict(), "model": getattr(sem, "model_name", "?"),
                       "chunk": getattr(sem, "chunk_tokens", None), "overlap": getattr(sem, "overlap_tokens", None),
                       "top_k": getattr(sem, "top_k", None), "tfidf": ranker.tfidf.fit_scope}, sort_keys=True)


def match(method: str, job_description: str, extractions: Sequence[ExtractionResult],
          ranker: CandidateRanker) -> MatchResult:
    """Run one method on the usable, non-duplicate resumes. Raises ValueError on invalid inputs
    (the interface validates first), and SemanticModelError if the model cannot load."""
    problems = validate_inputs(job_description, len(extractions), method)
    if problems:
        raise ValueError(" ".join(problems))
    pairs, excluded = partition_extractions(extractions)
    job, _ = ranker.analyse_job(job_description)
    keys = METHOD_SCORES[method]
    if not pairs:
        return MatchResult(method, [], excluded, job, keys,
                           warnings=("None of the uploaded files had usable text, so nothing was ranked.",))
    if method == METHOD_COMBINED:
        return _run_combined(job_description, pairs, excluded, job, ranker)
    if method == METHOD_TFIDF:
        return _run_tfidf(job_description, pairs, excluded, job, ranker)
    return _run_semantic(job_description, pairs, excluded, job, ranker)


def _run_combined(jd, pairs, excluded, job, ranker) -> MatchResult:
    report = ranker.rank(jd, pairs, excluded)
    rows = []
    for c in report.candidates:
        text_evidence = (("Best-matching passage (semantic)", c.semantic_evidence),) if c.semantic_evidence else ()
        rows.append(UiRow(
            candidate_id=c.candidate_id, rank=c.rank, empty=c.empty,
            scores={"combined_index": c.combined_index, "semantic_cosine": c.semantic_similarity,
                    "tfidf_cosine": c.tfidf_similarity, "required_skill_coverage": c.required_skill_coverage},
            required_total=len(c.required_skills), matched_required=c.matched_required,
            missing_required=c.missing_required, matched_preferred=c.matched_preferred,
            missing_preferred=c.missing_preferred, skill_evidence=dict(c.skill_evidence),
            text_evidence=text_evidence, warnings=tuple(c.warnings)))
    return MatchResult(METHOD_COMBINED, rows, list(report.excluded), report.job, METHOD_SCORES[METHOD_COMBINED],
                       weights=dict(report.weights), notes=tuple(report.notes))


def _run_tfidf(jd, pairs, excluded, job, ranker) -> MatchResult:
    """TF-IDF alone, fitted per (job description, resume) pair exactly as in the evaluation."""
    scored = []
    for position, (cid, text) in enumerate(pairs):
        s = ranker.tfidf.rank(jd, [(cid, text)]).scores[0]
        scored.append((s.token_count == 0, s.score, position, cid, s))
    scored.sort(key=lambda r: (r[0], -round(r[1], _TIE_DECIMALS), r[2]))
    rows, previous, previous_rank = [], None, 0
    for pos, (empty, score, _, cid, s) in enumerate(scored, start=1):
        key = (empty, round(score, _TIE_DECIMALS))
        rank = previous_rank if key == previous else pos
        previous, previous_rank = key, rank
        terms = "; ".join(f"{t} ({w:.3f})" for t, w in s.top_terms)
        warnings = ("No usable words in this resume after preprocessing: score 0, ranked last.",) if empty else ()
        evidence = (("Top shared terms (contribution to the cosine)", terms),) if terms else ()
        rows.append(UiRow(candidate_id=cid, rank=rank, empty=empty, scores={"tfidf_cosine": score},
                          text_evidence=evidence, extra=terms, warnings=warnings))
    return MatchResult(METHOD_TFIDF, rows, excluded, job, METHOD_SCORES[METHOD_TFIDF])


def _run_semantic(jd, pairs, excluded, job, ranker) -> MatchResult:
    ranking = ranker.semantic.rank(jd, pairs)
    rows = []
    for s in ranking.scores:
        evidence = ()
        if s.best_matches and s.chunks:
            best = max(s.best_matches, key=lambda m: m.cosine)
            evidence = (("Best-matching passage", s.chunks[best.resume_chunk].text[:_EVIDENCE_CHARS]),)
        warnings: List[str] = []
        if s.empty:
            warnings.append("No usable text in this resume: not scored and ranked last.")
        elif s.score < 0:
            warnings.append("The raw cosine is negative (the texts are dissimilar in the model's embedding space).")
        rows.append(UiRow(candidate_id=s.candidate_id, rank=s.rank, empty=s.empty,
                          scores={"semantic_cosine": s.score}, text_evidence=evidence, warnings=tuple(warnings)))
    return MatchResult(METHOD_SEMANTIC, rows, excluded, job, METHOD_SCORES[METHOD_SEMANTIC],
                       warnings=tuple(ranking.warnings))


# =============================================================================
# Table and CSV
# =============================================================================
def results_table(result: MatchResult) -> List[dict]:
    """Rows for the on-screen table; only columns for the scores the method calculated."""
    table = []
    for row in result.rows:
        record = {"Rank": row.rank, "Candidate": row.candidate_id}
        for key in result.score_keys:
            record[SCORE_LABELS[key]] = row.scores.get(key)
        if row.required_total:
            record["Required skills matched"] = f"{len(row.matched_required or ())}/{row.required_total}"
        if row.empty:
            record["Flag"] = "No usable text"
        table.append(record)
    return table


def _clean_cell(value) -> str:
    """Blank for missing values; guards against spreadsheet formula injection from file names/text."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    text = str(value)
    return "'" + text if text.startswith(_CSV_FORMULA_PREFIXES) else text


CSV_COLUMNS = {
    METHOD_COMBINED: ["method", "rank", "candidate_id", "status", "combined_matching_index", "semantic_cosine_raw",
                      "tfidf_cosine", "required_skill_coverage", "required_skills_total", "matched_required_skills",
                      "missing_required_skills", "matched_preferred_skills", "warnings"],
    METHOD_TFIDF: ["method", "rank", "candidate_id", "status", "tfidf_cosine", "top_shared_terms", "warnings"],
    METHOD_SEMANTIC: ["method", "rank", "candidate_id", "status", "semantic_cosine_raw", "warnings"],
}


def result_to_csv(result: MatchResult) -> str:
    """CSV of the ranking. Only scores the method calculated are present; missing values are blank,
    never 0. Excluded files are listed after the ranked ones with an 'excluded' status and no scores."""
    columns = CSV_COLUMNS[result.method]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    join = "; ".join
    for row in result.rows:
        record = {"method": result.method, "rank": row.rank, "candidate_id": row.candidate_id,
                  "status": "no_usable_text" if row.empty else "ranked", "warnings": " | ".join(row.warnings)}
        if result.method == METHOD_COMBINED:
            record.update({
                "combined_matching_index": row.scores.get("combined_index"),
                "semantic_cosine_raw": row.scores.get("semantic_cosine"),
                "tfidf_cosine": row.scores.get("tfidf_cosine"),
                "required_skill_coverage": row.scores.get("required_skill_coverage"),
                "required_skills_total": row.required_total,
                "matched_required_skills": join(row.matched_required or ()),
                "missing_required_skills": join(row.missing_required or ()),
                "matched_preferred_skills": join(row.matched_preferred or ())})
        elif result.method == METHOD_TFIDF:
            record.update({"tfidf_cosine": row.scores.get("tfidf_cosine"), "top_shared_terms": row.extra})
        else:
            record.update({"semantic_cosine_raw": row.scores.get("semantic_cosine")})
        writer.writerow({k: _clean_cell(v) for k, v in record.items()})
    for ex in result.excluded:
        writer.writerow({k: _clean_cell(v) for k, v in {
            "method": result.method, "rank": None, "candidate_id": ex.file_id,
            "status": f"excluded_{ex.reason}", "warnings": ex.message}.items()})
    return out.getvalue()


# =============================================================================
# Synthetic demo material (clearly synthetic; no scores are stored anywhere)
# =============================================================================
def demo_scenarios(path: Path = DATASET_PATH) -> List[Tuple[str, str]]:
    """[(job_id, title)] of the synthetic jobs, or [] if the evaluation dataset is not present."""
    if not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(j["job_id"], j["title"]) for j in data["jobs"]]


def demo_job_description(job_id: str, path: Path = DATASET_PATH) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for job in data["jobs"]:
        if job["job_id"] == job_id:
            return job["description"]
    raise ValueError(f"unknown demo scenario {job_id!r}")


def text_to_pdf_bytes(text: str, width: int = 85, lines_per_page: int = 50) -> bytes:
    """A simple text-only PDF (used to make the synthetic demo resumes)."""
    lines: List[str] = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width) or [""])
    doc = pymupdf.open()
    for start in range(0, max(len(lines), 1), lines_per_page):
        page = doc.new_page()
        y = 72
        for line in lines[start:start + lines_per_page]:
            page.insert_text((60, y), line, fontsize=10)
            y += 14
    data = doc.tobytes()
    doc.close()
    return data


def demo_resumes_zip(job_id: str, path: Path = DATASET_PATH) -> bytes:
    """ZIP of the SYNTHETIC resumes of one demo scenario as PDFs (labels are not included)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    resumes = [r for r in data["resumes"] if r["job_id"] == job_id]
    if not resumes:
        raise ValueError(f"unknown demo scenario {job_id!r}")
    # Neutral file names in a label-independent order, so the demo does not hint at the answer.
    resumes.sort(key=lambda r: hashlib.sha256(r["resume_id"].encode("utf-8")).hexdigest())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for n, r in enumerate(resumes, start=1):
            archive.writestr(f"synthetic_resume_{n:02d}.pdf", text_to_pdf_bytes(r["text"]))
    return buffer.getvalue()
