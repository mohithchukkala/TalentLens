"""Evaluation of the three ranking methods on the SYNTHETIC evaluation set.

    python -m src.evaluate            # writes artifacts/evaluation_*.csv|json, prints a summary

Methods (all read from ONE CandidateRanker run per job, so no matcher logic is repeated here and
all methods see exactly the same inputs):
    A_tfidf     raw pairwise TF-IDF cosine
    B_semantic  raw MiniLM aggregated cosine (negative values are not clipped for ranking)
    C_combined  combined index 0.50 / 0.20 / 0.30

Metrics at K = 3: Precision@K, Recall@K, MRR, nDCG@K. Relevant = label >= 2; nDCG uses graded
labels with gain 2^label - 1 and discount 1 / log2(position + 1). Macro average over jobs.
The protocol is written down in PROJECT_LOG.md and EXPERIMENT_RESULTS.md and was fixed before the
dataset was created. Nothing here tunes any setting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "evaluation" / "dataset.json"
LABELS_PATH = ROOT / "data" / "evaluation" / "labels.csv"
ANNOTATIONS_PATH = ROOT / "data" / "evaluation" / "skill_annotations.json"
DICTIONARY_PATH = ROOT / "data" / "skill_dictionary.json"
RUBRIC_PATH = ROOT / "data" / "evaluation" / "RUBRIC.md"
OUT_DIR = ROOT / "artifacts"

K = 3
RELEVANT_MIN_LABEL = 2
VALID_LABELS = (0, 1, 2, 3)
METHODS = ("A_tfidf", "B_semantic", "C_combined")
METRICS = (f"precision@{K}", f"recall@{K}", "mrr", f"ndcg@{K}")
TIE_POLICIES = ("input", "best", "worst")
_TIE_DECIMALS = 12


# =============================================================================
# Dataset
# =============================================================================
@dataclass(frozen=True)
class Dataset:
    jobs: Dict[str, dict]                 # job_id -> {"job_id", "title", "description"}
    resumes: Dict[str, dict]              # resume_id -> {"resume_id", "job_id", "profile_type", "text"}
    labels: Dict[str, int]                # resume_id -> 0..3
    rationales: Dict[str, str]

    def pool(self, job_id: str) -> List[str]:
        return [rid for rid, r in self.resumes.items() if r["job_id"] == job_id]


def hashed_order(ids: Iterable[str]) -> List[str]:
    """A fixed, label-independent order (sorted by SHA-256 of the id), used as the input order so
    that ids such as 'BE-01' (a strong match) do not win ties just by sorting first."""
    return sorted(ids, key=lambda i: hashlib.sha256(i.encode("utf-8")).hexdigest())


def load_dataset(dataset_path: Path = DATASET_PATH, labels_path: Path = LABELS_PATH) -> Dataset:
    """Load and validate. Raises ValueError with a precise message for any problem."""
    raw = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    for key in ("meta", "jobs", "resumes"):
        if key not in raw:
            raise ValueError(f"dataset is missing the '{key}' section")
    if raw["meta"].get("synthetic") is not True:
        raise ValueError("dataset.meta.synthetic must be true: this project only evaluates labelled synthetic data")

    jobs: Dict[str, dict] = {}
    for job in raw["jobs"]:
        for field in ("job_id", "title", "description"):
            if not isinstance(job.get(field), str) or not job[field].strip():
                raise ValueError(f"job {job.get('job_id')!r}: field '{field}' must be a non-empty string")
        if job["job_id"] in jobs:
            raise ValueError(f"duplicate job_id {job['job_id']!r}")
        jobs[job["job_id"]] = job

    resumes: Dict[str, dict] = {}
    for res in raw["resumes"]:
        for field in ("resume_id", "job_id", "profile_type", "text"):
            if not isinstance(res.get(field), str) or not res[field].strip():
                raise ValueError(f"resume {res.get('resume_id')!r}: field '{field}' must be a non-empty string")
        if res["resume_id"] in resumes:
            raise ValueError(f"duplicate resume_id {res['resume_id']!r}")
        if res["job_id"] not in jobs:
            raise ValueError(f"resume {res['resume_id']!r} refers to unknown job_id {res['job_id']!r}")
        resumes[res["resume_id"]] = res

    labels: Dict[str, int] = {}
    rationales: Dict[str, str] = {}
    with open(labels_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        needed = {"resume_id", "job_id", "label", "rationale"}
        if not reader.fieldnames or not needed <= set(reader.fieldnames):
            raise ValueError(f"labels file needs columns {sorted(needed)}")
        for row in reader:
            rid = row["resume_id"]
            if rid not in resumes:
                raise ValueError(f"label for unknown resume_id {rid!r}")
            if rid in labels:
                raise ValueError(f"duplicate label for resume_id {rid!r}")
            if row["job_id"] != resumes[rid]["job_id"]:
                raise ValueError(f"label row for {rid!r} names job {row['job_id']!r} but the resume belongs to "
                                 f"{resumes[rid]['job_id']!r}")
            try:
                label = int(row["label"])
            except ValueError:
                raise ValueError(f"label for {rid!r} is not an integer: {row['label']!r}") from None
            if label not in VALID_LABELS:
                raise ValueError(f"label for {rid!r} must be one of {VALID_LABELS}, got {label}")
            if not row["rationale"].strip():
                raise ValueError(f"label for {rid!r} has no rationale")
            labels[rid] = label
            rationales[rid] = row["rationale"]
    missing = sorted(set(resumes) - set(labels))
    if missing:
        raise ValueError(f"resumes without a label: {missing}")
    for job_id in jobs:
        if not any(r["job_id"] == job_id for r in resumes.values()):
            raise ValueError(f"job {job_id!r} has no resumes")
    return Dataset(jobs, resumes, labels, rationales)


# =============================================================================
# Ordering (ties) and metrics: pure functions
# =============================================================================
def order_candidates(scores: Dict[str, float], input_order: Sequence[str],
                     labels: Optional[Dict[str, int]] = None, tie_break: str = "input") -> List[str]:
    """Candidate ids best first.

    Equal scores (12 decimals) form a tie group. tie_break:
      "input" keep the input order inside a tie group (the main, label-independent policy);
      "best"  order a tie group by label, highest first (optimistic bound);
      "worst" order a tie group by label, lowest first (pessimistic bound).
    """
    if tie_break not in TIE_POLICIES:
        raise ValueError(f"tie_break must be one of {TIE_POLICIES}")
    if tie_break != "input" and labels is None:
        raise ValueError("labels are needed for best/worst tie-breaking")
    position = {cid: i for i, cid in enumerate(input_order)}
    missing = [cid for cid in input_order if cid not in scores]
    if missing:
        raise ValueError(f"no score for candidates {missing}")

    def key(cid):
        s = -round(scores[cid], _TIE_DECIMALS)
        if tie_break == "input":
            return (s, position[cid])
        sign = -1 if tie_break == "best" else 1
        return (s, sign * labels[cid], position[cid])

    return sorted(input_order, key=key)


def _relevant(label: int) -> bool:
    return label >= RELEVANT_MIN_LABEL


def precision_at_k(ranked: Sequence[str], labels: Dict[str, int], k: int = K) -> Optional[float]:
    """(relevant in top k) / k. Missing slots (fewer than k candidates) count as non-relevant.
    None if the pool has no relevant candidate."""
    if not any(_relevant(labels[c]) for c in ranked):
        return None
    return sum(_relevant(labels[c]) for c in ranked[:k]) / k


def recall_at_k(ranked: Sequence[str], labels: Dict[str, int], k: int = K) -> Optional[float]:
    """(relevant in top k) / (all relevant in the pool). None if the pool has no relevant candidate."""
    total = sum(_relevant(labels[c]) for c in ranked)
    if total == 0:
        return None
    return sum(_relevant(labels[c]) for c in ranked[:k]) / total


def reciprocal_rank(ranked: Sequence[str], labels: Dict[str, int]) -> Optional[float]:
    """1 / (rank of the first relevant candidate). None if the pool has no relevant candidate."""
    if not any(_relevant(labels[c]) for c in ranked):
        return None
    for position, cid in enumerate(ranked, start=1):
        if _relevant(labels[cid]):
            return 1.0 / position
    return 0.0  # unreachable: a relevant candidate exists


def dcg(gains_labels: Sequence[int]) -> float:
    """sum over positions i = 1.. of (2^label - 1) / log2(i + 1)."""
    return sum((2 ** lab - 1) / math.log2(i + 1) for i, lab in enumerate(gains_labels, start=1))


def ndcg_at_k(ranked: Sequence[str], labels: Dict[str, int], k: int = K) -> Optional[float]:
    """DCG@k / IDCG@k with graded labels. IDCG uses the best k labels in the pool.
    None if IDCG is 0 (every label is 0)."""
    ideal = dcg(sorted((labels[c] for c in ranked), reverse=True)[:k])
    if ideal == 0:
        return None
    return dcg([labels[c] for c in ranked[:k]]) / ideal


def job_metrics(ranked: Sequence[str], labels: Dict[str, int], k: int = K) -> Dict[str, Optional[float]]:
    pool_labels = {c: labels[c] for c in ranked}
    return {
        f"precision@{k}": precision_at_k(ranked, pool_labels, k),
        f"recall@{k}": recall_at_k(ranked, pool_labels, k),
        "mrr": reciprocal_rank(ranked, pool_labels),
        f"ndcg@{k}": ndcg_at_k(ranked, pool_labels, k),
    }


def macro_average(values: Iterable[Optional[float]]) -> Tuple[Optional[float], int]:
    """(mean over the defined values, number of defined values). Undefined (None) jobs are excluded."""
    defined = [v for v in values if v is not None]
    if not defined:
        return None, 0
    return sum(defined) / len(defined), len(defined)


# =============================================================================
# Running the methods and evaluating
# =============================================================================
def run_methods(dataset: Dataset, ranker=None) -> List[dict]:
    """One CandidateRanker run per job. Returns one row per (job, resume) with the raw component
    scores of all three methods plus the skill details."""
    from src.ranker import CandidateRanker
    ranker = ranker or CandidateRanker()
    rows: List[dict] = []
    for job_id, job in dataset.jobs.items():
        order = hashed_order(dataset.pool(job_id))
        pairs = [(rid, dataset.resumes[rid]["text"]) for rid in order]
        report = ranker.rank(job["description"], pairs)
        for c in report.candidates:
            res = dataset.resumes[c.candidate_id]
            rows.append({
                "job_id": job_id, "resume_id": c.candidate_id, "profile_type": res["profile_type"],
                "label": dataset.labels[c.candidate_id],
                "tfidf": c.tfidf_similarity, "semantic": c.semantic_similarity,
                "semantic_component": c.semantic_component,
                "skill_coverage": c.required_skill_coverage, "combined": c.combined_index,
                "required_skills": "; ".join(c.required_skills),
                "matched_required": "; ".join(c.matched_required),
                "missing_required": "; ".join(c.missing_required),
                "input_position": order.index(c.candidate_id),
            })
    return rows


METHOD_SCORE_COLUMN = {"A_tfidf": "tfidf", "B_semantic": "semantic", "C_combined": "combined"}


def evaluate_scores(rows: List[dict], score_columns: Dict[str, str], tie_break: str = "input",
                    k: int = K) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """{method: {job_id | 'MACRO': {metric: value}}} from stored rows. Also returns, under the key
    'MACRO', the macro average and, under 'MACRO_defined_jobs', how many jobs were defined."""
    labels = {r["resume_id"]: r["label"] for r in rows}
    jobs = list(dict.fromkeys(r["job_id"] for r in rows))
    results: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for method, column in score_columns.items():
        per_job: Dict[str, Dict[str, Optional[float]]] = {}
        for job_id in jobs:
            job_rows = sorted((r for r in rows if r["job_id"] == job_id), key=lambda r: r["input_position"])
            scores = {r["resume_id"]: r[column] for r in job_rows}
            ranked = order_candidates(scores, [r["resume_id"] for r in job_rows], labels, tie_break)
            per_job[job_id] = job_metrics(ranked, labels, k)
        macro, defined = {}, {}
        for metric in per_job[jobs[0]]:
            macro[metric], defined[metric] = macro_average(per_job[j][metric] for j in jobs)
        per_job["MACRO"] = macro
        per_job["MACRO_defined_jobs"] = defined
        results[method] = per_job
    return results


def ties_at_boundary(rows: List[dict], column: str, k: int = K) -> Dict[str, bool]:
    """Per job: True if candidates at rank k and k+1 have equal scores (the top-k set depends on tie order)."""
    out = {}
    for job_id in dict.fromkeys(r["job_id"] for r in rows):
        scores = sorted((round(r[column], _TIE_DECIMALS) for r in rows if r["job_id"] == job_id), reverse=True)
        out[job_id] = len(scores) > k and scores[k - 1] == scores[k]
    return out


# ---- exploratory ablations (computed from stored raw components, never used to choose settings) ----
def exploratory_columns(rows: List[dict]) -> Dict[str, str]:
    """Adds ablation score columns to `rows` in place and returns {name: column}."""
    from src.ranker import ScoringConfig, combine_scores
    variants = {
        "X1_skills_only": None,
        "X2_semantic_tfidf_no_skills": ScoringConfig(0.5, 0.2, 0.3),   # coverage left out -> renormalised
        "X3_semantic_skills_no_tfidf": ScoringConfig(0.625, 0.0, 0.375),
        "X4_equal_weights": ScoringConfig(1 / 3, 1 / 3, 1 / 3),
    }
    columns = {}
    for name, config in variants.items():
        column = "x_" + name
        columns[name] = column
        for r in rows:
            if config is None:
                r[column] = r["skill_coverage"] if r["skill_coverage"] is not None else 0.0
            elif name == "X2_semantic_tfidf_no_skills":
                r[column] = combine_scores(r["semantic"], r["tfidf"], None, config).index
            else:
                r[column] = combine_scores(r["semantic"], r["tfidf"], r["skill_coverage"], config).index
    return columns


# ---- skill extraction ----
def skill_extraction_eval(dataset: Dataset, annotations: Dict[str, List[str]], extractor=None) -> dict:
    """Mention-level precision / recall / F1 of the dictionary extractor against hand annotations.
    Per resume the unit is the set of canonical skills. Micro-averaged over all annotated resumes."""
    from src.preprocessing import preprocess_for_skills
    from src.skill_extractor import get_default_extractor
    extractor = extractor or get_default_extractor()
    per_resume = []
    tp = fp = fn = 0
    for rid, gold_list in annotations.items():
        if rid not in dataset.resumes:
            raise ValueError(f"annotation for unknown resume {rid!r}")
        gold = set(gold_list)
        predicted = set(extractor.extract(preprocess_for_skills(dataset.resumes[rid]["text"])))
        t, f_p, f_n = len(gold & predicted), len(predicted - gold), len(gold - predicted)
        tp, fp, fn = tp + t, fp + f_p, fn + f_n
        per_resume.append({"resume_id": rid, "tp": t, "fp": f_p, "fn": f_n,
                           "false_positives": sorted(predicted - gold), "false_negatives": sorted(gold - predicted)})
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"n_resumes": len(annotations), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "per_resume": per_resume}


# =============================================================================
# Output
# =============================================================================
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_config(dataset: Dataset, ranker) -> dict:
    import numpy, sklearn, torch, sentence_transformers
    from src.tfidf_matcher import make_vectorizer
    vec = make_vectorizer()
    sem = ranker.semantic
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": {"numpy": numpy.__version__, "scikit-learn": sklearn.__version__,
                     "torch": torch.__version__, "sentence-transformers": sentence_transformers.__version__},
        "methods": {
            "A_tfidf": {"scope": "pairwise (IDF fitted on job description + that one resume)",
                        "vectorizer": {k: str(getattr(vec, k)) for k in
                                       ("lowercase", "token_pattern", "ngram_range", "sublinear_tf", "use_idf",
                                        "smooth_idf", "norm", "min_df", "max_df")}},
            "B_semantic": {"model": sem.model_name, "device": sem.device, "max_seq_length": sem.max_seq_length,
                           "chunk_tokens": sem.chunk_tokens, "overlap_tokens": sem.overlap_tokens,
                           "top_k": sem.top_k, "aggregation": "JD-chunk-token-weighted mean of best resume-chunk cosine",
                           "ranked_by": "raw cosine (not clipped)"},
            "C_combined": {"weights": ranker.config.as_dict(),
                           "formula": "0.50*max(0,semantic) + 0.20*tfidf + 0.30*required_skill_coverage"},
        },
        "metrics": {"k": K, "relevant_min_label": RELEVANT_MIN_LABEL, "gain": "2^label - 1",
                    "discount": "1/log2(position+1)", "aggregate": "macro average over jobs, undefined jobs excluded",
                    "tie_policy_main": "input order = SHA-256 order of candidate id, equal scores (12 decimals) keep it",
                    "tie_bounds": "best/worst: tie groups ordered by label"},
        "data": {"n_jobs": len(dataset.jobs), "n_resumes": len(dataset.resumes),
                 "sha256": {"dataset.json": _sha(DATASET_PATH), "labels.csv": _sha(LABELS_PATH),
                            "RUBRIC.md": _sha(RUBRIC_PATH), "skill_annotations.json": _sha(ANNOTATIONS_PATH),
                            "skill_dictionary.json": _sha(DICTIONARY_PATH)}},
    }


def _fmt(value: Optional[float]) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def write_outputs(out_dir: Path, rows: List[dict], main: dict, bounds: Dict[str, dict], exploratory: dict,
                  skill_eval: dict, config: dict, boundary_ties: Dict[str, Dict[str, bool]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    score_fields = ["job_id", "resume_id", "profile_type", "label", "tfidf", "semantic", "semantic_component",
                    "skill_coverage", "combined", "required_skills", "matched_required", "missing_required"]
    with open(out_dir / "evaluation_scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=score_fields + ["x_" + n for n in exploratory["columns"]], extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["job_id"], r["resume_id"])))

    def write_metric_table(path, results, extra=None):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow((["tie_policy"] if extra else []) + ["method", "scope", "metric", "value", "jobs_defined"])
            for method, per_job in results.items():
                for scope, metrics in per_job.items():
                    if scope == "MACRO_defined_jobs":
                        continue
                    for metric, value in metrics.items():
                        defined = per_job["MACRO_defined_jobs"][metric] if scope == "MACRO" else (0 if value is None else 1)
                        w.writerow(([extra] if extra else []) + [method, scope, metric,
                                                                  "" if value is None else f"{value:.6f}", defined])

    write_metric_table(out_dir / "evaluation_results.csv", main)
    with open(out_dir / "evaluation_tie_bounds.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tie_policy", "method", "metric", "macro_value"])
        for policy, results in bounds.items():
            for method, per_job in results.items():
                for metric, value in per_job["MACRO"].items():
                    w.writerow([policy, method, metric, "" if value is None else f"{value:.6f}"])
    write_metric_table(out_dir / "evaluation_exploratory.csv", exploratory["results"])
    (out_dir / "skill_extraction_eval.json").write_text(json.dumps(skill_eval, indent=2), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF, semantic and combined ranking on the synthetic set.")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output directory (default: artifacts)")
    args = parser.parse_args(argv)

    dataset = load_dataset()
    from src.ranker import CandidateRanker
    ranker = CandidateRanker()
    rows = run_methods(dataset, ranker)

    main_results = evaluate_scores(rows, METHOD_SCORE_COLUMN, "input")
    bounds = {policy: evaluate_scores(rows, METHOD_SCORE_COLUMN, policy) for policy in ("best", "worst")}
    columns = exploratory_columns(rows)
    exploratory = {"columns": columns, "results": evaluate_scores(rows, columns, "input")}
    annotations = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))["annotations"]
    skill_eval = skill_extraction_eval(dataset, annotations)
    boundary = {m: ties_at_boundary(rows, c) for m, c in METHOD_SCORE_COLUMN.items()}
    config = run_config(dataset, ranker)
    write_outputs(args.out, rows, main_results, bounds, exploratory, skill_eval, config, boundary)

    print(f"Evaluated {len(dataset.resumes)} judgments over {len(dataset.jobs)} jobs (SYNTHETIC data). Written to {args.out}")
    print(f"{'method':<12}{'scope':<9}" + "".join(f"{m:>14}" for m in METRICS))
    for method, per_job in main_results.items():
        for scope in [j for j in per_job if j not in ("MACRO", "MACRO_defined_jobs")] + ["MACRO"]:
            print(f"{method:<12}{scope:<9}" + "".join(f"{_fmt(per_job[scope][m]):>14}" for m in METRICS))
    print("ties at the rank-3 boundary:", {m: [j for j, t in b.items() if t] for m, b in boundary.items()})
    print("skill extraction (micro):", {k: skill_eval[k] for k in ("n_resumes", "tp", "fp", "fn", "precision", "recall", "f1")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
