"""Tests for src/evaluate.py: metric arithmetic against hand calculations, tie handling, dataset validation.

No model is loaded here. The measured results of the real evaluation are produced by
`python -m src.evaluate`, not by these tests.
"""

import copy
import csv
import json
import math

import pytest

from src import evaluate as ev
from src.evaluate import (dcg, evaluate_scores, hashed_order, job_metrics, load_dataset, macro_average,
                          ndcg_at_k, order_candidates, precision_at_k, recall_at_k, reciprocal_rank,
                          skill_extraction_eval, ties_at_boundary)

# ranking a, b, c, d, e with labels 0, 3, 2, 0, 1
RANKED = ["a", "b", "c", "d", "e"]
LABELS = {"a": 0, "b": 3, "c": 2, "d": 0, "e": 1}


# =============================================================================
# Metrics against hand-calculated values
# =============================================================================
def test_precision_recall_mrr_ndcg_hand_example():
    # top 3 = a(0) b(3) c(2): relevant (label >= 2) are b and c.
    assert precision_at_k(RANKED, LABELS, 3) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, LABELS, 3) == pytest.approx(2 / 2)      # pool has exactly 2 relevant
    assert reciprocal_rank(RANKED, LABELS) == pytest.approx(1 / 2)     # first relevant is at rank 2
    # DCG = (2^0-1)/log2(2) + (2^3-1)/log2(3) + (2^2-1)/log2(4) = 0 + 7/1.58496 + 3/2 = 5.91651
    # IDCG (labels 3,2,1) = 7/1 + 3/log2(3) + 1/log2(4) = 9.39279 ; nDCG = 0.62990
    assert ndcg_at_k(RANKED, LABELS, 3) == pytest.approx(0.6298989694, abs=1e-9)


def test_dcg_formula_directly():
    assert dcg([3, 2, 1]) == pytest.approx(7 / 1 + 3 / math.log2(3) + 1 / 2)
    assert dcg([0, 0, 0]) == 0.0
    assert dcg([]) == 0.0


def test_perfect_ranking_with_more_relevant_than_k_has_recall_below_one():
    labels = {c: 3 for c in "abcde"}
    r = list("abcde")
    assert precision_at_k(r, labels, 3) == 1.0
    assert recall_at_k(r, labels, 3) == pytest.approx(3 / 5)           # at most 3 of 5 fit in the top 3
    assert reciprocal_rank(r, labels) == 1.0
    assert ndcg_at_k(r, labels, 3) == pytest.approx(1.0)


def test_first_relevant_beyond_the_top_k():
    labels = {"a": 0, "b": 1, "c": 1, "d": 2}
    r = ["a", "b", "c", "d"]
    assert precision_at_k(r, labels, 3) == 0.0
    assert recall_at_k(r, labels, 3) == 0.0
    assert reciprocal_rank(r, labels) == pytest.approx(1 / 4)
    # DCG = 0 + 1/log2(3) + 1/2 ; ideal labels (2,1,1): 3 + 1/log2(3) + 1/2
    expected = (1 / math.log2(3) + 0.5) / (3 + 1 / math.log2(3) + 0.5)
    assert ndcg_at_k(r, labels, 3) == pytest.approx(expected)
    assert expected == pytest.approx(0.27376, abs=1e-4)


def test_ranking_order_changes_ndcg_as_expected():
    labels = {"x": 3, "y": 2, "z": 1}
    assert ndcg_at_k(["x", "y", "z"], labels, 3) == pytest.approx(1.0)
    swapped = ndcg_at_k(["y", "x", "z"], labels, 3)
    assert swapped == pytest.approx((3 / 1 + 7 / math.log2(3) + 1 / 2) / (7 / 1 + 3 / math.log2(3) + 1 / 2))
    assert swapped == pytest.approx(0.84283, abs=1e-4)
    assert swapped < 1.0


def test_fewer_candidates_than_k():
    labels = {"a": 2, "b": 0}
    r = ["a", "b"]
    assert precision_at_k(r, labels, 3) == pytest.approx(1 / 3)        # empty slots are not relevant
    assert recall_at_k(r, labels, 3) == 1.0
    assert reciprocal_rank(r, labels) == 1.0
    assert ndcg_at_k(r, labels, 3) == pytest.approx(1.0)


def test_no_relevant_candidate_makes_binary_metrics_undefined_not_zero():
    labels = {"a": 1, "b": 0, "c": 1}
    r = ["a", "b", "c"]
    assert precision_at_k(r, labels) is None and recall_at_k(r, labels) is None
    assert reciprocal_rank(r, labels) is None
    # graded labels still define nDCG: DCG = 1/1 + 0 + 1/2 ; ideal order a, c, b gives 1 + 1/log2(3)
    assert ndcg_at_k(r, labels) == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k(["a", "c", "b"], labels) == pytest.approx(1.0)
    assert ndcg_at_k(r, {"a": 0, "b": 0, "c": 0}) is None             # IDCG = 0


def test_job_metrics_keys_and_values():
    m = job_metrics(RANKED, LABELS, 3)
    assert set(m) == {"precision@3", "recall@3", "mrr", "ndcg@3"}
    assert m["precision@3"] == pytest.approx(2 / 3) and m["mrr"] == 0.5


def test_macro_average_excludes_undefined_jobs():
    assert macro_average([1.0, None, 0.5]) == (pytest.approx(0.75), 2)
    assert macro_average([None, None]) == (None, 0)
    assert macro_average([]) == (None, 0)


# =============================================================================
# Ties
# =============================================================================
def test_tie_policies():
    scores = {"a": 0.5, "b": 0.5, "c": 0.9, "d": 0.5}
    labels = {"a": 0, "b": 3, "c": 1, "d": 2}
    order = ["a", "b", "c", "d"]
    assert order_candidates(scores, order, tie_break="input") == ["c", "a", "b", "d"]
    assert order_candidates(scores, order, labels, "best") == ["c", "b", "d", "a"]
    assert order_candidates(scores, order, labels, "worst") == ["c", "a", "d", "b"]


def test_scores_equal_to_twelve_decimals_are_tied():
    scores = {"a": 0.5, "b": 0.5 + 1e-15, "c": 0.4}
    assert order_candidates(scores, ["a", "b", "c"]) == ["a", "b", "c"]            # tie keeps input order
    assert order_candidates(scores, ["b", "a", "c"]) == ["b", "a", "c"]


def test_tie_policy_errors():
    with pytest.raises(ValueError, match="tie_break"):
        order_candidates({"a": 1.0}, ["a"], tie_break="random")
    with pytest.raises(ValueError, match="labels are needed"):
        order_candidates({"a": 1.0}, ["a"], tie_break="best")
    with pytest.raises(ValueError, match="no score"):
        order_candidates({"a": 1.0}, ["a", "b"])


def test_best_and_worst_bounds_bracket_the_main_result():
    rows = []
    for i, (rid, label) in enumerate([("a", 3), ("b", 2), ("c", 0), ("d", 0), ("e", 1)]):
        rows.append({"job_id": "J", "resume_id": rid, "label": label, "input_position": i, "s": 0.0})  # all tied
    main = evaluate_scores(rows, {"M": "s"}, "input")["M"]["MACRO"]
    best = evaluate_scores(rows, {"M": "s"}, "best")["M"]["MACRO"]
    worst = evaluate_scores(rows, {"M": "s"}, "worst")["M"]["MACRO"]
    for metric in main:
        assert worst[metric] <= main[metric] + 1e-12 <= best[metric] + 2e-12
    assert best["precision@3"] == pytest.approx(2 / 3) and worst["precision@3"] == 0.0


def test_ties_at_boundary_flag():
    rows = [{"job_id": "J", "resume_id": c, "s": s} for c, s in zip("abcde", [0.9, 0.5, 0.4, 0.4, 0.1])]
    assert ties_at_boundary(rows, "s", 3) == {"J": True}               # rank 3 and 4 equal
    rows[3]["s"] = 0.3
    assert ties_at_boundary(rows, "s", 3) == {"J": False}


def test_hashed_order_is_deterministic_a_permutation_and_not_alphabetical():
    ids = [f"BE-0{i}" for i in range(1, 10)]
    first = hashed_order(ids)
    assert first == hashed_order(reversed(ids)) == hashed_order(ids)
    assert sorted(first) == sorted(ids)
    assert first != sorted(ids)                                        # ids do not decide the order


# =============================================================================
# evaluate_scores: macro average over jobs
# =============================================================================
def test_evaluate_scores_macro_is_the_mean_of_the_per_job_values():
    rows = []
    # job 1: method puts the relevant one first; job 2: puts it last of 4
    for pos, (rid, label, score) in enumerate([("a1", 3, 0.9), ("b1", 0, 0.5), ("c1", 0, 0.1)]):
        rows.append({"job_id": "J1", "resume_id": rid, "label": label, "input_position": pos, "s": score})
    for pos, (rid, label, score) in enumerate([("a2", 2, 0.1), ("b2", 0, 0.9), ("c2", 0, 0.8), ("d2", 0, 0.7)]):
        rows.append({"job_id": "J2", "resume_id": rid, "label": label, "input_position": pos, "s": score})
    res = evaluate_scores(rows, {"M": "s"}, "input")["M"]
    assert res["J1"]["mrr"] == 1.0 and res["J2"]["mrr"] == pytest.approx(1 / 4)
    assert res["MACRO"]["mrr"] == pytest.approx((1.0 + 0.25) / 2)
    assert res["J1"]["precision@3"] == pytest.approx(1 / 3) and res["J2"]["precision@3"] == 0.0
    assert res["MACRO"]["precision@3"] == pytest.approx(1 / 6)
    assert res["MACRO_defined_jobs"]["mrr"] == 2


def test_evaluate_scores_excludes_a_job_without_relevant_candidates():
    rows = [{"job_id": "J1", "resume_id": "a", "label": 3, "input_position": 0, "s": 1.0},
            {"job_id": "J1", "resume_id": "b", "label": 0, "input_position": 1, "s": 0.0},
            {"job_id": "J2", "resume_id": "c", "label": 1, "input_position": 0, "s": 1.0},
            {"job_id": "J2", "resume_id": "d", "label": 0, "input_position": 1, "s": 0.0}]
    res = evaluate_scores(rows, {"M": "s"}, "input")["M"]
    assert res["J2"]["mrr"] is None
    assert res["MACRO"]["mrr"] == 1.0 and res["MACRO_defined_jobs"]["mrr"] == 1   # only J1 counted


# =============================================================================
# Dataset loading and validation
# =============================================================================
def test_real_dataset_loads_and_has_the_documented_shape():
    ds = load_dataset()
    assert len(ds.jobs) == 5 and len(ds.resumes) == 46 and len(ds.labels) == 46
    assert set(ds.labels.values()) <= {0, 1, 2, 3}
    for job_id in ds.jobs:
        pool = ds.pool(job_id)
        assert 9 <= len(pool) <= 10
        assert sum(ds.labels[r] >= 2 for r in pool) >= 3          # every job has at least 3 relevant candidates
        assert any(ds.labels[r] == 0 for r in pool)
    assert all(r for r in ds.rationales.values())
    types = {r["profile_type"] for r in ds.resumes.values()}
    for needed in ("negation", "keyword_list_no_evidence", "long_relevant_at_end", "describes_role_from_outside",
                   "relevant_paraphrase_no_named_tools", "sparse_ambiguous"):
        assert needed in types


def test_real_dataset_labels_are_not_correlated_with_the_input_order():
    ds = load_dataset()
    for job_id in ds.jobs:
        labels = [ds.labels[r] for r in hashed_order(ds.pool(job_id))]
        assert labels != sorted(labels, reverse=True) and labels != sorted(labels)


@pytest.fixture()
def tiny(tmp_path):
    data = {"meta": {"synthetic": True},
            "jobs": [{"job_id": "J", "title": "T", "description": "Python"}],
            "resumes": [{"resume_id": "R1", "job_id": "J", "profile_type": "p", "text": "Python"},
                        {"resume_id": "R2", "job_id": "J", "profile_type": "p", "text": "Baking"}]}
    rows = [["resume_id", "job_id", "label", "rationale"], ["R1", "J", "3", "why"], ["R2", "J", "0", "why"]]

    def write(d=None, r=None):
        (tmp_path / "d.json").write_text(json.dumps(d or data), encoding="utf-8")
        with open(tmp_path / "l.csv", "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(r or rows)
        return tmp_path / "d.json", tmp_path / "l.csv"

    return data, rows, write


def test_valid_tiny_dataset_loads(tiny):
    _, _, write = tiny
    ds = load_dataset(*write())
    assert ds.labels == {"R1": 3, "R2": 0} and ds.pool("J") == ["R1", "R2"]


def test_validation_rejects_bad_datasets(tiny):
    data, rows, write = tiny

    def broken(mutate_data=None, mutate_rows=None):
        d, r = copy.deepcopy(data), copy.deepcopy(rows)
        if mutate_data:
            mutate_data(d)
        if mutate_rows:
            mutate_rows(r)
        return write(d, r)

    with pytest.raises(ValueError, match="synthetic"):
        load_dataset(*broken(lambda d: d["meta"].update(synthetic=False)))
    with pytest.raises(ValueError, match="missing the 'jobs'"):
        load_dataset(*broken(lambda d: d.pop("jobs")))
    with pytest.raises(ValueError, match="duplicate resume_id"):
        load_dataset(*broken(lambda d: d["resumes"].append(dict(d["resumes"][0]))))
    with pytest.raises(ValueError, match="unknown job_id"):
        load_dataset(*broken(lambda d: d["resumes"][0].update(job_id="X")))
    with pytest.raises(ValueError, match="non-empty string"):
        load_dataset(*broken(lambda d: d["resumes"][0].update(text="  ")))
    with pytest.raises(ValueError, match="duplicate job_id"):
        load_dataset(*broken(lambda d: d["jobs"].append(dict(d["jobs"][0]))))
    with pytest.raises(ValueError, match="must be one of"):
        load_dataset(*broken(mutate_rows=lambda r: r[1].__setitem__(2, "4")))
    with pytest.raises(ValueError, match="not an integer"):
        load_dataset(*broken(mutate_rows=lambda r: r[1].__setitem__(2, "high")))
    with pytest.raises(ValueError, match="without a label"):
        load_dataset(*broken(mutate_rows=lambda r: r.pop(2)))
    with pytest.raises(ValueError, match="unknown resume_id"):
        load_dataset(*broken(mutate_rows=lambda r: r.append(["R9", "J", "1", "why"])))
    with pytest.raises(ValueError, match="duplicate label"):
        load_dataset(*broken(mutate_rows=lambda r: r.append(["R1", "J", "1", "why"])))
    with pytest.raises(ValueError, match="names job"):
        load_dataset(*broken(mutate_rows=lambda r: r[1].__setitem__(1, "OTHER")))
    with pytest.raises(ValueError, match="no rationale"):
        load_dataset(*broken(mutate_rows=lambda r: r[1].__setitem__(3, " ")))


# =============================================================================
# Skill-extraction metric
# =============================================================================
class StubExtractor:
    def __init__(self, found):
        self.found = found

    def extract(self, text):
        return {name: None for name in self.found[text.split()[0]]}


def test_skill_extraction_precision_recall_f1_by_hand(tiny):
    _, _, write = tiny
    ds = load_dataset(*write())
    # R1 text "Python", R2 text "Baking". Gold R1 = {Python, SQL}; predicted {Python, Docker}.
    # Gold R2 = {} ; predicted {}.   TP=1 FP=1 FN=1  -> P=0.5 R=0.5 F1=0.5
    out = skill_extraction_eval(ds, {"R1": ["Python", "SQL"], "R2": []},
                                StubExtractor({"Python": ["Python", "Docker"], "Baking": []}))
    assert (out["tp"], out["fp"], out["fn"]) == (1, 1, 1)
    assert out["precision"] == out["recall"] == out["f1"] == 0.5
    by = {r["resume_id"]: r for r in out["per_resume"]}
    assert by["R1"]["false_positives"] == ["Docker"] and by["R1"]["false_negatives"] == ["SQL"]


def test_skill_extraction_edge_cases(tiny):
    _, _, write = tiny
    ds = load_dataset(*write())
    none_found = skill_extraction_eval(ds, {"R1": ["Python"]}, StubExtractor({"Python": []}))
    assert none_found["precision"] is None and none_found["recall"] == 0.0 and none_found["f1"] is None
    nothing = skill_extraction_eval(ds, {"R2": []}, StubExtractor({"Baking": []}))
    assert nothing["precision"] is None and nothing["recall"] is None and nothing["f1"] is None
    perfect = skill_extraction_eval(ds, {"R1": ["Python"]}, StubExtractor({"Python": ["Python"]}))
    assert perfect["f1"] == 1.0
    with pytest.raises(ValueError, match="unknown resume"):
        skill_extraction_eval(ds, {"R9": []}, StubExtractor({}))
