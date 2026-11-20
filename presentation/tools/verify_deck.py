"""Technical-accuracy checks for the deck, the charts and the diagrams (read-only; changes nothing).

    python presentation/tools/verify_deck.py        (needs python-pptx; tooling only, not in requirements.txt)

Checks: 12 slides, speaker notes on every slide; the evaluation table on slide 10 equals artifacts/evaluation_results.csv; the formula weights in the deck
equal the defaults in src/ranker.py; chart tooltips equal the CSV artifacts; the diagram's skill count equals the dictionary; every author cited on the
slides appears in docs/RELATED_WORK.md; flags phrases that could overclaim so that they can be read in context.
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

from pptx import Presentation

root = Path(__file__).resolve().parent.parent.parent
deck = Presentation(str(root / "presentation" / "Resume_Screening_Project.pptx"))
problems = []


def check(ok, message):
    print(("PASS  " if ok else "FAIL  ") + message)
    if not ok:
        problems.append(message)


def slide_text(slide):
    parts = []
    for shape in slide.shapes:
        if shape.has_text_frame:
            parts.append(shape.text_frame.text)
        if getattr(shape, "has_table", False) and shape.has_table:
            for row in shape.table.rows:
                parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


texts = [slide_text(s) for s in deck.slides]
notes = [s.notes_slide.notes_text_frame.text if s.has_notes_slide else "" for s in deck.slides]
check(len(texts) == 12, f"deck has {len(texts)} slides (expected 12)")
check(all(len(n.strip()) > 200 for n in notes), "every slide has speaker notes (minimum length 200 characters): " + str([len(n) for n in notes]))

# --- evaluation table on slide 10 vs stored artifacts
macro = {}
with open(root / "artifacts" / "evaluation_results.csv", newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if row["scope"] == "MACRO":
            macro[(row["method"], row["metric"])] = float(row["value"])
table = None
for shape in deck.slides[9].shapes:
    if getattr(shape, "has_table", False) and shape.has_table:
        table = [[c.text for c in r.cells] for r in shape.table.rows]
names = {"TF-IDF": "A_tfidf", "Semantic": "B_semantic", "Combined": "C_combined"}
metrics = ["precision@3", "recall@3", "mrr", "ndcg@3"]
ok = table is not None
for line in (table or [])[1:]:
    for metric, shown in zip(metrics, line[1:]):
        ok &= shown == f"{macro[(names[line[0]], metric)]:.3f}"
check(ok, "slide 10 evaluation table equals artifacts/evaluation_results.csv (3 decimals)")

# --- weights
ranker_src = (root / "src" / "ranker.py").read_text(encoding="utf-8")
weights = [float(x) for x in re.findall(r"w_(?:semantic|tfidf|skills): float = ([0-9.]+)", ranker_src)]
check(weights == [0.5, 0.2, 0.3], f"src/ranker.py default weights are {weights}")
check("0.50·max(0, S) + 0.20·T + 0.30·K" in texts[8] and "0.50 x max(0, S) + 0.20 x T + 0.30 x K" in (root / "docs/figures/diagram_architecture.svg").read_text(encoding="utf-8"),
      "formula and weights on slide 9 and in the architecture diagram match the defaults")

# --- model and chunk facts
combined_text = "\n".join(texts)
for fact in ("384", "256", "128-token", "200 tokens", "all-MiniLM-L6-v2", "Apache-2.0"):
    check(fact in combined_text, f"deck mentions '{fact}'")
check("not tested" in texts[7] and "128-token" in texts[7], "slide 8 states the 128-token vs 200-token question as not tested")

# --- skill dictionary count in the diagram
dictionary = json.loads((root / "data" / "skill_dictionary.json").read_text(encoding="utf-8"))
count = len(dictionary["skills"])
svg = (root / "docs" / "figures" / "diagram_architecture.svg").read_text(encoding="utf-8")
check(f"dictionary of {count} skills" in svg, f"architecture diagram says 'dictionary of {count} skills' (dictionary file has {count})")

# --- charts vs CSV artifacts (tooltips inside the SVG files)
figs = root / "docs" / "figures"
tips = re.findall(r"<title>([^<|]+) \| ([^:]+): ([0-9.]+)</title>", (figs / "fig_evaluation_macro.svg").read_text(encoding="utf-8"))
label = {"TF-IDF": "A_tfidf", "Semantic": "B_semantic", "Combined": "C_combined"}
metric_label = {"Precision@3": "precision@3", "Recall@3": "recall@3", "MRR": "mrr", "nDCG@3": "ndcg@3"}
ok = len(tips) == 12 and all(f"{macro[(label[m], metric_label[g])]:.3f}" == v for m, g, v in tips)
check(ok, f"fig_evaluation_macro.svg: {len(tips)} bar values equal the MACRO rows of the CSV")
per_job = {}
with open(root / "artifacts" / "evaluation_results.csv", newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if row["scope"].startswith("JD-") and row["metric"] == "ndcg@3":
            per_job[(row["method"], row["scope"])] = float(row["value"])
jobs = {"Backend": "JD-BE", "Frontend": "JD-FE", "Data analyst": "JD-DA", "Machine learning": "JD-ML", "Full-stack": "JD-FS"}
tips = re.findall(r"<title>([^<|]+) \| ([^:]+): ([0-9.]+)</title>", (figs / "fig_ndcg_per_job.svg").read_text(encoding="utf-8"))
ok = len(tips) == 15 and all(f"{per_job[(label[m], jobs[j])]:.3f}" == v for m, j, v in tips)
check(ok, f"fig_ndcg_per_job.svg: {len(tips)} bar values equal the per-job nDCG@3 rows of the CSV")
rows = list(csv.DictReader(open(root / "artifacts" / "evaluation_scores.csv", encoding="utf-8")))
sem = [max(0.0, float(r["semantic"])) * 0.5 for r in rows]
tf = [float(r["tfidf"]) * 0.2 for r in rows]
cov = [float(r["skill_coverage"]) * 0.3 for r in rows]
tot = [a + b + c for a, b, c in zip(sem, tf, cov)]
mt = statistics.mean(tot)
var = sum((x - mt) ** 2 for x in tot) / len(tot)
share = {name: sum((t - statistics.mean(term)) * (x - mt) for t, x in zip(term, tot)) / len(tot) / var for name, term in (("cov", cov), ("sem", sem), ("tf", tf))}
influence_svg = (figs / "fig_component_influence.svg").read_text(encoding="utf-8")
ok = all(f"{share[k]:.3f}" in influence_svg for k in share)
check(ok, f"fig_component_influence.svg shares {({k: round(v, 3) for k, v in share.items()})} equal the recomputed variance shares")
check("about 75%" in texts[9] and 0.74 < share["cov"] < 0.76, "slide 10 'about 75%' agrees with the recomputed coverage share")

# --- citations on the slides appear in the related-work document
related = (root / "docs" / "RELATED_WORK.md").read_text(encoding="utf-8")
for author in ("Spärck Jones", "Salton", "Reimers", "Wang", "Karpukhin", "Thakur", "Gugnani", "Raghavan", "Bolukbasi", "Järvelin"):
    on_slides = author in "\n".join(texts[3:4] + texts[11:12])
    check(on_slides and author in related, f"'{author}' is cited on the slides and listed in docs/RELATED_WORK.md")

# --- wording that could overclaim (printed for reading, not an automatic failure)
flag = re.compile(r"outperform|proven|state-of-the-art|novel|accurate|production|real-world|significan|best\b|better", re.I)
print("\nPhrases to read in context (slides and notes):")
for i, (t, n) in enumerate(zip(texts, notes), start=1):
    for source, body in (("slide", t), ("notes", n)):
        for m in flag.finditer(body):
            lo, hi = max(0, m.start() - 55), min(len(body), m.end() + 55)
            print(f"  {source} {i}: ...{body[lo:hi].replace(chr(10), ' ')}...")
print("\nResult:", "all automatic checks passed" if not problems else f"{len(problems)} check(s) failed")
sys.exit(1 if problems else 0)
