"""Validates the 10 generated PDFs: opens them, extracts text, checks sections, clipping, page counts, README/CSV consistency.
Run from the project root:  python data/sample_resumes/realistic_backend_set/generator/validate_resumes.py
"""
import csv, re, sys, unicodedata
from pathlib import Path
import pymupdf

here = Path(__file__).resolve().parent
folder = here.parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(folder.parent.parent.parent))
from resume_data import CANDIDATES
from src.pdf_parser import extract_text_from_path

SECTION_PATTERNS = {
    "summary": r"Professional Summary|Career Objective|Profile",
    "skills": r"Technical Skills",
    "experience": r"Internship Experience|Professional Experience|Work Experience|Internship\b",
    "projects": r"Projects",
    "education": r"Education",
    "certifications": r"Certifications and Courses",
}
norm = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().lower()
readme = (folder / "README.md").read_text(encoding="utf-8") if (folder / "README.md").exists() else ""
csv_rows = list(csv.DictReader(open(folder / "expected_profiles.csv", encoding="utf-8"))) if (folder / "expected_profiles.csv").exists() else []
csv_files = {r["Filename"] for r in csv_rows}
failures = 0
print(f"{'file':34} pages words status  sections")
for c in CANDIDATES:
    p = folder / c["file"]
    problems = []
    if not p.exists():
        print(c["file"], "MISSING"); failures += 1; continue
    doc = pymupdf.open(p); pages = len(doc)
    raw = "\n".join(pg.get_text() for pg in doc)
    res = extract_text_from_path(p)           # the application's own parser
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", res.text))  # the app also applies NFKC (ligature fi -> fi)
    raw = unicodedata.normalize("NFKC", raw)
    words = len(text.split())
    if str(res.status) != "Status.OK": problems.append(f"parser status {res.status}")
    if not (1 <= pages <= 2): problems.append(f"{pages} pages")
    if c["name"] not in text: problems.append("name missing")
    for k in (c["email"], c["phone"], c["location"]):
        if k not in raw: problems.append(f"contact item missing: {k}")
    found = [s for s, pat in SECTION_PATTERNS.items() if re.search(pat, text)]
    for s in SECTION_PATTERNS:
        if s not in found and not (s == "experience" and not c["experience"]): problems.append(f"section missing: {s}")
    if c["experience"] == [] and re.search(r"Experience", text): pass
    # clipping: every bullet / paragraph must be present in the extracted text
    flat = norm(raw)
    pieces = [c["summary"]] + [b for e in c["experience"] for b in e["bullets"] + e.get("responsibilities", [])] + \
             [b for pr in c["projects"] for b in pr["bullets"]] + c["certs"] + c["extras"]
    missing = [x for x in pieces if norm(x)[:60] not in flat or norm(x)[-40:] not in flat]
    if missing: problems.append(f"{len(missing)} text pieces not found (clipping?)")
    sentences = len(re.findall(r"[.!?](\s|$)", text))
    if words < 280 or sentences < 12: problems.append(f"too thin: {words} words, {sentences} sentences")
    if c["file"] not in readme: problems.append("not in README")
    if c["file"] not in csv_files: problems.append("not in CSV")
    print(f"{c['file']:34} {pages:5} {words:5} {'OK' if not problems else 'FAIL'}   {','.join(found)}")
    for pr in problems: print("     -", pr)
    failures += bool(problems)
extra = csv_files - {c["file"] for c in CANDIDATES}
if extra: print("CSV lists unknown files:", extra); failures += 1
print("\nRESULT:", "all 10 PDFs passed" if not failures else f"{failures} problem file(s)")
sys.exit(1 if failures else 0)
