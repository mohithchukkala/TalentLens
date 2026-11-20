"""Editable source of the two architecture diagrams.

    python docs/figures/make_diagrams.py

Writes diagram_architecture.svg and diagram_pipeline.svg next to this script (pure Python, no dependencies).
The SVG files are plain vector graphics (no scripts, no CSS variables) and can be opened and edited in Inkscape,
draw.io or PowerPoint; edit this script to change boxes, text or colours and re-run it. PNG versions for the slides
are rendered from the SVGs with a browser (presentation/tools/render_images.js).

The diagrams show the implemented system only: three selectable methods (TF-IDF, semantic, combined) and
skill coverage as a COMPONENT of the combined index, not a selectable method.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent

INK = "#1b2437"
MUTED = "#5b6475"
LINE = "#8b94a5"
SURFACE = "#ffffff"
NEUTRAL_FILL = "#eef1f6"
BLUE, ORANGE, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
BLUE_F, ORANGE_F, GREEN_F, VIOLET_F = "#e3eefb", "#fdebe3", "#dff5ec", "#ebe8fa"
FONT = "Segoe UI, Calibri, Arial, sans-serif"


class Svg:
    def __init__(self, width, height, title, desc):
        self.w, self.h = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-labelledby="t d" font-family="{FONT}">'
            f'<title id="t">{title}</title><desc id="d">{desc}</desc>'
            '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>'
            f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>']

    def text(self, x, y, s, size=15, weight="400", fill=INK, anchor="start"):
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>')

    def box(self, x, y, w, h, lines, fill=NEUTRAL_FILL, stroke=LINE, tag=None, tag_fill=None, dashed=False,
            title_size=15, sub_size=13):
        """lines[0] is the bold heading, the rest are smaller sub-lines. tag = small caption on the top edge."""
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        self.parts.append(f'<g><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
        n = len(lines)
        line_h = 20
        total = title_size + (n - 1) * line_h
        y0 = y + (h - total) / 2 + title_size - 3
        for i, line in enumerate(lines):
            if i == 0:
                self.text(x + w / 2, y0, line, title_size, "600", INK, "middle")
            else:
                self.text(x + w / 2, y0 + 1 + i * line_h, line, sub_size, "400", MUTED, "middle")
        if tag:
            tw = 7.0 * len(tag) + 18
            self.parts.append(f'<rect x="{x + 12}" y="{y - 12}" width="{tw}" height="22" rx="11" fill="{tag_fill or stroke}"/>')
            self.text(x + 12 + tw / 2, y + 3.5, tag, 12, "600", "#ffffff", "middle")
        self.parts.append("</g>")

    def arrow(self, points, dashed=False, color=MUTED, width=2):
        d = "M" + " L".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash} marker-end="url(#arr)"/>')

    def line(self, points, color=LINE, width=2):
        d = "M" + " L".join(f"{x},{y}" for x, y in points)
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"/>')

    def band(self, x, y, w, h, label, fill="#f7f8fb", label_x=None):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="#d9dde6" stroke-width="1.5"/>')
        self.text(label_x if label_x is not None else x + 16, y + 24, label, 13, "600", MUTED)

    def save(self, name):
        (OUT / name).write_text("".join(self.parts) + "</svg>", encoding="utf-8")
        print("wrote", OUT / name)


def architecture():
    s = Svg(1280, 840, "System architecture",
            "Streamlit interface with job description, PDF upload and a method selector; PDF parsing and validation; preprocessing; "
            "skill coverage, TF-IDF similarity and semantic similarity computed independently; combined index; ranking table and CSV export.")
    s.text(40, 38, "System architecture", 24, "700")
    s.text(40, 62, "All processing is local and in memory. Three selectable methods: A TF-IDF, B semantic, C combined (default).", 14, fill=MUTED)

    # UI inputs
    s.band(30, 80, 1220, 112, "Streamlit interface (app.py): inputs")
    s.box(60, 108, 270, 66, ["Job description", "pasted text"])
    s.box(360, 108, 270, 66, ["PDF resumes", "one or many files, in memory"])
    s.box(660, 108, 560, 66, ["Method selector (sidebar)", "A TF-IDF  |  B Semantic  |  C Combined (default):", "decides which score is ranked"],
          fill="#fff6dd", stroke="#d9a400")

    # validation and extraction
    s.band(30, 208, 1220, 118, "Input validation and PDF extraction (pdf_parser.py, app_logic.py)", label_x=520)
    s.box(60, 238, 400, 70, ["Input validation", "empty job description, no upload,", "TF-IDF job description with only stop words"])
    s.box(490, 238, 730, 70, ["PDF text extraction (PyMuPDF), one status per file",
                              "readable | no usable text | probably scanned (OCR needed) | password-protected |",
                              "corrupted | empty | too large | duplicate: listed with a reason, not ranked"])
    s.arrow([(195, 174), (195, 238)])
    s.arrow([(495, 174), (495, 238)])

    # preprocessing
    s.box(60, 350, 1160, 54, ["Preprocessing (preprocessing.py): three views of the text",
                              "skills view: case and punctuation kept   |   TF-IDF view: lowercase tokens (c++, node.js, ci/cd)   |   embedding view: near-original text, contact details removed"])
    s.arrow([(855, 308), (855, 350)])

    # parallel scoring paths (A | K | B)
    s.band(30, 424, 1220, 188, "Independent scoring paths, in parallel", label_x=270)
    s.box(60, 486, 350, 112, ["TF-IDF similarity T", "TF-IDF vectors, IDF fitted per",
                              "job-description / resume pair, cosine"], fill=BLUE_F, stroke=BLUE, tag="METHOD A", tag_fill=BLUE)
    s.box(465, 486, 350, 112, ["Skill coverage K", "dictionary of 114 skills, aliases, rules",
                               "required skills found / required skills"], fill=VIOLET_F, stroke=VIOLET,
          tag="COMPONENT ONLY, NOT SELECTABLE", tag_fill=VIOLET, dashed=True)
    s.box(870, 486, 350, 112, ["Semantic similarity S", "all-MiniLM-L6-v2 embeddings, chunks of",
                               "up to 200 tokens, cosine (raw)"], fill=ORANGE_F, stroke=ORANGE, tag="METHOD B", tag_fill=ORANGE)
    for x in (235, 760, 1045):          # the middle arrow enters right of the box tag
        s.arrow([(x, 404), (x, 486)])

    # combined
    s.box(260, 656, 760, 60, ["Combined matching index (ranker.py)",
                              "0.50 x max(0, S) + 0.20 x T + 0.30 x K     (weights fixed in advance, not tuned)"],
          fill=GREEN_F, stroke=GREEN, tag="METHOD C (DEFAULT)", tag_fill=GREEN)
    s.arrow([(235, 598), (235, 630), (440, 630), (440, 656)], color=BLUE)
    s.arrow([(640, 598), (640, 656)], color=VIOLET)
    s.arrow([(1045, 598), (1045, 630), (840, 630), (840, 656)], color=ORANGE)

    # ranking and outputs
    s.box(60, 752, 1160, 56, ["Ranking and outputs (app_logic.py, app.py)",
                              "ranked table  |  per-candidate details (matched / missing skills, evidence, warnings)  |  files not ranked  |  CSV export"])
    s.arrow([(640, 716), (640, 752)], color=GREEN)
    s.arrow([(60, 542), (44, 542), (44, 734), (100, 734), (100, 752)], dashed=True, color=BLUE)
    s.text(52, 726, "A alone: ranked by T", 12, "400", BLUE)
    s.arrow([(1220, 542), (1236, 542), (1236, 734), (1180, 734), (1180, 752)], dashed=True, color=ORANGE)
    s.text(1228, 726, "B alone: ranked by S", 12, "400", ORANGE, "end")
    s.arrow([(1220, 141), (1264, 141), (1264, 780), (1220, 780)], dashed=True, color="#d9a400")
    s.save("diagram_architecture.svg")


def pipeline():
    s = Svg(1300, 880, "End-to-end processing pipeline",
            "Sequential stages: upload, file validation, text extraction, preprocessing and skill identification; then three independent scoring paths in parallel; "
            "then combined score, ranking, result inspection and CSV export.")
    s.text(40, 38, "End-to-end processing pipeline", 24, "700")
    s.text(40, 62, "Sequential stages go left to right; the three scoring paths run in parallel for each resume and merge only in the combined score.", 14, fill=MUTED)

    labels = [("1  Upload", ["job description +", "PDF resumes"]),
              ("2  File validation", ["status per file;", "bad files listed, not ranked"]),
              ("3  Text extraction", ["PyMuPDF blocks in", "reading order"]),
              ("4  Preprocess + skills", ["3 text views;", "required / preferred skills"])]
    x = 40
    s.text(40, 92, "Sequential stages", 13, "600", MUTED)
    for i, (head, subs) in enumerate(labels):
        s.box(x, 104, 270, 86, [head] + subs)
        if i < 3:
            s.arrow([(x + 270, 147), (x + 300, 147)])
        x += 300

    # bus from stage 4 into the three parallel lanes
    s.line([(1105, 190), (1105, 224), (245, 224)])
    s.band(40, 244, 1220, 214, "Independent scoring paths, in parallel", label_x=270)
    lane_y = 296
    s.box(70, lane_y, 350, 140, ["Lane A: TF-IDF", "tokens -> TF-IDF vectors", "(IDF fitted per resume / job pair)", "cosine -> T"],
          fill=BLUE_F, stroke=BLUE, tag="METHOD A", tag_fill=BLUE)
    s.box(475, lane_y, 350, 140, ["Lane K: skill coverage", "dictionary match of required skills", "K = matched / required",
                                  "(unique canonical skills)"], fill=VIOLET_F, stroke=VIOLET, tag="COMPONENT, NOT A MODE", tag_fill=VIOLET, dashed=True)
    s.box(880, lane_y, 350, 140, ["Lane B: semantic", "chunks (<= 200 tokens) -> embeddings", "best-matching chunk per job part",
                                  "cosine -> S (raw)"], fill=ORANGE_F, stroke=ORANGE, tag="METHOD B", tag_fill=ORANGE)
    for xx in (245, 790, 1055):         # the middle arrow enters right of the box tag
        s.arrow([(xx, 224), (xx, lane_y)])

    # combined
    s.box(380, 520, 540, 74, ["5  Combined score (method C, default)", "index = 0.50 x max(0, S) + 0.20 x T + 0.30 x K"],
          fill=GREEN_F, stroke=GREEN)
    s.arrow([(245, 436), (245, 490), (470, 490), (470, 520)], color=BLUE)
    s.arrow([(650, 436), (650, 520)], color=VIOLET)
    s.arrow([(1055, 436), (1055, 490), (830, 490), (830, 520)], color=ORANGE)

    # ranking, inspection, export
    s.box(455, 650, 390, 80, ["6  Ranking", "best first; equal scores share a rank;", "blank resumes last"])
    s.arrow([(650, 594), (650, 650)], color=GREEN)
    s.arrow([(70, 366), (46, 366), (46, 690), (455, 690)], dashed=True, color=BLUE)
    s.text(54, 682, "method A alone: ranked by T", 12, "400", BLUE)
    s.arrow([(1230, 366), (1254, 366), (1254, 690), (845, 690)], dashed=True, color=ORANGE)
    s.text(1246, 682, "method B alone: ranked by S", 12, "400", ORANGE, "end")
    s.box(250, 770, 380, 80, ["7  Result inspection", "table, matched / missing skills,", "evidence passages, warnings"])
    s.box(690, 770, 360, 80, ["8  CSV export", "only the columns the method", "calculated; blanks stay blank"])
    s.arrow([(650, 730), (650, 750), (440, 750), (440, 770)])
    s.arrow([(630, 810), (690, 810)])
    s.text(40, 866, "Dashed arrows: choosing method A or B in the sidebar ranks by that score only. Skill coverage cannot be chosen on its own.", 13, fill=MUTED)
    s.save("diagram_pipeline.svg")


if __name__ == "__main__":
    architecture()
    pipeline()
