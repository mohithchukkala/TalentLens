"""Draw the evaluation figures for the presentation from the STORED evaluation artifacts.

    python docs/figures/make_figures.py

Reads artifacts/evaluation_results.csv and artifacts/evaluation_scores.csv (nothing is re-evaluated, nothing
is typed in by hand) and writes three SVG files next to this script. Pure Python, no extra dependencies.
Every figure carries the caveat that the data are SYNTHETIC. Colours: first three slots of a validated
categorical palette (all-pairs colour-vision separation checked); values are printed on the bars because the
aqua is below 3:1 contrast on a light background.
"""

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent
METHODS = [("A_tfidf", "TF-IDF", "--c1"), ("B_semantic", "Semantic", "--c2"), ("C_combined", "Combined", "--c3")]
METRICS = [("precision@3", "Precision@3"), ("recall@3", "Recall@3"), ("mrr", "MRR"), ("ndcg@3", "nDCG@3")]
JOB_NAMES = {"JD-BE": "Backend", "JD-FE": "Frontend", "JD-DA": "Data analyst", "JD-ML": "Machine learning", "JD-FS": "Full-stack"}

STYLE = """
<style>
  .fig { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e3e2de; --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --neutral:#6b6a66; }
  @media (prefers-color-scheme: dark) { .fig { --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#3a3a37; --c1:#3987e5; --c2:#d95926; --c3:#199e70; --neutral:#9a998f; } }
  text { font-family: Segoe UI, Helvetica, Arial, sans-serif; fill: var(--ink); }
  .t2 { fill: var(--ink2); }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--ink2); stroke-width: 1; }
</style>
"""


def read_results():
    macro, per_job = {}, {}
    with open(ROOT / "artifacts" / "evaluation_results.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r["value"]:
                continue
            target = macro if r["scope"] == "MACRO" else per_job
            key = (r["method"], r["metric"]) if r["scope"] == "MACRO" else (r["method"], r["scope"], r["metric"])
            target[key] = float(r["value"])
    return macro, per_job


def bar_path(x, y, w, base, r=4):
    h = base - y
    r = max(0.0, min(r, h, w / 2))
    return (f"M{x:.1f},{base:.1f} L{x:.1f},{y + r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + r:.1f},{y:.1f} "
            f"L{x + w - r:.1f},{y:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + w:.1f},{y + r:.1f} L{x + w:.1f},{base:.1f} Z")


def svg_open(width, height, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" class="fig" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}" role="img" aria-labelledby="t d">{STYLE}<title id="t">{title}</title><desc id="d">{desc}</desc>'
            f'<rect width="{width}" height="{height}" fill="var(--surface)"/>')


def header(title, subtitle):
    return (f'<text x="24" y="32" font-size="18" font-weight="600">{title}</text>'
            f'<text x="24" y="54" font-size="12.5" class="t2">{subtitle}</text>')


def footer(height, line1, line2):
    return (f'<text x="24" y="{height - 26}" font-size="11.5" class="t2">{line1}</text>'
            f'<text x="24" y="{height - 10}" font-size="11.5" class="t2">{line2}</text>')


def legend(x, y, items):
    out, cursor = [], x
    for label, colour in items:
        out.append(f'<rect x="{cursor}" y="{y - 10}" width="12" height="12" rx="3" fill="var({colour})"/>'
                   f'<text x="{cursor + 18}" y="{y}" font-size="12.5">{label}</text>')
        cursor += 28 + 7.2 * len(label)
    return "".join(out)


def grouped_bars(groups, series, values, title, subtitle, desc, file_name, width=780, height=498):
    """groups: [(key, label)], series: [(key, label, css colour var)], values[(series_key, group_key)] -> float."""
    left, right, top, bottom = 56, 20, 118, 74
    plot_w, plot_h = width - left - right, height - top - bottom
    base = top + plot_h
    parts = [svg_open(width, height, title, desc), header(title, subtitle),
             legend(24, 84, [(label, colour) for _, label, colour in series])]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = base - tick * plot_h
        parts.append(f'<line class="{"axis" if tick == 0 else "grid"}" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>'
                     f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" class="t2">{tick:.2f}</text>')
    group_w = plot_w / len(groups)
    bar_w, gap = 36, 2
    cluster = len(series) * bar_w + (len(series) - 1) * gap
    for gi, (gkey, glabel) in enumerate(groups):
        x0 = left + gi * group_w + (group_w - cluster) / 2
        for si, (skey, slabel, colour) in enumerate(series):
            v = values[(skey, gkey)]
            x = x0 + si * (bar_w + gap)
            y = base - v * plot_h
            parts.append(f'<path d="{bar_path(x, y, bar_w, base)}" fill="var({colour})"><title>{slabel} | {glabel}: {v:.3f}</title></path>'
                         f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" font-size="11" text-anchor="middle">{v:.3f}</text>')
        parts.append(f'<text x="{left + gi * group_w + group_w / 2:.1f}" y="{base + 22}" font-size="12.5" text-anchor="middle">{glabel}</text>')
    parts.append(footer(height, "Source: artifacts/evaluation_results.csv. SYNTHETIC data: 5 invented jobs, 46 invented resumes.",
                        "Not real-world performance; differences between methods are not shown to be significant.") + "</svg>")
    (OUT / file_name).write_text("".join(parts), encoding="utf-8")


def horizontal_bars(rows, title, subtitle, desc, file_name, width=780, height=342):
    """rows: [(label, value, note)] with values in 0..1, one neutral colour (identity is carried by the labels)."""
    left, right, top = 250, 170, 92
    bar_h, gap = 30, 22
    plot_w = width - left - right
    base_y = top + len(rows) * (bar_h + gap)
    parts = [svg_open(width, height, title, desc), header(title, subtitle)]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + tick * plot_w
        parts.append(f'<line class="{"axis" if tick == 0 else "grid"}" x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{base_y - gap + 4}"/>'
                     f'<text x="{x:.1f}" y="{base_y - gap + 22}" font-size="11" text-anchor="middle" class="t2">{tick:.2f}</text>')
    for i, (label, value, note) in enumerate(rows):
        y = top + i * (bar_h + gap)
        w = value * plot_w
        r = min(4, w)
        path = (f"M{left},{y} L{left + w - r:.1f},{y} A{r},{r} 0 0 1 {left + w:.1f},{y + r} L{left + w:.1f},{y + bar_h - r} "
                f"A{r},{r} 0 0 1 {left + w - r:.1f},{y + bar_h} L{left},{y + bar_h} Z")
        parts.append(f'<text x="{left - 10}" y="{y + bar_h / 2 + 4:.1f}" font-size="13" text-anchor="end">{label}</text>'
                     f'<path d="{path}" fill="var(--neutral)"><title>{label}: {value:.3f} ({note})</title></path>'
                     f'<text x="{left + w + 8:.1f}" y="{y + bar_h / 2 + 4:.1f}" font-size="12.5">{value:.3f}</text>'
                     f'<text x="{left + w + 52:.1f}" y="{y + bar_h / 2 + 4:.1f}" font-size="11.5" class="t2">{note}</text>')
    parts.append(footer(height, "Source: artifacts/evaluation_scores.csv (46 SYNTHETIC judgments).",
                        "Descriptive statistic of one small set; not a reason to change a weight.") + "</svg>")
    (OUT / file_name).write_text("".join(parts), encoding="utf-8")


def component_shares():
    rows = list(csv.DictReader(open(ROOT / "artifacts" / "evaluation_scores.csv", encoding="utf-8")))
    sem = [max(0.0, float(r["semantic"])) for r in rows]
    tf = [float(r["tfidf"]) for r in rows]
    cov = [float(r["skill_coverage"]) for r in rows]
    terms = {"Skill coverage (weight 0.30)": [0.30 * v for v in cov], "Semantic similarity (weight 0.50)": [0.50 * v for v in sem],
             "TF-IDF similarity (weight 0.20)": [0.20 * v for v in tf]}
    total = [sum(t[i] for t in terms.values()) for i in range(len(rows))]
    mean_total = statistics.mean(total)
    variance = sum((x - mean_total) ** 2 for x in total) / len(total)
    out = []
    for label, term in terms.items():
        mean_term = statistics.mean(term)
        share = sum((term[i] - mean_term) * (total[i] - mean_total) for i in range(len(total))) / len(total) / variance
        raw = {"Skill coverage (weight 0.30)": cov, "Semantic similarity (weight 0.50)": sem, "TF-IDF similarity (weight 0.20)": tf}[label]
        out.append((label, share, f"std. dev. of the raw score {statistics.pstdev(raw):.3f}"))
    return out


def main():
    macro, per_job = read_results()
    series = [(key, label, colour) for key, label, colour in METHODS]
    grouped_bars(
        groups=METRICS, series=series, values={(m, k): macro[(m, k)] for m, _, _ in METHODS for k, _ in METRICS},
        title="Ranking quality on the synthetic evaluation set (macro average, K = 3)",
        subtitle="Combined beat TF-IDF but did not clearly beat semantic similarity alone.",
        desc="Grouped bars of Precision@3, Recall@3, MRR and nDCG@3 for TF-IDF, semantic and combined ranking, macro-averaged over 5 synthetic jobs.",
        file_name="fig_evaluation_macro.svg")
    jobs = [(j, name) for j, name in JOB_NAMES.items()]
    grouped_bars(
        groups=jobs, series=series, values={(m, j): per_job[(m, j, "ndcg@3")] for m, _, _ in METHODS for j, _ in jobs},
        title="nDCG@3 per job: no method wins everywhere",
        subtitle="Combined is better on Backend and Data analyst, worse on Frontend, Machine learning and Full-stack than semantic alone.",
        desc="Grouped bars of nDCG@3 for each of the five synthetic jobs and three methods.",
        file_name="fig_ndcg_per_job.svg")
    horizontal_bars(
        rows=component_shares(),
        title="Which term drives the combined index? (share of its variance, 46 synthetic judgments)",
        subtitle="The terms are added without rescaling, so the widest-spread term controls the ordering.",
        desc="Horizontal bars of each weighted term's share of the variance of the combined matching index: skill coverage about 0.75, semantic about 0.20, TF-IDF about 0.05.",
        file_name="fig_component_influence.svg")
    for name in ("fig_evaluation_macro.svg", "fig_ndcg_per_job.svg", "fig_component_influence.svg"):
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
