import html, sys
from pathlib import Path
import pymupdf
from resume_data import CANDIDATES

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
E = html.escape
SECTION_TITLES = {"summary": "Professional Summary", "skills": "Technical Skills", "projects": "Projects",
                  "education": "Education", "certs": "Certifications and Courses (fictional sample credentials)",
                  "extras": "Achievements and Activities"}


def css(c):
    a = "#" + c["accent"]
    serif = c["layout"] in ("single_serif", "traditional")
    font = "serif" if serif else "sans-serif"
    trad = c["layout"] == "traditional"
    return f"""
    body {{ font-family: {font}; font-size: 9.6pt; color: #1b1b1b; line-height: 1.28; }}
    h1 {{ font-size: {'22pt' if not trad else '20pt'}; color: {a}; margin: 0 0 1pt 0; {'text-align:center;' if trad else ''} }}
    .headline {{ font-size: 11pt; color: #333333; margin: 0 0 3pt 0; {'text-align:center;' if trad else ''} }}
    .contact {{ font-size: 9pt; color: #444444; margin: 0 0 6pt 0; {'text-align:center;' if trad else ''} }}
    h2 {{ font-size: {'10.5pt' if not trad else '10.5pt'}; color: {a}; margin: 9pt 0 3pt 0; padding-bottom: 1pt;
          border-bottom: 1px solid {a}; {'text-transform: uppercase;' if trad else ''} }}
    p {{ margin: 0 0 3pt 0; }}
    ul {{ margin: 1pt 0 4pt 0; }}
    li {{ margin: 0 0 1.5pt 0; }}
    .role {{ font-weight: bold; }}
    .meta {{ color: #444444; }}
    .tech {{ color: #333333; font-style: italic; }}
    .item {{ margin: 0 0 5pt 0; }}
    """


def contact_line(c):
    return f"{E(c['email'])} | {E(c['phone'])} | {E(c['location'])}"


def sec_summary(c):
    t = "Career Objective" if c["name"] in ("Ananya Iyer", "Priya Sharma") else ("Profile" if c["name"] in ("Farhan Qureshi", "Sneha Kulkarni") else "Professional Summary")
    return f"<h2>{t}</h2><p>{E(c['summary'])}</p>"


def sec_skills(c):
    rows = "".join(f"<p><b>{E(k)}:</b> {E(v)}</p>" for k, v in c["skills"])
    return f"<h2>Technical Skills</h2>{rows}"


def sec_experience(c):
    if not c["experience"]:
        return ""
    out = f"<h2>{E(c.get('exp_title', 'Work Experience'))}</h2>"
    for e in c["experience"]:
        out += f"<div class='item'><p><span class='role'>{E(e['title'])}</span>, {E(e['org'])}</p>"
        out += f"<p class='meta'>{E(e['dates'])} | {E(e['place'])}</p>"
        if e.get("responsibilities"):
            out += "<p><b>Responsibilities</b></p><ul>" + "".join(f"<li>{E(b)}</li>" for b in e["responsibilities"]) + "</ul>"
            out += "<p><b>Achievements</b></p>"
        out += "<ul>" + "".join(f"<li>{E(b)}</li>" for b in e["bullets"]) + "</ul>"
        out += f"<p class='tech'>Technologies used: {E(e['tech'])}</p></div>"
    return out


def sec_projects(c):
    title = "Academic Projects" if not c["experience"] else "Projects"
    out = f"<h2>{title}</h2>"
    for p in c["projects"]:
        out += f"<div class='item'><p><span class='role'>{E(p['title'])}</span></p><ul>"
        out += "".join(f"<li>{E(b)}</li>" for b in p["bullets"]) + "</ul>"
        out += f"<p class='tech'>Technologies used: {E(p['tech'])}</p></div>"
    return out


def sec_education(c):
    out = "<h2>Education</h2>"
    for e in c["education"]:
        out += f"<div class='item'><p><span class='role'>{E(e['degree'])}</span></p><p>{E(e['inst'])}</p><p class='meta'>{E(e['year'])} | {E(e['score'])}</p>"
        if e["note"]:
            out += f"<p>{E(e['note'])}</p>"
        out += "</div>"
    return out


def sec_certs(c):
    return "<h2>Certifications and Courses (fictional sample credentials)</h2><ul>" + "".join(f"<li>{E(x)}</li>" for x in c["certs"]) + "</ul>"


def sec_extras(c):
    if not c["extras"]:
        return ""
    return "<h2>Achievements and Activities</h2><ul>" + "".join(f"<li>{E(x)}</li>" for x in c["extras"]) + "</ul>"


SEC = {"summary": sec_summary, "skills": sec_skills, "experience": sec_experience, "projects": sec_projects,
       "education": sec_education, "certs": sec_certs, "extras": sec_extras}


def header(c):
    return f"<h1>{E(c['name'])}</h1><p class='headline'>{E(c['headline'])}</p><p class='contact'>{contact_line(c)}</p>"


def build(c):
    path = OUT / c["file"]
    A4 = pymupdf.paper_rect("a4")
    style = css(c)
    raw = Path(str(path) + ".raw")
    writer = pymupdf.DocumentWriter(str(raw))
    if c["layout"] != "two_col":
        body = header(c) + "".join(SEC[s](c) for s in c["order"])
        story = pymupdf.Story(html=body, user_css=style)
        where = A4 + (50, 46, -50, -62)
        more = 1
        while more:
            dev = writer.begin_page(A4)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()
    else:
        head = pymupdf.Story(html=header(c), user_css=style)
        main = pymupdf.Story(html="".join(SEC[s](c) for s in c["order"]), user_css=style)
        side = pymupdf.Story(html="".join(SEC[s](c) for s in c["side_order"]), user_css=style)
        first = True
        more = 1
        while more:
            dev = writer.begin_page(A4)
            if first:
                _, filled = head.place(pymupdf.Rect(46, 40, 549, 140))
                head.draw(dev)
                top = filled[3] + 4
                moreside, _ = side.place(pymupdf.Rect(46, top, 190, 780))
                assert not moreside, "sidebar overflow in " + c["file"]
                side.draw(dev)
                more, _ = main.place(pymupdf.Rect(204, top, 549, 780))
                first = False
            else:
                more, _ = main.place(pymupdf.Rect(46, 46, 549, 780))
            main.draw(dev)
            writer.end_page()
    writer.close()
    # footer with page numbers
    doc = pymupdf.open(str(raw))
    n = len(doc)
    for i, page in enumerate(doc, 1):
        page.insert_text((50, 822), "Fictional sample document: all people, organisations and credentials are invented.", fontsize=7.5, color=(0.4, 0.4, 0.4))
        page.insert_text((500, 822), f"Page {i} of {n}", fontsize=7.5, color=(0.4, 0.4, 0.4))
    doc.save(str(path)); doc.close()
    try:
        raw.unlink()
    except OSError:
        pass
    return n


for c in CANDIDATES:
    print(c["file"], build(c))
