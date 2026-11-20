"""Tests for src/pdf_parser.py.

All PDFs are generated in memory with PyMuPDF, so no resume files are needed
and nothing is written to disk (except where tmp_path is used on purpose).
"""

import pymupdf
import pytest

from src import pdf_parser
from src.pdf_parser import (Status, extract_many, extract_text_from_bytes,
                            extract_text_from_path, make_unique_ids)

RESUME_LINES_1 = [
    "Asha Rao - Software Engineer",
    "Skills: Python, SQL, Docker, Git, Machine Learning",
    "Built REST APIs with Flask and deployed them on AWS.",
]
RESUME_LINES_2 = [
    "Education: B.Tech in Computer Science, 2025",
    "Project: Resume screening system using TF-IDF and cosine similarity.",
]


def make_pdf(pages=None, encrypt_pw=None, image_pages=()) -> bytes:
    """Build a small PDF. `pages` is a list of line-lists; each page's two groups of
    lines are placed far apart vertically so they become separate text blocks."""
    doc = pymupdf.open()
    pages = pages if pages is not None else []
    for i, lines in enumerate(pages):
        page = doc.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontsize=11)
            y += 16
        if i in image_pages:
            add_image(page)
    if not pages:
        doc.new_page()  # a blank page: valid PDF, no text
    kwargs = {}
    if encrypt_pw:
        kwargs = dict(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=encrypt_pw, owner_pw=encrypt_pw + "x")
    data = doc.tobytes(**kwargs)
    doc.close()
    return data


def add_image(page):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60), False)
    pix.clear_with(200)
    page.insert_image(pymupdf.Rect(300, 300, 400, 400), pixmap=pix)


# ---- valid PDFs -------------------------------------------------------------
def test_valid_pdf_extracts_text_and_metadata():
    r = extract_text_from_bytes(make_pdf([RESUME_LINES_1]), "asha.pdf")
    assert r.status is Status.OK
    assert r.usable
    assert r.page_count == 1
    assert "Python, SQL, Docker" in r.text
    assert r.char_count >= pdf_parser.MIN_TOTAL_CHARS
    assert len(r.content_hash) == 64
    assert r.file_id == "asha.pdf"


def test_line_breaks_are_preserved_and_order_kept():
    r = extract_text_from_bytes(make_pdf([RESUME_LINES_1]))
    assert "\n" in r.text
    first = r.text.index("Asha Rao")
    second = r.text.index("Skills:")
    third = r.text.index("REST APIs")
    assert first < second < third


def test_multi_page_pdf_keeps_page_order():
    r = extract_text_from_bytes(make_pdf([RESUME_LINES_1, RESUME_LINES_2]))
    assert r.status is Status.OK
    assert r.page_count == 2
    assert r.text.index("Asha Rao") < r.text.index("Education:")


def test_no_more_than_one_blank_line_in_a_row():
    r = extract_text_from_bytes(make_pdf([RESUME_LINES_1, RESUME_LINES_2]))
    assert "\n\n\n" not in r.text


# ---- no usable text ---------------------------------------------------------
def test_blank_pdf_is_no_text_not_a_valid_empty_resume():
    r = extract_text_from_bytes(make_pdf(), "blank.pdf")
    assert r.status is Status.NO_TEXT
    assert not r.usable
    assert "blank" in r.message.lower() or "no usable text" in r.message.lower()


def test_tiny_text_is_rejected():
    r = extract_text_from_bytes(make_pdf([["Hi"]]))
    assert r.status is Status.NO_TEXT
    assert not r.usable


def test_image_only_pdf_is_flagged_as_scanned_and_mentions_ocr():
    data = make_pdf(pages=[[]], image_pages=(0,))
    r = extract_text_from_bytes(data, "scan.pdf")
    assert r.status is Status.SCANNED_LIKELY
    assert not r.usable
    assert "OCR" in r.message


def test_text_pdf_with_one_scanned_page_warns_but_is_usable():
    data = make_pdf(pages=[RESUME_LINES_1, []], image_pages=(1,))
    r = extract_text_from_bytes(data)
    assert r.status is Status.OK_WITH_WARNINGS
    assert r.usable
    assert any("scanned" in w for w in r.warnings)


def test_zero_byte_file():
    r = extract_text_from_bytes(b"", "empty.pdf")
    assert r.status is Status.EMPTY_FILE
    assert not r.usable


# ---- broken files -----------------------------------------------------------
@pytest.mark.parametrize("payload", [
    b"this is definitely not a pdf",
    b"%PDF-1.7\n garbage garbage garbage",
    b"\x00\x01\x02\x03" * 100,
], ids=["plain_text", "pdf_header_then_garbage", "binary_noise"])
def test_corrupted_files_are_reported_not_raised(payload):
    r = extract_text_from_bytes(payload, "bad.pdf")
    assert r.status is Status.CORRUPTED
    assert not r.usable
    assert "corrupted" in r.message.lower()


def test_truncated_pdf_is_never_reported_as_a_clean_ok():
    full = make_pdf([RESUME_LINES_1])
    r = extract_text_from_bytes(full[: len(full) // 2], "cut.pdf")
    # PyMuPDF may repair it, recover little, or refuse it. Any of these is acceptable,
    # but a truncated file must never come back as a plain OK with no warning.
    assert r.status is not Status.OK
    assert not r.usable or r.warnings


def test_password_protected_pdf():
    r = extract_text_from_bytes(make_pdf([RESUME_LINES_1], encrypt_pw="secret"), "locked.pdf")
    assert r.status is Status.ENCRYPTED
    assert not r.usable
    assert "password" in r.message.lower()


def test_file_too_large(monkeypatch):
    monkeypatch.setattr(pdf_parser, "MAX_FILE_BYTES", 100)
    r = extract_text_from_bytes(make_pdf([RESUME_LINES_1]))
    assert r.status is Status.TOO_LARGE
    assert not r.usable


def test_extraction_failure_on_every_page_is_reported(monkeypatch):
    def boom(self, *args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(pymupdf.Page, "get_text", boom)
    r = extract_text_from_bytes(make_pdf([RESUME_LINES_1]))
    assert r.status is Status.EXTRACTION_FAILED
    assert not r.usable
    assert r.warnings


def test_one_failed_page_still_returns_other_pages(monkeypatch):
    original = pymupdf.Page.get_text

    def flaky(self, *args, **kwargs):
        if self.number == 1:
            raise RuntimeError("simulated failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_text", flaky)
    r = extract_text_from_bytes(make_pdf([RESUME_LINES_1, RESUME_LINES_2]))
    assert r.status is Status.OK_WITH_WARNINGS
    assert "Asha Rao" in r.text
    assert "Education:" not in r.text
    assert any("Page 2" in w for w in r.warnings)


# ---- batches, duplicates ----------------------------------------------------
def test_make_unique_ids():
    assert make_unique_ids(["cv.pdf", "cv.pdf", "cv.pdf"]) == ["cv.pdf", "cv (2).pdf", "cv (3).pdf"]
    assert make_unique_ids(["a.pdf", "b.pdf"]) == ["a.pdf", "b.pdf"]
    # an upload genuinely named like a generated id must not collide
    ids = make_unique_ids(["cv.pdf", "cv (2).pdf", "cv.pdf"])
    assert len(set(ids)) == 3


def test_same_filename_different_content_both_kept():
    a, b = make_pdf([RESUME_LINES_1]), make_pdf([RESUME_LINES_2 + RESUME_LINES_2])
    results = extract_many([("cv.pdf", a), ("cv.pdf", b)])
    assert [r.file_id for r in results] == ["cv.pdf", "cv (2).pdf"]
    assert all(r.filename == "cv.pdf" for r in results)
    assert all(r.duplicate_of is None for r in results)


def test_identical_content_marked_duplicate_under_different_names():
    data = make_pdf([RESUME_LINES_1])
    results = extract_many([("first.pdf", data), ("copy.pdf", data)])
    assert results[0].duplicate_of is None
    assert results[1].duplicate_of == "first.pdf"
    assert results[1].usable  # still extracted; caller decides to exclude


def test_one_bad_file_does_not_stop_the_batch():
    good = make_pdf([RESUME_LINES_1])
    results = extract_many([("good.pdf", good), ("bad.pdf", b"nope"), ("empty.pdf", b"")])
    assert [r.usable for r in results] == [True, False, False]
    assert len(results) == 3


def test_empty_files_are_not_treated_as_duplicates_of_each_other():
    results = extract_many([("x.pdf", b""), ("y.pdf", b"")])
    assert [r.status for r in results] == [Status.EMPTY_FILE, Status.EMPTY_FILE]
    assert all(r.duplicate_of is None for r in results)


# ---- path helper, no files left behind --------------------------------------
def test_extract_text_from_path(tmp_path):
    p = tmp_path / "resume.pdf"
    p.write_bytes(make_pdf([RESUME_LINES_1]))
    r = extract_text_from_path(p)
    assert r.status is Status.OK
    assert r.file_id == "resume.pdf"


def test_extract_text_from_missing_path(tmp_path):
    r = extract_text_from_path(tmp_path / "missing.pdf")
    assert not r.usable
    assert "Cannot read file" in r.message


def test_extraction_does_not_write_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    extract_many([("a.pdf", make_pdf([RESUME_LINES_1]))])
    assert list(tmp_path.iterdir()) == []
