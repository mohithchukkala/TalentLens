"""PDF text extraction with PyMuPDF.

Design rules
------------
* Work on bytes in memory. Nothing is written to disk, so uploaded resumes are
  never stored (the Streamlit file uploader already holds them as bytes).
* Never silently return empty text. Every file gets an ExtractionResult with a
  status, and only status == OK (or OK_WITH_WARNINGS) means the text is usable.
* Keep paragraph / line structure: text is read block by block in reading order,
  lines inside a block keep their newline, blocks are separated by a blank line.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pymupdf

# ---- Tunable limits (named so they can be explained and tested) -------------
MAX_FILE_BYTES = 10 * 1024 * 1024  # refuse files above 10 MB
MIN_TOTAL_CHARS = 100              # fewer non-whitespace characters than this = no usable text
# A page that has images and fewer non-whitespace characters than this is treated
# as an image-only page (likely scanned).
MIN_CHARS_FOR_TEXT_PAGE = 20


class Status(str, Enum):
    OK = "ok"
    OK_WITH_WARNINGS = "ok_with_warnings"      # usable text, but something needs attention
    SCANNED_LIKELY = "scanned_likely"          # images but no text layer: OCR needed
    NO_TEXT = "no_text"                        # blank / almost no text, no images
    ENCRYPTED = "encrypted"                    # password required
    CORRUPTED = "corrupted"                    # not a readable PDF
    EMPTY_FILE = "empty_file"                  # zero bytes
    TOO_LARGE = "too_large"
    EXTRACTION_FAILED = "extraction_failed"    # opened, but reading text failed


USABLE_STATUSES = {Status.OK, Status.OK_WITH_WARNINGS}


@dataclass
class ExtractionResult:
    file_id: str                  # unique within a batch (duplicates get " (2)", " (3)", ...)
    filename: str                 # name as uploaded
    status: Status
    message: str                  # human-readable explanation for the UI
    text: str = ""                # extracted text (original layout kept as far as possible)
    page_count: int = 0
    char_count: int = 0           # non-whitespace characters
    content_hash: str = ""        # SHA-256 of the file bytes
    duplicate_of: Optional[str] = None  # file_id of the first identical upload, if any
    warnings: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """True only if the text can be passed on to matching."""
        return self.status in USABLE_STATUSES and bool(self.text.strip())


# ---- Helpers ----------------------------------------------------------------
def _tidy(text: str) -> str:
    """Light, lossless-in-meaning cleanup. Heavy preprocessing belongs to Phase 4."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line in a row
    return text.strip()


def _page_text(page: "pymupdf.Page") -> str:
    """Text of one page: blocks in reading order, blank line between blocks."""
    blocks = page.get_text("blocks", sort=True)
    parts = []
    for block in blocks:
        # block = (x0, y0, x1, y1, text, block_no, block_type); type 0 = text, 1 = image
        if block[6] == 0 and block[4].strip():
            parts.append(block[4].strip("\n"))
    return "\n\n".join(parts)


def _failure(file_id: str, filename: str, status: Status, message: str,
             content_hash: str = "", page_count: int = 0) -> ExtractionResult:
    return ExtractionResult(file_id=file_id, filename=filename, status=status,
                            message=message, content_hash=content_hash,
                            page_count=page_count)


# ---- Core function ----------------------------------------------------------
def extract_text_from_bytes(data: bytes, filename: str = "resume.pdf",
                            file_id: Optional[str] = None) -> ExtractionResult:
    """Extract text from one PDF held in memory. Never raises for bad input."""
    file_id = file_id or filename
    if not data:
        return _failure(file_id, filename, Status.EMPTY_FILE, "The file is empty (0 bytes).")

    digest = hashlib.sha256(data).hexdigest()
    if len(data) > MAX_FILE_BYTES:
        return _failure(file_id, filename, Status.TOO_LARGE,
                        f"File is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB.", digest)

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises FileDataError or RuntimeError for bad files
        return _failure(file_id, filename, Status.CORRUPTED,
                        f"Could not open as a PDF (file may be corrupted or not a PDF): {exc}", digest)

    try:
        if doc.needs_pass:
            return _failure(file_id, filename, Status.ENCRYPTED,
                            "The PDF is password-protected. Remove the password and upload again.",
                            digest, doc.page_count)
        if doc.page_count == 0:
            return _failure(file_id, filename, Status.NO_TEXT, "The PDF has no pages.", digest)

        page_texts: List[str] = []
        warnings: List[str] = []
        if getattr(doc, "is_repaired", False):
            warnings.append("The PDF was damaged and had to be repaired; some content may be missing.")
        failed_pages: List[int] = []
        image_only_pages: List[int] = []

        for index, page in enumerate(doc, start=1):
            try:
                text = _page_text(page)
                has_images = bool(page.get_images(full=False))
            except Exception as exc:
                failed_pages.append(index)
                warnings.append(f"Page {index}: text extraction failed ({exc}).")
                continue
            if has_images and len(re.sub(r"\s", "", text)) < MIN_CHARS_FOR_TEXT_PAGE:
                image_only_pages.append(index)
            page_texts.append(text)

        if len(failed_pages) == doc.page_count:
            return ExtractionResult(file_id, filename, Status.EXTRACTION_FAILED,
                                    "Text extraction failed on every page.",
                                    content_hash=digest, page_count=doc.page_count,
                                    warnings=warnings)

        text = _tidy("\n\n".join(t for t in page_texts if t))
        char_count = len(re.sub(r"\s", "", text))
        page_count = doc.page_count
    except Exception as exc:  # last-resort guard so one bad file cannot crash a batch
        return _failure(file_id, filename, Status.EXTRACTION_FAILED,
                        f"Unexpected error while reading the PDF: {exc}", digest)
    finally:
        doc.close()

    if char_count < MIN_TOTAL_CHARS:
        if image_only_pages:
            return ExtractionResult(
                file_id, filename, Status.SCANNED_LIKELY,
                "Almost no text found, but the file contains images. It is probably a scanned "
                "resume: OCR is required to read it (OCR is not part of this version).",
                text=text, page_count=page_count, char_count=char_count,
                content_hash=digest, warnings=warnings)
        return ExtractionResult(
            file_id, filename, Status.NO_TEXT,
            f"Only {char_count} characters of text found (minimum {MIN_TOTAL_CHARS}). "
            "The file is blank or has no usable text.",
            text=text, page_count=page_count, char_count=char_count,
            content_hash=digest, warnings=warnings)

    if image_only_pages:
        warnings.append(f"Pages {image_only_pages} contain images but almost no text; "
                        "they may be scanned and were not read.")
    status = Status.OK_WITH_WARNINGS if warnings else Status.OK
    message = "Text extracted." if status is Status.OK else "Text extracted, with warnings."
    return ExtractionResult(file_id, filename, status, message, text=text,
                            page_count=page_count, char_count=char_count,
                            content_hash=digest, warnings=warnings)


# ---- Batches ----------------------------------------------------------------
def make_unique_ids(filenames: Sequence[str]) -> List[str]:
    """'cv.pdf', 'cv.pdf' -> 'cv.pdf', 'cv (2).pdf'. Deterministic, order-preserving."""
    seen: dict[str, int] = {}
    used: set[str] = set()
    ids: List[str] = []
    for name in filenames:
        count = seen.get(name, 0) + 1
        seen[name] = count
        candidate = name
        if count > 1:
            path = Path(name)
            candidate = f"{path.stem} ({count}){path.suffix}"
            while candidate in used:  # e.g. an upload really named 'cv (2).pdf'
                count += 1
                seen[name] = count
                candidate = f"{path.stem} ({count}){path.suffix}"
        used.add(candidate)
        ids.append(candidate)
    return ids


def extract_many(files: Iterable[Tuple[str, bytes]]) -> List[ExtractionResult]:
    """Extract a batch of (filename, bytes) pairs.

    * Same filename twice -> unique file_ids, both are kept.
    * Same *content* twice (SHA-256) -> the later one gets duplicate_of set, so
      the caller can exclude it from ranking instead of ranking one person twice.
    """
    files = list(files)
    ids = make_unique_ids([name for name, _ in files])
    results: List[ExtractionResult] = []
    first_by_hash: dict[str, str] = {}
    for (name, data), file_id in zip(files, ids):
        result = extract_text_from_bytes(data, filename=name, file_id=file_id)
        if result.content_hash:
            if result.content_hash in first_by_hash:
                result.duplicate_of = first_by_hash[result.content_hash]
                result.warnings.append(f"Identical content to '{result.duplicate_of}'.")
            else:
                first_by_hash[result.content_hash] = file_id
        results.append(result)
    return results


def extract_text_from_path(path: str | Path) -> ExtractionResult:
    """Convenience for scripts and manual checks (reads the file, does not copy it)."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return _failure(path.name, path.name, Status.CORRUPTED, f"Cannot read file: {exc}")
    return extract_text_from_bytes(data, filename=path.name)


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        r = extract_text_from_path(arg)
        print(f"{r.filename}: {r.status.value} | pages={r.page_count} chars={r.char_count}")
        print(f"  {r.message}")
        for w in r.warnings:
            print(f"  warning: {w}")
        if r.usable:
            print("  --- first 400 characters ---")
            print(r.text[:400])
