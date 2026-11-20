"""Text preprocessing: one shared base step, then one function per consumer.

Why three different functions?
------------------------------
The three methods read text differently, so each gets the cleanup it can use
and nothing more.

                      skills            TF-IDF               embeddings
  case                kept              lowercased           kept
  punctuation         kept              kept inside tokens   kept
                                        (c++, node.js, ci/cd)
  stop words          kept              removed (a kept-list  kept
                                        protects "front",
                                        "back", "full", ...)
  contact details     removed           removed              removed
  line breaks         kept              dropped (tokens)     kept (the Phase 6
                                                             chunker splits on
                                                             them)

The *original* extracted text stays untouched in ExtractionResult.text so the
interface can always show evidence from the real document.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---- Shared base step -------------------------------------------------------
_BULLET_CHARS = "•▪●◦‣∙▫■□➢➤►◆◇★☆✓✔➔→"
_DASH_MAP = {"–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-", "‒": "-"}
_QUOTE_MAP = {"‘": "'", "’": "'", "“": '"', "”": '"', "´": "'", "`": "'"}
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

_LEADING_BULLET = re.compile(rf"(?m)^[ \t]*[{re.escape(_BULLET_CHARS)}-·*]+[ \t]*")
_INLINE_BULLET = re.compile(rf"[ \t]*[{re.escape(_BULLET_CHARS)}-·][ \t]*")
_LINEBREAK_HYPHEN = re.compile(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])")


def normalize_text(text: str) -> str:
    """Character-level cleanup shared by all three pipelines.

    * Unicode NFKC: ligatures ("ﬁ" -> "fi"), full-width letters, non-breaking spaces.
    * Curly quotes / long dashes -> ASCII; zero-width and soft-hyphen characters removed.
    * Bullet glyphs (including the private-use bullet Word puts in PDFs) -> "- ".
    * "scikit-\\nlearn" (hyphen at a line break) -> "scikit-learn". The hyphen is kept
      on purpose: dropping it would turn real compounds into one word.
    * Spaces/tabs collapsed, at most one blank line in a row. Case is NOT changed.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_INVISIBLE)
    for table in (_DASH_MAP, _QUOTE_MAP):
        for old, new in table.items():
            text = text.replace(old, new)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = _LEADING_BULLET.sub("- ", text)
    text = _INLINE_BULLET.sub(" | ", text)
    text = _LINEBREAK_HYPHEN.sub("-", text)
    text = re.sub(r"[^\S\n]+", " ", text)          # runs of spaces/tabs -> one space
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---- Contact details --------------------------------------------------------
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github|gitlab)\.com/\S*", re.IGNORECASE)
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{7,}\d(?![\w])")


def _mask_phone(match: re.Match) -> str:
    # Only treat as a phone number if it has >= 9 digits, so "2019 - 2023" survives.
    return " " if sum(ch.isdigit() for ch in match.group()) >= 9 else match.group()


def remove_contact_info(text: str) -> str:
    """Remove e-mail addresses, URLs and phone numbers.

    Reasons: they are not skills, they use up the embedding model's 256-token window,
    they add noise to TF-IDF, and personal details should not influence ranking.
    """
    text = _EMAIL.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _PHONE.sub(_mask_phone, text)
    text = re.sub(r"[^\S\n]{2,}", " ", text)       # tidy gaps left by removals
    return "\n".join(line.strip() for line in text.split("\n"))


# ---- 1. Skill extraction ----------------------------------------------------
def preprocess_for_skills(text: str) -> str:
    """Minimal cleanup for the dictionary matcher.

    Keeps case, punctuation, stop words and line breaks, because skills such as
    "C++", "C#", ".NET", "Node.js" and "CI/CD" depend on punctuation and because the
    matcher uses capitalisation and nearby words to avoid false matches ("Go", "R").
    The evidence shown to the user is a snippet of this text.
    """
    return remove_contact_info(normalize_text(text)).strip()


# ---- 2. TF-IDF --------------------------------------------------------------
# Words sklearn lists as stop words but that carry meaning in technical text:
#   "front end", "back end", "full stack", "system design", and the language Go
# ("go" was found missing in Phase 5 when a demo showed the Go language vanishing).
TFIDF_KEEP_WORDS = frozenset({"front", "back", "full", "system", "go"})
TFIDF_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - TFIDF_KEEP_WORDS
_SINGLE_LETTER_KEEP = frozenset({"c", "r"})  # the languages C and R

# One token = letters/digits, optionally joined by . - /  (node.js, scikit-learn, ci/cd),
# optionally ending in ++, + or #  (c++, c#). ".net" is matched as its own token.
_TFIDF_TOKEN = re.compile(r"\.net\b|[a-z0-9]+(?:[.\-/][a-z0-9]+)*(?:\+\+|\+|#)?")
_NUMERIC_ONLY = re.compile(r"[\d.\-/]+")


def tfidf_tokens(text: str) -> List[str]:
    """Tokenize for TF-IDF.

    Strategy: normalise -> remove contact info -> lowercase -> regex tokens that keep
    c++, c#, .net, node.js, ci/cd, scikit-learn as ONE token each -> drop stop words
    (except TFIDF_KEEP_WORDS), purely numeric tokens and single letters other than c / r.

    Aliases are deliberately NOT merged ("sklearn" and "scikit-learn" stay different
    tokens). TF-IDF is the traditional baseline; the skill extractor is where aliases
    are normalised. Word pairs such as "machine learning" are captured later by
    ngram_range=(1, 2) in the vectorizer, not here.
    """
    cleaned = remove_contact_info(normalize_text(text)).lower()
    tokens = []
    for token in _TFIDF_TOKEN.findall(cleaned):
        if token in TFIDF_STOP_WORDS or _NUMERIC_ONLY.fullmatch(token):
            continue
        if len(token) == 1 and token not in _SINGLE_LETTER_KEEP:
            continue
        tokens.append(token)
    return tokens


def preprocess_for_tfidf(text: str) -> str:
    """Tokens joined by single spaces.

    The Phase 5 vectorizer must therefore split on whitespace only:
    TfidfVectorizer(lowercase=False, token_pattern=r"\\S+", ...), otherwise
    sklearn's default pattern would cut "c++" down to "c".
    """
    return " ".join(tfidf_tokens(text))


# ---- 3. Semantic embeddings -------------------------------------------------
def preprocess_for_embeddings(text: str) -> str:
    """Near-original text for the Sentence Transformer.

    Only the shared character cleanup plus removal of contact details. Case,
    punctuation, stop words and wording are untouched, because the pretrained model
    uses full sentences and context. Line breaks are kept: the Phase 6 chunker splits
    on lines and sentences before counting tokens.
    """
    return remove_contact_info(normalize_text(text)).strip()
