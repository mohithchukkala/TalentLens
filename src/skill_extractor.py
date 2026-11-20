"""Dictionary-based skill extraction.

This is a rule-based phrase matcher over a hand-written dictionary. It is NOT a
trained named-entity-recognition (NER) model: it only finds skills that are in the
dictionary, it does not understand sentences, and it cannot tell "expert in Python"
from "no Python experience".

How a skill is found
--------------------
1. Every alias of every skill becomes a regular expression with custom boundaries:
   the match may not touch a letter/digit on either side, and may not be directly
   preceded by "." (so "SQL" is not found inside "MySQL" / "NoSQL", "Java" not inside
   "JavaScript", ".NET" not inside "ASP.NET", "js" not inside "Node.js").
2. Spaces and hyphens inside an alias are interchangeable ("machine learning" also
   matches "machine-learning" and a line break between the words).
3. Ambiguous words (Go, R, C, React, Express, Spark...) carry a rule in the JSON file:
   exact capitalisation and/or a nearby context word is required.
4. If two matches overlap, the longest wins ("Microsoft SQL Server" beats "SQL",
   "C++" beats "C"). Every match is then mapped to its canonical name.

Every match keeps the wording found in the text and a snippet of evidence.
"""

from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

DEFAULT_DICTIONARY_PATH = Path(__file__).resolve().parent.parent / "data" / "skill_dictionary.json"
CONTEXT_WINDOW_CHARS = 80
MAX_EVIDENCE_PER_SKILL = 3
EVIDENCE_MAX_LINE_CHARS = 160

_LEFT_BOUNDARY = r"(?<![A-Za-z0-9.])"
_RIGHT_BOUNDARY = r"(?![A-Za-z0-9]|&[A-Za-z])"  # also blocks "R&D"


@dataclass(frozen=True)
class SkillMatch:
    """One occurrence of a skill in a text."""
    skill: str          # canonical name, used for comparison
    category: str
    matched_text: str   # wording exactly as written in the text (e.g. "sklearn")
    start: int
    end: int
    evidence: str       # the line (or a window of it) that contains the match


@dataclass
class SkillHit:
    """All occurrences of one skill, merged."""
    skill: str
    category: str
    count: int = 0
    mentions: List[str] = field(default_factory=list)   # distinct wordings, in order seen
    evidence: List[str] = field(default_factory=list)   # up to MAX_EVIDENCE_PER_SKILL snippets


@dataclass
class JobSkills:
    """Skills of a job description split by how the posting presents them."""
    required: Dict[str, SkillHit]
    preferred: Dict[str, SkillHit]   # appear only in 'preferred / nice to have' parts
    # True if the JD showed a heading or cue phrase ("Requirements:", "Nice to have", "a plus")
    # so the required/preferred split rests on something; False means every skill was put in
    # `required` only because that is the default, i.e. the split is uncertain.
    structured: bool = False

    @property
    def all(self) -> Dict[str, SkillHit]:
        return {**self.preferred, **self.required}


@dataclass
class _CompiledAlias:
    skill: str
    category: str
    alias: str
    pattern: "re.Pattern"
    case_sensitive: bool
    context: Optional["re.Pattern"]


def _alias_regex(alias: str) -> str:
    parts = [p for p in re.split(r"[ \-]+", alias.strip()) if p]
    body = r"[\s\-]+".join(re.escape(p) for p in parts)
    return f"{_LEFT_BOUNDARY}{body}{_RIGHT_BOUNDARY}"


def _context_regex(words: Iterable[str]) -> "re.Pattern":
    alternatives = "|".join(re.escape(w.lower()) for w in words)
    return re.compile(rf"(?<![a-z0-9])(?:{alternatives})(?![a-z0-9])")


class SkillExtractor:
    """Configurable dictionary matcher. Build it from JSON, a dict, or a list of entries."""

    def __init__(self, entries: List[dict]):
        self._aliases: List[_CompiledAlias] = []
        self.categories: Dict[str, str] = {}
        seen_alias_owner: Dict[str, str] = {}

        for entry in entries:
            name = entry.get("name", "").strip()
            if not name:
                raise ValueError(f"Skill entry without a name: {entry!r}")
            if name in self.categories:
                raise ValueError(f"Duplicate skill name: {name!r}")
            category = entry.get("category", "Other")
            self.categories[name] = category
            texts = [name] + [a for a in entry.get("aliases", []) if a.strip()]
            rules = entry.get("rules", {})
            for key in rules:
                if key not in texts:
                    raise ValueError(f"Rule key {key!r} of skill {name!r} is not its name or an alias")
            for text in texts:
                canon = re.sub(r"[\s\-]+", " ", text.lower())
                owner = seen_alias_owner.get(canon)
                if owner is not None and owner != name:
                    raise ValueError(f"Alias {text!r} is used by both {owner!r} and {name!r}")
                seen_alias_owner[canon] = name
                rule = rules.get(text, {})
                case_sensitive = bool(rule.get("case_sensitive", False))
                flags = 0 if case_sensitive else re.IGNORECASE
                context = _context_regex(rule["context"]) if rule.get("context") else None
                self._aliases.append(_CompiledAlias(
                    skill=name, category=category, alias=text,
                    pattern=re.compile(_alias_regex(text), flags),
                    case_sensitive=case_sensitive, context=context))

    # -- constructors ---------------------------------------------------------
    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_DICTIONARY_PATH) -> "SkillExtractor":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(data["skills"])

    @classmethod
    def from_dict(cls, data: dict) -> "SkillExtractor":
        return cls(data["skills"])

    @property
    def skill_names(self) -> List[str]:
        return list(self.categories)

    # -- matching -------------------------------------------------------------
    def find_matches(self, text: str) -> List[SkillMatch]:
        """All non-overlapping skill occurrences, in text order."""
        if not text or not text.strip():
            return []
        candidates: List[Tuple[int, int, _CompiledAlias]] = []
        lowered = text.lower()
        bounds = _sentence_boundaries(text)
        for alias in self._aliases:
            for m in alias.pattern.finditer(text):
                if alias.context is not None:
                    # context must be in the same sentence AND within the character window,
                    # so a word in the next sentence cannot "vouch" for an ambiguous match
                    idx = bisect.bisect_right(bounds, m.start())
                    lo = max(bounds[idx - 1], m.start() - CONTEXT_WINDOW_CHARS)
                    hi = min(bounds[idx], m.end() + CONTEXT_WINDOW_CHARS)
                    window = lowered[lo:m.start()] + " " + lowered[m.end():hi]
                    if not alias.context.search(window):
                        continue
                candidates.append((m.start(), m.end(), alias))

        # longest match first; ties broken by position
        candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
        accepted: List[Tuple[int, int, _CompiledAlias]] = []
        for start, end, alias in candidates:
            if all(end <= a_start or start >= a_end for a_start, a_end, _ in accepted):
                accepted.append((start, end, alias))
        accepted.sort(key=lambda c: c[0])

        return [SkillMatch(skill=a.skill, category=a.category, matched_text=text[s:e],
                           start=s, end=e, evidence=_evidence(text, s, e))
                for s, e, a in accepted]

    def extract(self, text: str) -> Dict[str, SkillHit]:
        """Skills found in `text`, keyed by canonical name (order = first appearance)."""
        return _merge(self.find_matches(text))

    def extract_names(self, text: str) -> Set[str]:
        return set(self.extract(text))

    # -- job descriptions -----------------------------------------------------
    def extract_job_skills(self, jd_text: str) -> JobSkills:
        """Split JD skills into required vs preferred using simple, visible rules.

        * A heading line such as "Nice to have:" / "Preferred qualifications" switches
          the following lines to *preferred*; "Requirements:" / "Must have" /
          "Skills:" switches back to *required*.
        * A line containing "preferred", "nice to have", "bonus", "a plus" ... is
          preferred on its own.
        * Everything else is *required* (the default when a JD has no structure).
        * A skill mentioned in both a required and a preferred place counts as required.
        """
        matches = self.find_matches(jd_text)
        if not matches:
            return JobSkills({}, {})
        line_starts = [0]
        for m in re.finditer(r"\n", jd_text):
            line_starts.append(m.end())
        lines = jd_text.split("\n")

        def line_index(pos: int) -> int:
            lo, hi = 0, len(line_starts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if line_starts[mid] <= pos:
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        lines_with_skills = {line_index(m.start) for m in matches}
        line_modes: List[str] = []
        state = "required"
        structured = False
        for i, line in enumerate(lines):
            if i not in lines_with_skills:
                heading = _heading_mode(line)
                if heading:
                    state = heading
                    structured = True
            line_modes.append(state)

        boundaries = _sentence_boundaries(jd_text)

        def mode_of(m: SkillMatch) -> str:
            # An inline cue ("a plus", "nice to have") affects only its own sentence,
            # so one cue in a single-paragraph JD does not demote every skill.
            idx = bisect.bisect_right(boundaries, m.start)
            seg_start, seg_end = boundaries[idx - 1], boundaries[idx]
            if _INLINE_PREFERRED.search(jd_text[seg_start:seg_end]):
                return "preferred"
            return line_modes[line_index(m.start)]

        modes = [mode_of(m) for m in matches]
        if not structured:  # an inline cue also counts as structure
            structured = any(
                _INLINE_PREFERRED.search(jd_text[boundaries[bisect.bisect_right(boundaries, m.start) - 1]:
                                                 boundaries[bisect.bisect_right(boundaries, m.start)]])
                for m in matches)
        required = _merge([m for m, mode in zip(matches, modes) if mode == "required"])
        preferred_all = _merge([m for m, mode in zip(matches, modes) if mode == "preferred"])
        preferred = {k: v for k, v in preferred_all.items() if k not in required}
        return JobSkills(required=required, preferred=preferred, structured=structured)


# ---- helpers ----------------------------------------------------------------
_PREFERRED_WORDS = (r"preferred|nice[- ]to[- ]have|good[- ]to[- ]have|bonus|a plus|"
                    r"desirable|advantage|optional")
_INLINE_PREFERRED = re.compile(rf"\b(?:{_PREFERRED_WORDS})\b", re.IGNORECASE)
_PREFERRED_HEADING = re.compile(rf"(?:{_PREFERRED_WORDS})", re.IGNORECASE)
_REQUIRED_HEADING = re.compile(
    r"requirement|required|must[- ]have|qualification|what you.{0,3}ll need|skills|"
    r"responsibilit|about you|you have|tech stack|experience", re.IGNORECASE)
_HEADING_SHAPE = re.compile(r"^\s*[A-Za-z][A-Za-z /&'-]{2,50}?\s*:?\s*$")


_SENTENCE_END = re.compile(r"[.;!?](?=\s|$)|\n")


def _sentence_boundaries(text: str) -> List[int]:
    """Sorted cut positions: ". ", "; ", "! ", "? " or a line break ("1.2" and "Node.js" do not cut)."""
    return [0] + [b.end() for b in _SENTENCE_END.finditer(text)] + [len(text)]


def _heading_mode(line: str) -> Optional[str]:
    """'required' / 'preferred' if the line is a short heading, else None."""
    stripped = line.strip()
    if not stripped or stripped[0] in "-*•|" or not _HEADING_SHAPE.match(line):
        return None  # bullet items are content, never headings
    if _PREFERRED_HEADING.search(line):
        return "preferred"
    if _REQUIRED_HEADING.search(line):
        return "required"
    return None


def _evidence(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end
    if line_end - line_start <= EVIDENCE_MAX_LINE_CHARS:
        snippet = text[line_start:line_end]
    else:
        a, b = max(line_start, start - 70), min(line_end, end + 70)
        snippet = ("..." if a > line_start else "") + text[a:b] + ("..." if b < line_end else "")
    return re.sub(r"\s+", " ", snippet).strip()


def _merge(matches: List[SkillMatch]) -> Dict[str, SkillHit]:
    hits: Dict[str, SkillHit] = {}
    for m in matches:
        hit = hits.setdefault(m.skill, SkillHit(skill=m.skill, category=m.category))
        hit.count += 1
        if m.matched_text not in hit.mentions:
            hit.mentions.append(m.matched_text)
        if m.evidence not in hit.evidence and len(hit.evidence) < MAX_EVIDENCE_PER_SKILL:
            hit.evidence.append(m.evidence)
    return hits


@lru_cache(maxsize=1)
def get_default_extractor() -> SkillExtractor:
    """The extractor built from data/skill_dictionary.json (compiled once)."""
    return SkillExtractor.from_json(DEFAULT_DICTIONARY_PATH)


if __name__ == "__main__":
    import sys

    sample = " ".join(sys.argv[1:]) or "Built APIs in Node.js and Python with scikit-learn, sklearn and Postgres."
    for name, hit in get_default_extractor().extract(sample).items():
        print(f"{name} [{hit.category}] x{hit.count} as {hit.mentions} | {hit.evidence[0]}")
