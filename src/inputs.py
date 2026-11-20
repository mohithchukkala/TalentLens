"""Input validation shared by the matchers (TF-IDF, semantic, and later the combined ranker)."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Tuple, Union

Candidates = Union[Mapping[str, str], Iterable[Tuple[str, str]]]


def normalise_candidates(candidates: Candidates) -> List[Tuple[str, str]]:
    """Return a list of (id, text) pairs, or raise TypeError / ValueError.

    Accepts a mapping {id: text} or an iterable of (id, text) tuples. Ids and texts must be
    strings and ids must be unique. An empty collection is valid and returns [].
    """
    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a mapping {id: text} or an iterable of (id, text) pairs, not a string")
    if isinstance(candidates, Mapping):
        pairs = list(candidates.items())
    else:
        try:
            pairs = list(candidates)
        except TypeError:
            raise TypeError("candidates must be a mapping {id: text} or an iterable of (id, text) pairs") from None
    seen = set()
    out: List[Tuple[str, str]] = []
    for item in pairs:
        if not (isinstance(item, tuple) and len(item) == 2):
            raise TypeError(f"each candidate must be an (id, text) pair, got {item!r}")
        cid, text = item
        if not isinstance(cid, str):
            raise TypeError(f"candidate id must be a string, got {type(cid).__name__}")
        if not isinstance(text, str):
            raise TypeError(f"text of candidate {cid!r} must be a string, got {type(text).__name__}")
        if cid in seen:
            raise ValueError(f"duplicate candidate id {cid!r}; ids must be unique")
        seen.add(cid)
        out.append((cid, text))
    return out
