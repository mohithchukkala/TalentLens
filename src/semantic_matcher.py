"""Method B: semantic similarity with a pretrained Sentence Transformer.

The model is `sentence-transformers/all-MiniLM-L6-v2`, trained by others and used here
exactly as published: this project does not train or fine-tune it. It maps a piece of text
to a 384-number vector (an *embedding*) so that texts with similar meaning get vectors that
point in similar directions, even when they share no words. Cosine similarity of two
embeddings measures that direction.

Score meaning
-------------
cosine(a, b) = a . b / (|a| |b|), range [-1, 1]. For unit vectors (this model normalises its
output) it is just the dot product. It is a similarity of two texts in the model's embedding
space. It is NOT a probability, NOT a percentage, NOT a measure of whether someone is
qualified, and it is not on the same scale as the TF-IDF cosine: MiniLM scores for loosely
related texts are often far from 0, TF-IDF scores for the same pair are often near 0. Phase 7
has to look at both distributions before combining them.

The 256-token limit and chunking
--------------------------------
The model reads at most `max_seq_length` (256) word-piece tokens, 2 of which are the special
[CLS] and [SEP] tokens, and silently ignores the rest. A resume longer than 254 tokens would
be judged on its first part only. So:

1. Preprocess with `preprocess_for_embeddings`.
2. If the whole text fits in 254 tokens, keep it as ONE chunk (short documents are not split).
3. Otherwise cut it into units (lines; a line that is itself too long is cut into sentences,
   then words, then characters) and pack consecutive units greedily into chunks of at most
   `chunk_tokens` (200) tokens. 200 leaves a safety margin below the 254 usable tokens.
4. Overlap: each new chunk starts with the last whole units of the previous chunk, as many as
   fit in `overlap_tokens` (40). The overlap is whole lines/sentences, so it is at most 40 tokens
   and can be smaller (zero if the last unit alone is longer than 40 tokens).
5. Token counts come from the model's own tokenizer. Counts are additive over whitespace-separated
   units, so a chunk's size is the sum of its units' sizes.
6. A chunk is an exact slice of the preprocessed text, with character offsets, so it can be
   traced back to its place in the document and shown as evidence.
Chunking reduces information loss; it does not guarantee that the model "understood" the whole
document. Each chunk is embedded separately, so meaning that spans chunks is not captured.

Aggregation (job description chunks J_1..J_m, resume chunks R_1..R_n)
---------------------------------------------------------------------
    best_i = mean of the `top_k` largest values of cos(J_i, R_j) over j     (top_k = 1: the maximum)
    score  = sum_i (w_i * best_i) / sum_i w_i,   w_i = number of tokens in J_i

Reading: "for every part of the job description, how well is it covered by the best-matching
part of the resume; then average, giving longer job-description parts more weight."
Trade-offs: the maximum avoids diluting a strong skills section with unrelated sections
(education, hobbies), which averaging all resume chunks would do. Its weakness: one resume
chunk can decide the best_i of a job-description part, and if the job description is a single
chunk (the common case) that one resume chunk decides the whole score. The matching resume chunk
is recorded in `best_matches` so a person can read it. `top_k=2` or more is available for
exploratory comparison; the default (1) is the pre-registered setting and is not tuned.
The result is a raw cosine in [-1, 1]; no clipping or rescaling happens here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.inputs import Candidates, normalise_candidates
from src.preprocessing import preprocess_for_embeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPECIAL_TOKENS = 2           # [CLS] and [SEP]
DEFAULT_CHUNK_TOKENS = 200
DEFAULT_OVERLAP_TOKENS = 40
DEFAULT_BATCH_SIZE = 32
_TIE_DECIMALS = 12

TokenCounter = Callable[[Sequence[str]], List[int]]


class SemanticModelError(RuntimeError):
    """The model could not be loaded or could not encode the text."""


# ---- Model loading (once) ---------------------------------------------------
def get_model(model_name: str = MODEL_NAME, device: str = "cpu"):
    """Load the pretrained model once per (name, device) and reuse it afterwards.

    A thin wrapper so that get_model(), get_model(MODEL_NAME) and get_model(MODEL_NAME, "cpu")
    all hit the same cache entry (lru_cache alone keys on how the call was written).
    """
    return _load_model(model_name, device)


@lru_cache(maxsize=2)
def _load_model(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:  # network, disk, version problems
        raise SemanticModelError(
            f"Could not load the model '{model_name}': {exc}. The first run needs an internet "
            "connection to download it (about 90 MB); afterwards it is read from the local cache."
        ) from exc


def make_token_counter(model) -> TokenCounter:
    """Token counts (without special tokens) from the model's own tokenizer."""
    tokenizer = model.tokenizer

    def count(texts: Sequence[str]) -> List[int]:
        if not texts:
            return []
        ids = tokenizer(list(texts), add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        return [len(x) for x in ids]

    return count


# ---- Chunking ---------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    doc_id: str          # "job_description" or the candidate id
    index: int           # position of the chunk inside its document
    text: str            # exact slice of the preprocessed document
    token_count: int
    start: int           # character offsets of `text` in the preprocessed document
    end: int


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    tokens: int


_SENTENCE_CUT = re.compile(r"(?<=[.!?])\s+")


def _line_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"[^\n]*\S[^\n]*", text)]


def _trim(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_span(text: str, span: Tuple[int, int], pattern: "re.Pattern") -> List[Tuple[int, int]]:
    start, end = span
    pieces, cursor = [], start
    for m in pattern.finditer(text, start, end):
        pieces.append(_trim(text, cursor, m.start()))
        cursor = m.end()
    pieces.append(_trim(text, cursor, end))
    return [p for p in pieces if p[0] < p[1]]


def _build_units(text: str, count: TokenCounter, max_tokens: int) -> List[_Unit]:
    """Cut text into units that each fit in max_tokens: lines -> sentences -> words -> characters."""
    def counted(spans):
        counts = count([text[s:e] for s, e in spans])
        return list(zip(spans, counts))

    units: List[_Unit] = []
    word_pattern = re.compile(r"\s+")

    def add(span_counts, level):
        for (s, e), n in span_counts:
            if n <= max_tokens:
                units.append(_Unit(s, e, n))
            elif level == "line":
                sentences = _split_span(text, (s, e), _SENTENCE_CUT)
                if len(sentences) > 1:
                    add(counted(sentences), "sentence")
                else:  # no sentence ends inside the line: go straight to words
                    add(counted(_split_span(text, (s, e), word_pattern)), "word")
            elif level == "sentence":
                add(counted(_split_span(text, (s, e), word_pattern)), "word")
            else:  # a single "word" longer than the budget: halve it until the pieces fit
                mid = (s + e) // 2
                if mid == s or mid == e:  # one character cannot be split further
                    units.append(_Unit(s, e, n))
                else:
                    add(counted([(s, mid), (mid, e)]), "word")

    add(counted(_line_spans(text)), "line")
    return units


def chunk_text(text: str, doc_id: str, count: TokenCounter,
               chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
               overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
               whole_doc_tokens: Optional[int] = None) -> List[Chunk]:
    """Split a preprocessed document into overlapping, traceable chunks.

    `whole_doc_tokens`: if the document has at most this many tokens it stays a single chunk
    (default: chunk_tokens). Returns [] for blank text.
    """
    if not text or not text.strip():
        return []
    whole_doc_tokens = chunk_tokens if whole_doc_tokens is None else whole_doc_tokens
    total = count([text])[0]
    if total <= whole_doc_tokens:
        return [Chunk(doc_id, 0, text, total, 0, len(text))]

    units = _build_units(text, count, chunk_tokens)
    groups: List[List[int]] = []
    current: List[int] = []
    current_tokens = 0
    for i, unit in enumerate(units):
        if current and current_tokens + unit.tokens > chunk_tokens:
            groups.append(current)
            carry: List[int] = []
            carried = 0
            for j in reversed(current):          # overlap: trailing whole units within the budget
                if carried + units[j].tokens > overlap_tokens:
                    break
                carry.insert(0, j)
                carried += units[j].tokens
            while carry and carried + unit.tokens > chunk_tokens:
                carried -= units[carry.pop(0)].tokens
            current, current_tokens = carry, carried
        current.append(i)
        current_tokens += unit.tokens
    if current:
        groups.append(current)

    chunks = []
    for index, group in enumerate(groups):
        start, end = units[group[0]].start, units[group[-1]].end
        chunks.append(Chunk(doc_id, index, text[start:end], sum(units[j].tokens for j in group), start, end))
    return chunks


# ---- Results ----------------------------------------------------------------
@dataclass(frozen=True)
class ChunkMatch:
    """For one job-description chunk: its best-matching resume chunk(s) and the cosine used."""
    jd_chunk: int
    resume_chunk: int        # index of the single best resume chunk
    cosine: float            # best_i (mean of the top_k cosines when top_k > 1)


@dataclass(frozen=True)
class SemanticScore:
    candidate_id: str
    score: float                         # aggregated raw cosine in [-1, 1]; 0.0 when `empty`
    rank: int                            # 1 = best; equal scores share a rank; empty resumes rank last
    empty: bool                          # True if the resume has no usable text (nothing was embedded)
    chunks: Tuple[Chunk, ...] = ()
    best_matches: Tuple[ChunkMatch, ...] = ()


@dataclass(frozen=True)
class SemanticRanking:
    scores: List[SemanticScore]
    model_name: str
    max_seq_length: int
    chunk_tokens: int
    overlap_tokens: int
    top_k: int
    jd_chunks: Tuple[Chunk, ...] = ()
    warnings: List[str] = field(default_factory=list)

    def score_by_id(self) -> Dict[str, float]:
        return {s.candidate_id: s.score for s in self.scores}


# ---- Matcher ----------------------------------------------------------------
def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class SemanticMatcher:
    """Rank resumes against a job description by embedding similarity.

    Create it once and reuse it. The model is loaded lazily on first use and shared through
    `get_model`; pass `model=` (any object with `encode`, plus `tokenizer` or a `token_counter`)
    to inject a different or fake model.
    """

    def __init__(self, model=None, model_name: str = MODEL_NAME, device: str = "cpu",
                 chunk_tokens: int = DEFAULT_CHUNK_TOKENS, overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
                 top_k: int = 1, batch_size: int = DEFAULT_BATCH_SIZE,
                 token_counter: Optional[TokenCounter] = None):
        for name, value in (("chunk_tokens", chunk_tokens), ("overlap_tokens", overlap_tokens),
                            ("top_k", top_k), ("batch_size", batch_size)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if chunk_tokens < 1 or top_k < 1 or batch_size < 1 or overlap_tokens < 0:
            raise ValueError("chunk_tokens, top_k and batch_size must be >= 1 and overlap_tokens >= 0")
        if overlap_tokens >= chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        self.model_name, self.device = model_name, device
        self.chunk_tokens, self.overlap_tokens = chunk_tokens, overlap_tokens
        self.top_k, self.batch_size = top_k, batch_size
        self._model = model
        self._token_counter = token_counter
        if model is not None:
            self._validate_chunk_size()

    # -- model access ---------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            self._model = get_model(self.model_name, self.device)
            self._validate_chunk_size()
        return self._model

    @property
    def max_seq_length(self) -> int:
        return int(getattr(self.model, "max_seq_length", 256))

    @property
    def usable_tokens(self) -> int:
        """Tokens available for text once [CLS] and [SEP] are added."""
        return self.max_seq_length - SPECIAL_TOKENS

    def _validate_chunk_size(self) -> None:
        limit = int(getattr(self._model, "max_seq_length", 256)) - SPECIAL_TOKENS
        if self.chunk_tokens > limit:
            raise ValueError(f"chunk_tokens={self.chunk_tokens} exceeds the model's usable "
                             f"{limit} tokens (max_seq_length - {SPECIAL_TOKENS})")

    @property
    def count_tokens(self) -> TokenCounter:
        if self._token_counter is None:
            self._token_counter = make_token_counter(self.model)
        return self._token_counter

    # -- chunking -------------------------------------------------------------
    def chunk_document(self, text: str, doc_id: str = "document") -> List[Chunk]:
        """Preprocess for embeddings, then chunk. Short documents stay whole (<= usable_tokens)."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        return chunk_text(preprocess_for_embeddings(text), doc_id, self.count_tokens,
                          self.chunk_tokens, self.overlap_tokens, whole_doc_tokens=self.usable_tokens)

    # -- embedding ------------------------------------------------------------
    def _embed(self, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        """Embed each distinct text once, in batches."""
        unique = list(dict.fromkeys(texts))
        if not unique:
            return {}
        try:
            vectors = np.asarray(self.model.encode(
                unique, batch_size=self.batch_size, convert_to_numpy=True,
                show_progress_bar=False, normalize_embeddings=True), dtype=np.float64)
        except SemanticModelError:
            raise
        except Exception as exc:
            raise SemanticModelError(f"The model failed while encoding text: {exc}") from exc
        if vectors.ndim != 2 or len(vectors) != len(unique) or not np.all(np.isfinite(vectors)):
            raise SemanticModelError("The model returned invalid embeddings (wrong shape or NaN/inf).")
        return dict(zip(unique, _unit_rows(vectors)))

    # -- ranking --------------------------------------------------------------
    def rank(self, job_description: str, candidates: Candidates) -> SemanticRanking:
        """Score and rank candidates. Raises TypeError / ValueError for invalid input,
        SemanticModelError if the model cannot load or encode."""
        if not isinstance(job_description, str):
            raise TypeError(f"job_description must be a string, got {type(job_description).__name__}")
        pairs = normalise_candidates(candidates)
        jd_chunks = self.chunk_document(job_description, "job_description")
        if not jd_chunks:
            raise ValueError("the job description is empty")

        warnings: List[str] = []
        config = dict(model_name=self.model_name, max_seq_length=self.max_seq_length,
                      chunk_tokens=self.chunk_tokens, overlap_tokens=self.overlap_tokens, top_k=self.top_k)
        if not pairs:
            return SemanticRanking(scores=[], jd_chunks=tuple(jd_chunks), warnings=warnings, **config)

        resume_chunks = {cid: self.chunk_document(text, cid) for cid, text in pairs}

        # Safety net: warn if any chunk would still be truncated by the model.
        all_chunks = list(jd_chunks) + [c for cs in resume_chunks.values() for c in cs]
        recount = self.count_tokens([c.text for c in all_chunks])
        too_long = [c.doc_id for c, n in zip(all_chunks, recount) if n > self.usable_tokens]
        if too_long:
            warnings.append(f"Some chunks exceed the model limit and will be truncated: {sorted(set(too_long))}")

        embeddings = self._embed([c.text for c in all_chunks])
        jd_matrix = np.stack([embeddings[c.text] for c in jd_chunks])
        weights = np.array([max(c.token_count, 1) for c in jd_chunks], dtype=np.float64)

        rows = []
        for position, (cid, _) in enumerate(pairs):
            chunks = resume_chunks[cid]
            if not chunks:
                rows.append((True, 0.0, position, cid, (), ()))
                warnings.append(f"No usable text for '{cid}'; not scored (ranked last).")
                continue
            resume_matrix = np.stack([embeddings[c.text] for c in chunks])
            sims = np.clip(jd_matrix @ resume_matrix.T, -1.0, 1.0)       # (jd chunks) x (resume chunks)
            k = min(self.top_k, sims.shape[1])
            top = np.sort(sims, axis=1)[:, -k:]
            best = top.mean(axis=1)                                        # best_i
            score = float(np.clip(np.sum(weights * best) / np.sum(weights), -1.0, 1.0))
            matches = tuple(ChunkMatch(i, int(np.argmax(sims[i])), float(best[i])) for i in range(len(jd_chunks)))
            rows.append((False, score, position, cid, tuple(chunks), matches))

        # non-empty first, then by score (descending), ties keep input order
        rows.sort(key=lambda r: (r[0], -r[1], r[2]))
        results: List[SemanticScore] = []
        previous_key, previous_rank = None, 0
        for pos, (empty, score, _, cid, chunks, matches) in enumerate(rows, start=1):
            key = (empty, round(score, _TIE_DECIMALS))
            rank = previous_rank if key == previous_key else pos
            previous_key, previous_rank = key, rank
            results.append(SemanticScore(cid, score, rank, empty, chunks, matches))
        return SemanticRanking(scores=results, jd_chunks=tuple(jd_chunks), warnings=warnings, **config)


def semantic_rank(job_description: str, candidates: Candidates,
                  matcher: Optional[SemanticMatcher] = None) -> SemanticRanking:
    """One-call helper; pass a long-lived `matcher` to avoid re-creating it."""
    return (matcher or SemanticMatcher()).rank(job_description, candidates)


if __name__ == "__main__":
    # SYNTHETIC demonstration data written for this project. Not real resumes.
    jd = ("Machine learning engineer to build text classification systems. Requirements: Python, "
          "natural language processing, training neural networks, deploying models with Docker.")
    demo = {
        "synthetic_A.pdf": "Researcher who trained transformer language models for sentiment analysis "
                           "in PyTorch and packaged the services in containers.",
        "synthetic_B.pdf": "Pastry chef. Baked bread, croissants and wedding cakes for a busy bakery.",
        "synthetic_C.pdf": "Frontend developer building responsive React interfaces and design systems.",
        "synthetic_D.pdf": "Data analyst writing SQL reports and Python scripts for sales dashboards.",
    }
    matcher = SemanticMatcher()
    ranking = matcher.rank(jd, demo)
    print(f"SYNTHETIC EXAMPLE | model={ranking.model_name} | max_seq_length={ranking.max_seq_length} | "
          f"chunk_tokens={ranking.chunk_tokens} overlap={ranking.overlap_tokens} top_k={ranking.top_k}")
    print("Scores are raw cosine similarities in embedding space, not percentages or probabilities.")
    for s in ranking.scores:
        print(f"  rank {s.rank}  {s.candidate_id:<16} cosine={s.score:+.4f}  chunks={len(s.chunks)}")
