"""
Step 3 — Hybrid (lexical + semantic) scoring of each candidate.
Step 4 — Filter by threshold, rank, keep top-3 per drug.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, NamedTuple

import google.generativeai as genai
import numpy as np
from rapidfuzz import fuzz

from src.candidate_retrieval import Candidate

EMBEDDING_MODEL = "models/text-embedding-004"
LEXICAL_WEIGHT = 0.7
SEMANTIC_WEIGHT = 0.3


class ScoredCandidate(NamedTuple):
    name: str
    source: str
    source_score: float
    lexical_score: float
    semantic_score: float
    hybrid_score: float


def lexical_score(a: str, b: str) -> float:
    a, b = a.lower().strip(), b.lower().strip()
    return max(
        fuzz.ratio(a, b),
        fuzz.partial_ratio(a, b),
        fuzz.token_sort_ratio(a, b),
    )


@lru_cache(maxsize=2048)
def _embed(text: str) -> tuple:
    """Cached Gemini embedding lookup. Returns a tuple (hashable for lru_cache)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (check your .env file).")
    genai.configure(api_key=api_key)
    result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
    return tuple(result["embedding"])


def _embed_and_maybe_charge(text: str, cost_tracker=None) -> np.ndarray:
    """Wraps the cached _embed() call and only records a cost event when it
    was an actual cache miss (a real API call), so repeated lookups of the
    same drug name within a run aren't double-charged."""
    before = _embed.cache_info().hits
    vec = _embed(text)
    was_cache_hit = _embed.cache_info().hits > before
    if cost_tracker is not None and not was_cache_hit:
        cost_tracker.record_estimated("embedding", EMBEDDING_MODEL, input_text=text)
    return np.array(vec)


def semantic_score(a: str, b: str, use_semantic: bool = True, cost_tracker=None) -> float:
    """Cosine similarity between embeddings, rescaled to 0-100. Fails soft to 0.
    If cost_tracker is passed, records an estimated cost per real (non-cached)
    embed call — the embedding endpoint doesn't return usage_metadata, so
    this uses the ~4-chars/token estimate rather than a real token count."""
    if not use_semantic:
        return 0.0
    try:
        va = _embed_and_maybe_charge(a, cost_tracker)
        vb = _embed_and_maybe_charge(b, cost_tracker)
    except Exception:
        return 0.0
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    cos_sim = float(np.dot(va, vb) / denom)  # roughly -1..1
    return max(0.0, min(100.0, (cos_sim + 1) / 2 * 100))


def rank_candidates(
    raw_guess: str,
    candidates: List[Candidate],
    score_threshold: float = 45.0,
    top_k: int = 3,
    use_semantic: bool = True,
    cost_tracker=None,
) -> List[ScoredCandidate]:
    """Step 3 + Step 4: score every candidate, drop below threshold, keep top_k."""
    scored: List[ScoredCandidate] = []
    for c in candidates:
        lex = lexical_score(raw_guess, c.name)
        sem = semantic_score(raw_guess, c.name, use_semantic=use_semantic, cost_tracker=cost_tracker)
        hybrid = LEXICAL_WEIGHT * lex + SEMANTIC_WEIGHT * sem
        scored.append(
            ScoredCandidate(
                name=c.name,
                source=c.source,
                source_score=c.score,
                lexical_score=round(lex, 2),
                semantic_score=round(sem, 2),
                hybrid_score=round(hybrid, 2),
            )
        )

    scored = [s for s in scored if s.hybrid_score >= score_threshold]
    scored.sort(key=lambda s: s.hybrid_score, reverse=True)
    return scored[:top_k]
