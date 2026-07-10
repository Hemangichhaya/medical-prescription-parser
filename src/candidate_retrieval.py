from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional

import requests
from bs4 import BeautifulSoup

RXNORM_URL = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
RXNORM_SPELLING_URL = "https://rxnav.nlm.nih.gov/REST/spellingsuggestions.json"
OPENFDA_URL = "https://api.fda.gov/drug/ndc.json"
DRUGSCOM_SEARCH_URL = "https://www.drugs.com/search.php"
ONEMG_SEARCH_URL = "https://www.1mg.com/search/all"

DEFAULT_TIMEOUT = 8
ENABLE_SCRAPED_SOURCES = False  # flip to True (or pass use_scraped_sources=True) to opt in


class Candidate(NamedTuple):
    name: str
    score: float  
    source: str


@dataclass
class SourceAttempt:
    """One retrieval attempt's outcome — lets you tell apart 'queried and
    found nothing' from 'request failed' from 'got a response but the HTML
    parser matched nothing', instead of every case silently looking like []."""

    source: str
    status: str  # "ok" | "http_error" | "network_error" | "parse_error" | "skipped"
    candidate_count: int
    detail: Optional[str] = None  # HTTP status code, exception message, etc.


def _log(debug_log: Optional[List[SourceAttempt]], attempt: SourceAttempt) -> None:
    if debug_log is not None:
        debug_log.append(attempt)


def rxnorm_candidates(term: str, max_entries: int = 20, debug_log: Optional[List[SourceAttempt]] = None) -> List[Candidate]:
    try:
        resp = requests.get(
            RXNORM_URL, params={"term": term, "maxEntries": max_entries}, timeout=DEFAULT_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        _log(debug_log, SourceAttempt("rxnorm", "network_error", 0, str(e)))
        return []
    except ValueError as e:
        _log(debug_log, SourceAttempt("rxnorm", "parse_error", 0, f"bad JSON: {e}"))
        return []

    out = []
    for c in data.get("approximateGroup", {}).get("candidate", []):
        name = c.get("name")
        if name:
            out.append(Candidate(name=name, score=float(c.get("score", 0)), source="rxnorm"))
    _log(debug_log, SourceAttempt("rxnorm", "ok", len(out), f"HTTP {resp.status_code}"))
    return out


def rxnorm_spelling_candidates(term: str, debug_log: Optional[List[SourceAttempt]] = None) -> List[Candidate]:
    """RxNorm's dedicated spelling-correction endpoint. Purpose-built for
    exactly this problem (garbled OCR text with no close approximateTerm
    match) — worth trying whenever approximateTerm comes back empty."""
    try:
        resp = requests.get(RXNORM_SPELLING_URL, params={"name": term}, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        _log(debug_log, SourceAttempt("rxnorm_spelling", "network_error", 0, str(e)))
        return []
    except ValueError as e:
        _log(debug_log, SourceAttempt("rxnorm_spelling", "parse_error", 0, f"bad JSON: {e}"))
        return []

    suggestions = (
        data.get("suggestionGroup", {}).get("suggestionList", {}).get("suggestion", []) or []
    )
    out = [Candidate(name=s, score=60.0, source="rxnorm_spelling") for s in suggestions if s]
    _log(debug_log, SourceAttempt("rxnorm_spelling", "ok", len(out), f"HTTP {resp.status_code}"))
    return out


def openfda_candidates(term: str, max_entries: int = 20, debug_log: Optional[List[SourceAttempt]] = None) -> List[Candidate]:
    query = f'generic_name:"{term}"+brand_name:"{term}"'
    try:
        resp = requests.get(
            OPENFDA_URL,
            params={"search": query, "limit": max_entries},
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        # openFDA returns HTTP 404 (not an error, just "no results") when a
        # search matches nothing — treat that specific case as "ok, 0 hits".
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 404:
            _log(debug_log, SourceAttempt("openfda", "ok", 0, "HTTP 404 (no matches)"))
        else:
            _log(debug_log, SourceAttempt("openfda", "network_error", 0, str(e)))
        return []
    except ValueError as e:
        _log(debug_log, SourceAttempt("openfda", "parse_error", 0, f"bad JSON: {e}"))
        return []

    out = []
    for r in data.get("results", []):
        for field in ("brand_name", "generic_name"):
            name = r.get(field)
            if name:
                out.append(Candidate(name=name, score=75.0, source="openfda"))
    _log(debug_log, SourceAttempt("openfda", "ok", len(out), f"HTTP {resp.status_code}"))
    return out


def drugscom_candidates(term: str, max_entries: int = 10, debug_log: Optional[List[SourceAttempt]] = None) -> List[Candidate]:
    """Best-effort scrape of drugs.com search results. Fails soft."""
    try:
        resp = requests.get(
            DRUGSCOM_SEARCH_URL,
            params={"searchterm": term},
            timeout=DEFAULT_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        _log(debug_log, SourceAttempt("drugs.com", "network_error", 0, str(e)))
        return []

    out = []
    for a in soup.select("a.ddc-search-suggestion, .search-result-title a, a[href*='/mtm/'], a[href*='/cdi/']"):
        name = re.sub(r"\s+", " ", a.get_text(strip=True))
        if name:
            out.append(Candidate(name=name, score=70.0, source="drugs.com"))
    out = out[:max_entries]
    detail = f"HTTP {resp.status_code}" if out else f"HTTP {resp.status_code}, selectors matched 0 elements"
    _log(debug_log, SourceAttempt("drugs.com", "ok", len(out), detail))
    return out


def onemg_candidates(term: str, max_entries: int = 10, debug_log: Optional[List[SourceAttempt]] = None) -> List[Candidate]:
    """Best-effort scrape of 1mg search results. Fails soft."""
    try:
        resp = requests.get(
            ONEMG_SEARCH_URL,
            params={"name": term},
            timeout=DEFAULT_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        _log(debug_log, SourceAttempt("1mg", "network_error", 0, str(e)))
        return []

    out = []
    for el in soup.select("[class*='style__pro-title'], a[href*='/drugs/'] div"):
        name = re.sub(r"\s+", " ", el.get_text(strip=True))
        if name:
            out.append(Candidate(name=name, score=65.0, source="1mg"))
    out = out[:max_entries]
    detail = f"HTTP {resp.status_code}" if out else f"HTTP {resp.status_code}, selectors matched 0 elements"
    _log(debug_log, SourceAttempt("1mg", "ok", len(out), detail))
    return out


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _retry_terms(term: str) -> List[str]:
    """
    Generate looser variants of a raw OCR guess to retry when the exact
    string returns zero candidates from every source. Handwriting OCR
    often injects stray digits/spaces into an otherwise-recognizable word
    (e.g. "VOLIM2N" -> "VOLIMN", "P MTMONT" -> "MTMONT"), which can push
    the term far enough from the real name that even RxNorm's fuzzy
    approximateTerm search returns nothing. These variants give the
    fuzzy/spelling APIs a second and third shot before giving up.
    """
    variants = []
    original_lower = term.strip().lower()

    cleaned = re.sub(r"[^A-Za-z\s]", "", term).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned and cleaned.lower() != original_lower:
        variants.append(cleaned)

    for token in re.findall(r"[A-Za-z]{3,}", term):
        if token.lower() not in {v.lower() for v in variants} and token.lower() != original_lower:
            variants.append(token)

    if not variants and re.fullmatch(r"[A-Za-z]+", term.strip()):
        w = term.strip()
        if len(w) > 4:
            variants.append(w[:-1])   # drop last letter
            variants.append(w[1:])    # drop first letter

    return variants


def get_candidate_pool(
    term: str,
    use_scraped_sources: bool = ENABLE_SCRAPED_SOURCES,
    scrape_as_fallback: bool = True,
    debug_log: Optional[List[SourceAttempt]] = None,
) -> List[Candidate]:
    """
    Merge RxNorm + openFDA (+ RxNorm spelling-suggestions retry, + optional
    drugs.com/1mg) candidates for one raw drug-name guess, deduped by
    normalized name, keeping the highest-scored occurrence.

    Retrieval order:
      1. RxNorm approximateTerm + openFDA on the raw guess as-is.
      2. If that returns nothing: RxNorm spellingsuggestions on the raw
         guess (purpose-built for garbled/misspelled input).
      3. If still nothing: retry approximateTerm + spellingsuggestions on
         cleaned/tokenized variants of the raw guess (strips stray OCR
         digits, tries individual tokens) — handles cases like "VOLIM2N"
         or "P MTMONT" where the exact string is too mangled to match
         anything, but a cleaned-up substring would.
      4. drugs.com/1mg scraping only as a last resort (see
         scrape_as_fallback) — these are unofficial and one of them
         (drugs.com) commonly returns 403; don't rely on it.

    use_scraped_sources=True  -> always also query drugs.com/1mg.
    scrape_as_fallback=True   -> only query drugs.com/1mg when every other
                                  source above returned nothing.
    debug_log                 -> pass a list to collect a SourceAttempt per
                                  API/scrape call made for this term, so you
                                  can see exactly what happened with each
                                  source instead of just an empty result.
    """
    pool: List[Candidate] = []
    pool.extend(rxnorm_candidates(term, debug_log=debug_log))
    pool.extend(openfda_candidates(term, debug_log=debug_log))

    if not pool:
        pool.extend(rxnorm_spelling_candidates(term, debug_log=debug_log))

    if not pool:
        for variant in _retry_terms(term):
            pool.extend(rxnorm_candidates(variant, debug_log=debug_log))
            pool.extend(rxnorm_spelling_candidates(variant, debug_log=debug_log))
            if pool:
                break  # stop as soon as a variant produces something

    needs_fallback = not pool and scrape_as_fallback
    if use_scraped_sources or needs_fallback:
        pool.extend(drugscom_candidates(term, debug_log=debug_log))
        pool.extend(onemg_candidates(term, debug_log=debug_log))
    else:
        _log(debug_log, SourceAttempt("drugs.com", "skipped", 0, "earlier sources found candidates; fallback not needed"))
        _log(debug_log, SourceAttempt("1mg", "skipped", 0, "earlier sources found candidates; fallback not needed"))

    best_by_key: Dict[str, Candidate] = {}
    for c in pool:
        key = _normalize(c.name)
        if not key:
            continue
        if key not in best_by_key or c.score > best_by_key[key].score:
            best_by_key[key] = c

    return list(best_by_key.values())