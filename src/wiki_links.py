"""Verified deeper-dive links for reader-card key terms. Pure stdlib — ``requests``
is not a dependency; ``urllib.request`` with an explicit User-Agent and an injectable opener
is the repo norm (``paper_titles.py``, ``retrieval/sources.py``).

Verify-before-link chain per term (never guess a sense, never fabricate a page):

1. Wikipedia REST summary — 200 + ``type=="standard"`` links the page;
   ``type=="disambiguation"`` skips straight to the scholar fallback.
2. 404 -> MediaWiki ``opensearch`` — a title hit is RE-VERIFIED through the summary
   endpoint before linking.
3. Otherwise Wikidata ``wbsearchentities`` — accepted ONLY when the hit's label or an
   alias casefold-equals or whole-phrase-contains the term.
4. Otherwise an honest Google-Scholar SEARCH link (it IS a search; no network needed).

Etiquette per Wikimedia policy: one serialized request at a time, ``sleep(0.15)`` between
HTTP calls, at most 3 HTTP calls per term, a proper User-Agent with a contact address, and
a network-down latch after 5 consecutive transport failures (stop fetching; the remaining
terms stay un-cached so a later run retries). The JSON cache negative-caches ``null`` ONLY
for definitive misses (chain exhausted with the network up) — NEVER transport errors.

This module runs DRIVER-side; renderers only read the persisted ``link``/``link_kind``
fields (dropped-candidate pages attach from the cache alone, zero network).
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus

_WIKI_LINK_CONTACT_ENV = "GRAPH_HYPOTH_WIKI_CONTACT"
_WIKI_LINK_DEFAULT_CONTACT = "https://github.com/Mingxue-Xu/graph-hypoth"

# Shared cache default used by the standalone Synthesist driver.
DEFAULT_WIKI_CACHE = Path("out/wiki-link-cache.json")


def _wiki_link_ua() -> str:
    """Wikimedia-etiquette User-Agent with a contact address, read from
    GRAPH_HYPOTH_WIKI_CONTACT at call time (never a personal address baked into source)."""
    contact = os.environ.get(_WIKI_LINK_CONTACT_ENV, _WIKI_LINK_DEFAULT_CONTACT)
    return f"graph-hypoth-orchestration/0.1 (reader-card link resolver; {contact})"


_SLEEP_SECONDS = 0.15
_LATCH_FAILURES = 5

_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_OPENSEARCH_URL = (
    "https://en.wikipedia.org/w/api.php?action=opensearch&limit=1&namespace=0"
    "&format=json&search="
)
_WIKIDATA_URL = (
    "https://www.wikidata.org/w/api.php?action=wbsearchentities&type=item&limit=1"
    "&format=json&language=en&uselang=en&search="
)
_SCHOLAR_URL = "https://scholar.google.com/scholar?q="


@dataclass(frozen=True)
class ResolvedLink:
    url: str
    kind: str            # "wikipedia" | "wikidata" | "arxiv-search" | "scholar-search"
    title: str = ""


def _fetch_json(
    url: str, *, timeout: float = 8.0, opener: Callable[..., Any] | None = None
) -> tuple[int, dict | list | None]:
    """GET ``url`` with the wiki User-Agent -> ``(status, parsed_json)``. NEVER raises:
    an HTTP error status is a definitive ANSWER and returns ``(status, None)`` (the chain
    needs to distinguish 404); any transport/parse failure returns ``(0, None)``."""
    request = urllib.request.Request(url, headers={"User-Agent": _wiki_link_ua()})
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 0) or 0)
            body = response.read()
        return status, json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as error:  # a real status (e.g. 404) — definitive
        return int(error.code), None
    except Exception:  # noqa: BLE001 — transport/parse trouble is never definitive
        return 0, None


def _scholar_link(term: str) -> ResolvedLink:
    return ResolvedLink(url=_SCHOLAR_URL + quote_plus(term), kind="scholar-search")


def _summary_url(title: str) -> str:
    # safe="" so a slash inside a term is %2F-encoded (the REST path requires it).
    return _SUMMARY_URL + quote(title.replace(" ", "_"), safe="")


def _wikipedia_link(data: Mapping[str, Any]) -> ResolvedLink | None:
    content_urls = data.get("content_urls")
    desktop = content_urls.get("desktop") if isinstance(content_urls, Mapping) else None
    url = str(desktop.get("page", "")) if isinstance(desktop, Mapping) else ""
    if not url:
        return None
    return ResolvedLink(url=url, kind="wikipedia", title=str(data.get("title", "") or ""))


def _canon(text: str) -> str:
    """Casefold + collapse hyphens/underscores to spaces + strip a trailing plural 's'/'es'
    per word — the equality basis for 'this page NAMES this term'."""
    words = re.split(r"[\s_-]+", text.casefold().strip())
    out = []
    for word in words:
        if len(word) > 3 and word.endswith("es"):
            word = word[:-2]
        elif len(word) > 2 and word.endswith("s"):
            word = word[:-1]
        out.append(word)
    return " ".join(w for w in out if w)


def _names_term(term: str, candidate: str) -> bool:
    """STRICT title equality (modulo case/plural/hyphenation). The verify-before-link bar:
    a page whose TITLE does not name the term is never accepted on title alone."""
    return bool(term) and bool(candidate) and _canon(term) == _canon(candidate)


def _phrase_in(term: str, text: str) -> bool:
    """The term occurs verbatim (casefold, word-boundary) inside ``text``."""
    needle, hay = term.casefold(), text.casefold()
    if not needle or not hay:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", hay) is not None


_CONTEXT_WORD = re.compile(r"[a-z]{5,}")


def _context_overlap(context: str, data: Mapping[str, Any]) -> int:
    """How many content words (>=5 chars) of the term's own definition appear on the page."""
    words = set(_CONTEXT_WORD.findall(context.casefold()))
    if not words:
        return 0
    hay = f"{data.get('title', '')} {data.get('extract', '')}".casefold()
    return sum(1 for word in words if word in hay)


def _is_short_term(term: str) -> bool:
    canon = _canon(term)
    return " " not in canon and len(canon) <= 8


def _page_names_term(term: str, data: Mapping[str, Any], context: str = "") -> bool:
    """Does this summary page actually MEAN this term? Base bar: the returned title equals
    the term (mod case/plural/hyphen) OR the page's own extract mentions it verbatim — keeps
    legitimate redirects (SNAr -> 'Nucleophilic aromatic substitution') while rejecting fuzzy
    near-misses (DESs -> 'Dessert', 'low toxicity' -> 'Tonicity'). SHORT single-word terms
    (acronyms) additionally face the SENSE problem — a page can name the term and still be a
    different concept ('Nades', a village, vs NADES the solvents) — so when the caller supplies
    ``context`` (the term's own definition): a short term that passed the base bar is VETOED
    on zero context overlap, and one that failed it is RESCUED on strong overlap (>=2 content
    words — the SNAr redirect whose extract omits the acronym but matches the chemistry)."""
    named = _names_term(term, str(data.get("title", "") or "")) or _phrase_in(
        term, str(data.get("extract", "") or "")
    )
    if not _is_short_term(term):
        return named
    if not context:
        return named
    overlap = _context_overlap(context, data)
    if named:
        return overlap >= 1
    return overlap >= 2


def _wikidata_link(term: str, data: Mapping[str, Any], context: str = "") -> ResolvedLink | None:
    hits = data.get("search")
    if not isinstance(hits, list) or not hits or not isinstance(hits[0], Mapping):
        return None
    hit = hits[0]
    label = str(hit.get("label", "") or "")
    # STRICT main-label equality only (no aliases, no containment): alias fuzz is exactly
    # how 'NADES' linked a gamer tag — an entity that doesn't NAME the term never links.
    if not _names_term(term, label):
        return None
    # Wikidata hits face the SENSE problem even on label equality — for ANY term length
    # ('Spectrum Control' the electronics company label-matches 'spectrum control' the
    # signal-processing notion): with a context (the term's own definition) require >=1
    # overlapping content word against label+description; no agreement -> honest fallback.
    if context:
        page_view = {"title": label, "extract": str(hit.get("description", "") or "")}
        if _context_overlap(context, page_view) < 1:
            return None
    entity_id = str(hit.get("id", "") or "")
    if not entity_id:
        return None
    return ResolvedLink(
        url=f"https://www.wikidata.org/wiki/{entity_id}", kind="wikidata", title=label
    )


def resolve_term(
    term: str,
    *,
    context: str = "",
    fetch_json: Callable[[str], tuple[int, dict | list | None]] = _fetch_json,
    sleep: Callable[[float], Any] = time.sleep,
) -> ResolvedLink | None:
    """Resolve ONE term through the verify-before-link chain (<= 3 HTTP calls,
    ``sleep(0.15)`` between them). Returns a verified wikipedia/wikidata link, the honest
    scholar-search fallback when the chain concludes definitively without one, or ``None``
    when a transport failure prevented a definitive conclusion (callers must NOT cache a
    ``None``)."""
    term = str(term or "").strip()
    if not term:
        return None
    calls = 0

    def get(url: str) -> tuple[int, dict | list | None]:
        nonlocal calls
        if calls:
            sleep(_SLEEP_SECONDS)
        calls += 1
        return fetch_json(url)

    status, data = get(_summary_url(term))
    if status == 200 and isinstance(data, Mapping):
        page_type = str(data.get("type", "") or "")
        if page_type == "standard":
            # A 200 can be a REDIRECT to a different concept (DESs -> 'Dessert'): accept
            # only when the page provably NAMES the term (title equality or verbatim
            # mention in its own extract); else fall through to Wikidata's strict guard.
            if _page_names_term(term, data, context):
                link = _wikipedia_link(data)
                return link if link is not None else _scholar_link(term)
            return _wikidata_step(term, get, context)
        if page_type == "disambiguation":
            return _scholar_link(term)  # never guess a sense
        # An unexpected page type is a definitive non-article -> try Wikidata (call 2).
        return _wikidata_step(term, get, context)
    if status == 404:
        status, data = get(_OPENSEARCH_URL + quote_plus(term))
        if status != 200 or not isinstance(data, list):
            return None if status == 0 else _scholar_link(term)
        titles = data[1] if len(data) > 1 and isinstance(data[1], list) else []
        if titles and str(titles[0] or "").strip():
            # A fuzzy hit is never linked blind: RE-VERIFY through the summary endpoint,
            # AND the re-verified page must NAME the term ('low toxicity' -> 'Tonicity'
            # passes opensearch but fails this bar -> honest scholar fallback).
            status, data = get(_summary_url(str(titles[0])))
            if status == 0:
                return None
            if (
                status == 200
                and isinstance(data, Mapping)
                and data.get("type") == "standard"
                and _page_names_term(term, data, context)
            ):
                link = _wikipedia_link(data)
                if link is not None:
                    return link
            return _scholar_link(term)  # budget spent; honest fallback
        return _wikidata_step(term, get, context)
    # Any other status (5xx/429/...) is transport-ish: inconclusive, never cached.
    return None


def _wikidata_step(
    term: str, get: Callable[[str], tuple[int, dict | list | None]], context: str = ""
) -> ResolvedLink | None:
    status, data = get(_WIKIDATA_URL + quote_plus(term))
    if status == 0:
        return None
    if status == 200 and isinstance(data, Mapping):
        link = _wikidata_link(term, data, context)
        if link is not None:
            return link
    return _scholar_link(term)


def load_cache(path: str | Path) -> dict[str, dict | None]:
    """The link cache: ``{casefolded_term: {url,kind,title,resolved_at} | null}``. A missing
    or corrupt file loads as empty (renderers reading cache-only stay offline-safe)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_cache(path: str | Path, cache: Mapping[str, dict | None]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(cache), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _cached_link(term: str, entry: Mapping[str, Any] | None) -> ResolvedLink:
    if entry is None:
        # null = a KNOWN definitive miss: the scholar-search link is derived offline.
        return _scholar_link(term)
    return ResolvedLink(
        url=str(entry.get("url", "") or ""),
        kind=str(entry.get("kind", "") or ""),
        title=str(entry.get("title", "") or ""),
    )


def resolve_terms(
    terms: Iterable[str],
    *,
    contexts: Mapping[str, str] | None = None,
    cache_path: str | Path | None = None,
    fetch_json: Callable[[str], tuple[int, dict | list | None]] = _fetch_json,
    sleep: Callable[[float], Any] = time.sleep,
) -> dict[str, ResolvedLink | None]:
    """Resolve many terms through one serialized loop with the shared cache and the
    network-down latch. Cache policy: verified wikipedia/wikidata links are stored as
    entries; a chain that exhausted definitively (scholar fallback) stores ``null``;
    transport-inconclusive terms store NOTHING (retried next run)."""
    cache: dict[str, dict | None] = load_cache(cache_path) if cache_path else {}
    dirty = False
    consecutive_failures = 0
    fetched_any = False

    def latched_fetch(url: str) -> tuple[int, dict | list | None]:
        nonlocal consecutive_failures
        status, data = fetch_json(url)
        consecutive_failures = consecutive_failures + 1 if status == 0 else 0
        return status, data

    out: dict[str, ResolvedLink | None] = {}
    batch: dict[str, ResolvedLink | None] = {}
    for raw_term in terms:
        term = str(raw_term or "").strip()
        if not term:
            out[str(raw_term)] = None
            continue
        key = term.casefold()
        if key in batch:
            out[str(raw_term)] = batch[key]
            continue
        if key in cache:
            link: ResolvedLink | None = _cached_link(term, cache[key])
        elif consecutive_failures >= _LATCH_FAILURES:
            link = None  # network-down latch: stop fetching, leave un-cached
        else:
            if fetched_any:
                sleep(_SLEEP_SECONDS)  # serialized spacing between terms' HTTP calls
            fetched_any = True
            link = resolve_term(
                term,
                context=str((contexts or {}).get(key, "") or ""),
                fetch_json=latched_fetch,
                sleep=sleep,
            )
            if link is not None:
                cache[key] = (
                    None  # definitive miss -> negative-cache null
                    if link.kind == "scholar-search"
                    else {
                        "url": link.url,
                        "kind": link.kind,
                        "title": link.title,
                        "resolved_at": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                    }
                )
                dirty = True
        batch[key] = link
        out[str(raw_term)] = link
    if cache_path and dirty:
        save_cache(cache_path, cache)
    return out


def attach_key_term_links(
    elaborations: Mapping[str, Any],
    *,
    cache_path: str | Path | None,
    enabled: bool = True,
    fetch_json: Callable[[str], tuple[int, dict | list | None]] = _fetch_json,
    sleep: Callable[[float], Any] = time.sleep,
) -> Mapping[str, Any]:
    """Collect every ``key_terms[].term`` across the cards (deduped casefold), resolve each
    once, and write ``link``/``link_kind`` into entries LACKING them (existing links are
    never overwritten). ``enabled=False`` returns the input untouched with zero fetches.
    The input mapping is never mutated."""
    if not enabled or not elaborations:
        return elaborations
    ordered_terms: list[str] = []
    contexts: dict[str, str] = {}
    seen: set[str] = set()
    for card in elaborations.values():
        if not isinstance(card, Mapping):
            continue
        for entry in card.get("key_terms") or []:
            if not isinstance(entry, Mapping) or entry.get("link"):
                continue  # only entries LACKING a link need resolution (write-only-absent)
            term = str(entry.get("term", "") or "").strip()
            if not term:
                continue
            key = term.casefold()
            # The term's own definition is the SENSE context for short/acronym terms.
            gloss = str(entry.get("plain_meaning", "") or entry.get("definition", "") or "")
            if key not in seen:
                seen.add(key)
                ordered_terms.append(term)
                contexts[key] = gloss
            elif gloss and not contexts.get(key):
                contexts[key] = gloss
    if not ordered_terms:
        return elaborations
    resolved = resolve_terms(
        ordered_terms, contexts=contexts, cache_path=cache_path,
        fetch_json=fetch_json, sleep=sleep,
    )
    by_key = {term.casefold(): link for term, link in resolved.items()}
    out = copy.deepcopy(dict(elaborations))
    for card in out.values():
        if not isinstance(card, Mapping):
            continue
        for entry in card.get("key_terms") or []:
            if not isinstance(entry, dict) or entry.get("link"):
                continue
            link = by_key.get(str(entry.get("term", "") or "").strip().casefold())
            if link is not None and link.url:
                entry["link"] = link.url
                entry["link_kind"] = link.kind
    return out
