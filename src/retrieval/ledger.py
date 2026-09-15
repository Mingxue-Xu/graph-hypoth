from __future__ import annotations

import html
import re
from datetime import datetime
from typing import Any

from src.paper_titles import (
    METADATA_HEADING_RE as _METADATA_HEADING_RE,
    METADATA_TO_SECTION_RE as _METADATA_TO_SECTION_RE,
    looks_like_metadata_or_title_excerpt as _looks_like_metadata_or_title_excerpt,
)
from src.retrieval.models import SourceResult
from src.retrieval.similarity import Embedder, _cosine
from src.retrieval.quote_verification import (
    QuoteVerificationResult,
    verify_candidate_excerpt,
)
from src.state import RetrievedEvidence


MAX_QUOTE_CHARS = 80000  # 5x baseline: feed the miner ~5x more of a rich full-text body
# Cap for the larger appraiser-facing body excerpt. The compact, verified ``quote``
# stays at MAX_QUOTE_CHARS for display/Q-scoring; when a source carries substantially more
# text (a fetched full-text body, or an abstract longer than the 3-sentence quote window)
# the appraiser additionally sees up to this many cleaned chars so it judges the body, not
# a snippet. The bound keeps verifier prompt size and cost controlled.
MAX_BODY_EXCERPT_CHARS = 16000
# Max sentences in a query-relevant quote window. The window scorer still tie-breaks toward
# FEWER sentences for a given relevance score, so this does NOT pull unrelated context in —
# it only lets the quote grow when the query-relevant content genuinely spans more sentences
# (e.g. a full-text body), instead of the old hard 3-sentence snippet.
MAX_QUOTE_SENTENCES = 60  # 5x baseline: the sentence window is the *binding* quote-length cap

_STITCHED_LOCATOR_RE = re.compile(r"\[\s*\.{3}\s*\]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_ABSTRACT_HEADING_RE = re.compile(r"^abstract\b", re.I)
_SECTION_HEADING_RE = re.compile(
    r"^(?:abstract\b|(?:\d+|[ivxlcdm]+)[.)]\s+[A-Za-z])",
    re.I,
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
}


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(value: str) -> str:
    # Cross-source titles carry stray markup (e.g. "... Using <i>R</i>"); strip tags
    # and unescape entities so the same title keys identically across sources.
    # NOTE: intentionally NOT paper_titles.strip_markup — this strips tags to ""
    # (merging adjacent words), while strip_markup replaces tags with " ". Title
    # keys are lowercased/whitespace-collapsed downstream so the merge is fine
    # here; kept local rather than forced into a shared helper.
    return html.unescape(_HTML_TAG_RE.sub("", value))


def _normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", _strip_html(value).strip().lower())


def _publication_year(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(datetime.fromisoformat(value.replace("Z", "+00:00")).year)
    except ValueError:
        return value[:4]


_ID_PRIORITY = ("doi", "pmid", "pmcid", "arxiv")

# Minimum normalized-title length for the cross-source title bridge key. Long enough to
# exclude generic/short headings (and never collide V1 vs V2, which differ in title)
# while covering real scholarly titles, which run far longer.
_TITLE_BRIDGE_MIN_CHARS = 16


def _dedupe_keys(result: SourceResult) -> list[tuple[str, str]]:
    # ALL of a result's identity keys, so a paper can be matched by any external
    # id it shares with an earlier sighting (e.g. OpenAlex {doi,pmid} bridges to a
    # later EuropePMC carrying only the shared pmid). Source-AGNOSTIC external ids
    # first (in priority order), then the source-prefixed source_id (intra-source),
    # then url. A normalized, HTML-stripped, year-free TITLE bridge key is added on
    # top so the same work from sources sharing no external id (e.g. an arXiv-id-only
    # preprint vs a DOI-only indexed record) still collapses to one record.
    external_ids = getattr(result, "external_ids", None) or {}
    cross_source = result.metadata.get("cross_source_dedupe") is not False
    keys: list[tuple[str, str]] = []
    if cross_source:
        for kind in _ID_PRIORITY:
            value = external_ids.get(kind)
            if value:
                keys.append((kind, value))
    if result.source_id:
        keys.append(("source_id", f"{result.source}:{result.source_id}"))
    elif result.url:
        url_key = result.url.rstrip("/").lower()
        keys.append(
            ("url", url_key if cross_source else f"{result.source}:{url_key}")
        )
    normalized_title = _normalize_title(result.title)
    if cross_source and len(normalized_title) >= _TITLE_BRIDGE_MIN_CHARS:
        keys.append(("title", normalized_title))
    if not keys:
        # No id, url, or distinctive title — fall back to title+year as the sole key.
        keys.append(
            ("title_year", f"{normalized_title}:{_publication_year(result.published_date)}")
        )
    return keys


def _union_preserving_order(*lists: list[Any]) -> list[Any]:
    # Merge lists into one, dropping duplicates while keeping first-seen order.
    seen: set[str] = set()
    merged: list[Any] = []
    for values in lists:
        for value in values:
            marker = str(value)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(value)
    return merged


def _evidence_index(value: str) -> int | None:
    match = re.fullmatch(r"ev_(\d{6})", value)
    if match is None:
        return None
    return int(match.group(1))


def _normalized_scores_by_index(results: list[SourceResult]) -> dict[int, float | None]:
    grouped: dict[str, list[tuple[int, SourceResult]]] = {}
    for index, result in enumerate(results):
        grouped.setdefault(result.source, []).append((index, result))

    normalized: dict[int, float | None] = {}
    for group in grouped.values():
        numeric = [(index, item.score) for index, item in group if item.score is not None]
        if not numeric:
            for index, _ in group:
                normalized[index] = None
            continue
        values = [float(score) for _, score in numeric]
        low = min(values)
        high = max(values)
        for index, item in group:
            if item.score is None:
                normalized[index] = None
            elif high == low:
                normalized[index] = 1.0
            else:
                normalized[index] = (float(item.score) - low) / (high - low)
    return normalized


def _with_unique_existing_ids(items: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
    used: set[int] = set()
    unique_items: list[RetrievedEvidence] = []
    for item in items:
        index = _evidence_index(item.evidence_id)
        if index is not None and index not in used:
            used.add(index)
            unique_items.append(item)
            continue

        next_index = max(used, default=0) + 1
        while next_index in used:
            next_index += 1
        used.add(next_index)
        unique_items.append(
            item.model_copy(update={"evidence_id": f"ev_{next_index:06d}"})
        )
    return unique_items


def _compact_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _source_text_for_quote_selection(value: str | None) -> str:
    if not value:
        return ""
    lines = value.splitlines()
    start_index = 0
    for index, line in enumerate(lines[:80]):
        if _ABSTRACT_HEADING_RE.match(line.strip()):
            start_index = index
            break

    kept: list[str] = []
    skipping_metadata = False
    for line in lines[start_index:]:
        stripped = line.strip()
        if skipping_metadata:
            if _SECTION_HEADING_RE.match(stripped):
                skipping_metadata = False
                continue
            else:
                continue
        if _SECTION_HEADING_RE.match(stripped) and len(stripped.split()) <= 5:
            continue
        if _METADATA_HEADING_RE.match(stripped):
            inline_remainder = _METADATA_TO_SECTION_RE.sub("", stripped, count=1)
            if inline_remainder != stripped and inline_remainder.strip():
                kept.append(inline_remainder.strip())
                skipping_metadata = False
                continue
            skipping_metadata = True
            continue
        kept.append(line)
    return "\n".join(kept)


def _sentences(value: str | None) -> list[str]:
    compact = _compact_text(value)
    if not compact:
        return []
    matches = [
        match.group(0).strip()
        for match in re.finditer(r"[^.!?]+(?:[.!?]+(?=\s|$)|$)", compact)
    ]
    return [sentence for sentence in matches if sentence]


def _truncate_quote(value: str) -> str:
    compact = _compact_text(value)
    if len(compact) <= MAX_QUOTE_CHARS:
        return compact

    truncated = compact[:MAX_QUOTE_CHARS].rstrip()
    last_sentence_end = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )
    if last_sentence_end >= MAX_QUOTE_CHARS // 3:
        return truncated[: last_sentence_end + 1].rstrip()

    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space].rstrip()
    return truncated


def _shape_quote(value: str | None) -> str:
    compact = _compact_text(value)
    if not compact:
        return ""

    complete_sentences = [
        sentence for sentence in _sentences(compact) if re.search(r"[.!?]$", sentence)
    ]
    if complete_sentences:
        for sentence_count in range(min(MAX_QUOTE_SENTENCES, len(complete_sentences)), 0, -1):
            candidate = _compact_text(" ".join(complete_sentences[:sentence_count]))
            if len(candidate) <= MAX_QUOTE_CHARS:
                return candidate

    return _truncate_quote(compact)


def _query_terms(query: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD_RE.findall(query)
        if token.lower() not in _STOPWORDS
    }


def _term_matches(term: str, token: str) -> bool:
    if token == term:
        return True
    if len(term) <= 4:
        return False
    return token.startswith(term) or term.startswith(token)


def _query_match_score(value: str, terms: set[str]) -> int:
    if not terms:
        return 0
    tokens = [token.lower() for token in _WORD_RE.findall(value)]
    matched_terms = {
        term for term in terms if any(_term_matches(term, token) for token in tokens)
    }
    total_matches = sum(
        1
        for term in terms
        for token in tokens
        if _term_matches(term, token)
    )
    return len(matched_terms) * 10 + total_matches


def _clean_highlight(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    compact = _compact_text(value)
    if not compact or _STITCHED_LOCATOR_RE.search(compact):
        return None
    return compact


def _highlight_score(metadata: dict[str, Any], index: int) -> float:
    scores = metadata.get("exa_highlight_scores")
    if not isinstance(scores, list) or index >= len(scores):
        return 0.0
    score = scores[index]
    if isinstance(score, int | float):
        return float(score)
    return 0.0


def _highlight_quote(result: SourceResult, query: str) -> tuple[str, int] | None:
    highlights = result.metadata.get("exa_highlights")
    if isinstance(highlights, str):
        highlights = [highlights]
    if not isinstance(highlights, list):
        return None

    terms = _query_terms(query)
    candidates: list[tuple[int, float, int, str]] = []
    for index, value in enumerate(highlights):
        clean = _clean_highlight(value)
        if clean is None:
            continue
        if _looks_like_metadata_or_title_excerpt(clean, title=result.title):
            continue
        score = _query_match_score(clean, terms)
        if terms and score == 0:
            continue
        candidates.append(
            (
                score,
                _highlight_score(result.metadata, index),
                -index,
                clean,
            )
        )

    if not candidates:
        return None
    best = max(candidates)
    *_, clean = best
    index = -best[2]
    return _shape_quote(clean), index


def _selection_metadata(
    *,
    source: str,
    highlight_index: int | None,
    query: str,
    candidate_excerpt: str,
    verified_quote: str | None,
    verification_status: str,
    match_type: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "highlight_index": highlight_index,
        "query": query,
        "candidate_excerpt": candidate_excerpt,
        "verified_quote": verified_quote,
        "verification_status": verification_status,
        "match_type": match_type,
    }


def _text_quote(text: str | None, query: str) -> str:
    source_text = _source_text_for_quote_selection(text)
    compact = _compact_text(source_text)
    if not compact:
        return ""

    sentences = _sentences(compact)
    if not sentences:
        return _truncate_quote(compact)

    terms = _query_terms(query)
    best_candidate: tuple[int, int, int, str] | None = None
    for start in range(len(sentences)):
        max_count = min(MAX_QUOTE_SENTENCES, len(sentences) - start)
        for sentence_count in range(1, max_count + 1):
            candidate = _compact_text(" ".join(sentences[start : start + sentence_count]))
            score = _query_match_score(candidate, terms)
            rank = (score, -sentence_count, -start, candidate)
            if best_candidate is None or rank > best_candidate:
                best_candidate = rank

    if best_candidate is None or best_candidate[0] == 0:
        return _shape_quote(compact)
    return _shape_quote(best_candidate[3])


def _quote_query_for(result: SourceResult, query: str) -> str:
    highlight_query = result.metadata.get("exa_highlight_query")
    if isinstance(highlight_query, str) and highlight_query.strip():
        return highlight_query.strip()
    return query


def _body_excerpt(text: str | None, quote: str) -> str:
    """Return a larger cleaned body excerpt for the appraiser.

    Returns up to ``MAX_BODY_EXCERPT_CHARS`` of cleaned body text when the source carries
    substantially more than the compact verified ``quote`` (a fetched full-text body, or an
    abstract longer than the 3-sentence quote window); empty when the body adds nothing
    beyond the quote (so no excerpt is stored and the appraiser falls back to the quote).
    """
    body = _compact_text(_source_text_for_quote_selection(text))
    if len(body) <= len(quote):
        return ""
    if len(body) <= MAX_BODY_EXCERPT_CHARS:
        return body
    truncated = body[:MAX_BODY_EXCERPT_CHARS].rstrip()
    cut = max(truncated.rfind(". "), truncated.rfind("? "), truncated.rfind("! "))
    if cut >= MAX_BODY_EXCERPT_CHARS // 2:
        return truncated[: cut + 1]
    return truncated


def _quote_for(result: SourceResult, query: str) -> tuple[str, dict[str, Any]]:
    quote_query = _quote_query_for(result, query)
    highlight = _highlight_quote(result, quote_query)
    if highlight is not None:
        candidate_excerpt, highlight_index = highlight
        verification = verify_candidate_excerpt(
            candidate_excerpt=candidate_excerpt,
            source_text=result.text,
        )
        if verification.verification_status == "accepted" and verification.verified_quote:
            return verification.verified_quote, _selection_metadata(
                source="highlight",
                highlight_index=highlight_index,
                query=quote_query,
                candidate_excerpt=candidate_excerpt,
                verified_quote=verification.verified_quote,
                verification_status=verification.verification_status,
                match_type=verification.match_type,
            )

        text_quote = _text_quote(result.text, quote_query)
        if text_quote:
            text_verification = verify_candidate_excerpt(
                candidate_excerpt=text_quote,
                source_text=result.text,
            )
            return text_quote, _candidate_rejection_metadata(
                verification=verification,
                final_verification=text_verification,
                source="text",
                highlight_index=highlight_index,
                query=quote_query,
                fallback_quote=text_verification.verified_quote or text_quote,
            )

        summary_quote = _shape_quote(result.summary)
        return summary_quote, _candidate_rejection_metadata(
            verification=verification,
            final_verification=None,
            source="summary",
            highlight_index=highlight_index,
            query=quote_query,
            fallback_quote=summary_quote or None,
        )

    text_quote = _text_quote(result.text, quote_query)
    if text_quote:
        verification = verify_candidate_excerpt(
            candidate_excerpt=text_quote,
            source_text=result.text,
        )
        return text_quote, _selection_metadata(
            source="text",
            highlight_index=None,
            query=quote_query,
            candidate_excerpt=text_quote,
            verified_quote=verification.verified_quote or text_quote,
            verification_status=verification.verification_status,
            match_type=verification.match_type,
        )

    summary_quote = _shape_quote(result.summary)
    return summary_quote, _selection_metadata(
        source="summary",
        highlight_index=None,
        query=quote_query,
        candidate_excerpt=summary_quote,
        verified_quote=summary_quote or None,
        verification_status="summary_fallback" if summary_quote else "unverified",
        match_type="abstractive_summary" if summary_quote else "no_source_text",
    )


def _candidate_rejection_metadata(
    *,
    verification: QuoteVerificationResult,
    final_verification: QuoteVerificationResult | None,
    source: str,
    highlight_index: int | None,
    query: str,
    fallback_quote: str | None,
) -> dict[str, Any]:
    metadata = _selection_metadata(
        source=source,
        highlight_index=highlight_index,
        query=query,
        candidate_excerpt=verification.candidate_excerpt,
        verified_quote=fallback_quote,
        verification_status=(
            final_verification.verification_status
            if final_verification is not None
            else verification.verification_status
        ),
        match_type=(
            final_verification.match_type
            if final_verification is not None
            else verification.match_type
        ),
    )
    metadata["candidate_verification_status"] = verification.verification_status
    metadata["candidate_match_type"] = verification.match_type
    return metadata


class EvidenceLedger:
    def __init__(
        self,
        *,
        trust_policy: dict[str, str],
        trust_tier_priority: dict[str, int],
        source_order: list[str],
        existing: list[RetrievedEvidence] | None = None,
    ) -> None:
        self.trust_policy = trust_policy
        self.trust_tier_priority = trust_tier_priority
        self.source_order = source_order
        self._items: list[RetrievedEvidence] = _with_unique_existing_ids(
            list(existing or [])
        )
        existing_indexes = [
            index
            for item in self._items
            if (index := _evidence_index(item.evidence_id)) is not None
        ]
        self._next_index = (
            max(existing_indexes) + 1 if existing_indexes else len(self._items) + 1
        )

    def add_many(
        self,
        *,
        results: list[SourceResult],
        query: str,
        retrieved_by: str,
        tool_call_id: str,
    ) -> list[RetrievedEvidence]:
        # Index each item under ALL of its identity keys so a later sighting can
        # match on any shared id (e.g. a pmid-only EuropePMC hit bridges to an
        # earlier OpenAlex {doi,pmid} item).
        by_key: dict[tuple[str, str], RetrievedEvidence] = {}
        for item in self._items:
            for item_key in _dedupe_keys(self._source_shadow(item)):
                by_key[item_key] = item
        added_or_updated: list[RetrievedEvidence] = []
        normalized_scores = _normalized_scores_by_index(results)

        for source_rank, result in enumerate(results, start=1):
            result_keys = _dedupe_keys(result)
            raw_score = result.score
            normalized_score = normalized_scores[source_rank - 1]
            provenance = self._provenance_for(
                result=result,
                source_rank=source_rank,
                tool_call_id=tool_call_id,
            )
            existing = next(
                (by_key[k] for k in result_keys if k in by_key), None
            )
            if existing is not None:
                existing.metadata.setdefault("provenance", []).append(provenance)
                # Union the identity map so later sightings enrich it, then
                # re-register the item under any newly-added keys.
                if result.external_ids:
                    existing.external_ids = {
                        **existing.external_ids,
                        **result.external_ids,
                    }
                    for new_key in _dedupe_keys(self._source_shadow(existing)):
                        by_key.setdefault(new_key, existing)
                # Highest-trust wins: a higher-trust source's record replaces the
                # displayed fields, keeping the stable evidence_id + provenance.
                new_tier = self.trust_policy[result.source]
                if self._tier_priority(new_tier) > self._tier_priority(existing.trust_tier):
                    quote, quote_selection = _quote_for(result, query)
                    provenance_list = existing.metadata.get("provenance", [])
                    existing.source = result.source
                    existing.source_id = result.source_id
                    existing.title = result.title
                    existing.authors = result.authors
                    existing.published_date = result.published_date
                    existing.url = result.url
                    # Keep the RICHEST quote: adopt the higher-trust source's quote only
                    # when it is strictly richer (longer) than the existing one. This
                    # never wipes a good quote with an empty one, AND never downgrades a
                    # rich Exa highlight or full passage to a shorter metadata snippet.
                    replace_quote = bool(quote) and len(quote) > len(existing.quote)
                    if replace_quote:
                        existing.quote = quote
                    if normalized_score is not None:
                        existing.score = normalized_score
                    existing.trust_tier = new_tier
                    # Merge metadata so the existing source's enrichment (e.g.
                    # references, r2) survives, with the incoming source winning
                    # where it has a value.
                    new_metadata = {**existing.metadata, **result.metadata}
                    # Union reference lists across both sources so citation
                    # coverage (read by the coherence layer) is not lost when the
                    # incoming source carries its own, different references.
                    existing_refs = existing.metadata.get("references")
                    incoming_refs = result.metadata.get("references")
                    if existing_refs or incoming_refs:
                        new_metadata["references"] = _union_preserving_order(
                            existing_refs or [], incoming_refs or []
                        )
                    # Adopt the incoming source's body excerpt when it is richer than the
                    # one already stored (e.g. a higher-trust source carries full text the
                    # original lacked); keep the richest, as with the quote.
                    incoming_excerpt = _body_excerpt(result.text, existing.quote)
                    existing_excerpt = existing.metadata.get("full_text_excerpt", "")
                    if len(incoming_excerpt) > len(existing_excerpt or ""):
                        new_metadata["full_text_excerpt"] = incoming_excerpt
                    new_metadata["source_rank"] = source_rank
                    new_metadata["provenance"] = provenance_list
                    if raw_score is not None:
                        new_metadata["raw_score"] = raw_score
                    # Keep ``quote_selection`` in lockstep with the retained quote:
                    # only overwrite it when the incoming quote actually replaced the
                    # existing one, so it keeps describing the displayed quote.
                    if replace_quote:
                        new_metadata["quote_selection"] = quote_selection
                    existing.metadata = new_metadata
                added_or_updated.append(existing)
                continue

            trust_tier = self.trust_policy[result.source]
            quote, quote_selection = _quote_for(result, query)
            # Rehydrate coherence fields from metadata["r2"], which (unlike the
            # top-level fields) survives the _source_shadow re-admit round-trip.
            r2 = result.metadata.get("r2")
            r2 = r2 if isinstance(r2, dict) else {}
            evidence = RetrievedEvidence(
                evidence_id=f"ev_{self._next_index:06d}",
                source=result.source,
                source_id=result.source_id,
                title=result.title,
                authors=result.authors,
                published_date=result.published_date,
                url=result.url,
                quote=quote,
                relevance=f"retrieved for query: {query}",
                retrieval_method="search_papers",
                retrieved_by=retrieved_by,
                tool_call_id=tool_call_id,
                score=normalized_score,
                rank=0,
                trust_tier=trust_tier,
                redacted=False,
                external_ids=dict(result.external_ids),
                relatedness_score=r2.get("relatedness_score"),
                relation_label=r2.get("relation_label"),
                domain_context=r2.get("domain_context"),
                synonym_links=r2.get("synonym_links"),
                citation_neighbors=r2.get("citation_neighbors"),
                metadata={
                    **result.metadata,
                    "raw_score": raw_score,
                    "source_rank": source_rank,
                    "quote_selection": quote_selection,
                    "provenance": [provenance],
                    # The fuller appraiser-facing body excerpt, omitted when the
                    # body adds nothing beyond the compact quote.
                    **(
                        {"full_text_excerpt": body_excerpt}
                        if (body_excerpt := _body_excerpt(result.text, quote))
                        else {}
                    ),
                },
            )
            self._next_index += 1
            self._items.append(evidence)
            # Register the new evidence under all of its identity keys so a later
            # sighting sharing any one of them merges instead of duplicating.
            for new_key in result_keys:
                by_key.setdefault(new_key, evidence)
            added_or_updated.append(evidence)

        self._sort_and_rank()
        return added_or_updated

    def add_retrieved(
        self,
        *,
        items: list[RetrievedEvidence],
        query: str,
        retrieved_by: str,
        tool_call_id: str,
    ) -> list[RetrievedEvidence]:
        results = [self._source_shadow(item) for item in items]
        return self.add_many(
            results=results,
            query=query,
            retrieved_by=retrieved_by,
            tool_call_id=tool_call_id,
        )

    def all_evidence(self) -> list[RetrievedEvidence]:
        return list(self._items)

    def score_relevance(self, embedder: Embedder | None, claim: str) -> None:
        """Score each item's claim-relevance (specter2 cosine of ``claim`` against the
        item's ``title + quote``) into ``metadata['claim_relevance']``, then re-rank.

        Cross-source comparable, unlike the per-source ``score`` — this is what lets an
        on-topic web_research_paper outrank an abstract-less higher-trust stub. A no-op
        when no embedder is available, so the deterministic (embedder-free) path keeps
        its trust-first order."""
        if embedder is None or not self._items:
            return
        claim_vector = embedder([claim])[0]
        texts = [f"{item.title or ''} {item.quote or ''}".strip() for item in self._items]
        vectors = embedder(texts)
        for item, vector in zip(self._items, vectors):
            item.metadata["claim_relevance"] = _cosine(claim_vector, vector)
        self._sort_and_rank()

    def drop_empty_quote_stubs(self) -> None:
        """Remove records that carry no usable source text (``quote_selection``
        match_type ``no_source_text``) — dead metadata stubs that cannot ground the
        miner — and re-rank the survivors."""
        self._items = [
            item
            for item in self._items
            if (item.metadata.get("quote_selection") or {}).get("match_type")
            != "no_source_text"
        ]
        self._sort_and_rank()

    def _sort_and_rank(self) -> None:
        self._items.sort(key=self._ranking_key)
        for rank, item in enumerate(self._items, start=1):
            item.rank = rank

    def _ranking_key(self, item: RetrievedEvidence) -> tuple[Any, ...]:
        # Relevance-first: claim-relevance (specter2 cosine, set by score_relevance)
        # dominates; trust tier is only a tiebreaker. When relevance is unscored
        # (the embedder-free path) it is 0.0 for every item, so the key degrades to
        # the original trust-first ordering and existing goldens are unaffected.
        return (
            -float(item.metadata.get("claim_relevance", 0.0)),
            -self.trust_tier_priority.get(item.trust_tier, 0),
            self._source_index(item.source),
            # RRF fan-out signal (set by the planned multi-query path): rewards papers
            # surfaced by several sub-queries. 0.0 when single-query, so unaffected.
            -float(item.metadata.get("rrf_score", 0.0)),
            -self._source_rank_score(item),
            int(item.metadata.get("source_rank", 1)),
            item.title,
        )

    def _tier_priority(self, tier: str) -> int:
        return int(self.trust_tier_priority.get(tier, 0))

    def _source_index(self, source: str) -> int:
        try:
            return self.source_order.index(source)
        except ValueError:
            return len(self.source_order)

    @staticmethod
    def _source_rank_score(item: RetrievedEvidence) -> float:
        if item.score is not None:
            return item.score
        source_rank = int(item.metadata.get("source_rank", 1))
        return 1 / source_rank

    @staticmethod
    def _source_shadow(item: RetrievedEvidence) -> SourceResult:
        return SourceResult(
            source=item.source,
            source_id=item.source_id,
            external_ids=dict(item.external_ids),
            title=item.title,
            authors=item.authors,
            published_date=item.published_date,
            url=item.url,
            text=item.quote,
            summary=None,
            score=item.metadata.get("raw_score", item.score),
            metadata=dict(item.metadata),
        )

    @staticmethod
    def _provenance_for(
        *,
        result: SourceResult,
        source_rank: int,
        tool_call_id: str,
    ) -> dict[str, Any]:
        return {
            "source": result.source,
            "source_id": result.source_id,
            "url": result.url,
            "source_rank": source_rank,
            "score": result.score,
            "tool_call_id": tool_call_id,
        }
