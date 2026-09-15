"""Shared deterministic similarity primitives — single source of truth.

These pure functions are reused by claim-scoped coherence and target-scoped association scoring,
so the two paths cannot drift on tokenization, cosine clamping, identifier normalization, or
citation overlap. They are model-free and network-free.

Two token surfaces are intentionally distinct:
- `_tokenize` — raw lowercase `[a-z0-9]+` tokens (legacy coherence behavior).
- `normalized_terms` — `_tokenize` MINUS the canonical scoring stopword list
  (``scoring_defaults.STOPWORDS``) for the lexical channel.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

from src.retrieval import scoring_defaults as sd

# An embedder maps a batch of texts to vectors (same contract as coherence.py).
Embedder = Callable[[list[str]], "list[list[float]]"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str | None) -> set[str]:
    """Raw lowercase alphanumeric tokens (legacy coherence tokenizer)."""
    return set(_TOKEN_RE.findall((text or "").lower()))


def normalized_terms(text: str | None) -> set[str]:
    """Return lowercase alphanumeric tokens minus scoring stopwords.

    Uses the canonical, version-pinned ``scoring_defaults.STOPWORDS`` rather than the
    ledger's own stopword list.
    """
    return {token for token in _tokenize(text) if token not in sd.STOPWORDS}


def clip01(value: float) -> float:
    """Clamp a numeric value to ``[0, 1]``."""
    return min(1.0, max(0.0, value))


def jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard index over two token sets; 0.0 when either side is empty."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _lexical_similarity(left: str, right: str) -> float:
    """Jaccard over RAW tokens of two strings (legacy coherence form)."""
    return jaccard(_tokenize(left), _tokenize(right))


def _cosine(left: list[float], right: list[float]) -> float:
    """Return cosine similarity clamped to ``[0, 1]``.

    Matches the shipped coherence transform exactly: an orthogonal pair scores
    0, a negative cosine clamps to 0 (NOT the  `(1+cos)/2` rescale).
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_left * norm_right)))


def _normalize_identifier(value: Any) -> str | None:
    """Normalize a DOI/OpenAlex-ID/other identifier for set-equality matching.

    Lowercases and strips the common ``https://doi.org/`` and
    ``https://openalex.org/`` prefixes so identities and reference lists compare
    consistently within a namespace.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://openalex.org/",
        "http://openalex.org/",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text or None


def references_of(item: Any) -> set[str]:
    """Normalized reference identifiers from ``item.metadata['references']``."""
    metadata = getattr(item, "metadata", None) or {}
    return {
        normalized
        for ref in (metadata.get("references") or [])
        if (normalized := _normalize_identifier(ref))
    }


def identifiers_of(item: Any) -> set[str]:
    """All known identifiers for an item, normalized to match references.

    Identity spans ALL ids (external_ids, openalex_id, doi, source_id), not just
    the primary key, so cross-namespace co-citation can land (coherence.py:287).
    """
    metadata = getattr(item, "metadata", None) or {}
    if metadata.get("cross_source_dedupe") is False:
        # Some sources expose model-reported identity only for audit. Do not let
        # it create citation/coherence edges through URL or external-id overlap.
        return set()
    raw = [
        *(getattr(item, "external_ids", None) or {}).values(),
        metadata.get("openalex_id"),
        metadata.get("doi"),
        getattr(item, "source_id", None),
    ]
    return {
        normalized for value in raw if (normalized := _normalize_identifier(value))
    }


def overlap(
    refs_a: set[str],
    ids_a: set[str],
    refs_b: set[str],
    ids_b: set[str],
) -> float:
    """Return citation overlap as ``max(coupling, co_citation)``.

    - coupling = Jaccard of the two reference sets (shared bibliography).
    - co_citation = 1.0 iff either paper references one of the other's ids.
    """
    union = refs_a | refs_b
    coupling = len(refs_a & refs_b) / len(union) if union else 0.0
    co_citation = 1.0 if (refs_a & ids_b) or (refs_b & ids_a) else 0.0
    return max(coupling, co_citation)
