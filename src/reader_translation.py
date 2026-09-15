"""Rewrite reader cards into a specific reader's vocabulary.

Driver-side, after ``elaborate_all`` and before ``write_trace`` (renderers are contractually
offline — they only read what this seam persisted into plain_language.json). Two LLM calls per
card: (1) a GENERATIVE translator (role ``reader_translator``) proposes the rewritten card plus
explicit ``{original -> familiar}`` pairs; (2) a SECOND, critical-tier verifier (role
``translation_verifier``) grades every pair against the original term's definition and a verbatim
source quote — ``confirmed | approximate | rejected``, reject-when-in-doubt (a wrong analogy is
worse than jargon). Any rejection triggers ONE retry excluding the failed pairs; still rejected
means the IDENTITY card ships (the attempt stays auditable in the ``translations`` record, but no
unfaithful analogy ever reaches the reader). Deterministic post-guards (no LLM): headline length,
original-term first-use presence, canonical ``key_terms[].term`` restoration, empty-card
passthrough with zero backend calls.

The per-card ``translations`` record schema is FROZEN (Evidence Reviewer's terminology auditor may later
re-audit exactly this record): ``[{original, familiar, first_use_rendering, faithfulness,
rationale}]``. ``key_terms[].term`` always stays the ORIGINAL term (the canonical, link-resolution
key); familiar phrasing lives in the prose and in the per-term ``familiar_gloss``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from src import graph_config_defaults as gcd
from src.progress import report_progress

# Per-quote prompt cap; retrieved quotes can be very large.
_QUOTE_MAX_CHARS = gcd.PASSAGE_PROMPT_MAX_CHARS
_HEADLINE_MAX_WORDS = 20
_RECORD_KEYS = ("original", "familiar", "first_use_rendering", "faithfulness", "rationale")

_TRANSLATOR_SYSTEM_PROMPT = (
    "You translate a plain-language research-hypothesis reader card into the vocabulary of ONE "
    "specific reader, whose profile (home field, familiar terms, analogy domains, papers they "
    "know) is provided. Leave alone every term the reader plausibly already knows. Replace a "
    "term ONLY when a FAITHFUL equivalent exists in the reader's vocabulary; render the first "
    'use as "<familiar phrasing> (<ORIGINAL TERM>)" so the original term appears literally. '
    "FAITHFULNESS IS THE BAR: when no faithful mapping exists, keep the original term and make "
    "its key_terms definition self-contained instead. key_terms rules: keep every \"term\" "
    "field EXACTLY as the original term (never rename it); rewrite plain_meaning to be "
    "SELF-CONTAINED (no undefined jargon inside a definition — unpack any needed technical term "
    "inline in plain words); add a \"familiar_gloss\" phrased in the reader's vocabulary only "
    "when it is faithful. Do NOT change factual content, hedging, or claims. Ground every "
    "mapping in the provided concept definitions and verbatim source quotes — never in outside "
    "knowledge. Return STRICT JSON only (no markdown): the SAME card keys, plus "
    '"translations": [{"original": "...", "familiar": "...", '
    '"first_use_rendering": "<familiar phrasing> (<ORIGINAL TERM>)", '
    '"rationale": "<why this mapping is faithful, grounded in the definition/quote>"}].'
)

_VERIFIER_SYSTEM_PROMPT = (
    "You are a skeptical faithfulness verifier for term translations in a research reader card. "
    "For each pair you receive the original term, the proposed familiar phrasing, the "
    "translator's rationale, the original term's definition, and a verbatim source quote. Judge "
    "ONLY from that provided material. Verdicts: \"confirmed\" = the familiar phrasing denotes "
    "the SAME object/method/property (a practitioner of either field would accept the "
    "identification); \"approximate\" = the same core idea but it loses a stated qualification "
    "(name the lost qualification in the note); \"rejected\" = a different object, the wrong "
    "direction, or the mapping is unverifiable from the provided material. WHEN IN DOUBT, "
    "REJECT — an unfaithful analogy is worse than jargon. Return STRICT JSON only (no "
    'markdown): {"verdicts": [{"original": "...", "verdict": '
    '"confirmed|approximate|rejected", "note": "..."}]}.'
)


def source_quote_index(evidence: Any) -> dict[str, str]:
    """``{lowercased title: quote}`` over the retrieved-evidence pool (items may be
    ``RetrievedEvidence`` objects or mappings). First occurrence of a title wins (the pool is
    rank-ordered). The index keys are matched to key-term ``source`` strings by the same
    title-containment rule the trace renderer uses for citations."""
    index: dict[str, str] = {}
    for item in evidence or ():
        if isinstance(item, Mapping):
            title, quote = item.get("title", ""), item.get("quote", "")
        else:
            title = getattr(item, "title", "")
            quote = getattr(item, "quote", "")
        key = str(title or "").strip().lower()
        if key and str(quote or "").strip() and key not in index:
            index[key] = str(quote)
    return index


def _quote_for_source(source: Any, quote_index: Mapping[str, str] | None) -> str:
    """The verbatim quote whose title containment-matches a key-term ``source`` string —
    the local twin of the trace renderer's ``_cite_html`` matching rule (``title in source
    or source in title``, both lowercased)."""
    text = str(source or "").strip().lower()
    if not text or not quote_index:
        return ""
    for title, quote in quote_index.items():
        key = str(title or "").strip().lower()
        if key and (key in text or text in key):
            return str(quote)[:_QUOTE_MAX_CHARS]
    return ""


def _key_term_entry(card: Mapping[str, Any], term: str) -> Mapping[str, Any] | None:
    wanted = term.casefold()
    for entry in card.get("key_terms") or []:
        if isinstance(entry, Mapping) and str(entry.get("term", "")).casefold() == wanted:
            return entry
    return None


def _definition_for_term(
    term: str, concept_definitions: Mapping[str, str] | None, card: Mapping[str, Any]
) -> str:
    wanted = term.casefold()
    for label, definition in (concept_definitions or {}).items():
        if str(label).casefold() == wanted and str(definition or "").strip():
            return str(definition)
    entry = _key_term_entry(card, term)
    if entry is not None:
        return str(entry.get("plain_meaning", "") or "")
    return ""


def _quote_for_term(
    term: str, card: Mapping[str, Any], quote_index: Mapping[str, str] | None
) -> str:
    entry = _key_term_entry(card, term)
    if entry is None:
        return ""
    return _quote_for_source(entry.get("source", ""), quote_index)


def _lexicon_block(lexicon: Any) -> str:
    lines: list[str] = []
    home = str(getattr(lexicon, "home_field", "") or "")
    if home:
        lines.append(f"Home field: {home}")
    terms = getattr(lexicon, "familiar_terms", ()) or ()
    rendered_terms = []
    for term in terms:
        label = str(getattr(term, "term", "") or "")
        gloss = str(getattr(term, "gloss", "") or "")
        if label:
            rendered_terms.append(f"{label} ({gloss})" if gloss else label)
    if rendered_terms:
        lines.append("Familiar terms: " + "; ".join(rendered_terms))
    domains = [str(d) for d in (getattr(lexicon, "analogy_domains", ()) or ()) if str(d)]
    if domains:
        lines.append("Analogy domains: " + "; ".join(domains))
    titles = [
        str(getattr(paper, "title", "") or "")
        for paper in (getattr(lexicon, "papers", ()) or ())
    ]
    titles = [t for t in titles if t]
    if titles:
        lines.append("Papers the reader knows: " + "; ".join(titles))
    return "\n".join(lines)


def _quotes_block(card: Mapping[str, Any], quote_index: Mapping[str, str] | None) -> str:
    lines: list[str] = []
    for entry in card.get("key_terms") or []:
        if not isinstance(entry, Mapping):
            continue
        term = str(entry.get("term", "") or "")
        quote = _quote_for_source(entry.get("source", ""), quote_index)
        if term and quote:
            lines.append(f'- {term} (from: {entry.get("source", "")}): "{quote}"')
    return "\n".join(lines)


def _translator_user_prompt(
    card: Mapping[str, Any],
    lexicon: Any,
    concept_definitions: Mapping[str, str] | None,
    quote_index: Mapping[str, str] | None,
    exclude: Sequence[str],
) -> str:
    parts = [f"Reader profile:\n{_lexicon_block(lexicon)}"]
    parts.append(
        "Reader card (JSON):\n"
        + json.dumps(dict(card), ensure_ascii=False, default=str)
    )
    if concept_definitions:
        definition_lines = "\n".join(
            f"- {label}: {definition}"
            for label, definition in concept_definitions.items()
            if str(definition or "").strip()
        )
        if definition_lines:
            parts.append(f"Concept definitions:\n{definition_lines}")
    quotes = _quotes_block(card, quote_index)
    if quotes:
        parts.append(f"Verbatim source quotes:\n{quotes}")
    if exclude:
        parts.append(
            "Do NOT propose translations for these terms — they failed the faithfulness "
            "check; keep each in its original form: " + ", ".join(exclude) + "."
        )
    parts.append(
        "Rewrite the card for this reader. Return STRICT JSON with the same card keys "
        'plus "translations".'
    )
    return "\n\n".join(parts)


def _verifier_user_prompt(
    pairs: Sequence[Mapping[str, str]],
    card: Mapping[str, Any],
    concept_definitions: Mapping[str, str] | None,
    quote_index: Mapping[str, str] | None,
) -> str:
    blocks: list[str] = []
    for i, pair in enumerate(pairs, 1):
        original = pair["original"]
        definition = _definition_for_term(original, concept_definitions, card)
        quote = _quote_for_term(original, card, quote_index)
        blocks.append(
            f"### Pair {i}\n"
            f"original: {original}\n"
            f"familiar: {pair['familiar']}\n"
            f"translator rationale: {pair['rationale']}\n"
            f"definition of the original term: {definition or '(none provided)'}\n"
            f'verbatim source quote: {(chr(34) + quote + chr(34)) if quote else "(none provided)"}'
        )
    blocks.append("Judge every pair. Return STRICT JSON.")
    return "\n\n".join(blocks)


def _run_json(backend: Any, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
    """One completion -> parsed JSON object, or None on empty/malformed output (the shipped
    LLM-seam degrade convention; parsing reuses ``camel_adapter``)."""
    from src.camel_adapter import backend_json

    return backend_json(backend, system_prompt, user_prompt) or None


def _records(
    pairs: Sequence[Mapping[str, str]], faithfulness: Sequence[str]
) -> list[dict[str, str]]:
    return [
        {
            "original": pair["original"],
            "familiar": pair["familiar"],
            "first_use_rendering": pair["first_use_rendering"],
            "faithfulness": verdict,
            "rationale": pair["rationale"],
        }
        for pair, verdict in zip(pairs, faithfulness)
    ]


def _identity_with_records(
    card: Mapping[str, Any], records: list[dict[str, str]]
) -> dict[str, Any]:
    out = dict(card)
    out["translations"] = records
    return out


def _restore_canonical_terms(key_terms: Any, records: Sequence[Mapping[str, str]]) -> list:
    """Deterministic guard: a key_terms entry named by a pair's FAMILIAR phrasing is renamed
    back to the pair's ORIGINAL term (the canonical, searchable, link-resolution key)."""
    by_familiar = {r["familiar"].casefold(): r["original"] for r in records}
    out = []
    for entry in key_terms or []:
        if not isinstance(entry, Mapping):
            out.append(entry)
            continue
        entry = dict(entry)
        original = by_familiar.get(str(entry.get("term", "")).casefold())
        if original:
            entry["term"] = original
        out.append(entry)
    return out


class LLMReaderTranslator:
    """The reader-translation seam over two CAMEL-style backends (``.run(messages)``): the
    generative translator plus the MANDATORY second-call verifier (self-checking one's own
    analogy is the exact failure mode this design rejects). Malformed output at either seam
    degrades to the identity card — never crashes, never ships unverified phrasing."""

    def __init__(
        self,
        model_backend: Any,
        *,
        verifier_backend: Any,
        system_prompt: str | None = None,
        verifier_system_prompt: str | None = None,
    ) -> None:
        self._backend = model_backend
        self._verifier_backend = verifier_backend
        self._system_prompt = system_prompt or _TRANSLATOR_SYSTEM_PROMPT
        self._verifier_system_prompt = verifier_system_prompt or _VERIFIER_SYSTEM_PROMPT

    def _attempt(
        self,
        card: Mapping[str, Any],
        lexicon: Any,
        concept_definitions: Mapping[str, str] | None,
        quote_index: Mapping[str, str] | None,
        exclude: Sequence[str],
    ) -> tuple[dict[str, Any], list[dict[str, str]]] | None:
        data = _run_json(
            self._backend,
            self._system_prompt,
            _translator_user_prompt(card, lexicon, concept_definitions, quote_index, exclude),
        )
        if data is None:
            return None
        pairs: list[dict[str, str]] = []
        raw = data.get("translations")
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, Mapping):
                    continue
                original = str(item.get("original", "") or "").strip()
                familiar = str(item.get("familiar", "") or "").strip()
                if not original or not familiar:
                    continue
                pairs.append(
                    {
                        "original": original,
                        "familiar": familiar,
                        "first_use_rendering": str(item.get("first_use_rendering", "") or ""),
                        "rationale": str(item.get("rationale", "") or ""),
                    }
                )
        translated = {k: v for k, v in data.items() if k != "translations"}
        return translated, pairs

    def _verify(
        self,
        pairs: Sequence[Mapping[str, str]],
        card: Mapping[str, Any],
        concept_definitions: Mapping[str, str] | None,
        quote_index: Mapping[str, str] | None,
    ) -> list[str] | None:
        """One batched completion grading ALL pairs. Returns per-pair faithfulness aligned
        with ``pairs`` (a pair the verifier skipped or graded outside the enum is REJECTED —
        when in doubt, reject), or None when the verifier output is malformed."""
        data = _run_json(
            self._verifier_backend,
            self._verifier_system_prompt,
            _verifier_user_prompt(pairs, card, concept_definitions, quote_index),
        )
        if data is None:
            return None
        raw = data.get("verdicts")
        if not isinstance(raw, list):
            return None
        by_original: dict[str, str] = {}
        for item in raw:
            if isinstance(item, Mapping):
                key = str(item.get("original", "") or "").casefold()
                by_original[key] = str(item.get("verdict", "") or "")
        out: list[str] = []
        for pair in pairs:
            verdict = by_original.get(pair["original"].casefold(), "")
            out.append(verdict if verdict in ("confirmed", "approximate") else "rejected")
        return out

    def _resolve(
        self,
        card: Mapping[str, Any],
        translated: Mapping[str, Any],
        records: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Deterministic post-guards (no LLM) + card assembly."""
        shipped: dict[str, Any] = dict(card)
        for key, value in translated.items():
            if key in ("translations", "pre_translation"):
                continue  # these are OURS to write, never the model's
            shipped[key] = value
        if shipped.get("key_terms") is not None:
            shipped["key_terms"] = _restore_canonical_terms(shipped.get("key_terms"), records)
        headline = str(shipped.get("headline", "") or "")
        if len(headline.split()) > _HEADLINE_MAX_WORDS:
            shipped["headline"] = card.get("headline")
        # The first-use guard scans the card's PROSE — top-level strings AND list-of-string
        # fields (next_steps): a verifier-approved translation whose original term lands
        # inside a list is legitimate, not a policy violation. key_terms are EXCLUDED — they
        # mechanically carry the canonical original term, which would make the guard vacuous.
        prose_parts: list[str] = []
        for key, value in shipped.items():
            if key in ("key_terms", "translations", "pre_translation"):
                continue
            if isinstance(value, str):
                prose_parts.append(value)
            elif isinstance(value, list):
                prose_parts.extend(item for item in value if isinstance(item, str))
        prose = " ".join(prose_parts).casefold()
        for record in records:
            if record["faithfulness"] == "rejected":
                continue
            if record["original"].casefold() not in prose:
                # the "(ORIGINAL)" first-use policy was violated -> identity, auditable
                return _identity_with_records(card, records)
        if any(shipped.get(key) != value for key, value in card.items()):
            shipped["pre_translation"] = dict(card)
        shipped["translations"] = records
        return shipped

    def translate(
        self,
        card: Mapping[str, Any],
        *,
        lexicon: Any,
        concept_definitions: Mapping[str, str] | None = None,
        source_quotes: Mapping[str, str] | None = None,
    ) -> dict[str, Any] | Mapping[str, Any]:
        if not isinstance(card, Mapping) or not str(card.get("headline", "") or "").strip():
            return card  # empty/heading-less card: untouched, ZERO backend calls
        first = self._attempt(card, lexicon, concept_definitions, source_quotes, exclude=())
        if first is None:
            return card  # malformed translator output -> identity
        translated1, pairs1 = first
        if not pairs1:
            return card  # rewritten prose without declared pairs is unauditable -> identity
        verdicts1 = self._verify(pairs1, card, concept_definitions, source_quotes)
        if verdicts1 is None:
            return card  # unverifiable -> reject-when-in-doubt -> identity
        rejected1 = [p for p, v in zip(pairs1, verdicts1) if v == "rejected"]
        if not rejected1:
            return self._resolve(card, translated1, _records(pairs1, verdicts1))
        # ONE retry, excluding the failed pairs; the failed attempt stays auditable.
        rejected_records = _records(rejected1, ["rejected"] * len(rejected1))
        second = self._attempt(
            card,
            lexicon,
            concept_definitions,
            source_quotes,
            exclude=tuple(p["original"] for p in rejected1),
        )
        if second is None:
            return _identity_with_records(card, rejected_records)
        translated2, pairs2 = second
        if not pairs2:
            return _identity_with_records(card, rejected_records)
        verdicts2 = self._verify(pairs2, card, concept_definitions, source_quotes)
        if verdicts2 is None:
            return _identity_with_records(card, rejected_records)
        records = rejected_records + _records(pairs2, verdicts2)
        if "rejected" in verdicts2:
            return _identity_with_records(card, records)  # still rejected -> identity
        return self._resolve(card, translated2, records)


def translate_all(
    translator: LLMReaderTranslator | None,
    elaborations: Mapping[str, Any],
    *,
    lexicon: Any,
    concept_definitions: Mapping[str, str] | None = None,
    source_quotes: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Translate every card (identity when the translator or the lexicon is absent — the
    no-lexicon world is bit-for-bit today's behavior)."""
    if translator is None or lexicon is None or not elaborations:
        return elaborations
    translated = {}
    for index, (cid, card) in enumerate(elaborations.items(), 1):
        report_progress(
            "Writing reports", "translating hypothesis",
            current=index, total=len(elaborations),
        )
        translated[cid] = translator.translate(
            card,
            lexicon=lexicon,
            concept_definitions=concept_definitions,
            source_quotes=source_quotes,
        )
    return translated
