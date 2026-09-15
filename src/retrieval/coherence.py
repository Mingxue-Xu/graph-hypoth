from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

from src.retrieval import scoring_defaults as sd
from src.retrieval._util import get_attr_or_key as _get
from src.retrieval.similarity import (
    Embedder,
    _cosine,
    _lexical_similarity,
    identifiers_of,
    overlap,
    references_of,
)
from src.state import RetrievedEvidence

logger = logging.getLogger(__name__)

OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"

# A judge maps the input claim + candidate evidence to per-evidence verdicts.
Judge = Callable[[str, list[RetrievedEvidence]], "dict[str, JudgeVerdict]"]


@lru_cache(maxsize=4)
def load_specter2_embedder(model_name: str, revision: str | None = None) -> Embedder | None:
    """Load a local SPECTER2 (sentence-transformers) embedder, cached per (model, revision).

    ``revision`` pins the exact HuggingFace snapshot: it is threaded
    through to ``SentenceTransformer`` so the embedding channel cannot drift to an
    unpinned model version. Returns None when the optional ``retrieval-coherence`` extra
    is absent so the enricher transparently falls back to lexical similarity.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - optional extra absent.
        logger.debug("sentence-transformers unavailable; lexical fallback: %s", exc)
        return None
    try:
        model = SentenceTransformer(model_name, revision=revision)
    except Exception as exc:  # pragma: no cover - model download/load failure.
        logger.warning("failed to load embedding model %s@%s: %s", model_name, revision, exc)
        return None

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = model.encode(texts, normalize_embeddings=False)
        return [[float(value) for value in vector] for vector in vectors]

    return embed


@dataclass
class JudgeVerdict:
    relation_label: str | None = None
    domain_context: str | None = None
    synonym_links: list[str] = field(default_factory=list)


class EvidenceEnricher:
    """Adds R2 coherence signals to retrieved evidence via three channels.

    1. embeddings (local SPECTER2, lexical fallback) -> claim<->candidate recall
    2. citation graph overlap (model-free) -> wording-independent relatedness
    3. LLM-as-judge (top-K only) -> relation label / domain context / synonyms

    Channels degrade independently: a missing embedder falls back to lexical, a
    disabled/absent judge simply leaves relation fields unset.
    """

    def __init__(
        self,
        config: Any,
        *,
        embedder: Embedder | None = None,
        judge: Judge | None = None,
    ) -> None:
        self.config = config
        self._judge = judge
        if embedder is not None:
            self._embedder: Embedder | None = embedder
        elif _get(config, "embedding_enabled", True):
            self._embedder = load_specter2_embedder(
                str(_get(config, "embedding_model", sd.EMBEDDING_MODEL)),
                revision=str(_get(config, "embedding_revision", sd.EMBEDDING_REVISION)),
            )
        else:
            self._embedder = None

    def enrich(
        self, claim: str, items: list[RetrievedEvidence]
    ) -> list[RetrievedEvidence]:
        if not items or not _get(self.config, "enabled", True):
            return items

        embed_scores = self._embedding_scores(claim, items)
        citation_scores, citation_neighbors = self._citation_scores(items)

        w_embed = float(_get(self.config, "embedding_weight", 0.6))
        w_cite = float(_get(self.config, "citation_weight", 0.4))
        total_weight = w_embed + w_cite
        for item in items:
            embed = embed_scores.get(item.evidence_id, 0.0)
            cite = citation_scores.get(item.evidence_id, 0.0)
            score = (w_embed * embed + w_cite * cite) / total_weight if total_weight else 0.0
            item.relatedness_score = round(score, 6)
            neighbors = citation_neighbors.get(item.evidence_id)
            if neighbors:
                item.citation_neighbors = list(neighbors)
            # Persist into metadata["r2"] too: it is the only carrier that
            # survives the ledger re-admit round-trip (_source_shadow copies
            # metadata but not top-level fields). See plan Risk #3.
            r2 = item.metadata.setdefault("r2", {})
            r2["embedding"] = round(embed, 6)
            r2["citation"] = round(cite, 6)
            r2["relatedness_score"] = item.relatedness_score
            if item.citation_neighbors:
                r2["citation_neighbors"] = list(item.citation_neighbors)

        self._apply_judge(claim, items)
        return items

    def _apply_judge(self, claim: str, items: list[RetrievedEvidence]) -> None:
        judge = self._resolve_judge()
        if judge is None:
            return
        top_k = int(_get(self.config, "judge_top_k", 5))
        ranked = sorted(
            items, key=lambda item: item.relatedness_score or 0.0, reverse=True
        )[:top_k]
        if not ranked:
            return
        try:
            verdicts = judge(claim, ranked)
        except Exception as exc:  # noqa: BLE001 - judge is best-effort.
            logger.warning("coherence judge failed: %s", exc)
            return
        for item in ranked:
            verdict = verdicts.get(item.evidence_id)
            if verdict is None:
                continue
            r2 = item.metadata.setdefault("r2", {})
            # Only assign when the verdict carries a value, so a partial/None
            # verdict never clobbers a previously set label.
            if verdict.relation_label:
                item.relation_label = verdict.relation_label
                r2["relation_label"] = verdict.relation_label
            if verdict.domain_context:
                item.domain_context = verdict.domain_context
                r2["domain_context"] = verdict.domain_context
            if verdict.synonym_links:
                item.synonym_links = list(verdict.synonym_links)
                r2["synonym_links"] = list(verdict.synonym_links)
            # Record the anchor (keep-first) for later sightings; relation labels
            # above are last-write within a single enrich pass.
            r2.setdefault("anchor", claim)

    def _embedding_scores(
        self, claim: str, items: list[RetrievedEvidence]
    ) -> dict[str, float]:
        if self._embedder is not None:
            try:
                texts = [claim] + [self._text_for(item) for item in items]
                vectors = self._embedder(texts)
                claim_vector, item_vectors = vectors[0], vectors[1:]
                return {
                    item.evidence_id: _cosine(claim_vector, vector)
                    for item, vector in zip(items, item_vectors)
                }
            except Exception as exc:  # noqa: BLE001 - fall back to lexical.
                logger.debug("embedder failed; lexical fallback: %s", exc)
        return {
            item.evidence_id: _lexical_similarity(claim, self._text_for(item))
            for item in items
        }

    def _text_for(self, item: RetrievedEvidence) -> str:
        return " ".join(part for part in (item.title, item.quote) if part)

    def _citation_scores(
        self, items: list[RetrievedEvidence]
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        if not _get(self.config, "citation_enabled", True):
            return {}, {}
        # References and identifiers share the SAME normalization so co-citation
        # works within a namespace: OpenAlex stores references as work-ID URLs
        # while Crossref stores them as DOIs. An item's identity therefore spans
        # ALL of its known ids (external_ids, openalex_id, doi, source_id), not
        # just the primary key (see similarity.identifiers_of/references_of/overlap,
        # the single source of truth). Cross-namespace coupling (an OpenAlex W-ID
        # ref vs a Crossref item known only by DOI) still cannot match without
        # external id resolution -- an inherent limit, out of scope here.
        references = {item.evidence_id: references_of(item) for item in items}
        identifiers = {item.evidence_id: identifiers_of(item) for item in items}
        # Identify neighbors by a STABLE primary key (doi/source_id), not
        # evidence_id: evidence_ids are renumbered on ledger re-admission, which
        # would leave the stored neighbor ids dangling.
        identity = {item.evidence_id: self._identity(item) for item in items}
        scores: dict[str, float] = {}
        neighbors: dict[str, list[str]] = {}
        for item in items:
            own_refs = references[item.evidence_id]
            own_ids = identifiers[item.evidence_id]
            ranked_neighbors: list[tuple[float, str]] = []
            best = 0.0
            for other in items:
                if other.evidence_id == item.evidence_id:
                    continue
                pair_overlap = overlap(
                    own_refs,
                    own_ids,
                    references[other.evidence_id],
                    identifiers[other.evidence_id],
                )
                if pair_overlap > 0:
                    ranked_neighbors.append((pair_overlap, identity[other.evidence_id]))
                best = max(best, pair_overlap)
            scores[item.evidence_id] = best
            if ranked_neighbors:
                ranked_neighbors.sort(reverse=True)
                neighbors[item.evidence_id] = [key for _, key in ranked_neighbors[:5]]
        return scores, neighbors

    @staticmethod
    def _identity(item: RetrievedEvidence) -> str:
        """Stable primary key used to label citation neighbors."""
        return str(item.metadata.get("doi") or item.source_id or item.evidence_id)

    def _resolve_judge(self) -> Judge | None:
        if self._judge is not None:
            return self._judge
        if not _get(self.config, "judge_enabled", False):
            return None
        model_config = _get(self.config, "judge_model")
        if model_config is None:
            return None
        return _build_llm_judge(model_config)


def _build_llm_judge(model_config: Any) -> Judge | None:
    try:
        import os

        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - openai is a core dep.
        logger.debug("openai unavailable for coherence judge: %s", exc)
        return None

    api_key_env = _get(model_config, "api_key_env", "OPENROUTER_API_KEY")
    api_key = os.environ.get(str(api_key_env)) if api_key_env else None
    if not api_key:
        return None
    model_id = _get(model_config, "model_id")
    if not model_id:
        return None
    base_url = _get(model_config, "base_url", OPENROUTER_API_BASE_URL)
    client = OpenAI(api_key=api_key, base_url=base_url)

    def judge(claim: str, items: list[RetrievedEvidence]) -> dict[str, JudgeVerdict]:
        prompt = _judge_prompt(claim, items)
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            logger.warning("coherence judge returned no choices")
            return {}
        try:
            payload = json.loads(choices[0].message.content or "{}")
        except (ValueError, TypeError) as exc:
            logger.warning("coherence judge returned non-JSON content: %s", exc)
            return {}
        verdicts = _parse_judge_payload(payload)
        if not verdicts:
            # Valid JSON whose shape doesn't match the verdict schema (e.g. a list
            # or a dict without a "verdicts" array) parses fine but yields nothing.
            # Surface it so a systematic schema mismatch isn't silently dropped.
            logger.warning("coherence judge payload had no usable verdicts")
        return verdicts

    return judge


_JUDGE_SYSTEM = (
    "You assess how each retrieved excerpt relates to a scientific claim. For "
    "each evidence id, return relation (one of supports, contradicts, qualifies, "
    "background, unrelated), a short domain_context note, and any synonym_links "
    "(terms equivalent across contexts). Reply as JSON: {\"verdicts\": [{...}]}."
)


def _judge_prompt(claim: str, items: list[RetrievedEvidence]) -> str:
    lines = [f"Claim: {claim}", "", "Evidence:"]
    for item in items:
        lines.append(f"- id={item.evidence_id} title={item.title!r} quote={item.quote!r}")
    return "\n".join(lines)


def _parse_judge_payload(payload: Any) -> dict[str, JudgeVerdict]:
    verdicts: dict[str, JudgeVerdict] = {}
    rows = payload.get("verdicts") if isinstance(payload, dict) else None
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        evidence_id = row.get("evidence_id") or row.get("id")
        if not evidence_id:
            continue
        synonyms = row.get("synonym_links") or []
        verdicts[str(evidence_id)] = JudgeVerdict(
            relation_label=row.get("relation") or row.get("relation_label"),
            domain_context=row.get("domain_context"),
            synonym_links=[str(term) for term in synonyms if term],
        )
    return verdicts
