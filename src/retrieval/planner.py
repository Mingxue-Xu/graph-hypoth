"""Pre-retrieval query planner for the Research Synthesist pipeline.

The single claim-query retrieval misses a fast-moving field's lexically-novel
terminology. This module turns a ``ResearchProfile`` into several targeted sub-queries
(each with its own source set + recency/venue filters) that the driver fans out and
merges into one ledger.

Two tiers, sharing the ``RetrievalPlan`` contract:

- ``plan_t1`` — a pure, deterministic, field-agnostic heuristic (claim + claim+topic
  variants). The test fallback; invents no recency window.
- ``LLMRetrievalPlanner`` — a real LLM seam over any CAMEL-shaped ``model_backend``
  (``.run(messages)``), mirroring the shipped ``LLMExtractor`` STRICT-JSON pattern. It reads
  the profile and emits STRICT JSON sub-queries, choosing how many and what recency window suit.
  Any malformed/empty response degrades to ``plan_t1`` so a run never loses retrieval.

Generality: nothing here encodes a field, venue, or year — all field context flows in
through the profile, and the recency window is the LLM's decision, never a constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from src.research_profile import ResearchProfile
from src.retrieval.models import SearchPaperFilters

# Operational fan-out bound (like N_max): caps sub-queries so fan-out stays within
# arxiv's ~1-req/3s limit and Exa pricing. Not a scoring coefficient; overridable.
N_CAP = 8


@dataclass(frozen=True)
class RetrievalPlan:
    """One targeted retrieval request: a query, an optional source subset (None = all
    configured), and per-query filters (recency / venue bias)."""

    query: str
    sources: list[str] | None = None
    filters: SearchPaperFilters = field(default_factory=SearchPaperFilters)


def plan_t1(profile: ResearchProfile, *, n_cap: int = N_CAP) -> list[RetrievalPlan]:
    """Deterministic, field-agnostic fallback: the raw anchor plus an anchor+topic variant
    for each authored concept (highest weight first), capped at ``n_cap``. No
    recency window because the fallback must not invent one. The anchor is the claim or, in
    claimless discovery, the lens, so a no-claim profile still produces queries."""
    anchor = profile.anchor()
    plans = [RetrievalPlan(query=anchor)]
    topics = [concept.term for concept in sorted(profile.concepts, key=lambda item: -item.weight)]
    for topic in topics:
        if len(plans) >= n_cap:
            break
        plans.append(RetrievalPlan(query=f"{anchor} {topic}".strip()))
    return plans


def rrf_scores(rankings: list[list[str]], *, k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion (RAG-Fusion): ``score(id) = Σ 1/(k + rank_i)`` over each
    sub-query's ranked id list. Rewards papers surfaced highly by several sub-queries."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


_PLANNER_SYSTEM_PROMPT = """\
You are a retrieval query planner for a scientific-literature pipeline. Given a \
researcher's claim and interests, produce a small set of targeted search sub-queries \
that together maximize recall of on-topic, recent papers — reformulate terminology, \
cover the distinct sub-topics, and DO NOT just repeat the claim verbatim.

For EACH sub-query also decide, from the pace of the field implied by the claim and \
expertise, an optional recency window in HOURS (`max_age_hours`); omit it to search all \
time. Optionally bias toward venues/terms via `include_text` (short phrases, <=5 words \
each). Leave `sources` null to query every configured source.

Respond with STRICT JSON only, no prose:
{"queries": [
  {"query": "<reformulated sub-query>", "sources": null,
   "max_age_hours": <int or omit>, "include_text": ["<short phrase>"]}
]}"""


def _safe_filters(
    *,
    max_age_hours: Any = None,
    include_text: Any = None,
    include_domains: Any = None,
) -> SearchPaperFilters:
    """Build SearchPaperFilters from LLM-proposed values, defensively. A value that
    fails validation (e.g. an include_text phrase >5 words) must not crash the plan:
    fall back to recency-only, then to empty filters."""
    kwargs: dict[str, Any] = {}
    if isinstance(max_age_hours, int) and not isinstance(max_age_hours, bool):
        kwargs["max_age_hours"] = max_age_hours
    if isinstance(include_text, list) and include_text and all(
        isinstance(t, str) and t.strip() for t in include_text
    ):
        kwargs["include_text"] = include_text
    if isinstance(include_domains, list) and include_domains and all(
        isinstance(d, str) and d.strip() for d in include_domains
    ):
        kwargs["include_domains"] = include_domains
    try:
        return SearchPaperFilters(**kwargs)
    except Exception:
        try:
            return SearchPaperFilters(max_age_hours=kwargs.get("max_age_hours"))
        except Exception:
            return SearchPaperFilters()


class LLMRetrievalPlanner:
    """A real retrieval planner over any CAMEL model backend (``.run(messages)``).

    Mirrors the shipped ``LLMExtractor`` seam: one LLM call returns STRICT JSON sub-queries; the
    parser enforces ``n_cap`` and drops malformed entries. An empty/malformed response degrades
    to ``plan_t1`` so a run never loses retrieval. ``system_prompt`` overrides the
    scaffold entirely."""

    def __init__(
        self,
        model_backend: Any,
        *,
        system_prompt: str | None = None,
        n_cap: int = N_CAP,
    ) -> None:
        self._backend = model_backend
        self._system_prompt = system_prompt or _PLANNER_SYSTEM_PROMPT
        self._n_cap = n_cap

    def plan(self, profile: ResearchProfile) -> list[RetrievalPlan]:
        from src.camel_adapter import backend_json

        try:
            data = cast(
                dict[str, Any],
                backend_json(self._backend, self._system_prompt, self._build_user_prompt(profile)),
            )
            plans = self._parse(data)
        except Exception:
            plans = []
        return plans or plan_t1(profile, n_cap=self._n_cap)

    def _build_user_prompt(self, profile: ResearchProfile) -> str:
        priorities = ", ".join(
            f"{concept.term} ({concept.weight})" for concept in profile.concepts
        )
        return "\n".join(
            line
            for line in (
                (f"Research question (open, not an established claim): {profile.claim}"
                 if profile.seed_kind == "research_question" else f"Claim: {profile.claim}")
                if profile.claim
                else f"Research focus (claimless discovery): {profile.anchor()}",
                f"Expertise: {profile.expertise}" if profile.expertise else "",
                f"Research field: {profile.field}" if profile.field else "",
                f"Priority topics: {priorities}" if priorities else "",
                f"Priority focus: {profile.priority_focus}" if profile.priority_focus else "",
                self._venue_line(profile),
            )
            if line
        )

    @staticmethod
    def _venue_line(profile: ResearchProfile) -> str:
        venue = getattr(profile, "venue_preference", "")
        if isinstance(venue, (list, tuple)):
            venue = ", ".join(str(v) for v in venue)
        venue = str(venue or "").strip()
        return f"Preferred venues (bias toward, do not exclude): {venue}" if venue else ""

    def _parse(self, obj: dict[str, Any]) -> list[RetrievalPlan]:
        raw_queries = obj.get("queries")
        if not isinstance(raw_queries, list):
            return []
        plans: list[RetrievalPlan] = []
        for item in raw_queries:
            if not isinstance(item, dict):
                continue
            query = item.get("query")
            if not isinstance(query, str) or not query.strip():
                continue
            sources = item.get("sources")
            if sources is not None and not (
                isinstance(sources, list) and all(isinstance(s, str) for s in sources)
            ):
                sources = None
            filters = _safe_filters(
                max_age_hours=item.get("max_age_hours"),
                include_text=item.get("include_text"),
                include_domains=item.get("include_domains"),
            )
            plans.append(RetrievalPlan(query=query.strip(), sources=sources, filters=filters))
            if len(plans) >= self._n_cap:
                break
        return plans
