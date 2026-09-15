from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievedEvidence(BaseModel):
    evidence_id: str
    source: str
    source_id: str | None = None
    title: str
    authors: list[str] = Field(default_factory=list)
    published_date: str | None = None
    url: str | None = None
    quote: str
    relevance: str
    retrieval_method: Literal["search_papers"] = "search_papers"
    retrieved_by: str
    tool_call_id: str
    score: float | None = None
    rank: int
    trust_tier: str
    redacted: bool = False
    # R2 coherence enrichments. They stay None when enrichment is off/empty.
    relatedness_score: float | None = None
    relation_label: str | None = None
    domain_context: str | None = None
    synonym_links: list[str] | None = None
    citation_neighbors: list[str] | None = None
    # Normalized cross-source identity map (doi/pmid/pmcid/arxiv) used for dedup;
    # survives the _source_shadow re-admit round-trip.
    external_ids: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
