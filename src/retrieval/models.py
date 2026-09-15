from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class SearchPaperFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    urls: list[str] | None = None
    target_urls: list[str] | None = None
    anchor_queries: list[str] | None = None
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    include_text: list[str] | None = None
    exclude_text: list[str] | None = None
    text: bool | None = None
    highlights: bool | None = None
    highlight_query: str | None = None
    highlight_max_characters: int | None = Field(default=None, ge=1)
    text_max_characters: int | None = Field(default=None, ge=1)
    max_age_hours: int | None = Field(default=None, ge=-1)
    livecrawl_timeout: int | None = Field(default=None, ge=1)
    include_html_tags: bool | None = None
    target_scope: str | None = None
    verify_quotes: bool = True

    @model_validator(mode="after")
    def validate_filters(self) -> "SearchPaperFilters":
        for field_name in ("include_text", "exclude_text"):
            values = getattr(self, field_name)
            if values is None:
                continue
            if len(values) > 1:
                raise ValueError(f"{field_name} allows at most one string")
            if values and len(values[0].split()) > 5:
                raise ValueError(f"{field_name} allows at most five words")
        if self.urls is not None and self.target_urls is not None:
            if self.urls != self.target_urls:
                raise ValueError("urls and target_urls must match when both are set")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def request_kind(self) -> Literal["search", "contents"]:
        return "contents" if self.urls or self.target_urls else "search"

    @field_validator("urls", "target_urls")
    @classmethod
    def validate_urls(
        cls,
        values: list[str] | None,
        info: Any,
    ) -> list[str] | None:
        if values is None:
            return None
        if not values:
            raise ValueError(f"{info.field_name} must contain at least one URL")
        normalized: list[str] = []
        for value in values:
            stripped = value.strip()
            if not stripped.startswith(("http://", "https://")):
                raise ValueError(f"{info.field_name} must be http or https URLs")
            normalized.append(stripped)
        return normalized

    @field_validator("anchor_queries")
    @classmethod
    def validate_anchor_queries(
        cls,
        values: list[str] | None,
    ) -> list[str] | None:
        if values is None:
            return None
        if len(values) > 8:
            raise ValueError("anchor_queries allows at most eight entries")
        normalized = [" ".join(value.split()) for value in values]
        if any(not value for value in normalized):
            raise ValueError("anchor_queries entries must be non-empty")
        return normalized

    @field_validator("include_domains", "exclude_domains")
    @classmethod
    def validate_domains(
        cls,
        values: list[str] | None,
        info: Any,
    ) -> list[str] | None:
        if values is None:
            return None
        normalized = [value.strip().lower() for value in values]
        if any(not value for value in normalized):
            raise ValueError(f"{info.field_name} entries must be non-empty")
        return normalized

    def ignored_for_source(self, source: str) -> list[str]:
        if source == "exa":
            return []
        if source == "codex_web":
            supported = {
                "urls",
                "target_urls",
                "anchor_queries",
                "include_domains",
                "exclude_domains",
                "include_text",
                "exclude_text",
                "highlight_query",
                "highlight_max_characters",
                "max_age_hours",
                "target_scope",
            }
            ignored: list[str] = []
            for field_name in (
                "urls",
                "target_urls",
                "anchor_queries",
                "include_domains",
                "exclude_domains",
                "include_text",
                "exclude_text",
                "text",
                "highlights",
                "highlight_query",
                "highlight_max_characters",
                "text_max_characters",
                "max_age_hours",
                "livecrawl_timeout",
                "include_html_tags",
                "target_scope",
                "verify_quotes",
            ):
                if field_name in supported:
                    continue
                if field_name == "verify_quotes":
                    if self.verify_quotes is not True:
                        ignored.append(field_name)
                    continue
                if getattr(self, field_name) is not None:
                    ignored.append(field_name)
            return ignored
        # arxiv consumes max_age_hours via a post-fetch published_date cutoff; the
        # metadata-only sources (crossref/openalex/fake/...) still ignore it.
        recency_aware = source == "arxiv"
        ignored: list[str] = []
        for field_name in (
            "urls",
            "target_urls",
            "anchor_queries",
            "include_domains",
            "exclude_domains",
            "include_text",
            "exclude_text",
            "text",
            "highlights",
            "highlight_query",
            "highlight_max_characters",
            "text_max_characters",
            "max_age_hours",
            "livecrawl_timeout",
            "include_html_tags",
            "target_scope",
            "verify_quotes",
        ):
            if field_name == "max_age_hours" and recency_aware:
                continue
            if field_name == "verify_quotes":
                if getattr(self, field_name) is not True:
                    ignored.append(field_name)
                continue
            if getattr(self, field_name) is not None:
                ignored.append(field_name)
        return ignored


class SourceResult(BaseModel):
    source: str
    source_id: str | None = None
    title: str
    authors: list[str] = Field(default_factory=list)
    published_date: str | None = None
    url: str | None = None
    text: str | None = None
    summary: str | None = None
    score: float | None = None
    # Normalized cross-source identity map (e.g. {"doi","pmid","pmcid","arxiv"}).
    # Typed identity, distinct from the free-form `metadata` enrichment bag.
    external_ids: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceStatus(BaseModel):
    source: str
    status: Literal["success", "partial_failure", "failed", "skipped"]
    source_query: str
    result_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    unused_filters: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalToolResult(BaseModel):
    tool_call_id: str
    query: str
    filters: SearchPaperFilters = Field(default_factory=SearchPaperFilters)
    sources: list[str]
    evidence: list[dict[str, Any]]
    source_statuses: list[SourceStatus]
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    elapsed_ms: int
    input_hash: str
    retrieval_config_hash: str
