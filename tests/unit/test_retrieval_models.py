import json

import pytest
from pydantic import ValidationError

from src.retrieval.models import (
    RetrievalToolResult,
    SearchPaperFilters,
    SourceStatus,
)


def test_search_paper_filters_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        SearchPaperFilters(unknown_filter=True)


@pytest.mark.parametrize("field_name", ["include_text", "exclude_text"])
def test_search_paper_filters_allow_at_most_one_string(field_name: str) -> None:
    with pytest.raises(ValidationError, match=f"{field_name} allows at most one string"):
        SearchPaperFilters(**{field_name: ["transformer", "attention"]})


@pytest.mark.parametrize("field_name", ["include_text", "exclude_text"])
def test_search_paper_filters_allow_at_most_five_words(field_name: str) -> None:
    with pytest.raises(ValidationError, match=f"{field_name} allows at most five words"):
        SearchPaperFilters(
            **{field_name: ["one two three four five six"]},
        )


def test_search_paper_filters_report_ignored_fields_for_non_exa_sources() -> None:
    filters = SearchPaperFilters(
        urls=["https://example.test/paper"],
        target_urls=["https://example.test/paper"],
        anchor_queries=["claim A", "claim B"],
        include_domains=["arxiv.org"],
        exclude_domains=["example.com"],
        include_text=["transformer"],
        exclude_text=["survey"],
        text=True,
        highlights=True,
        highlight_query="attention compression",
        highlight_max_characters=600,
        text_max_characters=5000,
        max_age_hours=0,
        livecrawl_timeout=3000,
        include_html_tags=True,
        target_scope="known_url",
        verify_quotes=False,
    )

    assert filters.request_kind == "contents"
    assert filters.ignored_for_source("exa") == []
    # arxiv now CONSUMES max_age_hours via a post-fetch published_date cutoff, so it is
    # no longer reported as ignored for arxiv (still ignored by the metadata-only sources).
    assert filters.ignored_for_source("arxiv") == [
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
        "livecrawl_timeout",
        "include_html_tags",
        "target_scope",
        "verify_quotes",
    ]
    assert filters.ignored_for_source("fake") == [
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
    ]


def test_search_paper_filters_accept_recipe_shaped_known_url_request() -> None:
    filters = SearchPaperFilters(
        target_urls=["https://arxiv.org/html/2309.10668v2"],
        anchor_queries=[
            "impressive compression rates",
            "models trained on text generalize to image audio compression",
        ],
        text=True,
        highlights=True,
        max_age_hours=0,
        include_html_tags=True,
        verify_quotes=True,
    )

    assert filters.request_kind == "contents"
    assert filters.urls is None
    assert filters.target_urls == ["https://arxiv.org/html/2309.10668v2"]
    assert filters.anchor_queries == [
        "impressive compression rates",
        "models trained on text generalize to image audio compression",
    ]


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"target_urls": []}, "target_urls must contain at least one URL"),
        ({"urls": ["ftp://example.test/paper"]}, "urls must be http"),
        ({"target_urls": ["ftp://example.test/paper"]}, "target_urls must be http"),
        ({"anchor_queries": ["q"] * 9}, "anchor_queries allows at most eight"),
        ({"anchor_queries": [""]}, "anchor_queries entries must be non-empty"),
        ({"include_domains": [""]}, "include_domains entries must be non-empty"),
    ],
)
def test_recipe_filter_validation_errors(
    payload: dict[str, object],
    expected: str,
) -> None:
    with pytest.raises(ValidationError, match=expected):
        SearchPaperFilters(**payload)


def test_retrieval_tool_result_model_dump_json_mode_is_json_safe() -> None:
    result = RetrievalToolResult(
        tool_call_id="call-1",
        query="retrieval augmented verification",
        filters=SearchPaperFilters(
            urls=["https://example.test/paper"],
            target_urls=["https://example.test/paper"],
            anchor_queries=["verification evidence"],
            include_domains=["arxiv.org"],
            include_text=["verification"],
            text=False,
            highlights=True,
            highlight_query="verification evidence",
            highlight_max_characters=600,
            text_max_characters=5000,
            max_age_hours=0,
            livecrawl_timeout=3000,
            include_html_tags=True,
            target_scope="known_url",
            verify_quotes=True,
        ),
        sources=["arxiv", "exa"],
        evidence=[
            {
                "evidence_id": "ev_000001",
                "source": "arxiv",
                "quote": "Evidence quote.",
            }
        ],
        source_statuses=[
            SourceStatus(
                source="arxiv",
                status="success",
                source_query="retrieval augmented verification",
                result_count=1,
            )
        ],
        elapsed_ms=25,
        input_hash="input-hash",
        retrieval_config_hash="config-hash",
    )

    dumped = result.model_dump(mode="json")

    assert dumped["filters"] == {
        "include_text": ["verification"],
        "include_domains": ["arxiv.org"],
        "exclude_domains": None,
        "urls": ["https://example.test/paper"],
        "target_urls": ["https://example.test/paper"],
        "anchor_queries": ["verification evidence"],
        "exclude_text": None,
        "text": False,
        "highlights": True,
        "highlight_query": "verification evidence",
        "highlight_max_characters": 600,
        "text_max_characters": 5000,
        "max_age_hours": 0,
        "livecrawl_timeout": 3000,
        "include_html_tags": True,
        "target_scope": "known_url",
        "verify_quotes": True,
        "request_kind": "contents",
    }
    assert dumped["source_statuses"][0]["status"] == "success"
    json.dumps(dumped)
