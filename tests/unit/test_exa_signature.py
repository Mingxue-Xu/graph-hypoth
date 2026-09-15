from __future__ import annotations

import inspect

from src.retrieval.sources import ExaPaperSource


def test_direct_exa_source_signature_exposes_recipe_params_without_deprecated_controls() -> None:
    signature = inspect.signature(ExaPaperSource._call_exa_sdk)

    for name in (
        "query",
        "urls",
        "max_age_hours",
        "livecrawl_timeout",
        "include_domains",
        "exclude_domains",
        "include_text",
        "exclude_text",
        "text",
        "highlights",
    ):
        assert name in signature.parameters
    assert "use_autoprompt" not in signature.parameters
    assert "livecrawl" not in signature.parameters


def test_exa_source_text_filter_validator_tracks_exa_constraints() -> None:
    source = ExaPaperSource(config={})

    assert source._validate_text_filters(["one"], None) is None
    assert source._validate_text_filters(["one", "two"], None) == (
        "exa include_text accepts at most one string"
    )
    assert source._validate_text_filters(None, ["one two three four five six"]) == (
        "exa exclude_text string must be at most five words"
    )
