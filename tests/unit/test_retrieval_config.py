import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import (
    ApifySourceConfig,
    ArxivSourceConfig,
    CodexWebSourceConfig,
    ExaSourceConfig,
    ResilienceConfig,
    RetrievalArtifactConfig,
    RetrievalConfig,
    RetrievalToolBudgetConfig,
)


def test_retrieval_defaults_to_bounded_crossref_with_optional_exa() -> None:
    config = RetrievalConfig()

    assert config.enabled is True
    assert config.mode == "source_only"
    assert config.sources == ["crossref"]
    assert config.optional_sources == ["exa"]
    assert config.selected_source_names() == ["crossref", "exa"]


def test_retrieval_profile_is_metadata_only() -> None:
    config = RetrievalConfig(profile="research-heavy")

    assert config.profile == "research-heavy"
    assert config.selected_source_names() == ["crossref", "exa"]


def test_shipped_config_does_not_select_unbounded_arxiv_loader() -> None:
    from src.config import load_config

    config = load_config(Path("config/evidence-evaluation.yaml")).retrieval

    assert config.sources == ["crossref"]
    assert "arxiv" not in config.selected_source_names()


def test_fake_mode_replaces_configured_sources_with_fake_source() -> None:
    config = RetrievalConfig(
        fake_mode=True, sources=["arxiv"], optional_sources=["exa"]
    )

    assert config.selected_source_names() == ["fake"]


@pytest.mark.parametrize(
    "payload, expected_error",
    [
        ({"mode": "hybrid"}, "mode"),
        ({"sources": ["arxiv", "semantic_scholar"]}, "unsupported retrieval sources"),
        ({"optional_sources": ["local_corpus"]}, "unsupported retrieval sources"),
    ],
)
def test_rejects_invalid_retrieval_modes_and_sources(
    payload: dict[str, object],
    expected_error: str,
) -> None:
    with pytest.raises(ValidationError, match=expected_error):
        RetrievalConfig(**payload)


def test_fake_source_cannot_be_selected_except_by_fake_mode() -> None:
    with pytest.raises(ValidationError, match="fake source is selected by fake_mode"):
        RetrievalConfig(fake_mode=True, sources=["fake"])


def test_arxiv_source_enforces_single_connection() -> None:
    with pytest.raises(ValidationError, match="single_connection"):
        ArxivSourceConfig(single_connection=False)


def test_exa_source_config_defaults_per_call_cap() -> None:
    assert ExaSourceConfig().max_cost_dollars_per_call == 0.05


def test_exa_source_config_supports_known_url_contents_recipe_options() -> None:
    config = ExaSourceConfig()

    assert config.max_age_hours is None
    assert config.livecrawl_timeout is None
    assert config.include_html_tags is False
    assert config.text_max_characters == 20000
    assert config.highlight_max_characters == 1200
    assert config.include_domains is None


def test_retrieval_artifact_config_defaults_to_runtime_trace_artifacts() -> None:
    config = RetrievalArtifactConfig()

    assert config.enabled is True
    assert config.write_json is True
    assert config.write_html is True


def test_retrieval_config_carries_artifact_settings() -> None:
    config = RetrievalConfig()

    assert config.artifacts.enabled is True
    assert config.artifacts.write_html is True


def test_retrieval_tool_budget_defaults_cancel_per_agent_round_cap() -> None:
    config = RetrievalToolBudgetConfig()

    # The run cap was raised from 12 to 24 because each facet-style logical retrieval
    # issues several ``search_papers`` calls. It remains a guard rather than a driver.
    assert config.max_calls_per_run == 24
    assert config.max_calls_per_agent_round is None


def test_retrieval_tool_budget_accepts_null_to_disable_caps() -> None:
    config = RetrievalToolBudgetConfig(
        max_calls_per_run=None,
        max_calls_per_agent_round=None,
    )

    assert config.max_calls_per_run is None
    assert config.max_calls_per_agent_round is None


def test_codex_web_default_timeout_allows_multi_search_research_turn() -> None:
    assert CodexWebSourceConfig().timeout_seconds == 900.0


def test_exa_source_config_rejects_negative_per_call_cap() -> None:
    with pytest.raises(ValidationError, match="max_cost_dollars_per_call"):
        ExaSourceConfig(max_cost_dollars_per_call=-0.01)


def test_retrieval_mvp_extra_declares_bounded_dependencies() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    retrieval_extra = project["project"]["optional-dependencies"]["retrieval"]

    assert "langchain-community>=0.3,<0.4" in retrieval_extra
    assert "arxiv>=2,<3" in retrieval_extra
    assert "pymupdf>=1.24,<2" in retrieval_extra
    assert "exa-py>=2.12,<3" in retrieval_extra


def test_selected_source_names_dedupes_overlap_between_sources() -> None:
    from src.config import RetrievalConfig

    config = RetrievalConfig(sources=["arxiv", "crossref"], optional_sources=["crossref"])
    assert config.selected_source_names() == ["arxiv", "crossref"]


def test_retrieval_config_rejects_unknown_top_level_key() -> None:
    import pytest

    from src.config import RetrievalConfig

    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        RetrievalConfig.model_validate({"optional_source": ["crossref"]})


def test_retrieval_config_rejects_unknown_source_in_source_order() -> None:
    import pytest

    from src.config import RetrievalConfig

    with pytest.raises(ValueError, match="ranking.source_order"):
        RetrievalConfig.model_validate(
            {"ranking": {"source_order": ["arxiv", "semantic_scholar"]}}
        )


def test_retrieval_config_rejects_selected_source_without_trust_tier() -> None:
    import pytest

    from src.config import RetrievalConfig

    with pytest.raises(ValueError, match="trust_policy"):
        RetrievalConfig.model_validate(
            {
                "sources": ["arxiv"],
                "optional_sources": ["crossref"],
                "trust_policy": {"arxiv": "authoritative_preprint"},  # crossref missing
            }
        )


def test_apify_source_config_timeout_survives_synchronous_browser_actor() -> None:
    # `run-sync-get-dataset-items` blocks for the entire actor run; a Google-Scholar
    # browser scrape measured ~292s live, so the shipped 60s timed out every call
    # (client-side timeout -> circuit breaker -> 0 records). Raised to 400s with margin.
    assert ApifySourceConfig().timeout_seconds == 400.0


def test_apify_source_config_caps_actor_maxresults() -> None:
    # `max_items` only pages the dataset read; it does NOT bound the actor's scrape
    # (asked 8, got 50 live). Cap the actor input directly so each run stays bounded
    # in time/cost on the free tier.
    assert ApifySourceConfig().extra_input.get("maxResults") == 15


def test_apify_resilience_policy_makes_a_single_attempt() -> None:
    # Retrying a ~5-min synchronous actor multiplies wall-clock (3 x 400s). One attempt
    # for apify; every other source keeps the default 3 (policy_for falls back to default).
    resilience = ResilienceConfig()

    assert resilience.policy_for("apify").max_attempts == 1
    assert resilience.policy_for("arxiv").max_attempts == 3
