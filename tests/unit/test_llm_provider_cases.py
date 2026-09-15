from tests.e2e.test_llm_provider_swap_live_e2e import (
    PROVIDER_CASES,
    _dependency_available,
    _is_quota_error,
    _provider_agent,
    _requested_provider_cases,
)


def test_live_provider_cases_default_to_provider_swap_matrix(monkeypatch) -> None:
    """The default live matrix is the supported OpenRouter path plus the
    OpenAI-wire endpoint probe. ``openai-compatible`` is a live-test target
    only, not a supported ``provider`` literal for a shipped config."""
    monkeypatch.delenv("GRAPH_HYPOTH_LIVE_LLM_PROVIDERS", raising=False)

    requested = {provider_case.name for provider_case in _requested_provider_cases()}

    assert "openrouter" in requested
    assert "openai-compatible" in requested


def test_openrouter_provider_config_defaults_base_url(monkeypatch) -> None:
    monkeypatch.setenv("GRAPH_HYPOTH_OPENROUTER_MODEL", "x/test-model")
    monkeypatch.delenv("OPENROUTER_API_BASE_URL", raising=False)

    agent = _provider_agent(PROVIDER_CASES["openrouter"])

    assert agent.model is not None
    assert agent.model.provider == "openrouter"
    assert agent.model.model_id == "x/test-model"
    assert agent.model.base_url == "https://openrouter.ai/api/v1"


def test_provider_dependency_check_handles_missing_namespace_package() -> None:
    assert _dependency_available("definitely_missing_namespace.child") is False


def test_provider_quota_error_detection() -> None:
    assert _is_quota_error(
        RuntimeError("Error code: 429 - RESOURCE_EXHAUSTED quota exceeded")
    ) is True
