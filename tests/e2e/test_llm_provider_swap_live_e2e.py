from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

import pytest

from src.camel_adapter import (
    _create_camel_model_backend,
    _first_response_message,
    _message_content,
)
from src.config import AgentConfig, ModelConfig


pytestmark = [pytest.mark.e2e, pytest.mark.live, pytest.mark.live_llm_provider]


@dataclass(frozen=True)
class ProviderCase:
    name: str
    provider: str
    env_key: str
    model_env: str
    default_model: str | None = None
    base_url_env: str | None = None
    default_base_url: str | None = None
    dependency: str | None = None
    marker: object | None = None


# The swap matrix covers the supported direct-LLM path (OpenRouter) and a generic
# OpenAI-wire-compatible endpoint (self-hosted or proxy). No model is recommended:
# supply the slug yourself via the case's ``model_env`` — https://openrouter.ai/models
PROVIDER_CASES: dict[str, ProviderCase] = {
    "openrouter": ProviderCase(
        "openrouter",
        "openrouter",
        "OPENROUTER_API_KEY",
        "GRAPH_HYPOTH_OPENROUTER_MODEL",
        base_url_env="OPENROUTER_API_BASE_URL",
        default_base_url="https://openrouter.ai/api/v1",
        marker=pytest.mark.live_openrouter,
    ),
    # ``openai-compatible-model`` is a live-only probe of a self-hosted or proxy
    # OpenAI-wire endpoint. It is NOT a fourth supported configuration path: the
    # supported ``provider`` literals are ``openrouter``, ``claude-cli``, and
    # ``codex-cli`` (see src/config.py and docs/guides/model-configuration.md).
    "openai-compatible": ProviderCase(
        "openai-compatible",
        "openai-compatible-model",
        "OPENAI_COMPATIBILITY_API_KEY",
        "GRAPH_HYPOTH_OPENAI_COMPATIBLE_MODEL",
        base_url_env="OPENAI_COMPATIBILITY_API_BASE_URL",
        marker=pytest.mark.live_openai_compatible,
    ),
}


def _requested_provider_cases() -> list[ProviderCase]:
    requested = os.environ.get(
        "GRAPH_HYPOTH_LIVE_LLM_PROVIDERS", ",".join(PROVIDER_CASES)
    )
    names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = [name for name in names if name not in PROVIDER_CASES]
    if unknown:
        raise ValueError(f"unsupported live provider test case: {unknown[0]}")
    return [PROVIDER_CASES[name] for name in names]


def _dependency_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _is_quota_error(exc: Exception) -> bool:
    normalized = str(exc).lower()
    return any(
        marker in normalized
        for marker in ("429", "quota", "rate limit", "resource_exhausted", "billing")
    )


def _provider_agent(provider_case: ProviderCase) -> AgentConfig:
    model_id = os.environ.get(provider_case.model_env) or provider_case.default_model
    if not model_id:
        pytest.skip(
            f"{provider_case.model_env} is required for {provider_case.name} "
            "(OpenRouter catalog slugs: https://openrouter.ai/models)"
        )
    base_url = (
        os.environ.get(provider_case.base_url_env)
        if provider_case.base_url_env
        else None
    ) or provider_case.default_base_url
    return AgentConfig(
        temperature=0.0,
        model=ModelConfig(
            provider=provider_case.provider,
            model_id=model_id,
            api_key_env=provider_case.env_key,
            base_url=base_url,
            max_tokens=64,
            timeout_seconds=90,
        ),
    )


@pytest.mark.parametrize(
    "provider_case",
    [
        pytest.param(case, marks=case.marker or (), id=case.name)
        for case in _requested_provider_cases()
    ],
)
def test_live_llm_provider_can_run_a_graph_state_backend(
    provider_case: ProviderCase,
) -> None:
    if not os.environ.get(provider_case.env_key):
        pytest.skip(f"{provider_case.env_key} is required for {provider_case.name}")
    if provider_case.dependency and not _dependency_available(provider_case.dependency):
        pytest.skip(f"{provider_case.dependency} is required for {provider_case.name}")

    try:
        backend = _create_camel_model_backend(
            _provider_agent(provider_case), role_name="evidence_reviewer"
        )
        response = backend.run(
            [
                {"role": "system", "content": "Reply with the single word OK."},
                {"role": "user", "content": "Connectivity check."},
            ]
        )
    except Exception as exc:
        if _is_quota_error(exc):
            pytest.skip(f"{provider_case.name} live provider quota exhausted")
        raise

    content = _message_content(_first_response_message(response))
    assert str(content).strip()
