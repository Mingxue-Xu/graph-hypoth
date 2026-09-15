"""``provider: claude-cli`` as a configurable direct-backend provider.

The Codex equivalents live in ``test_camel_adapter.py``; these cover the Claude
provider's dispatch, its shared guards, and the provider-aware effort vocabulary
that keeps a study from silently running at the CLI's default effort.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.camel_adapter import (
    _create_camel_model_backend,
    _create_direct_model_backend,
)
from src.claude_cli_backend import ClaudeSubagentBackend
from src.codex_cli_backend import CodexSubagentBackend
from src.config import AgentConfig, ModelConfig


def _agent(**model_overrides) -> AgentConfig:
    fields = {
        "provider": "claude-cli",
        "model_id": "claude-model-id",
        "api_key_env": None,
        "base_url": None,
        "reasoning_effort": "medium",
        "timeout_seconds": 600.0,
    }
    fields.update(model_overrides)
    return AgentConfig(temperature=0.0, model=ModelConfig(**fields))


# --- dispatch --------------------------------------------------------------


def test_claude_cli_provider_builds_the_claude_subagent_backend():
    with patch(
        "src.claude_cli_backend.resolve_claude_executable", return_value="/bin/claude"
    ):
        backend = _create_direct_model_backend(_agent(), role_name="builder")

    assert isinstance(backend, ClaudeSubagentBackend)
    assert backend.model == "claude-model-id"
    assert backend.reasoning_effort == "medium"
    assert backend.timeout_seconds == 600.0
    assert backend.role_name == "builder"


def test_codex_cli_provider_still_builds_the_codex_backend():
    agent = _agent(
        provider="codex-cli", model_id="codex-model-id", reasoning_effort="high"
    )
    with patch(
        "src.codex_cli_backend.resolve_codex_executable", return_value="/bin/codex"
    ):
        backend = _create_direct_model_backend(agent, role_name="builder")

    assert isinstance(backend, CodexSubagentBackend)


def test_claude_cli_is_rejected_by_camel_model_factory():
    with pytest.raises(RuntimeError, match="claude-cli"):
        _create_camel_model_backend(_agent(), role_name="builder")


# --- shared guards ---------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("api_key_env", "TEST_MODEL_API_KEY", "api_key_env"),
        ("base_url", "https://example.test", "base_url"),
        ("max_tokens", 1024, "max_tokens"),
    ],
)
def test_claude_cli_rejects_settings_the_backend_cannot_honor(field, value, expected):
    with pytest.raises(RuntimeError, match=expected):
        _create_direct_model_backend(_agent(**{field: value}), role_name="builder")


def test_the_saved_login_hint_names_the_right_cli():
    with pytest.raises(RuntimeError, match="claude auth login"):
        _create_direct_model_backend(
            _agent(api_key_env="TEST_MODEL_API_KEY"), role_name="builder"
        )
    with pytest.raises(RuntimeError, match="codex login"):
        _create_direct_model_backend(
            _agent(provider="codex-cli", api_key_env="TEST_MODEL_API_KEY"),
            role_name="builder",
        )


# --- effort vocabulary -----------------------------------------------------


def test_each_cli_accepts_only_its_own_effort_levels():
    # Claude has max and no minimal; Codex is the reverse.
    assert ModelConfig(
        provider="claude-cli", model_id="claude-model-id", reasoning_effort="max"
    ).reasoning_effort == "max"
    assert ModelConfig(
        provider="codex-cli", model_id="codex-model-id", reasoning_effort="minimal"
    ).reasoning_effort == "minimal"

    with pytest.raises(ValueError, match="not supported by provider 'claude-cli'"):
        ModelConfig(
            provider="claude-cli", model_id="claude-model-id", reasoning_effort="minimal"
        )
    with pytest.raises(ValueError, match="not supported by provider 'codex-cli'"):
        ModelConfig(
            provider="codex-cli", model_id="codex-model-id", reasoning_effort="max"
        )


def test_non_cli_providers_keep_the_unrestricted_vocabulary():
    # CAMEL providers ignore the field entirely; validation must not narrow them.
    for effort in ("minimal", "max"):
        assert ModelConfig(
            provider="openrouter", model_id="x/test-model",
            reasoning_effort=effort,
        ).reasoning_effort == effort


def test_effort_is_optional_for_cli_providers():
    assert ModelConfig(provider="claude-cli", model_id="m").reasoning_effort is None
