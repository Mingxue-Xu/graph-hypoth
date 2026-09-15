"""Shared test helpers.

``openrouter_provider_config`` exists because the shipped configuration and the
tests that exercise provider credentials deliberately disagree about providers.

``config/evidence-evaluation.yaml`` runs the documented default execution mode:
key-free ``claude-cli`` subagents answered by the direct ``backend.run(messages)``
seams. Those providers take no API key, so a test that asserts on credential
preflight — or on an OpenRouter-shaped model slug — has nothing to assert against
when it loads the shipped config directly.

Such tests take the shipped config's retrieval settings and override only the two
base model configurations onto an OpenRouter provider, so credential coverage
stays real without pinning the user-facing default to it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.config import OrchestrationConfig

DEFAULT_CONFIG_FILE = "config/evidence-evaluation.yaml"

# The two required base model configurations.
_BASE_ROLES = ("builder", "skeptical_verifier")


def runnable_default_config(
    path: str | Path = DEFAULT_CONFIG_FILE,
) -> OrchestrationConfig:
    """The shipped config with its ``<MODEL_ID>`` placeholders replaced.

    The shipped file is deliberately not runnable: it names no model, and
    ``_preflight_model_credentials`` rejects the placeholder so a live run stops
    in a second instead of after the whole retrieval stage. A test that drives a
    CLI entry point *through* that preflight therefore has to name a model, the
    same way a user's ``--config`` copy does.
    """
    from src.config import load_config

    config = load_config(Path(path)).model_copy(deep=True)
    for role in _BASE_ROLES:
        model = getattr(config.agents, role).model
        assert model is not None
        model.model_id = f"test-{role.replace('_', '-')}-model"
    return config


def openrouter_provider_config(
    path: str | Path = DEFAULT_CONFIG_FILE,
) -> OrchestrationConfig:
    """The shipped config with its base model configurations moved to OpenRouter.

    Model ids follow the ``x/<role>-model`` convention documented in ``conftest``:
    an OpenRouter-shaped slug in a fake vendor namespace, pairwise distinct so
    role-fallback assertions stay discriminating.
    """
    from src.config import ModelConfig, load_config

    config = load_config(Path(path)).model_copy(deep=True)
    for role in _BASE_ROLES:
        agent = getattr(config.agents, role)
        agent.model = ModelConfig(
            provider="openrouter",
            model_id=f"x/{role}-model",
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
            max_tokens=1024,
            timeout_seconds=60,
        )
    return config
