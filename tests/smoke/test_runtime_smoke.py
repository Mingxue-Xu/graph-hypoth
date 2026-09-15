from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

from src.config import load_config


pytestmark = pytest.mark.smoke


def test_default_config_and_backend_dependency_surface() -> None:
    config = load_config(Path("config/evidence-evaluation.yaml"))
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert config.agents.builder.model is not None
    assert config.agents.skeptical_verifier.model is not None
    # The shipped config runs the default execution mode: key-free Claude Code CLI
    # subagents. The CLI providers require no API key.
    for role in ("builder", "skeptical_verifier"):
        agent = getattr(config.agents, role)
        assert agent.model.provider == "claude-cli", role
        assert agent.model.api_key_env is None, role
        assert agent.model.base_url is None, role
        assert agent.model.max_tokens is None, role
    # Each model-backed role carries a model slug (not pinned to a specific model — operators tune
    # these; the exact per-role slugs are exercised by the config + reconciliation tests).
    assert config.agents.builder.model.model_id
    assert config.agents.skeptical_verifier.model.model_id

    serialized = config.model_dump(mode="json")
    assert "critic_controller" not in serialized["agents"]
    assert "graph" not in serialized
    assert "debate" not in serialized

    dependencies = project["project"]["dependencies"]
    assert "camel-ai==0.2.90" in dependencies
    assert not any(dependency.startswith("langgraph") for dependency in dependencies)

    camel_models = importlib.import_module("camel.models")
    assert camel_models.ModelFactory is not None


def test_current_graph_roles_resolve_to_configured_tiers() -> None:
    agents = load_config(Path("config/evidence-evaluation.yaml")).agents

    for role in (
        "extractor",
        "evidence_reviewer",
        "research_synthesist",
        "critic_panel",
        "experiment_designer",
        "experiment_validator",
        "elaboration_writer",
    ):
        assert agents.for_role(role).model is not None
