from pathlib import Path
import tomllib

import pytest

from src.config import (
    DEFAULT_CONFIG_PATH,
    AgentConfig,
    AgentsConfig,
    CoherenceConfig,
    ModelConfig,
    load_config,
)
from src.retrieval import scoring_defaults as sd


def test_model_config_defaults_provider_to_openrouter() -> None:
    assert ModelConfig(model_id="x/test-model").provider == "openrouter"


def test_default_config_path() -> None:
    # Consolidated constant (cli-paths unit): every caller default MUST still
    # resolve to this exact path — a pure name for the repeated literal.
    assert DEFAULT_CONFIG_PATH == Path("config/evidence-evaluation.yaml")


def test_coherence_embedding_default_matches_scoring_defaults() -> None:
    # The CoherenceConfig embedding default MUST be the pinned `_base` model + revision
    # (the bare `allenai/specter2` adapter id silently lexical-falls-back) and MUST stay
    # in lockstep with the scoring-defaults source of truth.
    coherence = CoherenceConfig()
    assert coherence.embedding_model == "allenai/specter2_base"
    assert coherence.embedding_model == sd.EMBEDDING_MODEL
    assert coherence.embedding_revision == sd.EMBEDDING_REVISION
    assert coherence.embedding_revision  # non-empty pinned HF revision


def test_shipped_yaml_loads_current_graph_state_schema() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)

    # The shipped config MUST NOT override the loadable `_base` id back to the bare
    # `allenai/specter2` PEFT-adapter id, which silently lexical-falls-back at runtime.
    assert config.retrieval.coherence.embedding_model == "allenai/specter2_base"
    assert config.agents.builder.model is not None
    assert config.agents.skeptical_verifier.model is not None
    assert config.agents.builder.model.model_id
    assert config.agents.skeptical_verifier.model.model_id
    assert config.agents.skeptical_verifier.temperature == 0.0
    assert config.retrieval.artifacts.enabled is True
    assert config.retrieval.artifacts.write_html is True
    assert config.retrieval.source_limits.exa.highlights is True
    assert config.retrieval.source_limits.exa.highlight_max_characters == 1200
    assert config.retrieval.source_limits.exa.text_max_characters == 20000
    assert config.retrieval.source_limits.exa.livecrawl_timeout is None
    assert config.retrieval.source_limits.exa.include_html_tags is False

    values = config.model_dump(mode="json")
    assert "graph" not in values
    assert "debate" not in values
    assert "cost_tracking" not in values
    assert "critic_controller" not in values["agents"]
    assert "tools" not in values["agents"]["builder"]
    assert "candidate_count" not in values["agents"]["builder"]


def test_load_config_accepts_str_path() -> None:  # string-path compatibility
    # load_config(str) must not raise "'str' object has no attribute 'open'".
    assert load_config("config/evidence-evaluation.yaml").retrieval is not None


def test_camel_backend_is_runtime_dependency_but_langgraph_is_not() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]

    assert "camel-ai==0.2.90" in dependencies
    assert "openai>=1,<2" in dependencies
    assert "camel" not in project["project"].get("optional-dependencies", {})
    assert not any(dependency.startswith("langgraph") for dependency in dependencies)


def test_rejects_invalid_agent_model_max_tokens(tmp_path: Path) -> None:
    values = load_config(DEFAULT_CONFIG_PATH).model_dump(mode="json")
    values["agents"]["builder"]["model"]["max_tokens"] = 0
    bad_config = tmp_path / "bad.yaml"
    import yaml

    bad_config.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="max_tokens"):
        load_config(bad_config)


def test_rejects_missing_required_tier_model() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    values = config.model_dump(mode="json")
    values["agents"]["skeptical_verifier"]["model"] = None

    with pytest.raises(ValueError, match="required base block") as exc_info:
        type(config).model_validate(values)
    assert "agents.skeptical_verifier.model" in str(exc_info.value)


@pytest.mark.parametrize(
    ("scope", "field"),
    [
        ("root", "graph"),
        ("root", "debate"),
        ("root", "cost_tracking"),
        ("agents", "critic_controller"),
        ("builder", "tools"),
        ("builder", "candidate_count"),
    ],
)
def test_removed_legacy_config_fields_are_rejected(scope, field) -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    values = config.model_dump(mode="json")
    target = values if scope == "root" else values["agents"]
    if scope == "builder":
        target = values["agents"]["builder"]
    target[field] = {} if field in {"graph", "debate", "critic_controller"} else 1

    with pytest.raises(ValueError, match=field):
        type(config).model_validate(values)


def _agent(model_id: str) -> AgentConfig:
    return AgentConfig(
        temperature=0.0,
        model=ModelConfig(model_id=model_id, api_key_env="OPENROUTER_API_KEY"),
    )


def _agents(**overrides) -> AgentsConfig:
    values = {
        "builder": _agent("x/builder-model"),
        "skeptical_verifier": _agent("x/verifier-model"),
    }
    values.update(overrides)
    return AgentsConfig(**values)


def test_current_roles_fall_back_to_tier_defaults() -> None:
    agents = _agents()

    for role in (
        "builder",
        "extractor",
        "elaboration_writer",
        "reader_translator",
        "research_synthesist",
        "experiment_designer",
    ):
        assert agents.for_role(role).model.model_id == "x/builder-model"
    for role in (
        "skeptical_verifier",
        "evidence_reviewer",
        "translation_verifier",
        "critic_panel",
        "experiment_validator",
    ):
        assert agents.for_role(role).model.model_id == "x/verifier-model"


def test_role_override_takes_precedence() -> None:
    agents = _agents(critic_panel=_agent("x/strong-judge"))

    assert agents.for_role("critic_panel").model.model_id == "x/strong-judge"
    assert agents.for_role("research_synthesist").model.model_id == "x/builder-model"


def test_for_role_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="unknown agent role"):
        _agents().for_role("nonsense_role")


def test_reader_translation_roles_remain_deliberate_optional_seams() -> None:
    agents = _agents(reader_translator=_agent("x/translator"))

    assert agents.for_role("reader_translator").model.model_id == "x/translator"
    assert agents.for_role("translation_verifier").model.model_id == "x/verifier-model"


def test_experiment_roles_accept_overrides() -> None:
    # Experiment Designer writes; Experiment Validator judges independently, so
    # overriding the designer must not move the validator off its tier default.
    agents = _agents(experiment_designer=_agent("x/designer"))

    assert agents.for_role("experiment_designer").model.model_id == "x/designer"
    assert agents.for_role("experiment_validator").model.model_id == "x/verifier-model"
