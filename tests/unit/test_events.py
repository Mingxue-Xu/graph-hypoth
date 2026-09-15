from src.config import AgentConfig, ModelConfig
from src.events import EventType, build_event
from src.events import (
    hash_prompt,
    model_config_hash,
    stable_hash_payload,
)


def test_event_hash_changes_when_content_changes() -> None:
    first_event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        node_name="review_evidence",
        round_index=1,
        event_type=EventType.AGENT_MESSAGE,
        sender_role="builder",
        receiver_role="evidence_reviewer",
        content="Initial claim analysis.",
        previous_hash="previous-hash",
        timestamp="2026-05-05T12:00:00+00:00",
    )
    second_event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        node_name="review_evidence",
        round_index=1,
        event_type=EventType.AGENT_MESSAGE,
        sender_role="builder",
        receiver_role="evidence_reviewer",
        content="Revised claim analysis.",
        previous_hash="previous-hash",
        timestamp="2026-05-05T12:00:00+00:00",
    )

    assert first_event.event_hash != second_event.event_hash
    assert first_event.prev_hash == "previous-hash"
    assert second_event.prev_hash == "previous-hash"


def test_idempotency_key_is_stable_for_same_event_identity() -> None:
    event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        node_name="critic_panel",
        round_index=2,
        event_type=EventType.CRITIC_DECISION,
        sender_role="critic_panel",
        receiver_role="research_synthesist",
        content="Reject until evidence is stronger.",
        previous_hash=None,
        timestamp="2026-05-05T12:00:00+00:00",
    )

    assert (
        event.idempotency_key
        == "run-1:checkpoint-1:critic_panel:2:critic_decision:critic_panel"
    )


def test_build_event_accepts_explicit_idempotency_key() -> None:
    event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint",
        node_name="node",
        round_index=0,
        event_type=EventType.TOOL_EVENT,
        sender_role="builder",
        receiver_role="graph",
        content="tool call",
        previous_hash="root",
        idempotency_key="explicit-key",
    )

    assert event.idempotency_key == "explicit-key"


def test_build_event_falls_back_to_derived_key() -> None:
    event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint",
        node_name="node",
        round_index=0,
        event_type=EventType.TOOL_EVENT,
        sender_role="builder",
        receiver_role="graph",
        content="tool call",
        previous_hash="root",
    )

    assert event.idempotency_key == "run-1:checkpoint:node:0:tool_event:builder"


def test_event_idempotency_key_changes_across_checkpoints() -> None:
    first_event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        node_name="critic_panel",
        round_index=2,
        event_type=EventType.CRITIC_DECISION,
        sender_role="critic_panel",
        receiver_role="research_synthesist",
        content="Reject until evidence is stronger.",
        previous_hash=None,
        timestamp="2026-05-05T12:00:00+00:00",
    )
    second_event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-2",
        node_name="critic_panel",
        round_index=2,
        event_type=EventType.CRITIC_DECISION,
        sender_role="critic_panel",
        receiver_role="research_synthesist",
        content="Reject until evidence is stronger.",
        previous_hash=None,
        timestamp="2026-05-05T12:00:00+00:00",
    )

    assert first_event.idempotency_key != second_event.idempotency_key


def test_event_hash_is_stable_for_same_payload() -> None:
    first_event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        node_name="critic_panel",
        round_index=2,
        event_type=EventType.CRITIC_DECISION,
        sender_role="critic_panel",
        receiver_role="research_synthesist",
        content="Accept with caveats.",
        previous_hash="previous-hash",
        structured_payload={
            "decision": "accept",
            "rationale": "grounded",
            "nested": {"status": "ok"},
        },
        token_usage={"input": 11, "output": 7},
        timestamp="2026-05-05T12:00:00+00:00",
    )
    second_event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        node_name="critic_panel",
        round_index=2,
        event_type=EventType.CRITIC_DECISION,
        sender_role="critic_panel",
        receiver_role="research_synthesist",
        content="Accept with caveats.",
        previous_hash="previous-hash",
        structured_payload={
            "decision": "accept",
            "rationale": "grounded",
            "nested": {"status": "ok"},
        },
        token_usage={"input": 11, "output": 7},
        timestamp="2026-05-05T12:00:00+00:00",
    )

    assert first_event.event_hash == second_event.event_hash


def test_event_cost_usd_is_recorded_without_changing_event_hash() -> None:
    first_event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        node_name="critic_panel",
        round_index=2,
        event_type=EventType.CRITIC_DECISION,
        sender_role="critic_panel",
        receiver_role="research_synthesist",
        content="Accept with caveats.",
        previous_hash="previous-hash",
        structured_payload={
            "decision": "accept",
            "rationale": "grounded",
            "nested": {"status": "ok"},
        },
        token_usage={"total_tokens": 18},
        timestamp="2026-05-05T12:00:00+00:00",
    )
    second_event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        node_name="critic_panel",
        round_index=2,
        event_type=EventType.CRITIC_DECISION,
        sender_role="critic_panel",
        receiver_role="research_synthesist",
        content="Accept with caveats.",
        previous_hash="previous-hash",
        structured_payload={
            "decision": "accept",
            "rationale": "grounded",
            "nested": {"status": "ok"},
        },
        token_usage={"total_tokens": 18},
        cost_usd=0.0123,
        timestamp="2026-05-05T12:00:00+00:00",
    )

    assert second_event.cost_usd == 0.0123
    assert first_event.event_hash == second_event.event_hash

    structured_cost_event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        node_name="critic_panel",
        round_index=2,
        event_type=EventType.CRITIC_DECISION,
        sender_role="critic_panel",
        receiver_role="research_synthesist",
        content="Accept with caveats.",
        previous_hash="previous-hash",
        structured_payload={
            "decision": "accept",
            "rationale": "grounded",
            "cost_usd": 0.0123,
            "nested": {"status": "ok", "cost_usd": 0.0456},
        },
        token_usage={"total_tokens": 18},
        timestamp="2026-05-05T12:00:00+00:00",
    )

    assert structured_cost_event.structured_payload["cost_usd"] == 0.0123
    assert structured_cost_event.event_hash == first_event.event_hash


def test_stable_hash_payload_uses_canonical_json_order() -> None:
    first_hash = stable_hash_payload({"b": 2, "a": {"d": 4, "c": 3}})
    second_hash = stable_hash_payload({"a": {"c": 3, "d": 4}, "b": 2})

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_hash_prompt_hashes_prompt_text() -> None:
    assert hash_prompt("rubric prompt") == hash_prompt("rubric prompt")
    assert hash_prompt("rubric prompt") != hash_prompt("different prompt")


def test_model_config_hash_keeps_env_name_but_not_env_value(monkeypatch) -> None:
    agent_config = AgentConfig(
        temperature=0.1,
        model=ModelConfig(
            provider="test-provider",
            model_id="x/test-model",
            api_key_env="TEST_MODEL_API_KEY",
        ),
    )

    monkeypatch.setenv("TEST_MODEL_API_KEY", "first-secret")
    first_hash = model_config_hash(agent_config)
    monkeypatch.setenv("TEST_MODEL_API_KEY", "second-secret")
    second_hash = model_config_hash(agent_config)

    assert first_hash == second_hash


def test_model_config_hash_changes_for_provider_and_base_url() -> None:
    agent_config = AgentConfig(
        temperature=0.1,
        model=ModelConfig(
            provider="test-provider",
            model_id="x/test-model",
            api_key_env="TEST_MODEL_API_KEY",
            base_url=None,
        ),
    )
    base_hash = model_config_hash(agent_config)

    provider_config = agent_config.model_copy(deep=True)
    assert provider_config.model is not None
    provider_config.model.provider = "test-provider-alt"

    base_url_config = agent_config.model_copy(deep=True)
    assert base_url_config.model is not None
    base_url_config.model.base_url = "https://openrouter.ai/api/v1"

    assert model_config_hash(provider_config) != base_hash
    assert model_config_hash(base_url_config) != base_hash


def test_model_config_hash_changes_for_model_id() -> None:
    agent_config = AgentConfig(
        temperature=0.1,
        model=ModelConfig(
            provider="test-provider",
            model_id="x/test-model",
            api_key_env="TEST_MODEL_API_KEY",
        ),
    )
    model_id_config = agent_config.model_copy(deep=True)
    assert model_id_config.model is not None
    model_id_config.model.model_id = "x/test-model-alt"

    assert model_config_hash(model_id_config) != model_config_hash(agent_config)
