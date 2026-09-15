from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.config import AgentConfig


def stable_hash_payload(payload: object) -> str:
    serialized_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()


def _hashable_structured_payload(payload: object) -> object:
    if isinstance(payload, dict):
        return {
            key: _hashable_structured_payload(value)
            for key, value in payload.items()
            if key != "cost_usd"
        }
    if isinstance(payload, list):
        return [_hashable_structured_payload(item) for item in payload]
    return payload


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def model_config_hash(agent_config: AgentConfig) -> str:
    return stable_hash_payload(agent_config.model_dump(mode="json", exclude_none=True))


def model_id(agent_config: AgentConfig) -> str | None:
    if agent_config.model is None:
        return None
    return agent_config.model.model_id


class EventType(StrEnum):
    # The current graph-state path emits AGENT_MESSAGE and TOOL_EVENT. The other
    # values remain solely so existing SQLite audit logs can still be decoded.
    AGENT_MESSAGE = "agent_message"
    CRITIC_DECISION = "critic_decision"
    STATE_UPDATE = "state_update"
    TOOL_EVENT = "tool_event"
    HUMAN_EVENT = "human_event"


class AuditEvent(BaseModel):
    event_id: str
    run_id: str
    thread_id: str
    checkpoint_id: str
    node_name: str
    round_index: int
    event_type: EventType
    sender_role: str
    receiver_role: str | None = None
    content: str
    structured_payload: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    provider: str | None = None
    model_config_hash: str | None = None
    prompt_hash: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: int | None = None
    cost_usd: float | None = None
    timestamp: str
    termination_reason: str | None = None
    prev_hash: str | None = None
    event_hash: str
    idempotency_key: str


def build_event(
    *,
    run_id: str,
    thread_id: str,
    checkpoint_id: str,
    node_name: str,
    round_index: int,
    event_type: EventType,
    sender_role: str,
    receiver_role: str | None,
    content: str,
    previous_hash: str | None,
    structured_payload: dict[str, Any] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    model: str | None = None,
    provider: str | None = None,
    model_config_hash: str | None = None,
    prompt_hash: str | None = None,
    token_usage: dict[str, int] | None = None,
    latency_ms: int | None = None,
    cost_usd: float | None = None,
    termination_reason: str | None = None,
    timestamp: str | None = None,
    idempotency_key: str | None = None,
) -> AuditEvent:
    assigned_timestamp = timestamp or datetime.now(UTC).isoformat()
    assigned_structured_payload = structured_payload or {}
    assigned_tool_calls = tool_calls or []
    assigned_idempotency_key = idempotency_key or (
        f"{run_id}:{checkpoint_id}:{node_name}:{round_index}:"
        f"{event_type.value}:{sender_role}"
    )
    event_fields = {
        "run_id": run_id,
        "thread_id": thread_id,
        "checkpoint_id": checkpoint_id,
        "node_name": node_name,
        "round_index": round_index,
        "event_type": event_type.value,
        "sender_role": sender_role,
        "receiver_role": receiver_role,
        "content": content,
        "structured_payload": assigned_structured_payload,
        "tool_calls": assigned_tool_calls,
        "model": model,
        "provider": provider,
        "model_config_hash": model_config_hash,
        "prompt_hash": prompt_hash,
        "token_usage": token_usage or {},
        "latency_ms": latency_ms,
        "timestamp": assigned_timestamp,
        "termination_reason": termination_reason,
        "prev_hash": previous_hash,
    }
    hash_fields = {
        **event_fields,
        "structured_payload": _hashable_structured_payload(
            assigned_structured_payload
        ),
    }
    event_hash = stable_hash_payload(hash_fields)

    # Cost can be observed or reconciled asynchronously. Keep it out of the
    # hashed payload so attaching measured cost does not invalidate the chain.
    return AuditEvent(
        event_id=str(uuid4()),
        event_hash=event_hash,
        idempotency_key=assigned_idempotency_key,
        cost_usd=cost_usd,
        **event_fields,
    )
