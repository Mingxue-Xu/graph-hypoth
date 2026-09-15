from pathlib import Path
import sqlite3

import pytest

from src.events import EventType, build_event
from src.log_store import SQLiteLogStore


def test_sqlite_log_store_deduplicates_by_idempotency_key(tmp_path: Path) -> None:
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        node_name="review_evidence",
        round_index=1,
        event_type=EventType.AGENT_MESSAGE,
        sender_role="builder",
        receiver_role="evidence_reviewer",
        content="Initial claim analysis.",
        previous_hash=None,
        timestamp="2026-05-05T12:00:00+00:00",
    )

    store.append(event)
    store.append(event)

    assert len(store.list_events("run-1")) == 1


def test_sqlite_log_store_orders_events_by_insertion(tmp_path: Path) -> None:
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
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
        previous_hash=None,
        timestamp="2026-05-05T12:00:00+00:00",
    )
    second_event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-2",
        node_name="review_evidence",
        round_index=2,
        event_type=EventType.AGENT_MESSAGE,
        sender_role="verifier",
        receiver_role="builder",
        content="Verification challenge.",
        previous_hash=first_event.event_hash,
        timestamp="2026-05-05T12:01:00+00:00",
    )

    store.append(first_event)
    store.append(second_event)

    assert store.list_events("run-1") == [first_event, second_event]


def test_sqlite_log_store_rejects_divergent_duplicate_idempotency_key(
    tmp_path: Path,
) -> None:
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    first_event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        node_name="review_evidence",
        round_index=1,
        event_type=EventType.STATE_UPDATE,
        sender_role="critic_panel",
        receiver_role="builder",
        content="rubric v1",
        previous_hash=None,
        timestamp="2026-05-05T12:00:00+00:00",
    )
    second_event = build_event(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        node_name="review_evidence",
        round_index=1,
        event_type=EventType.STATE_UPDATE,
        sender_role="critic_panel",
        receiver_role="builder",
        content="rubric v2",
        previous_hash=None,
        timestamp="2026-05-05T12:00:00+00:00",
    )

    store.append(first_event)

    with pytest.raises(ValueError, match="idempotency_key.*event_hash"):
        store.append(second_event)


def test_sqlite_log_store_retrieval_cache_round_trip(tmp_path: Path) -> None:
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    payload = {
        "tool_call_id": "tool-1",
        "query": "retrieval augmented debate",
        "evidence": [{"evidence_id": "ev_000001", "title": "Useful paper"}],
    }

    store.put_retrieval_cache(
        run_id="run-1",
        input_hash="input-hash",
        retrieval_config_hash="config-hash",
        payload=payload,
    )

    assert (
        store.get_retrieval_cache(
            run_id="run-1",
            input_hash="input-hash",
            retrieval_config_hash="config-hash",
        )
        == payload
    )


def test_sqlite_log_store_missing_retrieval_cache_returns_none(tmp_path: Path) -> None:
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()

    assert (
        store.get_retrieval_cache(
            run_id="run-1",
            input_hash="missing-input",
            retrieval_config_hash="config-hash",
        )
        is None
    )


def test_sqlite_log_store_rejects_divergent_retrieval_cache_payload(
    tmp_path: Path,
) -> None:
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    cache_key = {
        "run_id": "run-1",
        "input_hash": "input-hash",
        "retrieval_config_hash": "config-hash",
    }

    store.put_retrieval_cache(payload={"evidence": ["first"]}, **cache_key)

    with pytest.raises(ValueError, match="retrieval replay conflict"):
        store.put_retrieval_cache(payload={"evidence": ["second"]}, **cache_key)


def test_sqlite_log_store_rejects_non_dict_retrieval_cache_payload(
    tmp_path: Path,
) -> None:
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    with sqlite3.connect(tmp_path / "events.sqlite") as connection:
        connection.execute(
            """
            INSERT INTO retrieval_cache (
                run_id,
                input_hash,
                retrieval_config_hash,
                payload_hash,
                payload
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            ("run-1", "input-hash", "config-hash", "payload-hash", "[]"),
        )

    with pytest.raises(ValueError, match="retrieval cache payload must be a JSON object"):
        store.get_retrieval_cache(
            run_id="run-1",
            input_hash="input-hash",
            retrieval_config_hash="config-hash",
        )
