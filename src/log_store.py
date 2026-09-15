from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.events import AuditEvent
from src.events import stable_hash_payload
from src.runtime_trace import record_runtime_event


class SQLiteLogStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def setup(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_cache (
                    run_id TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    retrieval_config_hash TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (
                        run_id,
                        input_hash,
                        retrieval_config_hash
                    )
                    )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_source_attempts (
                    run_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    PRIMARY KEY (run_id, source)
                )
                """
            )
        record_runtime_event(
            "artifact",
            target="sqlite_log_store",
            direction="setup",
            artifact_path=self.path,
            payload={"path": str(self.path)},
        )

    def append(self, event: AuditEvent) -> None:
        payload = event.model_dump_json()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_row = connection.execute(
                """
                SELECT event_hash
                FROM audit_events
                WHERE idempotency_key = ?
                """,
                (event.idempotency_key,),
            ).fetchone()
            if existing_row is not None:
                if existing_row["event_hash"] == event.event_hash:
                    return
                raise ValueError(
                    "Conflicting audit event for idempotency_key "
                    f"{event.idempotency_key}: existing event_hash "
                    f"{existing_row['event_hash']} differs from incoming "
                    f"event_hash {event.event_hash}"
                )

            connection.execute(
                """
                INSERT INTO audit_events (
                    idempotency_key,
                    run_id,
                    event_hash,
                    payload
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    event.idempotency_key,
                    event.run_id,
                    event.event_hash,
                    payload,
                ),
            )
        record_runtime_event(
            "artifact",
            run_id=event.run_id,
            thread_id=event.thread_id,
            actor=event.sender_role,
            target="sqlite_audit_events",
            direction="write",
            artifact_path=self.path,
            payload=event.model_dump(mode="json"),
        )

    def list_events(self, run_id: str) -> list[AuditEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload
                FROM audit_events
                WHERE run_id = ?
                ORDER BY sequence ASC
                """,
                (run_id,),
            ).fetchall()

        return [AuditEvent.model_validate_json(row["payload"]) for row in rows]

    def put_retrieval_cache(
        self,
        *,
        run_id: str,
        input_hash: str,
        retrieval_config_hash: str,
        payload: dict,
    ) -> None:
        serialized_payload = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload_hash = stable_hash_payload(payload)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_row = connection.execute(
                """
                SELECT payload_hash
                FROM retrieval_cache
                WHERE run_id = ?
                  AND input_hash = ?
                  AND retrieval_config_hash = ?
                """,
                (run_id, input_hash, retrieval_config_hash),
            ).fetchone()
            if existing_row is not None:
                if existing_row["payload_hash"] == payload_hash:
                    return
                raise ValueError("retrieval replay conflict for input hash")

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
                (
                    run_id,
                    input_hash,
                    retrieval_config_hash,
                    payload_hash,
                    serialized_payload,
                ),
            )
        record_runtime_event(
            "artifact",
            run_id=run_id,
            target="sqlite_retrieval_cache",
            direction="write",
            artifact_path=self.path,
            payload={
                "run_id": run_id,
                "input_hash": input_hash,
                "retrieval_config_hash": retrieval_config_hash,
                "payload_hash": payload_hash,
                "payload": payload,
            },
        )

    def get_retrieval_cache(
        self,
        *,
        run_id: str,
        input_hash: str,
        retrieval_config_hash: str,
    ) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload
                FROM retrieval_cache
                WHERE run_id = ?
                  AND input_hash = ?
                  AND retrieval_config_hash = ?
                """,
                (run_id, input_hash, retrieval_config_hash),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        if not isinstance(payload, dict):
            raise ValueError("retrieval cache payload must be a JSON object")
        record_runtime_event(
            "artifact",
            run_id=run_id,
            target="sqlite_retrieval_cache",
            direction="read_hit",
            artifact_path=self.path,
            payload={
                "run_id": run_id,
                "input_hash": input_hash,
                "retrieval_config_hash": retrieval_config_hash,
                "payload": payload,
            },
        )
        return payload

    def reserve_retrieval_source_attempt(
        self,
        *,
        run_id: str,
        source: str,
        maximum: int,
    ) -> bool:
        """Atomically reserve one durable source attempt for a logical run."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT attempt_count
                FROM retrieval_source_attempts
                WHERE run_id = ? AND source = ?
                """,
                (run_id, source),
            ).fetchone()
            current = int(row["attempt_count"]) if row is not None else 0
            reserved = current < maximum
            next_count = current + 1 if reserved else current
            if reserved and row is None:
                connection.execute(
                    """
                    INSERT INTO retrieval_source_attempts (
                        run_id, source, attempt_count
                    )
                    VALUES (?, ?, ?)
                    """,
                    (run_id, source, next_count),
                )
            elif reserved:
                connection.execute(
                    """
                    UPDATE retrieval_source_attempts
                    SET attempt_count = ?
                    WHERE run_id = ? AND source = ?
                    """,
                    (next_count, run_id, source),
                )
        record_runtime_event(
            "budget",
            run_id=run_id,
            actor="retrieval",
            target=source,
            direction="reserve" if reserved else "reject",
            artifact_path=self.path,
            payload={
                "source": source,
                "attempt_count": next_count,
                "maximum": maximum,
                "reserved": reserved,
            },
        )
        return reserved

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection
