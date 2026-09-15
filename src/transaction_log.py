"""Audit receipts, the append-only transaction log, and replay.

Transaction rows retain base, result, and delta hashes. The validator is injected to avoid an
import cycle. Cost metadata is deliberately excluded from hashes and validation decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from src.delta import GraphDeltaProposal
from src.graph_store import CausalClaimGraphStore
from src.validator import GateResults, GraphDeltaValidator


@dataclass(frozen=True)
class Receipt:
    """``receipt_i = (tx_id, hash(G_i), hash(Δ_i), status, author, timestamp)`` (CT-05).

    Carries only the base and delta hashes; the result hash lives on the log row.
    """

    tx_id: str
    base_hash: str
    delta_hash: str
    status: str
    author: str
    timestamp: str

    def as_tuple(self) -> tuple[str, str, str, str, str, str]:
        return (self.tx_id, self.base_hash, self.delta_hash, self.status, self.author, self.timestamp)


class TransactionRow(BaseModel):
    """One append-only log row: the base/result/delta triple plus the receipt,
    the reason, the idempotency key, and the delta itself (so replay can re-apply)."""

    tx_id: str
    base_graph_hash: str
    result_graph_hash: str
    delta_hash: str
    validation_status: str
    author: str
    timestamp: str
    receipt: dict[str, Any]
    applied_reason: str
    idempotency_key: str
    delta: GraphDeltaProposal
    # Cost is recorded beside the receipt and excluded from every hash.
    run_cost: float | None = None


@dataclass(frozen=True)
class TransactionResult:
    accepted: bool
    receipt: Receipt
    gate_results: GateResults
    result_graph_hash: str
    row: TransactionRow


class GraphTransactionLog(BaseModel):
    """Append-only, replayable record of every accepted or rejected transaction."""

    rows: list[TransactionRow] = Field(default_factory=list)

    def committed_idempotency_keys(self) -> set[str]:
        return {row.idempotency_key for row in self.rows if row.validation_status == "accepted"}

    def commit(
        self,
        store: CausalClaimGraphStore,
        delta: GraphDeltaProposal,
        validator: GraphDeltaValidator,
        *,
        author: str,
        timestamp: str | None = None,
        run_cost: float | None = None,
    ) -> TransactionResult:
        """Evaluate ``Accept_i``; on accept apply + advance the store; always log one row."""
        tx_id = f"tx-{len(self.rows):06d}"
        timestamp = timestamp or datetime.now(UTC).isoformat()
        base_hash = store.base_hash
        delta_hash = delta.delta_hash()
        decision = validator.evaluate(
            store, delta, committed_keys=self.committed_idempotency_keys()
        )

        if decision.accepted:
            applied = validator.apply(store, delta, transaction_id=tx_id)
            _adopt(store, applied)
            result_hash = store.base_hash
            status, reason = "accepted", "applied"
        else:
            result_hash = base_hash  # A rejected proposal leaves the graph byte-identical.
            status, reason = "rejected", f"gate_failed:{decision.failing_gate}"

        receipt = Receipt(
            tx_id=tx_id, base_hash=base_hash, delta_hash=delta_hash,
            status=status, author=author, timestamp=timestamp,
        )
        row = TransactionRow(
            tx_id=tx_id, base_graph_hash=base_hash, result_graph_hash=result_hash,
            delta_hash=delta_hash, validation_status=status, author=author,
            timestamp=timestamp, receipt=dict(zip(
                ("tx_id", "base_hash", "delta_hash", "status", "author", "timestamp"),
                receipt.as_tuple(),
            )),
            applied_reason=reason, idempotency_key=delta.idempotency_key(),
            delta=delta, run_cost=run_cost,
        )
        self.rows.append(row)
        return TransactionResult(
            accepted=decision.accepted, receipt=receipt, gate_results=decision,
            result_graph_hash=result_hash, row=row,
        )

    def replay(self, validator: GraphDeltaValidator) -> list[str]:
        """Re-apply the accepted deltas in order from version 0; return their result hashes.

        AUDIT-ONLY: re-applies past committed deltas to verify the version history
        reproduces the recorded results. A mismatch is a real bug.
        """
        store = CausalClaimGraphStore()
        hashes: list[str] = []
        for row in self.rows:
            if row.validation_status != "accepted":
                continue
            applied = validator.apply(store, row.delta, transaction_id=row.tx_id)
            _adopt(store, applied)
            hashes.append(store.base_hash)
        return hashes


def _adopt(store: CausalClaimGraphStore, applied: CausalClaimGraphStore) -> None:
    """Advance ``store`` in place to the applied next-state (used by commit + replay)."""
    store.nodes = applied.nodes
    store.edges = applied.edges
    store.evidence_links = applied.evidence_links
    store.scope_context = applied.scope_context
    store.experiment_plans = applied.experiment_plans
    store.version = applied.version
