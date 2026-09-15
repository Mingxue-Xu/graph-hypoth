"""User priority annotation cycle.

The user-event layer on top of the shared delta, validator, and transaction-log core turns an
authored ``UserPriorityAnnotation`` into a typed
priority delta committed through shared validation. A read-only scheduling projection surfaces
priority to verification.

The authored value is stored verbatim on the ``[0,1]`` scale. The schema rejects out-of-range
values. ``P_user`` is never derived from system signals: the projection and builder read only the
authored field.
A priority event carries no status or confidence, so it cannot alter verification truth. The
fallback default is never written back into ``P_user``.

The shared validator and delta modules own the commit boundary, ``PriorityPayload``, and node-only
``user_priority`` storage. This cycle records a user event through validation rather than mutating
graph state directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, field_validator, model_validator

from src.delta import (
    DeltaFamily,
    GraphDeltaProposal,
    PriorityPayload,
)
from src.graph_config_defaults import P_DEFAULT
from src.graph_store import CausalClaimGraphStore, ConceptNode
from src.transaction_log import GraphTransactionLog, TransactionResult
from src.validator import GraphDeltaValidator


class UserPriorityAnnotation(BaseModel):
    """A user-authored priority event for one or more graph targets.

    Authority is limited to priority and focus; the model carries no status or
    confidence field, so a priority annotation can never alter verification truth.
    ``priority_values`` are on the ``[0,1]`` scale and rejected out of range. Each value aligns
    one-to-one with ``target_ids``, and ``author`` is required.
    """

    target_ids: list[str]
    priority_values: list[float]
    author: str
    focus_notes: str = ""

    @field_validator("priority_values")
    @classmethod
    def _within_unit_interval(cls, values: list[float]) -> list[float]:
        """Reject priority values outside ``[0,1]``."""
        for value in values:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"priority value {value} outside the [0,1] priority scale")
        return values

    @model_validator(mode="after")
    def _author_present_and_aligned(self) -> "UserPriorityAnnotation":
        if not self.target_ids:
            raise ValueError("a priority annotation must name at least one target")
        if len(self.target_ids) != len(self.priority_values):
            raise ValueError("target_ids and priority_values must align 1:1")
        if not self.author:
            raise ValueError("author is required")
        return self


def build_priority_from_labels(
    node_labels: Mapping[str, str],
    spec: Sequence[tuple[str, float]],
    *,
    author: str,
    focus_notes: str = "",
) -> UserPriorityAnnotation | None:
    """Build a ``UserPriorityAnnotation`` from a researcher-authored substring->weight ``spec``
    (the standalone profile intake; analogous to the e2e harness's hardcoded priority mapping).

    Each committed node label is matched against the spec substrings IN ORDER — most-specific
    first wins — and the node gets that substring's authored weight; unmatched nodes are omitted
    without fabricating priority. Values are stored verbatim.
    Returns ``None`` when nothing matches, so the caller can simply skip the priority step.
    """
    target_ids: list[str] = []
    priority_values: list[float] = []
    for node_id, label in node_labels.items():
        low = label.lower()
        for substring, weight in spec:
            if substring.lower() in low:
                target_ids.append(node_id)
                priority_values.append(float(weight))
                break  # most-specific substring (listed first) wins
    if not target_ids:
        return None
    return UserPriorityAnnotation(
        target_ids=target_ids, priority_values=priority_values,
        author=author, focus_notes=focus_notes,
    )


def build_priority_delta(
    annotation: UserPriorityAnnotation, *, base_graph_hash: str
) -> GraphDeltaProposal:
    """Map an authored annotation onto a priority delta.

    Maps the authored annotation onto the ``PriorityPayload`` envelope: ``target_ids``
    map to the payload's ``node_or_edge_ids`` and ``author`` rides the envelope's
    ``author_role`` (the receipt author). ``priority_values`` are passed through unchanged.
    """
    return GraphDeltaProposal(
        family=DeltaFamily.PRIORITY,
        base_graph_hash=base_graph_hash,
        payload=PriorityPayload(
            node_or_edge_ids=list(annotation.target_ids),
            priority_values=list(annotation.priority_values),
            focus_notes=annotation.focus_notes,
        ),
        author_role=annotation.author,
    )


def p_sched(target: ConceptNode, *, p_default: float = P_DEFAULT) -> float:
    """Return authored priority when present, otherwise the configured default.

    This never writes the fallback into the authored field. The
    fallback is the configured ``P_DEFAULT`` from the graph defaults module, not a literal
    (Principle 6). ``P_user`` is the node's authored ``user_priority`` and is never derived
    from system signals; this projection reads only that field. ``0.0`` is
    a legitimate authored value, so the set/unset test is ``is not None``, never truthiness.
    """
    p_user = target.user_priority
    return p_user if p_user is not None else p_default


def record_user_priority_annotations(
    store: CausalClaimGraphStore,
    log: GraphTransactionLog,
    validator: GraphDeltaValidator,
    annotation: UserPriorityAnnotation,
    *,
    timestamp: str | None = None,
) -> TransactionResult:
    """Intake: commit a ``UserPriorityAnnotation`` as ``Δ^priority`` through the shared
    transaction boundary.

    The author flows from the annotation onto the receipt; the delta is built against the
    store's current base hash so stale proposals are rejected.
    """
    delta = build_priority_delta(annotation, base_graph_hash=store.base_hash)
    return log.commit(store, delta, validator, author=annotation.author, timestamp=timestamp)
