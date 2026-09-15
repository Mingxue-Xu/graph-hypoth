"""Tests for the user-priority annotation cycle.

The cycle builds on the transaction core (delta, validator, and log). It proves that
authored values are stored verbatim; ``P_sched`` falls back to ``P_default`` without
writing it back; authored priorities drive ``P_sched``; system signals never derive
``P_user``; commits are auditable; and priority events never alter verification status
or confidence.

The commit boundary, ``PriorityPayload``, and node-only ``user_priority`` storage are
covered in ``test_validator.py`` and ``test_transaction_log.py``. This file tests:
``UserPriorityAnnotation`` (schema), ``build_priority_delta`` (the ``Δ^priority``
builder), ``p_sched`` (the projection), and ``record_user_priority_annotations`` (the
intake). LLM-free and deterministic; floats compared at ±1e-6.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from src.cycles.priority import (
    UserPriorityAnnotation,
    build_priority_delta,
    p_sched,
    record_user_priority_annotations,
)
from src.delta import (
    DeltaFamily,
    GraphDeltaProposal,
    PriorityPayload,
    build_edge,
    build_node,
)
from src.graph_config_defaults import P_DEFAULT
from src.graph_store import (
    CausalClaimGraphStore,
    ConceptNode,
    EdgeStatus,
)
from src.transaction_log import GraphTransactionLog
from src.validator import GraphDeltaValidator


def _store_with_node(label: str = "smoking", **node_overrides):
    store = CausalClaimGraphStore()
    node = build_node(label=label, type="exposure/intervention")
    for field, value in node_overrides.items():
        setattr(node, field, value)
    store.nodes[node.node_id] = node
    return store, node.node_id


def _commit(store, annotation):
    log, validator = GraphTransactionLog(), GraphDeltaValidator()
    result = record_user_priority_annotations(store, log, validator, annotation)
    return result, log


# --- The [0,1] schema guard at both the annotation and transaction gates ----------------
def test_priority_values_outside_unit_interval_rejected_at_annotation_schema():
    # The user-facing priority scale is [0,1]; the annotation schema rejects out-of-range
    # at the authoring entry point so nothing out of range is ever authored.
    with pytest.raises(ValidationError):
        UserPriorityAnnotation(target_ids=["n1"], priority_values=[1.5], author="alice")
    with pytest.raises(ValidationError):
        UserPriorityAnnotation(target_ids=["n1"], priority_values=[-0.01], author="alice")
    # the interval endpoints are valid
    ann = UserPriorityAnnotation(target_ids=["n1"], priority_values=[0.0], author="alice")
    assert ann.priority_values == [0.0]


def test_raw_priority_delta_out_of_range_rejected_by_schema_gate():  # schema gate
    # The [0,1] guard must also live at the transaction schema gate, not only at the
    # annotation: a raw PriorityPayload that bypasses UserPriorityAnnotation must still be
    # rejected by the validator and never applied.
    store, node_id = _store_with_node()
    delta = GraphDeltaProposal(
        family=DeltaFamily.PRIORITY, base_graph_hash=store.base_hash,
        payload=PriorityPayload(node_or_edge_ids=[node_id], priority_values=[1.5]),
    )
    decision = GraphDeltaValidator().evaluate(store, delta)
    assert decision.schema is False
    assert decision.accept_i == 0
    # through the commit boundary: rejected, graph left unchanged (not applied)
    log = GraphTransactionLog()
    result = log.commit(store, delta, GraphDeltaValidator(), author="alice")
    assert result.accepted is False
    assert store.version == 0
    assert store.nodes[node_id].user_priority is None


def test_priority_payload_rejects_smuggled_status_field():  # closed authority
    # A priority delta has no authority over verification status. A smuggled status
    # field must be REJECTED at the schema, not silently dropped — otherwise a caller could
    # believe it set a status that was quietly discarded.
    with pytest.raises(ValidationError):
        PriorityPayload(node_or_edge_ids=["n"], priority_values=[0.5], status="contradicted")


def test_in_range_priority_stored_verbatim_as_user_priority():  # identity projection
    store, node_id = _store_with_node()
    ann = UserPriorityAnnotation(
        target_ids=[node_id], priority_values=[0.7], author="alice", focus_notes="check first"
    )
    result, log = _commit(store, ann)
    assert result.accepted is True
    # stored exactly, no transformation (identity projection)
    assert math.isclose(store.nodes[node_id].user_priority, 0.7, abs_tol=1e-6)
    # The change is an auditable, accepted transaction row.
    assert len(log.rows) == 1
    assert log.rows[0].validation_status == "accepted"


# --- The P_sched projection --------------------------------------------------------------
def test_p_sched_returns_authored_priority_when_set():  # P_sched = P_user
    node = ConceptNode(node_id="n", label="x", user_priority=0.7)
    assert math.isclose(p_sched(node), 0.7, abs_tol=1e-6)
    # 0.0 is a legitimate authored priority and must NOT trigger the fallback
    node_zero = ConceptNode(node_id="z", label="z", user_priority=0.0)
    assert math.isclose(p_sched(node_zero), 0.0, abs_tol=1e-6)


def test_p_sched_falls_back_to_p_default_without_writeback():
    node = ConceptNode(node_id="n", label="x")  # user_priority unset
    assert math.isclose(p_sched(node), P_DEFAULT, abs_tol=1e-6)
    assert math.isclose(P_DEFAULT, 0.5, abs_tol=1e-6)  # imported default, not a local literal
    # reading the projection never writes the default back into the authored field
    p_sched(node)
    p_sched(node)
    assert node.user_priority is None


# --- The exclusion invariant -------------------------------------------------------------
def test_priority_excluded_from_system_signals():
    # A target with high uncertainty but NO authored priority: nothing derives user_priority
    # from uncertainty/gap/cost/assoc, and P_sched falls back to the neutral default rather
    # than a function of those signals.
    node = ConceptNode(node_id="n", label="x", uncertainty=0.99)
    assert node.user_priority is None
    assert math.isclose(p_sched(node), P_DEFAULT, abs_tol=1e-6)  # == default, NOT f(uncertainty)
    assert node.user_priority is None  # still unset; no signal written into the authored field
    # The Δ^priority builder's only priority source is the authored values (no signal inputs).
    ann = UserPriorityAnnotation(target_ids=["n"], priority_values=[0.3], author="alice")
    delta = build_priority_delta(ann, base_graph_hash="BASE")
    assert delta.payload.priority_values == [0.3]


# --- A priority event never alters verification truth -----------------------------------
def test_priority_commit_leaves_verification_status_untouched():
    store = CausalClaimGraphStore()
    n1 = build_node(label="smoking", type="exposure/intervention")
    n2 = build_node(label="cancer", type="outcome")
    edge = build_edge(
        source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
        direction="causal", relation_type="increases",
    )
    edge.status = EdgeStatus.SUPPORTED
    edge.confidence = 0.82
    store.nodes[n1.node_id] = n1
    store.nodes[n2.node_id] = n2
    store.edges[edge.edge_id] = edge

    ann = UserPriorityAnnotation(target_ids=[n1.node_id], priority_values=[0.9], author="alice")
    result, _ = _commit(store, ann)
    assert result.accepted is True
    assert store.edges[edge.edge_id].status == EdgeStatus.SUPPORTED  # unchanged
    assert math.isclose(store.edges[edge.edge_id].confidence, 0.82, abs_tol=1e-6)  # unchanged
    # The annotation has no authority to carry status or confidence.
    assert "status" not in UserPriorityAnnotation.model_fields
    assert "confidence" not in UserPriorityAnnotation.model_fields


# --- Done-when: priority reorders the scheduler queue ----------------------------------
def test_p_sched_reorders_targets_by_authored_priority():  # Done-when (reorder scheduler queue)
    a = ConceptNode(node_id="a", label="a")  # unset -> P_default (0.5)
    b = ConceptNode(node_id="b", label="b", user_priority=0.9)
    c = ConceptNode(node_id="c", label="c", user_priority=0.1)
    order = sorted([a, b, c], key=p_sched, reverse=True)
    assert [n.node_id for n in order] == ["b", "a", "c"]  # 0.9 > 0.5 (default) > 0.1


# --- The Δ^priority builder ---------------------------------------------------------------
def test_build_priority_delta_maps_payload_and_requires_aligned_author():
    store, node_id = _store_with_node()
    ann = UserPriorityAnnotation(
        target_ids=[node_id], priority_values=[0.4], author="alice", focus_notes="why"
    )
    delta = build_priority_delta(ann, base_graph_hash=store.base_hash)
    assert delta.family is DeltaFamily.PRIORITY
    assert delta.payload.node_or_edge_ids == [node_id]  # target_ids -> node_or_edge_ids
    assert delta.payload.priority_values == [0.4]
    assert delta.payload.focus_notes == "why"
    assert delta.author_role == "alice"  # author rides the envelope (the receipt author)
    # A well-formed priority delta passes the transaction schema gate.
    assert GraphDeltaValidator().evaluate(store, delta).schema is True
    # The schema enforces a required author and one-to-one target/value alignment.
    with pytest.raises(ValidationError):
        UserPriorityAnnotation(target_ids=[node_id, "x"], priority_values=[0.4], author="alice")
    with pytest.raises(ValidationError):
        UserPriorityAnnotation(target_ids=[node_id], priority_values=[0.4], author="")


# --- build_priority_from_labels: substring spec -> annotation (standalone profile input) --
def test_build_priority_from_labels_matches_most_specific_first():  # standalone priority intake
    from src.cycles.priority import build_priority_from_labels

    spec = [("calibration data", 0.86), ("calibration", 0.85)]
    nodes = {"n1": "Calibration data trade-offs", "n2": "model calibration error"}
    ann = build_priority_from_labels(nodes, spec, author="Mingxue Xu", focus_notes="behavioral first")
    by_id = dict(zip(ann.target_ids, ann.priority_values))
    assert by_id["n1"] == 0.86  # most-specific substring (listed first) wins
    assert by_id["n2"] == 0.85
    assert ann.author == "Mingxue Xu" and ann.focus_notes == "behavioral first"


def test_build_priority_from_labels_skips_unmatched_nodes():
    from src.cycles.priority import build_priority_from_labels

    spec = [("calibration", 0.85)]
    ann = build_priority_from_labels({"n1": "calibration", "n2": "kv cache size"}, spec, author="X")
    assert ann.target_ids == ["n1"]  # the unmatched node is omitted


def test_build_priority_from_labels_returns_none_when_nothing_matches():
    from src.cycles.priority import build_priority_from_labels

    ann = build_priority_from_labels({"n1": "kv cache"}, [("calibration", 0.85)], author="X")
    assert ann is None
