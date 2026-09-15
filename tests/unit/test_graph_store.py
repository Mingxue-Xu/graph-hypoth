"""Tests for the graph object model and store.

Deterministic, LLM-free. The store content hash reuses the pinned
``stable_hash_payload`` canonicalizer from ``events.py``.
"""

from __future__ import annotations

import pytest

from src.events import stable_hash_payload
from src.graph_store import (
    CausalClaimGraphStore,
    CausalEdge,
    ConceptNode,
    EdgeStatus,
    EvidenceLink,
    EvidenceRole,
    ExperimentDesign,
    ExperimentPlan,
    GraphTarget,
    compute_target_id,
)


def _node(node_id: str = "n1", label: str = "smoking") -> ConceptNode:
    return ConceptNode(node_id=node_id, label=label, definition="cigarette use", type="exposure/intervention")


def _edge(edge_id: str = "e1") -> CausalEdge:
    return CausalEdge(
        edge_id=edge_id,
        source_node_ids=["n1"],
        target_node_ids=["n2"],
        direction="causal",
        relation_type="increases",
    )


def test_concept_node_has_required_fields_and_no_status():
    node = _node()
    for field in (
        "node_id", "label", "aliases", "definition", "type",
        "scope_qualifiers", "provenance", "user_priority", "uncertainty",
    ):
        assert hasattr(node, field), field
    # Status lives on ``CausalEdge`` only.
    assert not hasattr(node, "status")


def test_causal_edge_required_fields():
    edge = _edge()
    for field in (
        "edge_id", "source_node_ids", "target_node_ids", "direction",
        "relation_type", "mechanism", "conditions", "confounders",
        "status", "confidence", "open_risks",
    ):
        assert hasattr(edge, field), field
    # New edges land unverified by default.
    assert edge.status == EdgeStatus.UNVERIFIED


def test_edge_status_enum_is_the_closed_six_member_set():
    assert {s.value for s in EdgeStatus} == {
        "unverified", "supported", "contradicted",
        "qualified", "insufficient", "not_causal",
    }


def test_evidence_role_enum_is_the_closed_relation_role_set():
    assert {r.value for r in EvidenceRole} == {
        "support", "contradiction", "qualification", "mechanism", "confounder",
    }


def test_empty_store_is_version_zero_with_deterministic_hash():  # empty-graph edge case
    a = CausalClaimGraphStore()
    b = CausalClaimGraphStore()
    assert a.version == 0
    # An empty graph at version 0 has a stable, reproducible content hash.
    assert a.base_hash == b.base_hash
    assert isinstance(a.base_hash, str) and len(a.base_hash) == 64


def test_adding_a_node_changes_the_content_hash():
    store = CausalClaimGraphStore()
    before = store.base_hash
    store.nodes[_node().node_id] = _node()
    assert store.base_hash != before


def test_content_hash_is_independent_of_evidence_link_insertion_order():  # content-addressed
    link_a = EvidenceLink(
        target_id="t1", verification_task_id="vt1", evidence_id="E1",
        evidence_role=EvidenceRole.SUPPORT, retrieval_event_id="tc1",
        committed_transaction_id="tx-1", trust_tier="high",
    )
    link_b = EvidenceLink(
        target_id="t2", verification_task_id="vt2", evidence_id="E2",
        evidence_role=EvidenceRole.MECHANISM, retrieval_event_id="tc2",
        committed_transaction_id="tx-1", trust_tier="medium",
    )
    forward = CausalClaimGraphStore(evidence_links=[link_a, link_b])
    reverse = CausalClaimGraphStore(evidence_links=[link_b, link_a])
    assert forward.base_hash == reverse.base_hash


def test_graph_target_id_is_stable_hash_of_ref_and_role():
    target = GraphTarget(
        node_or_edge_id="e1",
        target_type="edge",
        evidence_role=EvidenceRole.SUPPORT,
    )
    # target_id is an opaque, replay-stable handle derived from the pinned hasher;
    # the node/edge ref and role remain separate explicit fields (never parsed out).
    assert target.target_id == compute_target_id("e1", EvidenceRole.SUPPORT)
    assert target.target_id == stable_hash_payload(
        {"node_or_edge_id": "e1", "evidence_role": "support"}
    )
    assert target.node_or_edge_id == "e1"
    assert target.evidence_role == EvidenceRole.SUPPORT


def test_same_ref_different_role_yields_distinct_target_ids():  #  boundary
    assert compute_target_id("e1", EvidenceRole.SUPPORT) != compute_target_id(
        "e1", EvidenceRole.MECHANISM
    )


def test_evidence_link_carries_the_committed_bridge_fields():
    link = EvidenceLink(
        target_id="t1", verification_task_id="vt1", evidence_id="E1",
        evidence_role=EvidenceRole.SUPPORT, retrieval_event_id="tc1",
        committed_transaction_id="tx-7", trust_tier="high",
    )
    for field in (
        "target_id", "verification_task_id", "evidence_id", "evidence_role",
        "signal", "retrieval_event_id", "committed_transaction_id", "trust_tier",
    ):
        assert hasattr(link, field), field
    # The body stays in the ledger; the link carries no copied paper object.
    assert not hasattr(link, "quote")
    assert not hasattr(link, "abstract")


def test_evidence_role_required_with_no_default():  # no ``direct`` default
    with pytest.raises(Exception):
        GraphTarget(node_or_edge_id="e1", target_type="edge")  # missing evidence_role


# --- Durable experiment-plan content -----------------------------------------------------
def _plan(hut: str = "edge e1: X --increases--> Y") -> ExperimentPlan:
    return ExperimentPlan(
        hypothesis_under_test=hut,
        design=ExperimentDesign.RANDOMIZED_CONTROLLED,
    )


def test_experiment_design_enum_is_the_closed_five_member_set():
    assert {d.value for d in ExperimentDesign} == {
        "randomized_controlled", "controlled_observational", "ablation",
        "benchmark_comparison", "simulation",
    }


def test_empty_experiment_plans_are_excluded_from_the_content_hash():  # byte-identity
    # A plans-free graph MUST hash byte-identically to the pre-Δ^experiment store: the empty
    # experiment_plans map must NOT enter the content-hash payload (flag-OFF byte-identity).
    store = CausalClaimGraphStore(nodes={"n1": _node()})
    expected = stable_hash_payload({
        "nodes": {"n1": _node().model_dump(mode="json")},
        "edges": {},
        "evidence_links": [],
        "scope_context": {},
    })
    assert store.experiment_plans == {}
    assert store.content_hash() == expected


def test_storing_an_experiment_plan_changes_the_content_hash():  # content-addressed
    store = CausalClaimGraphStore(edges={"e1": _edge()})
    before = store.content_hash()
    store.experiment_plans["e1"] = _plan()
    assert store.content_hash() != before
