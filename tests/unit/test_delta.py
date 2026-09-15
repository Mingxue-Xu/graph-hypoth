"""Delta envelope, mutation families, operation union, and content-addressed IDs."""

from __future__ import annotations

from src.graph_store import (
    EdgeStatus,
    EvidenceRole,
    ExperimentDesign,
    ExperimentPlan,
)
from src.delta import (
    VERDICT_TO_STATUS,
    AddEdgeOp,
    AddNodeOp,
    DeltaFamily,
    ExperimentPayload,
    ExtractPayload,
    GraphDeltaProposal,
    HypothesisPayload,
    MergeNodesOp,
    PriorityPayload,
    VerificationVerdict,
    VerifyPayload,
    build_edge,
    build_node,
    compute_edge_id,
    compute_node_id,
    norm,
)
from src.retrieval import scoring_defaults as sd


def _extract(operations, base_hash="BASE", **envelope) -> GraphDeltaProposal:
    return GraphDeltaProposal(
        family=DeltaFamily.EXTRACT,
        base_graph_hash=base_hash,
        payload=ExtractPayload(operations=operations),
        **envelope,
    )


def test_norm_lowercases_alnum_tokens_minus_stopwords():
    assert norm("The Smoking of Cigarettes!") == ("smoking", "cigarettes")
    # ``the`` and ``of`` are vendored sklearn stopwords used by scoring defaults.
    assert "the" in sd.STOPWORDS and "of" in sd.STOPWORDS


def test_node_id_is_content_addressed_and_idempotent():  # repeated construction is idempotent
    a = build_node(label="Smoking", definition="cigarette use", type="exposure/intervention")
    b = build_node(label="Smoking", definition="cigarette use", type="exposure/intervention")
    assert a.node_id == b.node_id
    assert a.node_id == compute_node_id("exposure/intervention", "Smoking", "cigarette use")


def test_node_id_ignores_label_case_and_stopwords():
    a = build_node(label="The Smoking", definition="", type="exposure/intervention")
    b = build_node(label="smoking", definition="", type="exposure/intervention")
    assert a.node_id == b.node_id


def test_node_id_differs_on_different_content():  # content sensitivity
    a = build_node(label="smoking", definition="", type="exposure/intervention")
    b = build_node(label="alcohol", definition="", type="exposure/intervention")
    assert a.node_id != b.node_id


def test_edge_id_is_content_addressed_and_endpoint_order_independent():  # canonical endpoint order
    a = build_edge(source_node_ids=["n1", "n2"], target_node_ids=["n3"],
                   direction="causal", relation_type="increases")
    b = build_edge(source_node_ids=["n2", "n1"], target_node_ids=["n3"],
                   direction="causal", relation_type="increases")
    assert a.edge_id == b.edge_id
    assert a.edge_id == compute_edge_id(["n1", "n2"], ["n3"], "causal", "increases")


def test_extract_builder_edges_land_unverified():  # extraction starts without a verdict
    edge = build_edge(source_node_ids=["n1"], target_node_ids=["n2"],
                      direction="causal", relation_type="increases")
    assert edge.status == EdgeStatus.UNVERIFIED


def test_delta_family_set_is_the_closed_five_members():
    assert {f.value for f in DeltaFamily} == {
        "extract", "priority", "verify", "hypothesis", "experiment",
    }


def test_experiment_payload_round_trips_in_the_delta_union():
    plan = ExperimentPlan(hypothesis_under_test="e1", design=ExperimentDesign.ABLATION)
    delta = GraphDeltaProposal(
        family=DeltaFamily.EXPERIMENT,
        base_graph_hash="BASE",
        payload=ExperimentPayload(
            hypothesis_id="e1", experiment_plan=plan, referenced_evidence_ids=["E1"],
        ),
    )
    clone = GraphDeltaProposal.model_validate(delta.model_dump(mode="json"))
    assert type(clone.payload) is ExperimentPayload  # discriminated by kind="experiment"
    assert clone.payload.kind == "experiment"
    assert clone.idempotency_key() == delta.idempotency_key()
    assert clone.delta_hash() == delta.delta_hash()


def test_experiment_retrieval_batch_id_scopes_delta_identity():
    legacy_plan = ExperimentPlan(
        hypothesis_under_test="e1", design=ExperimentDesign.ABLATION
    )
    scoped_plan = legacy_plan.model_copy(
        update={"retrieval_batch_id": "batch-methods-1"}
    )
    legacy = GraphDeltaProposal(
        family=DeltaFamily.EXPERIMENT,
        base_graph_hash="BASE",
        payload=ExperimentPayload(
            hypothesis_id="e1", experiment_plan=legacy_plan,
            referenced_evidence_ids=["E1"],
        ),
    )
    scoped = GraphDeltaProposal(
        family=DeltaFamily.EXPERIMENT,
        base_graph_hash="BASE",
        payload=ExperimentPayload(
            hypothesis_id="e1", experiment_plan=scoped_plan,
            referenced_evidence_ids=["E1"],
        ),
    )

    assert "retrieval_batch_id" not in legacy.payload.experiment_plan.model_dump(mode="json")
    assert scoped.idempotency_key() != legacy.idempotency_key()
    assert scoped.delta_hash() != legacy.delta_hash()


def test_verdict_set_is_the_five_label_panel_b_set():
    assert {v.value for v in VerificationVerdict} == {
        "support", "contradict", "qualify", "insufficient", "not_causal",
    }


def test_verdict_to_status_is_the_one_to_one_map():
    assert VERDICT_TO_STATUS == {
        VerificationVerdict.SUPPORT: EdgeStatus.SUPPORTED,
        VerificationVerdict.CONTRADICT: EdgeStatus.CONTRADICTED,
        VerificationVerdict.QUALIFY: EdgeStatus.QUALIFIED,
        VerificationVerdict.INSUFFICIENT: EdgeStatus.INSUFFICIENT,
        VerificationVerdict.NOT_CAUSAL: EdgeStatus.NOT_CAUSAL,
    }


def test_operation_union_carries_the_three_closed_op_types():  # closed operation union
    n = build_node(label="x", definition="", type="construct")
    e = build_edge(source_node_ids=["n1"], target_node_ids=["n2"],
                   direction="causal", relation_type="increases")
    assert AddNodeOp(node=n).op_type == "add_node"
    assert AddEdgeOp(edge=e).op_type == "add_edge"
    assert MergeNodesOp(survivor_node_id="a", merged_node_id="b").op_type == "merge_nodes"


def test_idempotency_key_is_over_operations_only_not_volatile_metadata():
    n = build_node(label="smoking", definition="", type="exposure/intervention")
    ops = [AddNodeOp(node=n)]
    one = _extract(ops, author_role="extractor", rationale="r1", provenance=[{"p": 1}])
    two = _extract(ops, author_role="DIFFERENT", rationale="r2", provenance=[{"p": 2}])
    # Author / rationale / provenance are excluded -> same key.
    assert one.idempotency_key() == two.idempotency_key()


def test_idempotency_key_changes_with_base_hash_and_operations():
    n = build_node(label="smoking", definition="", type="exposure/intervention")
    base = _extract([AddNodeOp(node=n)], base_hash="BASE_A")
    other_base = _extract([AddNodeOp(node=n)], base_hash="BASE_B")
    other_ops = _extract(
        [AddNodeOp(node=build_node(label="alcohol", definition="", type="exposure/intervention"))],
        base_hash="BASE_A",
    )
    assert base.idempotency_key() != other_base.idempotency_key()
    assert base.idempotency_key() != other_ops.idempotency_key()


def test_proposal_round_trips_through_model_dump_for_replay():
    n = build_node(label="smoking", definition="", type="exposure/intervention")
    verify = GraphDeltaProposal(
        family=DeltaFamily.VERIFY,
        base_graph_hash="BASE",
        payload=VerifyPayload(
            edge_id="e1", evidence_role=EvidenceRole.SUPPORT,
            verdict=VerificationVerdict.SUPPORT, confidence=0.8,
        ),
    )
    for proposal in (_extract([AddNodeOp(node=n)]), verify):
        clone = GraphDeltaProposal.model_validate(proposal.model_dump(mode="json"))
        assert clone.idempotency_key() == proposal.idempotency_key()
        assert clone.delta_hash() == proposal.delta_hash()
        assert type(clone.payload) is type(proposal.payload)


def test_priority_payload_has_no_status_field():  # cannot express a status change
    payload = PriorityPayload(node_or_edge_ids=["n1"], priority_values=[0.9])
    assert not hasattr(payload, "status")


def test_hypothesis_payload_desugars_to_the_operation_union():  # operation-union sugar
    n1 = build_node(label="chronic stress", type="exposure/intervention")
    n2 = build_node(label="hypertension", type="outcome")
    edge = build_edge(source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
                      direction="causal", relation_type="increases")
    ops = HypothesisPayload(new_nodes=[n1, n2], new_edges=[edge]).to_operations()
    # The new_nodes/new_edges surface is sugar over the closed Operation union.
    assert [op.op_type for op in ops] == ["add_node", "add_node", "add_edge"]
    assert [op.node.node_id for op in ops if isinstance(op, AddNodeOp)] == [n1.node_id, n2.node_id]
    assert [op.edge.edge_id for op in ops if isinstance(op, AddEdgeOp)] == [edge.edge_id]
