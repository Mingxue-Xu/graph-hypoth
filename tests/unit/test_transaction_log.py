"""Tests for receipts, the append-only transaction log, and deterministic replay.

The suite checks one row per proposal, replayed result hashes, and the absence of dangling
evidence links. A receipt is the six-tuple ``(tx_id, hash(G_i), hash(Δ_i), status,
author, timestamp)``; the result hash lives only on the log row. Tests are LLM-free and
deterministic.
"""

from __future__ import annotations

from src.graph_store import (
    CausalClaimGraphStore,
    EvidenceLink,
    EvidenceRole,
    ExperimentDesign,
    ExperimentPlan,
)
from src.delta import (
    AddEdgeOp,
    AddNodeOp,
    DeltaFamily,
    ExperimentPayload,
    ExtractPayload,
    GraphDeltaProposal,
    VerificationVerdict,
    VerifyPayload,
    build_edge,
    build_node,
)
from src.validator import GraphDeltaValidator
from src.transaction_log import GraphTransactionLog


def _extract(operations, base_hash, **env) -> GraphDeltaProposal:
    return GraphDeltaProposal(
        family=DeltaFamily.EXTRACT, base_graph_hash=base_hash,
        payload=ExtractPayload(operations=operations), **env,
    )


def _two_nodes_one_edge_extract(store):
    n1 = build_node(label="smoking", type="exposure/intervention")
    n2 = build_node(label="cancer", type="outcome")
    edge = build_edge(source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
                      direction="causal", relation_type="increases")
    return _extract(
        [AddNodeOp(node=n1), AddNodeOp(node=n2), AddEdgeOp(edge=edge)], store.base_hash
    ), edge


def _seed_unverified_edge():
    store = CausalClaimGraphStore()
    n1 = build_node(label="smoking", type="exposure/intervention")
    n2 = build_node(label="cancer", type="outcome")
    edge = build_edge(source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
                      direction="causal", relation_type="increases")
    store.nodes[n1.node_id] = n1
    store.nodes[n2.node_id] = n2
    store.edges[edge.edge_id] = edge
    return store, edge


# --- Durable experiment-plan content -----------------------------------------------------
def test_commit_experiment_delta_carries_the_plan_into_the_store_and_result_hash():
    # A committed Δ^experiment MUST (a) leave its validated plan durable on the store and
    # (b) record result_graph_hash = hash(apply(G_i, Δ_i)) — the PLAN-INCLUSIVE hash. Regression
    # guard: `_adopt` formerly dropped `experiment_plans`, silently losing the plan and hashing
    # the result plan-free; commit/replay parity alone could therefore mask the bug.
    v = GraphDeltaValidator(ledger_evidence_ids={"E1"})
    log = GraphTransactionLog()
    store = CausalClaimGraphStore()
    extract, edge = _two_nodes_one_edge_extract(store)
    log.commit(store, extract, v, author="extractor")  # commit the hypothesis edge into the log

    plan = ExperimentPlan(
        hypothesis_under_test=f"edge {edge.edge_id}", design=ExperimentDesign.RANDOMIZED_CONTROLLED
    )
    delta = GraphDeltaProposal(
        family=DeltaFamily.EXPERIMENT, base_graph_hash=store.base_hash,
        payload=ExperimentPayload(
            hypothesis_id=edge.edge_id, experiment_plan=plan, referenced_evidence_ids=["E1"],
        ),
    )
    expected_hash = v.apply(store, delta).content_hash()  # pure apply -> the plan-inclusive next state
    result = log.commit(store, delta, v, author="experiment_designer")

    assert result.accepted is True
    assert edge.edge_id in store.experiment_plans          # the durable plan survives the commit
    assert result.result_graph_hash == expected_hash       # plan-inclusive result hash
    assert store.base_hash == expected_hash
    assert log.replay(v)[-1] == expected_hash              # replay reproduces the plan-inclusive hash


# --- Receipts ----------------------------------------------------------------------------
def test_accepted_commit_advances_version_and_logs_exactly_one_row():
    v, log = GraphDeltaValidator(), GraphTransactionLog()
    store = CausalClaimGraphStore()
    delta, _ = _two_nodes_one_edge_extract(store)
    result = log.commit(store, delta, v, author="extractor")
    assert result.accepted is True
    assert store.version == 1
    assert len(log.rows) == 1
    row = log.rows[0]
    assert row.validation_status == "accepted"
    # The receipt is exactly the six-tuple; the result hash lives on the row, not the receipt.
    assert result.receipt.as_tuple() == (
        row.tx_id, row.base_graph_hash, row.delta_hash, "accepted", "extractor", row.timestamp,
    )
    assert not hasattr(result.receipt, "result_graph_hash")
    assert row.result_graph_hash == store.base_hash  # applied content hash


def test_rejected_commit_leaves_graph_byte_identical_and_still_logs_a_row():
    v, log = GraphDeltaValidator(), GraphTransactionLog()
    store = CausalClaimGraphStore()
    before_hash, before_version = store.base_hash, store.version
    bad = _extract([AddNodeOp(node=build_node(label="x", type="construct"))], "WRONG_BASE")
    result = log.commit(store, bad, v, author="extractor")
    assert result.accepted is False
    assert store.base_hash == before_hash and store.version == before_version
    assert len(log.rows) == 1
    assert log.rows[0].validation_status == "rejected"
    assert "base_hash" in log.rows[0].applied_reason
    assert log.rows[0].result_graph_hash == before_hash  # unchanged


def test_one_row_per_proposal_across_accept_then_reject():
    v, log = GraphDeltaValidator(), GraphTransactionLog()
    store = CausalClaimGraphStore()
    good, _ = _two_nodes_one_edge_extract(store)
    log.commit(store, good, v, author="a")
    bad = _extract([AddNodeOp(node=build_node(label="y", type="construct"))], "STALE_BASE")
    log.commit(store, bad, v, author="a")
    assert len(log.rows) == 2


# --- Idempotency -------------------------------------------------------------------------
def test_idempotency_gate_blocks_recommit_against_the_same_base():
    v, log = GraphDeltaValidator(), GraphTransactionLog()
    store1 = CausalClaimGraphStore()
    delta, _ = _two_nodes_one_edge_extract(store1)
    log.commit(store1, delta, v, author="a")  # commits key K against the empty base
    # A second store still at the SAME (empty) base re-submits the identical delta.
    store2 = CausalClaimGraphStore()
    result = log.commit(store2, delta, v, author="a")
    assert result.accepted is False
    assert result.gate_results.failing_gate == "idempotent"
    assert store2.version == 0  # not double-applied


def test_resubmit_against_advanced_store_does_not_double_apply():
    v, log = GraphDeltaValidator(), GraphTransactionLog()
    store = CausalClaimGraphStore()
    delta, _ = _two_nodes_one_edge_extract(store)
    log.commit(store, delta, v, author="a")
    version_after_first = store.version
    again = log.commit(store, delta, v, author="a")  # base hash now stale
    assert again.accepted is False
    assert store.version == version_after_first


# --- Replay ------------------------------------------------------------------------------
def test_replay_reproduces_every_logged_result_hash():
    v, log = GraphDeltaValidator(), GraphTransactionLog()
    store = CausalClaimGraphStore()
    extract, edge = _two_nodes_one_edge_extract(store)
    log.commit(store, extract, v, author="a")
    verify = GraphDeltaProposal(
        family=DeltaFamily.VERIFY, base_graph_hash=store.base_hash,
        payload=VerifyPayload(edge_id=edge.edge_id, evidence_role=EvidenceRole.SUPPORT,
                              verdict=VerificationVerdict.SUPPORT, confidence=0.7),
    )
    log.commit(store, verify, v, author="b")
    # one interleaved rejection must not perturb replay
    log.commit(store, _extract([AddNodeOp(node=build_node(label="z", type="construct"))], "STALE"),
               v, author="a")

    accepted_result_hashes = [r.result_graph_hash for r in log.rows if r.validation_status == "accepted"]
    assert log.replay(v) == accepted_result_hashes
    assert log.replay(v)[-1] == store.base_hash  # replayed final graph == live graph


# --- Evidence links ----------------------------------------------------------------------
def test_committed_evidence_link_resolves_and_carries_committing_tx_id():
    store, edge = _seed_unverified_edge()
    v = GraphDeltaValidator(ledger_evidence_ids={"E1"})
    log = GraphTransactionLog()
    link = EvidenceLink(
        target_id="t", verification_task_id="vt", evidence_id="E1",
        evidence_role=EvidenceRole.SUPPORT, retrieval_event_id="tc-1",
        committed_transaction_id="UNSET", trust_tier="high",
    )
    verify = GraphDeltaProposal(
        family=DeltaFamily.VERIFY, base_graph_hash=store.base_hash,
        payload=VerifyPayload(edge_id=edge.edge_id, evidence_role=EvidenceRole.SUPPORT,
                              verdict=VerificationVerdict.SUPPORT, evidence_links=[link]),
    )
    result = log.commit(store, verify, v, author="verifier")
    assert result.accepted is True
    assert len(store.evidence_links) == 1
    committed = store.evidence_links[0]
    assert committed.evidence_id == "E1"               # resolves to a ledger record
    assert committed.committed_transaction_id == result.receipt.tx_id  # stamped with committer


def test_verify_with_unknown_ledger_evidence_is_rejected_with_no_dangling_link():
    store, edge = _seed_unverified_edge()
    v = GraphDeltaValidator(ledger_evidence_ids={"E1"})
    log = GraphTransactionLog()
    link = EvidenceLink(
        target_id="t", verification_task_id="vt", evidence_id="E_MISSING",
        evidence_role=EvidenceRole.SUPPORT, retrieval_event_id="tc-1",
        committed_transaction_id="UNSET", trust_tier="high",
    )
    verify = GraphDeltaProposal(
        family=DeltaFamily.VERIFY, base_graph_hash=store.base_hash,
        payload=VerifyPayload(edge_id=edge.edge_id, evidence_role=EvidenceRole.SUPPORT,
                              verdict=VerificationVerdict.SUPPORT, evidence_links=[link]),
    )
    result = log.commit(store, verify, v, author="verifier")
    assert result.accepted is False
    assert result.gate_results.refs is False
    assert store.evidence_links == []


# --- Cost isolation ----------------------------------------------------------------------
def test_cost_is_provably_isolated_from_every_hash_and_decision():
    v = GraphDeltaValidator()

    def commit_once(run_cost):
        store, log = CausalClaimGraphStore(), GraphTransactionLog()
        delta, _ = _two_nodes_one_edge_extract(store)
        return log.commit(store, delta, v, author="a", run_cost=run_cost)

    with_cost = commit_once(run_cost=12.34)
    without_cost = commit_once(run_cost=None)
    assert with_cost.accepted == without_cost.accepted
    assert with_cost.result_graph_hash == without_cost.result_graph_hash
    assert with_cost.receipt.base_hash == without_cost.receipt.base_hash
    assert with_cost.receipt.delta_hash == without_cost.receipt.delta_hash
    # cost is recorded beside the receipt but never enters a hash
    assert with_cost.row.run_cost == 12.34
    assert without_cost.row.run_cost is None


def test_commit_advances_experiment_plans_and_records_the_plan_hash():
    # A committed Δ^experiment must SURVIVE the store-advance: _adopt has to carry experiment_plans
    # onto the live store, else the accepted plan is silently dropped and result_graph_hash records
    # the pre-commit no-op hash instead of ``G_{i+1} = apply(G_i, Δ_i)``.
    store, edge = _seed_unverified_edge()
    log = GraphTransactionLog()
    v = GraphDeltaValidator(ledger_evidence_ids={"E1"})
    plan = ExperimentPlan(
        hypothesis_under_test=f"edge {edge.edge_id}",
        design=ExperimentDesign.RANDOMIZED_CONTROLLED,
    )
    delta = GraphDeltaProposal(
        family=DeltaFamily.EXPERIMENT, base_graph_hash=store.base_hash,
        payload=ExperimentPayload(
            hypothesis_id=edge.edge_id, experiment_plan=plan, referenced_evidence_ids=["E1"],
        ),
    )
    expected_hash = v.apply(store, delta).base_hash  # apply() is pure — the correct post-commit hash (WITH the plan)
    result = log.commit(store, delta, v, author="experiment_designer")
    assert result.accepted
    assert edge.edge_id in store.experiment_plans                    # the plan survived onto the live store
    assert result.result_graph_hash == expected_hash                # not the plan-free no-op hash
    assert store.base_hash == expected_hash
    # No replay assertion here: this fixture seeds the edge directly, outside the log,
    # so replay cannot reconstruct it. Plan-inclusive replay is covered above by
    # test_commit_experiment_delta_carries_the_plan_into_the_store_and_result_hash,
    # which first commits the edge through the log.
