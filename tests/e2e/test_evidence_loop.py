"""Live Evidence Reviewer loop: real LLM agents drive one target to a committed verdict.

The deterministic relation-scoring, verdict-selection, and commit layers are exhaustively
covered by ``tests/unit/test_verification.py`` over FIXTURED sub-signals; this test exercises
the ONE thing the unit suite cannot — the real LLM-backed bounded-judgment agents producing
their sub-signals end-to-end (Evidence Coherence Judge, Methods/Measurement Appraiser,
and Causal Evidence Verifier), then asserts only label-agnostic structural
invariants (the model is stochastic; never assert its exact text/verdict — determinism draft).

Cross-run committed-decision stability is covered by a pre-registered flip-rate release
gate run outside
``pytest -q`` — not asserted here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.cycles import verification as v
from src.delta import VerificationVerdict, build_edge, build_node
from src.graph_store import (
    CausalClaimGraphStore,
    EdgeStatus,
    EvidenceRole,
)
from src.retrieval.association import CandidateEvidence, EvidenceSignalBundle
from src.retrieval import scoring_defaults as sd
from src.state import RetrievedEvidence
from src.transaction_log import GraphTransactionLog
from src.validator import GraphDeltaValidator


def _candidate(evidence_id: str, *, target_id: str, source_id: str, quote: str) -> CandidateEvidence:
    evidence = RetrievedEvidence(
        evidence_id=evidence_id, source="openalex", source_id=source_id,
        title="Prospective cohort study", quote=quote, relevance="high",
        retrieved_by="retriever", tool_call_id=f"tool-{evidence_id}", rank=1, trust_tier="green",
    )
    bundle = EvidenceSignalBundle(
        target_id=target_id, verification_task_id="vt-evidence-loop", evidence_id=evidence_id,
        evidence_role="support", lexical_score=0.4, embedding_score=0.7, citation_score=0.1,
        source_prior=0.5, association_score=0.55, matched_terms=("exercise", "mortality"),
        matched_anchor="exercise", matched_quote_span=quote, citation_neighbors=(),
        scoring_defaults_version=sd.SCORING_DEFAULTS_VERSION,
    )
    return CandidateEvidence(evidence=evidence, bundle=bundle)


@pytest.mark.live
def test_live_evidence_loop_verifies_an_edge():
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is required for the live evidence-review agents")

    from src.camel_adapter import _create_camel_model_backend
    from src.config import load_config

    config = load_config(Path("config/evidence-evaluation.yaml"))
    backend = _create_camel_model_backend(
        config.agents.skeptical_verifier, role_name="skeptical_verifier"
    )

    # A committed unverified edge, as the extraction cycle would leave it, plus its support target.
    cause = build_node(label="regular physical activity", type="exposure/intervention")
    effect = build_node(label="all-cause mortality", type="outcome")
    edge = build_edge(
        source_node_ids=[cause.node_id], target_node_ids=[effect.node_id],
        direction="causal", relation_type="reduces",
    )
    store = CausalClaimGraphStore(
        nodes={cause.node_id: cause, effect.node_id: effect}, edges={edge.edge_id: edge}
    )
    task = v.VerificationTask(
        verification_task_id="vt-evidence-loop", edge_id=edge.edge_id,
        evidence_role=EvidenceRole.SUPPORT,
        question="Does regular physical activity reduce all-cause mortality in older adults?",
    )
    # Two CLEAN, unconditional causal-support passages on a SUPPORT-role target — chosen so the
    # real agents reliably yield a verdict in the COMMITTING set {support, insufficient} (both
    # land a verdict and move the edge off `unverified`; only qualify/contradict/not_causal
    # would overclaim a SUPPORT target and route to revision). We do NOT assert the exact label
    # (the model is stochastic) — only that an edge actually gets verified.
    candidates = [
        _candidate(
            "ev-1", target_id=task.target_id, source_id="s1",
            quote=(
                "In a randomized controlled trial, a structured physical-activity program "
                "reduced all-cause mortality compared with usual care over five years of "
                "follow-up."
            ),
        ),
        _candidate(
            "ev-2", target_id=task.target_id, source_id="s2",
            quote=(
                "A second randomized trial confirmed that increasing regular physical activity "
                "lowered all-cause mortality in older adults relative to a sedentary control arm."
            ),
        ),
    ]
    log = GraphTransactionLog()
    validator = GraphDeltaValidator(ledger_evidence_ids={"ev-1", "ev-2"})

    cycle = v.run_verification_cycle(
        task, candidates,
        evidence_reviewer=v.LLMEvidenceReviewer(backend),
        store=store, log=log, validator=validator,
        timestamp="2026-06-14T00:00:00+00:00",
    )

    # The real agents drove a claim to verified edges by committing a
    # COMMITTED Δ^verify and the edge actually left `unverified`. No gap-escape: a route-to-
    # A revision outcome here means nothing was verified and is a genuine failure.
    assert cycle.gap is False, (
        f"overclaim guard routed v*={cycle.selection.verdict} to revision — no edge verified"
    )
    assert cycle.transaction is not None and cycle.transaction.accepted is True
    assert store.version == 1
    assert store.edges[edge.edge_id].status != EdgeStatus.UNVERIFIED  # the edge is now verified
    # label-agnostic: the committing verdicts under a SUPPORT target are exactly these two.
    assert cycle.selection.verdict in {
        VerificationVerdict.SUPPORT, VerificationVerdict.INSUFFICIENT
    }
    assert len(store.evidence_links) == len(candidates)  # one EvidenceLink per (target, evidence)
    tx_id = cycle.transaction.receipt.tx_id
    assert all(link.committed_transaction_id == tx_id for link in store.evidence_links)
