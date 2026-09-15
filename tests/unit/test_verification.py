"""Tests for the deterministic evidence-verification cycle.

The suite covers scheduler selection and continuation, appraiser fusion, evidence
strength and confidence, verdict selection with insufficient and near-tie outcomes,
the overclaim guard, and the transactional verdict-to-status commit path. Every
appraiser sub-signal is a fixture rather than a real LLM call; every coefficient comes
from ``scoring_defaults`` or ``graph_config_defaults``. Floats are checked at ±1e-6.
"""

from __future__ import annotations

import pytest

from src import graph_config_defaults as gcd
from src.cycles import verification as v
from src.delta import DeltaFamily, VerificationVerdict, build_edge, build_node
from src.graph_store import (
    CausalClaimGraphStore,
    CausalEdge,
    EdgeStatus,
    EvidenceRole,
    compute_target_id,
)
from src.retrieval import scoring_defaults as sd
from src.retrieval.association import CandidateEvidence
from src.state import RetrievedEvidence
from src.transaction_log import GraphTransactionLog
from src.validator import GraphDeltaValidator


# =====================================================================================
# ``clip01`` consolidation regression: preserve the public re-export
# from retrieval.similarity, callers must still see `v.clip01`.
# =====================================================================================
def test_clip01_is_still_a_public_name_on_the_module():
    assert v.clip01(-0.2) == 0.0
    assert v.clip01(1.5) == 1.0
    assert v.clip01(0.3) == pytest.approx(0.3, abs=1e-6)


# =====================================================================================
# Verification scheduler
# =====================================================================================
def test_u_target_is_one_for_an_unverified_edge():
    edge = CausalEdge(
        edge_id="e", source_node_ids=["a"], target_node_ids=["b"], direction="causal",
        relation_type="increases", status=EdgeStatus.UNVERIFIED,
    )
    assert v.u_target(edge) == pytest.approx(1.0, abs=1e-6)


def test_u_target_is_one_minus_confidence_for_a_settled_edge():
    edge = CausalEdge(
        edge_id="e", source_node_ids=["a"], target_node_ids=["b"], direction="causal",
        relation_type="increases", status=EdgeStatus.SUPPORTED, confidence=0.8,
    )
    assert v.u_target(edge) == pytest.approx(0.2, abs=1e-6)


def test_gap_evidence_is_fraction_of_missing_roles_clamped():
    assert v.gap_evidence({"support", "mechanism"}, {"support"}) == pytest.approx(0.5, abs=1e-6)
    assert v.gap_evidence({"support"}, set()) == pytest.approx(1.0, abs=1e-6)
    # observed superset of required -> no gap; empty required -> no gap (no divide-by-zero)
    assert v.gap_evidence({"support"}, {"support", "mechanism"}) == pytest.approx(0.0, abs=1e-6)
    assert v.gap_evidence(set(), {"support"}) == pytest.approx(0.0, abs=1e-6)


def test_selection_score_equal_weight_average():
    # 1/3*(0.5 + 1.0 + 0.0) = 0.5 with the equal-weight proposal.
    score = v.selection_score(p_sched=0.5, u_target=1.0, gap_evidence=0.0)
    assert score == pytest.approx(0.5, abs=1e-6)


def test_selection_score_is_clipped_to_unit_interval():
    score = v.selection_score(p_sched=1.0, u_target=1.0, gap_evidence=1.0)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_selection_score_uses_exactly_three_weights_no_w_c():
    # Exactly {w_p, w_u, w_g}; cost is enforced separately as a hard guard.
    assert set(gcd.SEL_WEIGHTS) == {"p", "u", "g"}


def test_select_work_set_keeps_selectable_affordable_above_threshold():
    cands = [
        v.SchedulerCandidate(target_id="t1", selection_score=0.6, cost=1.0),
        v.SchedulerCandidate(target_id="t2", selection_score=0.4, cost=1.0),  # below theta_sel
        v.SchedulerCandidate(target_id="t3", selection_score=0.7, cost=1.0),
    ]
    selected = v.select_work_set(cands, budget=2.0)
    assert [c.target_id for c in selected] == ["t1", "t3"]


def test_select_work_set_excludes_unaffordable_target():
    cands = [v.SchedulerCandidate(target_id="t1", selection_score=0.9, cost=5.0)]
    assert v.select_work_set(cands, budget=2.0) == ()


def test_select_work_set_excludes_unselectable_target():
    cands = [v.SchedulerCandidate(target_id="t1", selection_score=0.9, cost=1.0, selectable=False)]
    assert v.select_work_set(cands, budget=2.0) == ()


def test_continue_loop_true_when_all_conditions_hold():
    assert v.continue_loop(selected_count=2, iteration=0, stall=0) is True


def test_continue_loop_stops_when_no_selectable_target():
    # |P_i| = 0 dominates: stop regardless of iteration index, stall, clarification.
    assert v.continue_loop(selected_count=0, iteration=0, stall=0) is False


def test_continue_loop_stops_at_iteration_cap():
    assert v.continue_loop(selected_count=5, iteration=gcd.N_MAX, stall=0) is False


def test_continue_loop_stops_on_clarification_required():
    assert v.continue_loop(
        selected_count=5, iteration=0, stall=0, clarification_required=True
    ) is False


def test_continue_loop_stops_at_stall_cap():
    assert v.continue_loop(selected_count=5, iteration=0, stall=gcd.N_STALL) is False


# =====================================================================================
# Evidence-appraisal signal fusion
# =====================================================================================
def _coherence(entailment=None, contradiction_risk=None, context_fit=0.0, construct_match=0.0):
    return v.CoherenceSignal(
        entailment=entailment or {},
        contradiction_risk=contradiction_risk or {},
        context_fit=context_fit,
        construct_match=construct_match,
    )


def test_relation_label_set_is_the_closed_six():
    assert {label.value for label in v.RelationLabel} == {
        "supports", "contradicts", "qualifies", "mechanism", "confounder", "not relevant"
    }


def test_rel_score_weighted_clip01_fusion():
    # 0.5*1.0 + 0.2*0.5 + 0.3*0.4 - 0.5*0.0 = 0.72.
    sig = _coherence(entailment={"supports": 1.0}, context_fit=0.5, construct_match=0.4)
    assert v.rel_score(v.RelationLabel.SUPPORTS, sig) == pytest.approx(0.72, abs=1e-6)


def test_rel_score_contradiction_risk_is_subtracted():  # contradiction-risk penalty
    # 0.72 - 0.5*0.4 = 0.52.
    sig = _coherence(
        entailment={"supports": 1.0}, contradiction_risk={"supports": 0.4},
        context_fit=0.5, construct_match=0.4,
    )
    assert v.rel_score(v.RelationLabel.SUPPORTS, sig) == pytest.approx(0.52, abs=1e-6)


def test_relation_label_is_argmax_over_the_closed_set():
    sig = _coherence(entailment={"contradicts": 1.0, "supports": 0.3})
    assert v.relation_label(sig) == v.RelationLabel.CONTRADICTS


def test_relation_label_ties_break_by_fixed_enum_order():  # deterministic argmax
    # supports and contradicts tie at RelScore 0.5; the first in enum order (supports) wins.
    sig = _coherence(entailment={"supports": 1.0, "contradicts": 1.0})
    assert v.relation_label(sig) == v.RelationLabel.SUPPORTS


def test_methods_score_full_design_no_bias():
    sig = v.MethodsSignal(
        design_strength=1.0, measurement_validity=1.0, population_fit=1.0,
        confounder_adjustment=1.0, bias_risk=0.0,
    )
    # 0.35 + 0.20 + 0.15 + 0.30 - 0 = 1.0 (positives sum to 1.0).
    assert v.methods_score(sig) == pytest.approx(1.0, abs=1e-6)


def test_methods_score_bias_penalty_lowers_and_clips():  # bias penalty
    sig = v.MethodsSignal(
        design_strength=0.5, measurement_validity=0.5, population_fit=0.5,
        confounder_adjustment=0.5, bias_risk=1.0,
    )
    # 0.5*(0.35+0.20+0.15+0.30) - 0.40*1.0 = 0.5 - 0.4 = 0.1.
    assert v.methods_score(sig) == pytest.approx(0.1, abs=1e-6)
    only_bias = v.MethodsSignal(0.0, 0.0, 0.0, 0.0, bias_risk=1.0)
    assert v.methods_score(only_bias) == pytest.approx(0.0, abs=1e-6)  # clipped at 0


def test_relmap_support_and_counter_sets():
    L, V = v.RelationLabel, VerificationVerdict
    assert v.supports_verdict(L.SUPPORTS, V.SUPPORT) is True
    assert v.supports_verdict(L.MECHANISM, V.SUPPORT) is True       # mechanism ∈ Support(support)
    assert v.supports_verdict(L.CONTRADICTS, V.SUPPORT) is False
    assert v.counters_verdict(L.CONTRADICTS, V.SUPPORT) is True     # contradicts ∈ Counter(support)
    assert v.counters_verdict(L.CONFOUNDER, V.SUPPORT) is True      # confounder ∈ Counter(support)
    assert v.supports_verdict(L.CONFOUNDER, V.NOT_CAUSAL) is True   # confounder ∈ Support(not_causal)
    assert v.counters_verdict(L.SUPPORTS, V.NOT_CAUSAL) is True
    assert v.supports_verdict(L.NOT_RELEVANT, V.SUPPORT) is False   # not relevant -> ∅
    assert v.counters_verdict(L.QUALIFIES, V.QUALIFY) is False      # Counter(qualify) = ∅


def test_evidence_strength_relation_bonus_present():
    # rel supports the verdict -> indicator 1: 0.45*1 + 0.45*0.8 + 0.10*0.5 = 0.86.
    s = v.evidence_strength(
        rel=v.RelationLabel.SUPPORTS, methods_score=0.8, quote_quality=0.5,
        verdict=VerificationVerdict.SUPPORT,
    )
    assert s == pytest.approx(0.86, abs=1e-6)


def test_evidence_strength_relation_bonus_absent():  # counter evidence
    # rel does NOT support the verdict -> indicator 0: 0 + 0.45*0.8 + 0.10*0.5 = 0.41.
    s = v.evidence_strength(
        rel=v.RelationLabel.CONTRADICTS, methods_score=0.8, quote_quality=0.5,
        verdict=VerificationVerdict.SUPPORT,
    )
    assert s == pytest.approx(0.41, abs=1e-6)


def test_appraiser_weights_resolve_from_scoring_defaults():
    assert sd.RELSCORE_WEIGHTS == {"entail": 0.5, "ctx": 0.2, "construct": 0.3, "contra": 0.5}
    assert sd.SE_WEIGHTS == {"rel": 0.45, "method": 0.45, "quote": 0.10}


# =====================================================================================
# Verdict confidence
# =====================================================================================
def _appraised(evidence_id, rel, methods_score=1.0, quote_quality=1.0, source_id=None):
    return v.AppraisedEvidence(
        evidence_id=evidence_id,
        source_id=source_id if source_id is not None else evidence_id,
        rel=rel,
        methods_score=methods_score,
        quote_quality=quote_quality,
    )


def test_top3_is_zero_for_empty_supporting_set():
    irrelevant = [_appraised("e1", v.RelationLabel.NOT_RELEVANT)]
    assert v.top3(irrelevant, VerificationVerdict.SUPPORT) == pytest.approx(0.0, abs=1e-6)


def test_top3_means_top_three_strongest():
    # four supporting evidences with s_e = 1.0 each (M=Q=1 -> clip01(0.45+0.45+0.10)=1.0);
    # Top3 = mean of the strongest 3 = 1.0.
    ev = [_appraised(f"e{i}", v.RelationLabel.SUPPORTS, source_id=f"s{i}") for i in range(4)]
    assert v.top3(ev, VerificationVerdict.SUPPORT) == pytest.approx(1.0, abs=1e-6)
    assert sd.K_CONF == 3


def test_balance_support_over_support_plus_counter():
    # e1 supports (s_e=1.0); e2 contradicts -> counter for SUPPORT, s_e=0.45*0+0.45+0.10=0.55.
    ev = [
        _appraised("e1", v.RelationLabel.SUPPORTS, source_id="s1"),
        _appraised("e2", v.RelationLabel.CONTRADICTS, source_id="s2"),
    ]
    # 1.0 / (1.0 + 0.55) = 0.64516129.
    assert v.balance(ev, VerificationVerdict.SUPPORT) == pytest.approx(0.64516129, abs=1e-6)


def test_breadth_counts_distinct_sources_in_support_set():
    same = [
        _appraised("e1", v.RelationLabel.SUPPORTS, source_id="shared"),
        _appraised("e2", v.RelationLabel.SUPPORTS, source_id="shared"),
    ]
    diff = [
        _appraised("e1", v.RelationLabel.SUPPORTS, source_id="s1"),
        _appraised("e2", v.RelationLabel.SUPPORTS, source_id="s2"),
    ]
    # K_breadth = 3: one distinct source -> 1/3; two distinct -> 2/3.
    assert v.breadth(same, VerificationVerdict.SUPPORT) == pytest.approx(1 / 3, abs=1e-6)
    assert v.breadth(diff, VerificationVerdict.SUPPORT) == pytest.approx(2 / 3, abs=1e-6)


def test_open_risk_and_limits_saturate():
    assert v.open_risk_term(["r1", "r2"]) == pytest.approx(2 / 3, abs=1e-6)
    assert v.open_risk_term(["r1", "r2", "r3", "r4"]) == pytest.approx(1.0, abs=1e-6)  # min(1,4/3)
    assert v.limits_term(["q1"]) == pytest.approx(1 / 3, abs=1e-6)
    assert v.limits_term([]) == pytest.approx(0.0, abs=1e-6)


def test_confidence_doc_pinned_weighted_sum():
    # one supporting evidence s_e=1.0: Top3=1.0, Balance=1.0, Breadth=1/3, no risks/limits.
    # 0.55*1 + 0.25*1 + 0.20*(1/3) = 0.86666667.
    ev = [_appraised("e1", v.RelationLabel.SUPPORTS)]
    assert v.confidence(ev, VerificationVerdict.SUPPORT) == pytest.approx(0.86666667, abs=1e-6)
    assert sd.CONF_WEIGHTS == {
        "top3": 0.55, "balance": 0.25, "breadth": 0.20, "open_risk": -0.25, "limits": -0.15
    }


def test_confidence_penalties_lower_the_score():
    ev = [_appraised("e1", v.RelationLabel.SUPPORTS)]
    base = v.confidence(ev, VerificationVerdict.SUPPORT)
    # 3 open risks -> OpenRisk=1.0 -> -0.25; 3 qualifiers -> Limits=1.0 -> -0.15.
    penalized = v.confidence(
        ev, VerificationVerdict.SUPPORT,
        open_risks=["r1", "r2", "r3"], qualifiers=["q1", "q2", "q3"],
    )
    assert penalized == pytest.approx(base - 0.25 - 0.15, abs=1e-6)


# =====================================================================================
# Deterministic verdict selection, confidence floor, and near-tie hold
# =====================================================================================
def test_select_verdict_clean_winner():
    ev = [_appraised("e1", v.RelationLabel.SUPPORTS)]  # only support has a supporting set
    result = v.select_verdict(ev)
    assert result.verdict == VerificationVerdict.SUPPORT
    assert result.held is False
    assert result.confidence == pytest.approx(0.86666667, abs=1e-6)


def test_select_verdict_floor_to_insufficient():
    # weak single read pushed below theta_verdict by open risks -> insufficient (no new enum).
    ev = [_appraised("e1", v.RelationLabel.SUPPORTS, methods_score=0.0, quote_quality=0.0)]
    result = v.select_verdict(ev, open_risks=["r1", "r2", "r3"])
    assert result.verdict == VerificationVerdict.INSUFFICIENT
    assert result.held is True
    assert result.hold_reason == "floor"
    assert result.confidence < sd.THETA_VERDICT


def test_select_verdict_every_support_set_empty_is_insufficient():
    ev = [_appraised("e1", v.RelationLabel.NOT_RELEVANT)]
    result = v.select_verdict(ev)
    assert result.verdict == VerificationVerdict.INSUFFICIENT
    assert result.held is True


def test_select_verdict_near_tie_holds_to_insufficient():
    # support and qualify both well above the floor but within delta_margin -> HOLD.
    ev = [
        _appraised("e1", v.RelationLabel.SUPPORTS, methods_score=0.5, quote_quality=1.0, source_id="s1"),
        _appraised("e2", v.RelationLabel.QUALIFIES, methods_score=0.5, quote_quality=0.8, source_id="s2"),
    ]
    result = v.select_verdict(ev)
    conf = result.per_verdict_confidence
    margin = abs(conf[VerificationVerdict.SUPPORT] - conf[VerificationVerdict.QUALIFY])
    assert 0.0 < margin < sd.DELTA_MARGIN                       # a genuine near-tie band
    assert min(conf[VerificationVerdict.SUPPORT],
               conf[VerificationVerdict.QUALIFY]) >= sd.THETA_VERDICT  # both above the floor
    assert result.verdict == VerificationVerdict.INSUFFICIENT   # routed to HOLD (re-verifiable)
    assert result.held is True and result.hold_reason == "near_tie"


def test_select_verdict_exact_tie_holds():
    ev = [
        _appraised("e1", v.RelationLabel.SUPPORTS, source_id="s1"),
        _appraised("e2", v.RelationLabel.QUALIFIES, source_id="s2"),
    ]  # symmetric strong reads -> equal Conf -> margin 0 -> HOLD
    result = v.select_verdict(ev)
    assert result.verdict == VerificationVerdict.INSUFFICIENT
    assert result.held is True and result.hold_reason == "near_tie"


def test_select_verdict_is_reproducible():
    ev = [_appraised("e1", v.RelationLabel.SUPPORTS)]
    first = v.select_verdict(ev)
    second = v.select_verdict(ev)
    assert first.verdict == second.verdict
    assert first.confidence == pytest.approx(second.confidence, abs=1e-6)


# =====================================================================================
# Overclaim guard
# =====================================================================================
def test_overclaim_gap_zero_when_observed_covers_required():
    assert v.overclaim_gap({"support"}, {"support", "mechanism"}) is False


def test_overclaim_gap_one_when_a_required_role_missing():
    assert v.overclaim_gap({"support", "mechanism"}, {"support"}) is True


def test_required_roles_for_verdict_maps_to_primary_role():
    assert v.required_roles_for_verdict(VerificationVerdict.SUPPORT) == frozenset({EvidenceRole.SUPPORT})
    assert v.required_roles_for_verdict(VerificationVerdict.CONTRADICT) == frozenset(
        {EvidenceRole.CONTRADICTION}
    )
    # insufficient is an abstention -> no required role (never flagged as an overclaim).
    assert v.required_roles_for_verdict(VerificationVerdict.INSUFFICIENT) == frozenset()


# =====================================================================================
# Evidence-link construction and verification commit path
# =====================================================================================
def _bundle(evidence_id="ev-1", target_id="t-1", role="support"):
    from src.retrieval.association import EvidenceSignalBundle

    return EvidenceSignalBundle(
        target_id=target_id, verification_task_id="vt-1", evidence_id=evidence_id,
        evidence_role=role, lexical_score=0.4, embedding_score=0.7, citation_score=0.1,
        source_prior=0.5, association_score=0.55, matched_terms=("smoking",),
        matched_anchor="smoking", matched_quote_span="smoking raises risk",
        citation_neighbors=(), scoring_defaults_version=sd.SCORING_DEFAULTS_VERSION,
    )


def _store_with_edge():
    n1 = build_node(label="smoking", type="exposure/intervention")
    n2 = build_node(label="lung cancer", type="outcome")
    edge = build_edge(
        source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
        direction="causal", relation_type="increases",
    )
    store = CausalClaimGraphStore(
        nodes={n1.node_id: n1, n2.node_id: n2}, edges={edge.edge_id: edge}
    )
    return store, edge


def _link(edge, *, evidence_id="ev-1", role=EvidenceRole.SUPPORT):
    target_id = compute_target_id(edge.edge_id, role)
    return v.build_evidence_link(
        target_id=target_id, verification_task_id="vt-1", evidence_id=evidence_id,
        evidence_role=role, signal=_bundle(evidence_id=evidence_id, target_id=target_id),
        retrieval_event_id="tool-call-1", trust_tier="green",
    )


def test_build_evidence_link_carries_the_eight_fields():
    store, edge = _store_with_edge()
    link = _link(edge)
    assert link.target_id == compute_target_id(edge.edge_id, EvidenceRole.SUPPORT)
    assert link.evidence_id == "ev-1"
    assert link.evidence_role == EvidenceRole.SUPPORT
    assert link.retrieval_event_id == "tool-call-1"
    assert link.trust_tier == "green"
    # The complete deterministic signal bundle carries its matched quote span.
    assert link.signal["scoring_defaults_version"] == sd.SCORING_DEFAULTS_VERSION
    assert link.signal["matched_quote_span"] == "smoking raises risk"


def test_build_verify_delta_packages_the_five_tuple():
    store, edge = _store_with_edge()
    delta = v.build_verify_delta(
        base_graph_hash=store.base_hash, edge_id=edge.edge_id,
        evidence_role=EvidenceRole.SUPPORT, verdict=VerificationVerdict.SUPPORT,
        confidence=0.87, evidence_links=[_link(edge)], open_risks=["residual confounding"],
    )
    assert delta.family == DeltaFamily.VERIFY
    assert delta.payload.edge_id == edge.edge_id
    assert delta.payload.verdict == VerificationVerdict.SUPPORT
    assert delta.payload.confidence == pytest.approx(0.87, abs=1e-6)
    assert delta.payload.target_id == compute_target_id(edge.edge_id, EvidenceRole.SUPPORT)
    assert delta.payload.open_risks == ["residual confounding"]


def test_verify_delta_commits_and_moves_edge_to_verdict_status():
    store, edge = _store_with_edge()
    log, validator = GraphTransactionLog(), GraphDeltaValidator(ledger_evidence_ids={"ev-1"})
    delta = v.build_verify_delta(
        base_graph_hash=store.base_hash, edge_id=edge.edge_id,
        evidence_role=EvidenceRole.SUPPORT, verdict=VerificationVerdict.SUPPORT,
        confidence=0.87, evidence_links=[_link(edge)], open_risks=["r"],
    )
    result = log.commit(store, delta, validator, author="causal_evidence_verifier",
                        timestamp="2026-06-14T00:00:00+00:00")
    assert result.accepted is True
    assert store.version == 1
    assert store.edges[edge.edge_id].status == EdgeStatus.SUPPORTED   # unverified -> supported
    assert store.edges[edge.edge_id].confidence == pytest.approx(0.87, abs=1e-6)


def test_committed_evidence_link_resolves_and_carries_committing_tx():
    store, edge = _store_with_edge()
    log, validator = GraphTransactionLog(), GraphDeltaValidator(ledger_evidence_ids={"ev-1"})
    delta = v.build_verify_delta(
        base_graph_hash=store.base_hash, edge_id=edge.edge_id,
        evidence_role=EvidenceRole.SUPPORT, verdict=VerificationVerdict.SUPPORT,
        confidence=0.9, evidence_links=[_link(edge)],
    )
    result = log.commit(store, delta, validator, author="verifier",
                        timestamp="2026-06-14T00:00:00+00:00")
    assert len(store.evidence_links) == 1
    committed = store.evidence_links[0]
    assert committed.evidence_id == "ev-1"                       # resolves to the ledger record
    assert committed.committed_transaction_id == result.receipt.tx_id  # carries its committing tx


def test_verify_delta_rejected_when_evidence_not_in_ledger():
    store, edge = _store_with_edge()
    log, validator = GraphTransactionLog(), GraphDeltaValidator(ledger_evidence_ids=set())
    delta = v.build_verify_delta(
        base_graph_hash=store.base_hash, edge_id=edge.edge_id,
        evidence_role=EvidenceRole.SUPPORT, verdict=VerificationVerdict.SUPPORT,
        confidence=0.9, evidence_links=[_link(edge, evidence_id="ghost")],
    )
    result = log.commit(store, delta, validator, author="verifier",
                        timestamp="2026-06-14T00:00:00+00:00")
    assert result.accepted is False
    assert result.gate_results.failing_gate == "refs"
    assert store.version == 0  # graph unchanged on reject


def test_build_verification_result_wraps_delta_with_derived_lists():
    store, edge = _store_with_edge()
    support_link = _link(edge, evidence_id="ev-1", role=EvidenceRole.SUPPORT)
    counter_link = _link(edge, evidence_id="ev-2", role=EvidenceRole.CONTRADICTION)
    delta = v.build_verify_delta(
        base_graph_hash=store.base_hash, edge_id=edge.edge_id,
        evidence_role=EvidenceRole.SUPPORT, verdict=VerificationVerdict.SUPPORT,
        confidence=0.87, evidence_links=[support_link, counter_link], open_risks=["r1"],
    )
    result = v.build_verification_result(
        proposed_delta=delta, qualifiers=["older adults only"],
    )
    assert result.verdict == VerificationVerdict.SUPPORT
    assert result.confidence == pytest.approx(0.87, abs=1e-6)
    # Evidence and counterevidence are partitioned by each link's evidence role.
    assert result.evidence_ids == ["ev-1"]
    assert result.counterevidence_ids == ["ev-2"]
    assert result.qualifiers == ["older adults only"]
    assert result.open_risks == ["r1"]


# =====================================================================================
# Quote quality
# =====================================================================================
def test_quote_quality_verified_span_vs_summary_fallback():
    # span present: 0.4(span) + 0.2(status) = 0.6 (context/target conservatively 0 here).
    assert v.quote_quality(has_span=True) == pytest.approx(0.6, abs=1e-6)
    # no span -> SummaryFallback penalty -0.5 -> clipped to 0.
    assert v.quote_quality(has_span=False) == pytest.approx(0.0, abs=1e-6)
    # full window coverage with a span saturates to 1.0.
    assert v.quote_quality(
        has_span=True, context_coverage=1.0, target_mention=1.0
    ) == pytest.approx(1.0, abs=1e-6)


# =====================================================================================
# Verification-cycle orchestration with fake agents
# =====================================================================================
def _retrieved(evidence_id, *, source_id="src", quote="exercise reduces mortality"):
    return RetrievedEvidence(
        evidence_id=evidence_id, source="openalex", source_id=source_id,
        title="A cohort study", quote=quote, relevance="high", retrieved_by="retriever",
        tool_call_id=f"tool-{evidence_id}", rank=1, trust_tier="green",
    )


def _candidate(evidence_id, *, target_id, source_id="src"):
    return CandidateEvidence(
        evidence=_retrieved(evidence_id, source_id=source_id),
        bundle=_bundle(evidence_id=evidence_id, target_id=target_id),
    )


class _FakeReviewer:
    """Return fixed per-passage signals and edge-level risks and qualifiers."""

    def __init__(self, coherence, methods, *, open_risks=(), qualifiers=()):
        self._coherence = coherence
        self._methods = methods
        self._appraisal = v.VerifierAppraisal(
            open_risks=tuple(open_risks), qualifiers=tuple(qualifiers)
        )

    def review(self, task, evidences):
        return v.EdgeReview(
            coherence={ev.evidence_id: self._coherence for ev in evidences},
            methods={ev.evidence_id: self._methods for ev in evidences},
            appraisal=self._appraisal,
        )


def test_build_triage_signal_wraps_retrieval_triage_output():
    store, edge = _store_with_edge()
    target_id = compute_target_id(edge.edge_id, EvidenceRole.SUPPORT)
    forwarded = [
        _candidate("ev-1", target_id=target_id, source_id="s1"),
        _candidate("ev-2", target_id=target_id, source_id="s2"),
    ]
    triage = v.build_triage_signal(forwarded, reason="top-K by association_score")
    assert triage.forwarded == tuple(forwarded)        # preserve the ranked top-K set
    assert triage.triage_decision == "forward"
    assert triage.reason == "top-K by association_score"
    assert triage.uncertainty_flag is False


def test_run_verification_cycle_accepts_a_triage_signal():
    store, edge = _store_with_edge()
    task = v.VerificationTask(
        verification_task_id="vt-1", edge_id=edge.edge_id, evidence_role=EvidenceRole.SUPPORT,
    )
    triage = v.build_triage_signal([_candidate("ev-1", target_id=task.target_id)])
    reviewer = _strong_support_reviewer()
    log, validator = GraphTransactionLog(), GraphDeltaValidator(ledger_evidence_ids={"ev-1"})
    cycle = v.run_verification_cycle(
        task, triage, evidence_reviewer=reviewer,
        store=store, log=log, validator=validator, timestamp="2026-06-14T00:00:00+00:00",
    )
    assert cycle.transaction is not None and cycle.transaction.accepted is True
    assert store.edges[edge.edge_id].status == EdgeStatus.SUPPORTED


def _strong_support_reviewer():
    return _FakeReviewer(
        _coherence(entailment={"supports": 1.0}, context_fit=1.0, construct_match=1.0),
        v.MethodsSignal(1.0, 1.0, 1.0, 1.0, 0.0),
    )


def test_run_verification_cycle_commits_a_support_verdict():
    store, edge = _store_with_edge()
    task = v.VerificationTask(
        verification_task_id="vt-1", edge_id=edge.edge_id, evidence_role=EvidenceRole.SUPPORT,
    )
    candidates = [
        _candidate("ev-1", target_id=task.target_id, source_id="s1"),
        _candidate("ev-2", target_id=task.target_id, source_id="s2"),
    ]
    reviewer = _strong_support_reviewer()
    log, validator = GraphTransactionLog(), GraphDeltaValidator(ledger_evidence_ids={"ev-1", "ev-2"})

    cycle = v.run_verification_cycle(
        task, candidates, evidence_reviewer=reviewer, store=store, log=log, validator=validator,
        timestamp="2026-06-14T00:00:00+00:00",
    )
    assert cycle.selection.verdict == VerificationVerdict.SUPPORT
    assert cycle.gap is False
    assert cycle.transaction is not None and cycle.transaction.accepted is True
    assert store.edges[edge.edge_id].status == EdgeStatus.SUPPORTED
    assert len(store.evidence_links) == 2  # one EvidenceLink per target/evidence pair
    assert cycle.result.verdict == VerificationVerdict.SUPPORT


def test_run_verification_cycle_routes_overclaim_to_revision():
    store, edge = _store_with_edge()
    # target is scoped SUPPORT but the evidence reads as a confounder -> v* = not_causal,
    # which requires a CONFOUNDER role the proposed links (SUPPORT) do not cover -> Gap=1.
    task = v.VerificationTask(
        verification_task_id="vt-1", edge_id=edge.edge_id, evidence_role=EvidenceRole.SUPPORT,
    )
    candidates = [_candidate("ev-1", target_id=task.target_id)]
    reviewer = _FakeReviewer(
        _coherence(entailment={"confounder": 1.0}, context_fit=1.0, construct_match=1.0),
        v.MethodsSignal(1.0, 1.0, 1.0, 1.0, 0.0),
    )
    log, validator = GraphTransactionLog(), GraphDeltaValidator(ledger_evidence_ids={"ev-1"})

    cycle = v.run_verification_cycle(
        task, candidates, evidence_reviewer=reviewer, store=store, log=log, validator=validator,
    )
    assert cycle.selection.verdict == VerificationVerdict.NOT_CAUSAL
    assert cycle.gap is True
    assert cycle.transaction is None          # routed to revision, NOT committed
    assert store.version == 0                 # graph unchanged
    assert store.edges[edge.edge_id].status == EdgeStatus.UNVERIFIED


def test_loop_terminates_when_settled_edge_is_no_longer_selectable():
    store, edge = _store_with_edge()
    task = v.VerificationTask(
        verification_task_id="vt-1", edge_id=edge.edge_id, evidence_role=EvidenceRole.SUPPORT,
    )
    reviewer = _strong_support_reviewer()
    log, validator = GraphTransactionLog(), GraphDeltaValidator(ledger_evidence_ids={"ev-1"})
    v.run_verification_cycle(
        task, [_candidate("ev-1", target_id=task.target_id)], evidence_reviewer=reviewer,
        store=store, log=log, validator=validator,
        timestamp="2026-06-14T00:00:00+00:00",
    )
    settled = store.edges[edge.edge_id]
    assert settled.status == EdgeStatus.SUPPORTED
    assert v.is_selectable(settled) is False  # resolved edges leave the selectable set
    work = v.select_work_set(
        [v.SchedulerCandidate(target_id=task.target_id, selection_score=1.0, cost=1.0,
                              selectable=v.is_selectable(settled))],
        budget=10.0,
    )
    assert work == ()
    assert v.continue_loop(selected_count=len(work), iteration=1, stall=0) is False


# =====================================================================================
# Live LLM agents — hermetic via a fake model backend (the real path is the live e2e)
# =====================================================================================
class _FakeBackend:
    def __init__(self, content):
        self._content = content
        self.seen = None

    def run(self, messages, tools=None):
        self.seen = messages
        return {"choices": [{"message": {"content": self._content}}]}


def test_llm_evidence_reviewer_parses_json_per_passage_and_edge_level():
    # One reviewer call returns per-passage coherence and method signals keyed by evidence_id,
    # plus edge-level risks and qualifiers. Sub-signals are clipped into [0,1].
    content = (
        '{"passages": [{"evidence_id": "ev-1", '
        '"entailment": {"supports": 0.9, "contradicts": 0.1}, '
        '"contradiction_risk": {"supports": 0.0}, "context_fit": 1.5, "construct_match": 0.7, '
        '"design_strength": 0.8, "measurement_validity": 0.7, "population_fit": 0.6, '
        '"confounder_adjustment": 0.5, "bias_risk": 0.2}], '
        '"open_risks": ["residual confounding"], "qualifiers": ["older adults only"]}'
    )
    reviewer = v.LLMEvidenceReviewer(_FakeBackend(content))
    task = v.VerificationTask(verification_task_id="vt", edge_id="e", evidence_role=EvidenceRole.SUPPORT)
    review = reviewer.review(task, [_retrieved("ev-1")])
    coherence = review.coherence["ev-1"]
    assert coherence.entailment["supports"] == pytest.approx(0.9, abs=1e-6)
    assert coherence.context_fit == pytest.approx(1.0, abs=1e-6)   # 1.5 clipped into [0,1]
    assert v.relation_label(coherence) == v.RelationLabel.SUPPORTS  # fusion unchanged
    methods = review.methods["ev-1"]
    assert methods.design_strength == pytest.approx(0.8, abs=1e-6)
    assert methods.bias_risk == pytest.approx(0.2, abs=1e-6)
    assert 0.0 <= v.methods_score(methods) <= 1.0
    # Edge-level risk and qualifier lists are returned without a verdict; selection is downstream.
    assert review.appraisal.open_risks == ("residual confounding",)
    assert review.appraisal.qualifiers == ("older adults only",)
    assert not hasattr(review, "verdict")


def test_llm_evidence_reviewer_sees_the_passage_not_the_association_floats():
    backend = _FakeBackend('{"passages": [], "open_risks": [], "qualifiers": []}')
    task = v.VerificationTask(
        verification_task_id="vt", edge_id="e", evidence_role=EvidenceRole.SUPPORT,
        question="does exercise reduce mortality?",
    )
    v.LLMEvidenceReviewer(backend).review(
        task, [_retrieved("ev-1", quote="exercise lowers all-cause mortality")]
    )
    fed = " ".join(str(m.get("content", "")) for m in backend.seen).lower()
    assert "exercise lowers all-cause mortality" in fed         # SEES: the passage verbatim
    assert "ev-1" in fed                                        # the evidence_id marker is echoed
    # The reviewer must not see deterministic association floats or how downstream verdict
    # selection will use its sub-signals; that separation is part of the input contract.
    assert "association_score" not in fed and "0.55" not in fed


def test_llm_evidence_reviewer_uses_full_text_excerpt_when_present():  # full-text preference
    # When a fuller body excerpt is available, the reviewer judges the body rather than
    # the shorter quote snippet.
    long_body = "Detailed methods and results spanning the full text. " * 20
    ev = _retrieved("ev-1", quote="short snippet").model_copy(
        update={"metadata": {"full_text_excerpt": long_body}}
    )
    task = v.VerificationTask(
        verification_task_id="vt", edge_id="e", evidence_role=EvidenceRole.SUPPORT
    )
    backend = _FakeBackend('{"passages": [], "open_risks": [], "qualifiers": []}')
    v.LLMEvidenceReviewer(backend).review(task, [ev])
    prompt = " ".join(str(m.get("content", "")) for m in backend.seen)
    assert long_body[:200] in prompt
