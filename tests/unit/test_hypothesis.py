"""Research Synthesist hypothesis-expansion behavior.

The suite exercises the fail-closed eligibility gate, deterministic signal fusion,
hypothesis ranking under a budget, unverified graph commits, and the disabled path.
Every threshold and weight comes from ``graph_config_defaults``; floating-point
results are checked to within 1e-6.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src import graph_config_defaults as gcd
from src.cycles.hypothesis import (
    HypothesisCandidate,
    build_hypothesis_candidates,
    build_hypothesis_delta,
    candidate_anchor_text,
    det_duplication,
    det_novelty,
    det_scope_fit,
    eligible_ranked,
    fuse_subsignal,
    gate_h,
    hyp_score,
    rank_score,
    run_hypothesis_cycle,
)
from src.delta import DeltaFamily, build_edge, build_node
from src.graph_store import CausalClaimGraphStore, EdgeStatus


# --- deterministic stub embedder (LLM/model-free idiom; mirrors test_extraction) --------
def _stub_embedder(mapping):
    def embed(texts):
        return [mapping[t] for t in texts]

    return embed


def _candidate(candidate_id, *, novelty=0.9, testability=0.9, scope_fit=0.9, duplication=0.1,
               plausibility=0.9, expected_yield=0.9, centrality=0.9, mechanism_specificity=0.9,
               new_nodes=(), new_edges=()):
    return HypothesisCandidate(
        candidate_id=candidate_id, new_nodes=tuple(new_nodes), new_edges=tuple(new_edges),
        novelty=novelty, testability=testability, scope_fit=scope_fit, duplication=duplication,
        plausibility=plausibility, expected_yield=expected_yield, centrality=centrality,
        mechanism_specificity=mechanism_specificity,
    )


def _two_node_candidate(candidate_id, src_label="chronic stress", tgt_label="hypertension",
                        **signals):
    """A committable candidate whose two endpoint references both resolve."""
    n1 = build_node(label=src_label, type="exposure/intervention")
    n2 = build_node(label=tgt_label, type="outcome")
    edge = build_edge(source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
                      direction="causal", relation_type="increases")
    return _candidate(candidate_id, new_nodes=(n1, n2), new_edges=(edge,), **signals)


# --- Eligibility gate: conjunctive, fail-closed threshold operators --------------------
def test_gate_h_eligible_when_all_four_bars_clear():
    assert gate_h(novelty=0.5, testability=0.6, scope_fit=0.7, duplication=0.2) == 1


def test_gate_h_is_conjunctive_fail_closed_on_each_single_bar():
    # Three bars maxed; the failing one alone blocks the candidate (no rescue).
    assert gate_h(novelty=0.0, testability=1.0, scope_fit=1.0, duplication=0.0) == 0  # novelty
    assert gate_h(novelty=1.0, testability=0.0, scope_fit=1.0, duplication=0.0) == 0  # testability
    assert gate_h(novelty=1.0, testability=1.0, scope_fit=0.0, duplication=0.0) == 0  # scope
    assert gate_h(novelty=1.0, testability=1.0, scope_fit=1.0, duplication=1.0) == 0  # duplication


def test_gate_h_threshold_boundaries_are_exact():
    t = gcd.HYP_GATE_THRESHOLDS
    # at-threshold on nov/test/scope PASSES (>=); at-threshold on dup is BLOCKED (strict <).
    assert gate_h(novelty=t["novelty"], testability=t["testability"],
                  scope_fit=t["scope"], duplication=t["dup"] - 1e-6) == 1
    assert gate_h(novelty=t["novelty"], testability=t["testability"],
                  scope_fit=t["scope"], duplication=t["dup"]) == 0  # dup AT threshold -> blocked
    assert gate_h(novelty=t["novelty"] - 1e-6, testability=t["testability"],
                  scope_fit=t["scope"], duplication=0.0) == 0  # nov just below -> blocked


def test_gate_h_is_deterministic_for_fixed_inputs():
    args = dict(novelty=0.41, testability=0.51, scope_fit=0.52, duplication=0.79)
    assert gate_h(**args) == gate_h(**args) == 1


# --- _clip01 re-export from retrieval.similarity (consolidation regression) ------------
def test_clip01_still_clamps_to_unit_interval():
    import src.cycles.hypothesis as hyp_module

    assert hyp_module._clip01(-0.2) == 0.0
    assert hyp_module._clip01(1.5) == 1.0
    assert hyp_module._clip01(0.3) == pytest.approx(0.3, abs=1e-6)


# --- cycles._prompt shared fragment helpers (consolidation regression) ------------------
def test_prompt_helpers_produce_expected_fragments():  # candidate_brief/edge_brief/passage_block
    from src.cycles._prompt import candidate_brief, edge_brief, passage_block

    n1 = build_node(label="chronic stress", type="exposure/intervention", definition="d1")
    n2 = build_node(label="hypertension", type="outcome", definition="d2")
    edge = build_edge(source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
                      direction="causal", relation_type="increases", mechanism="m1")

    assert candidate_brief((n1, n2)) == "chronic stress (exposure/intervention): d1; hypertension (outcome): d2"
    assert edge_brief((edge,)) == f"{n1.node_id} --increases--> {n2.node_id} [m1]"
    assert passage_block(["p1", "p2"]) == "### Passage 1\np1\n\n### Passage 2\np2"


# --- Hypothesis score: clipped weighted sum --------------------------------------------
def test_hyp_score_equal_weight_average_of_six_graded_signals():
    # 1/6 each -> the mean of the six [0,1] inputs.
    score = hyp_score(novelty=0.9, plausibility=0.6, testability=0.5,
                      expected_yield=0.4, centrality=0.3, mechanism_specificity=0.2)
    assert score == pytest.approx((0.9 + 0.6 + 0.5 + 0.4 + 0.3 + 0.2) / 6, abs=1e-6)


def test_hyp_score_all_ones_clips_to_one():
    assert hyp_score(novelty=1.0, plausibility=1.0, testability=1.0,
                     expected_yield=1.0, centrality=1.0, mechanism_specificity=1.0) == pytest.approx(
        1.0, abs=1e-6)


def test_hyp_score_is_monotonic_in_each_signal():
    base = dict(novelty=0.5, plausibility=0.5, testability=0.5,
                expected_yield=0.5, centrality=0.5, mechanism_specificity=0.5)
    lower = hyp_score(**base)
    base["plausibility"] = 0.9
    assert hyp_score(**base) > lower


# --- Signal fusion and deterministic cosine metrics -----------------------------------
def test_fuse_subsignal_late_fuses_llm_ordinal_and_det_metric():
    assert fuse_subsignal(llm_ordinal=0.8, det_metric=0.6, w_llm=0.7, w_det=0.3) == pytest.approx(
        0.7 * 0.8 + 0.3 * 0.6, abs=1e-6)


def test_fuse_subsignal_is_clipped_to_unit_interval():
    assert fuse_subsignal(llm_ordinal=1.0, det_metric=1.0, w_llm=0.7, w_det=0.7) == pytest.approx(
        1.0, abs=1e-6)


def test_det_novelty_is_one_minus_max_cosine_to_existing():
    anchor = "chronic stress"
    embedder = _stub_embedder({anchor: [1.0, 0.0], "a": [1.0, 0.0], "b": [0.0, 1.0]})
    # max cos over {parallel, orthogonal} = 1.0 -> novelty det = 0.0
    assert det_novelty(anchor, ["a", "b"], embedder=embedder) == pytest.approx(0.0, abs=1e-6)
    # orthogonal-only existing -> max cos 0 -> novelty 1.0
    assert det_novelty(anchor, ["b"], embedder=embedder) == pytest.approx(1.0, abs=1e-6)
    # empty graph -> nothing to overlap -> maximally novel
    assert det_novelty(anchor, [], embedder=embedder) == pytest.approx(1.0, abs=1e-6)


def test_det_duplication_is_max_cosine_over_reference_pool():
    anchor = "chronic stress"
    embedder = _stub_embedder({anchor: [1.0, 0.0], "dup": [1.0, 0.0], "far": [0.0, 1.0]})
    assert det_duplication(anchor, ["far", "dup"], embedder=embedder) == pytest.approx(1.0, abs=1e-6)
    assert det_duplication(anchor, ["far"], embedder=embedder) == pytest.approx(0.0, abs=1e-6)


def test_det_scope_fit_is_cosine_to_scope_anchor():
    anchor = "chronic stress"
    embedder = _stub_embedder({anchor: [1.0, 0.0], "scope": [1.0, 0.0], "off": [0.0, 1.0]})
    assert det_scope_fit(anchor, "scope", embedder=embedder) == pytest.approx(1.0, abs=1e-6)
    assert det_scope_fit(anchor, "off", embedder=embedder) == pytest.approx(0.0, abs=1e-6)


def test_candidate_anchor_text_joins_node_labels_and_aliases():
    n1 = build_node(label="chronic stress", type="exposure/intervention", aliases=["psychosocial stress"])
    n2 = build_node(label="hypertension", type="outcome")
    assert candidate_anchor_text((n1, n2)) == "chronic stress psychosocial stress hypertension"


# --- Ranking: eligible candidates, descending score, bounded output --------------------
def test_eligible_ranked_filters_by_gate_then_sorts_descending():
    a = _candidate("c-a", novelty=0.9, plausibility=0.9, testability=0.9,
                   expected_yield=0.9, centrality=0.9, mechanism_specificity=0.9)  # hyp 0.9
    b = _candidate("c-b", novelty=0.6, plausibility=0.6, testability=0.6,
                   expected_yield=0.6, centrality=0.6, mechanism_specificity=0.6)  # hyp 0.6
    blocked = _candidate("c-x", duplication=0.95)  # fails the dup bar -> ineligible
    ranked = eligible_ranked([b, blocked, a])
    assert [sc.candidate.candidate_id for sc in ranked] == ["c-a", "c-b"]  # blocked excluded
    assert ranked[0].hyp_score == pytest.approx(0.9, abs=1e-6)


def test_eligible_ranked_breaks_ties_deterministically_by_candidate_id():
    a = _candidate("c-2")
    b = _candidate("c-1")  # identical signals -> identical HypScore
    ranked = eligible_ranked([a, b])
    assert [sc.candidate.candidate_id for sc in ranked] == ["c-1", "c-2"]  # id ascending


def test_eligible_ranked_surfaces_only_top_k_under_budget():
    cands = [_candidate(f"c-{i}", novelty=0.9 - i * 0.05, plausibility=0.9 - i * 0.05,
                        testability=0.9 - i * 0.05, expected_yield=0.9 - i * 0.05,
                        centrality=0.9 - i * 0.05, mechanism_specificity=0.9 - i * 0.05)
             for i in range(4)]
    top = eligible_ranked(cands, budget=2)
    assert [sc.candidate.candidate_id for sc in top] == ["c-0", "c-1"]


# --- Δ^hypothesis builder -----------------------------------------------------------------
def test_build_hypothesis_delta_carries_new_nodes_edges_assumptions():
    n1 = build_node(label="chronic stress", type="exposure/intervention")
    n2 = build_node(label="hypertension", type="outcome")
    edge = build_edge(source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
                      direction="causal", relation_type="increases")
    delta = build_hypothesis_delta(base_graph_hash="h0", new_nodes=[n1, n2], new_edges=[edge],
                                   assumptions=["adults"], rationale="mediator hypothesis")
    assert delta.family == DeltaFamily.HYPOTHESIS
    assert delta.payload.new_nodes == [n1, n2]
    assert delta.payload.new_edges == [edge]
    assert delta.payload.assumptions == ["adults"]
    # The payload desugars to add-node and add-edge operations; every edge is unverified.
    op_types = {op.op_type for op in delta.payload.to_operations()}
    assert op_types == {"add_node", "add_edge"}
    assert all(e.status == EdgeStatus.UNVERIFIED for e in delta.payload.new_edges)


def test_build_hypothesis_delta_does_not_carry_public_idea_scaffold():  # architecture boundary
    n1 = build_node(label="compression floor", type="mechanism")
    delta = build_hypothesis_delta(base_graph_hash="h0", new_nodes=[n1])
    assert "idea_scaffold" not in delta.payload.model_dump()


# --- Research Synthesist grounding enforcement at the parse boundary -------------------
def _any_embedder(texts):
    return [[1.0, 0.0] for _ in texts]


_GROUNDED_SIGNALS = {"novelty": 0.5, "testability": 0.5, "scope_fit": 0.5, "duplication": 0.5}


def _raw_candidate(cid, *, mechanism_chain, source_quotes, new_nodes=None, new_edges=None):
    default_nodes = [
        {"label": f"{cid} mechanism", "type": "mechanism"},
        {"label": f"{cid} outcome", "type": "outcome"},
    ]
    default_edges = [{
        "source": f"{cid} mechanism",
        "target": f"{cid} outcome",
        "relation_type": "influences",
        "direction": "directed",
        "mechanism": "test fixture mechanism",
    }]
    return {
        "candidate_id": cid,
        "new_nodes": default_nodes if new_nodes is None else new_nodes,
        "new_edges": default_edges if new_edges is None else new_edges,
        "mechanism_chain": mechanism_chain,
        "source_quotes": source_quotes,
        "llm_signals": _GROUNDED_SIGNALS,
    }


def test_build_hypothesis_candidates_drops_empty_mechanism_chain(caplog):
    caplog.set_level("WARNING")
    raw = [
        _raw_candidate("h-bad", mechanism_chain=[], source_quotes=[
            {"evidence_id": "ev_1", "quote_span": "q", "role_in_hypothesis": "r"}
        ]),
        _raw_candidate("h-good", mechanism_chain=[
            {"from": "X", "relation": "influences", "to": "Y", "mechanism": "m"}
        ], source_quotes=[{"evidence_id": "ev_1", "quote_span": "q", "role_in_hypothesis": "r"}]),
    ]
    result = build_hypothesis_candidates(
        raw, store=CausalClaimGraphStore(), embedder=_any_embedder, claim="c"
    )
    assert [c.candidate_id for c in result] == ["h1"]  # h-bad dropped; h-good renumbered to h1
    assert "h-bad" in caplog.text and "mechanism_chain" in caplog.text  # per-candidate reason recorded


def test_build_hypothesis_candidates_drops_empty_source_quotes(caplog):
    caplog.set_level("WARNING")
    raw = [
        _raw_candidate("h-bad", mechanism_chain=[
            {"from": "X", "relation": "influences", "to": "Y", "mechanism": "m"}
        ], source_quotes=[]),
        _raw_candidate("h-good", mechanism_chain=[
            {"from": "X", "relation": "influences", "to": "Y", "mechanism": "m"}
        ], source_quotes=[{"evidence_id": "ev_1", "quote_span": "q", "role_in_hypothesis": "r"}]),
    ]
    result = build_hypothesis_candidates(
        raw, store=CausalClaimGraphStore(), embedder=_any_embedder, claim="c"
    )
    assert [c.candidate_id for c in result] == ["h1"]  # h-bad dropped; h-good renumbered to h1
    assert "h-bad" in caplog.text and "source_quotes" in caplog.text  # per-candidate reason recorded


@pytest.mark.parametrize(
    ("raw_relation", "expected"),
    [
        ("--bounds-->", "bounds"),
        ("→bounds→", "bounds"),
        ("dose-dependent", "dose-dependent"),
        (None, ""),
        (0, ""),
        (False, ""),
    ],
)
def test_build_hypothesis_candidates_normalizes_mechanism_relation_labels(
    raw_relation, expected
):
    raw = _raw_candidate(
        "h1",
        mechanism_chain=[
            {
                "from": "X",
                "relation": raw_relation,
                "to": "Y",
                "mechanism": "m",
            }
        ],
        source_quotes=[
            {"evidence_id": "ev_1", "quote_span": "q", "role_in_hypothesis": "r"}
        ],
    )

    result = build_hypothesis_candidates(
        [raw], store=CausalClaimGraphStore(), embedder=_any_embedder, claim="c"
    )

    assert result[0].mechanism_chain[0]["relation"] == expected


def test_build_hypothesis_candidates_drops_candidate_without_a_new_concept(caplog):
    """A hypothesis with no new node has no focus for the connected page."""
    caplog.set_level("WARNING")
    raw = [
        _raw_candidate("h-bad", new_nodes=[], mechanism_chain=[
            {"from": "X", "relation": "influences", "to": "Y", "mechanism": "m"}
        ], source_quotes=[{"evidence_id": "ev_1", "quote_span": "q", "role_in_hypothesis": "r"}]),
        _raw_candidate("h1", mechanism_chain=[
            {"from": "X", "relation": "influences", "to": "Y", "mechanism": "m"}
        ], source_quotes=[{"evidence_id": "ev_1", "quote_span": "q", "role_in_hypothesis": "r"}]),
    ]
    result = build_hypothesis_candidates(
        raw, store=CausalClaimGraphStore(), embedder=_any_embedder, claim="c"
    )
    assert [c.candidate_id for c in result] == ["h1"]
    assert "h-bad" in caplog.text and "new_nodes" in caplog.text


def test_build_hypothesis_candidates_drops_candidate_without_a_new_edge(caplog):
    """A node-only proposal has no hypothesis claim to commit or test."""
    caplog.set_level("WARNING")
    raw = [
        _raw_candidate("h-bad", new_edges=[], mechanism_chain=[
            {"from": "X", "relation": "influences", "to": "Y", "mechanism": "m"}
        ], source_quotes=[{"evidence_id": "ev_1", "quote_span": "q", "role_in_hypothesis": "r"}]),
        _raw_candidate("h1", mechanism_chain=[
            {"from": "X", "relation": "influences", "to": "Y", "mechanism": "m"}
        ], source_quotes=[{"evidence_id": "ev_1", "quote_span": "q", "role_in_hypothesis": "r"}]),
    ]
    result = build_hypothesis_candidates(
        raw, store=CausalClaimGraphStore(), embedder=_any_embedder, claim="c"
    )
    assert [c.candidate_id for c in result] == ["h1"]
    assert "h-bad" in caplog.text and "new_edges" in caplog.text


def test_build_hypothesis_candidates_normalizes_candidate_ids_to_h_n(caplog):
    """The connected renderer globs trace/h*.html, so a non-h<N> id would lose the page."""
    caplog.set_level("WARNING")

    def grounded(cid):
        return _raw_candidate(cid, mechanism_chain=[
            {"from": "X", "relation": "influences", "to": "Y", "mechanism": "m"}
        ], source_quotes=[{"evidence_id": "ev_1", "quote_span": "q", "role_in_hypothesis": "r"}])

    result = build_hypothesis_candidates(
        [grounded("H1"), grounded("h2"), grounded("cand-3"), grounded("h2")],
        store=CausalClaimGraphStore(), embedder=_any_embedder, claim="c",
    )
    # "h2" is well-formed and unique so it is kept; the rest are renumbered around it.
    assert [c.candidate_id for c in result] == ["h1", "h2", "h3", "h4"]
    assert "'H1'" in caplog.text and "'cand-3'" in caplog.text


# --- Disabled hypothesis cycle ---------------------------------------------------------
def test_cycle_default_off_proposes_nothing_and_leaves_graph_untouched():
    store = CausalClaimGraphStore()
    candidate = _two_node_candidate("c-1")
    result = run_hypothesis_cycle([candidate], store=store)  # enabled defaults to False
    assert result.enabled is False
    assert result.proposed == () and result.surfaced == () and result.transactions == ()
    assert store.version == 0  # graph byte-identical: the feature is inert by default


def test_cycle_enabled_but_precondition_unmet_skips_expansion():
    store = CausalClaimGraphStore()
    result = run_hypothesis_cycle([_two_node_candidate("c-1")], store=store,
                                  enabled=True, precondition=False)
    assert result.surfaced == () and result.transactions == ()
    assert store.version == 0


# --- Enabled hypothesis cycle ----------------------------------------------------------
def test_cycle_enabled_surfaces_ranked_eligible_under_budget():
    store = CausalClaimGraphStore()
    good = _two_node_candidate("c-good")
    blocked = _two_node_candidate("c-blocked", duplication=0.95)  # dup bar fails
    result = run_hypothesis_cycle([blocked, good], store=store, enabled=True)
    assert [sc.candidate.candidate_id for sc in result.surfaced] == ["c-good"]  #  fail-closed
    # Surfaced but unconfirmed candidates are not committed.
    assert result.transactions == () and store.version == 0


def test_cycle_confirmed_eligible_commits_unverified_with_receipt():
    store = CausalClaimGraphStore()
    good = _two_node_candidate("c-good")
    result = run_hypothesis_cycle([good], store=store, enabled=True,
                                  confirmed_ids=["c-good"],
                                  timestamp="2026-06-14T00:00:00+00:00")
    assert len(result.transactions) == 1
    tx = result.transactions[0]
    assert tx.accepted is True
    assert store.version == 1
    # Hypotheses land as unverified graph structure without evidence-backed status.
    assert all(e.status == EdgeStatus.UNVERIFIED for e in store.edges.values())
    assert all(e.confidence is None for e in store.edges.values())
    # The applied delta has both a receipt and a transaction-log row.
    assert tx.receipt.status == "accepted"
    assert tx.receipt.author == "hypothesis_expander"
    assert result.log.rows[-1].validation_status == "accepted"


def test_cycle_exposes_confirmed_committed_candidates():
    store = CausalClaimGraphStore()
    good = _two_node_candidate("c-good")
    result = run_hypothesis_cycle([good], store=store, enabled=True, confirmed_ids=["c-good"])
    assert [c.candidate_id for c in result.confirmed] == ["c-good"]  # confirmed AND committed
    # a surfaced-but-unconfirmed candidate is not exposed as confirmed
    other = CausalClaimGraphStore()
    unconfirmed = run_hypothesis_cycle([_two_node_candidate("c-x")], store=other, enabled=True)
    assert unconfirmed.surfaced and unconfirmed.confirmed == ()


def test_cycle_never_commits_an_ineligible_candidate_even_if_confirmed():
    store = CausalClaimGraphStore()
    blocked = _two_node_candidate("c-blocked", duplication=0.95)
    result = run_hypothesis_cycle([blocked], store=store, enabled=True,
                                  confirmed_ids=["c-blocked"])
    assert result.transactions == ()  # gate blocks it before any delta is built
    assert store.version == 0


def test_cycle_rejected_delta_leaves_graph_unchanged_and_is_auditable():
    store = CausalClaimGraphStore()
    # An eligible candidate with a dangling endpoint fails reference validation.
    n1 = build_node(label="chronic stress", type="exposure/intervention")
    dangling = build_edge(source_node_ids=[n1.node_id], target_node_ids=["ghost-node"],
                          direction="causal", relation_type="increases")
    candidate = _candidate("c-bad", new_nodes=(n1,), new_edges=(dangling,))
    result = run_hypothesis_cycle([candidate], store=store, enabled=True,
                                  confirmed_ids=["c-bad"], timestamp="2026-06-14T00:00:00+00:00")
    assert len(result.transactions) == 1
    tx = result.transactions[0]
    assert tx.accepted is False
    assert tx.gate_results.failing_gate == "refs"
    assert store.version == 0  # atomic: no partial mutation
    assert tx.receipt.status == "rejected"  # routable for revision, still audited


def test_cycle_reports_thresholds_and_weights_in_audit():
    store = CausalClaimGraphStore()
    result = run_hypothesis_cycle([_two_node_candidate("c-1")], store=store, enabled=True)
    audit = result.audit
    assert audit["graph_defaults_version"] == gcd.GRAPH_DEFAULTS_VERSION
    assert audit["hyp_gate_thresholds"] == gcd.HYP_GATE_THRESHOLDS
    assert audit["hypscore_weights"] == gcd.HYPSCORE_WEIGHTS
    assert audit["expansion_budget_b"] == gcd.EXPANSION_BUDGET_B


# --- Field-relative novelty and literature-saturation ranking --------------------------
# The eligibility gate keeps the proposer's novelty estimate, while an independent panel
# supplies field-relative novelty and saturation for ranking. The panel uses the median so
# the result is deterministic and robust to an outlying judge.
_FIELD_NOVELTY_SIGNALS = {
    "h1": dict(novelty=0.5769, testability=0.85, scope_fit=0.9243, duplication=0.6819,
               plausibility=0.92, expected_yield=0.80, centrality=0.88, mechanism_specificity=0.82),
    "h2": dict(novelty=0.6219, testability=0.72, scope_fit=0.8937, duplication=0.6652,
               plausibility=0.90, expected_yield=0.85, centrality=0.80, mechanism_specificity=0.78),
    "h3": dict(novelty=0.6007, testability=0.78, scope_fit=0.8796, duplication=0.7211,
               plausibility=0.90, expected_yield=0.82, centrality=0.75, mechanism_specificity=0.80),
    "h4": dict(novelty=0.5680, testability=0.80, scope_fit=0.8955, duplication=0.6955,
               plausibility=0.90, expected_yield=0.78, centrality=0.82, mechanism_specificity=0.85),
    "h5": dict(novelty=0.5429, testability=0.82, scope_fit=0.8831, duplication=0.7361,
               plausibility=0.92, expected_yield=0.78, centrality=0.78, mechanism_specificity=0.75),
    "h6": dict(novelty=0.6123, testability=0.70, scope_fit=0.8893, duplication=0.7035,
               plausibility=0.88, expected_yield=0.80, centrality=0.72, mechanism_specificity=0.78),
    "h7": dict(novelty=0.5631, testability=0.80, scope_fit=0.8680, duplication=0.7177,
               plausibility=0.90, expected_yield=0.72, centrality=0.65, mechanism_specificity=0.82),
    "h8": dict(novelty=0.5285, testability=0.82, scope_fit=0.9088, duplication=0.7366,
               plausibility=0.88, expected_yield=0.70, centrality=0.80, mechanism_specificity=0.74),
}

# Three independent ``(field_novelty, literature_saturation)`` grades per candidate.
_FIELD_NOVELTY_JUDGE_GRADES = {
    "h1": [(0.08, 0.97), (0.05, 0.97), (0.07, 0.95)],
    "h2": [(0.62, 0.50), (0.30, 0.80), (0.55, 0.70)],
    "h3": [(0.38, 0.72), (0.12, 0.92), (0.15, 0.90)],
    "h4": [(0.33, 0.68), (0.25, 0.78), (0.30, 0.85)],
    "h5": [(0.22, 0.85), (0.18, 0.90), (0.30, 0.85)],
    "h6": [(0.20, 0.90), (0.32, 0.76), (0.50, 0.75)],
    "h7": [(0.18, 0.92), (0.10, 0.94), (0.22, 0.92)],
    "h8": [(0.20, 0.60), (0.10, 0.90), (0.18, 0.85)],
}


def _field_novelty_candidates(apply_saturation: bool = True):
    """Build candidates with panel-median novelty and optional saturation."""
    from statistics import median

    candidates = []
    for cid, sig in _FIELD_NOVELTY_SIGNALS.items():
        base = _candidate(cid, **sig)
        grades = _FIELD_NOVELTY_JUDGE_GRADES[cid]
        novelty_graded = median(g[0] for g in grades)
        saturation = median(g[1] for g in grades) if apply_saturation else 0.0
        candidates.append(replace(base, novelty_graded=novelty_graded, saturation=saturation))
    return candidates


def _order(candidates):
    ranked = eligible_ranked(candidates, budget=len(candidates))
    return [sc.candidate.candidate_id for sc in ranked]


# --- Rank score: hypothesis score multiplied by one minus saturation ------------------
def test_rank_score_demotes_by_one_minus_saturation():
    assert rank_score(hyp_score=0.8, saturation=0.0) == pytest.approx(0.8, abs=1e-6)
    assert rank_score(hyp_score=0.8, saturation=0.5) == pytest.approx(0.4, abs=1e-6)
    assert rank_score(hyp_score=0.8, saturation=1.0) == pytest.approx(0.0, abs=1e-6)


def test_rank_score_clips_to_unit_interval():
    assert rank_score(hyp_score=1.2, saturation=0.0) == pytest.approx(1.0, abs=1e-6)
    assert rank_score(hyp_score=0.5, saturation=2.0) == pytest.approx(0.0, abs=1e-6)


def test_ranking_novelty_defaults_to_gate_novelty_when_unjudged():
    # An unjudged candidate ranks on its single novelty signal.
    c = _candidate("c-1", novelty=0.6)
    assert c.ranking_novelty == pytest.approx(0.6, abs=1e-6)


# --- Field-relative replay -------------------------------------------------------------
def test_field_novelty_ranking_drops_common_sense_out_of_top2():
    # Condition (b): independent field-relative novelty, no saturation. h1 (textbook common
    # sense, proposer-ranked #1) must fall out of the top-2.
    order = _order(_field_novelty_candidates(apply_saturation=False))
    assert order[:2] == ["h2", "h4"]      # the genuinely-novel pair leads
    assert "h1" not in order[:2]          # common sense demoted out of the top-2
    assert order.index("h1") == 2         # specifically to #3 (experiment condition b)


def test_saturation_penalty_sends_common_sense_to_last():
    # Condition (a)+(b): field-novelty + saturation penalty. h1 (saturation ~0.97) -> last.
    order = _order(_field_novelty_candidates(apply_saturation=True))
    assert order[0] == "h2"               # the under-modeled confounder leads
    assert order[-1] == "h1"              # textbook common sense ranked last


def test_saturation_demotes_without_gating_out_soft_demotion():
    # h1 stays ELIGIBLE (passes Gate_h on its permissive gate bar) — it is ranked last, not
    # silently dropped. All eight candidates remain in the ranked set.
    ranked = eligible_ranked(_field_novelty_candidates(apply_saturation=True), budget=8)
    assert len(ranked) == 8
    assert "h1" in {sc.candidate.candidate_id for sc in ranked}


def test_field_novelty_scores_stay_in_unit_interval():
    for sc in eligible_ranked(_field_novelty_candidates(apply_saturation=True), budget=8):
        assert 0.0 <= sc.hyp_score <= 1.0
        assert 0.0 <= sc.rank_score <= 1.0


def test_field_novelty_ranking_is_deterministic_within_1e6():
    a = eligible_ranked(_field_novelty_candidates(apply_saturation=True), budget=8)
    b = eligible_ranked(_field_novelty_candidates(apply_saturation=True), budget=8)
    assert [sc.candidate.candidate_id for sc in a] == [sc.candidate.candidate_id for sc in b]
    for sca, scb in zip(a, b, strict=True):
        assert sca.rank_score == pytest.approx(scb.rank_score, abs=1e-6)


def test_saturation_penalty_can_be_disabled_by_config_flag(monkeypatch):
    # The audited SATURATION_PENALTY_ENABLED flag drives reproducible behavior:
    # an audit record stamped flag=False must reproduce scores with no penalty applied.
    base = _candidate("c-1", novelty=0.9, plausibility=0.9, testability=0.9,
                      expected_yield=0.9, centrality=0.9, mechanism_specificity=0.9)
    saturated = replace(base, novelty_graded=0.9, saturation=0.9)
    on = eligible_ranked([saturated], budget=1)[0]
    assert on.rank_score < on.hyp_score                       # flag ON (default): penalty demotes
    monkeypatch.setattr(gcd, "SATURATION_PENALTY_ENABLED", False)
    off = eligible_ranked([saturated], budget=1)[0]
    assert off.rank_score == pytest.approx(off.hyp_score, abs=1e-6)  # flag OFF: penalty does NOT fire


# --- Audited saturation and panel configuration ---------------------------------------
def test_saturation_and_panel_config_present_and_audited():
    assert isinstance(gcd.SATURATION_PENALTY_ENABLED, bool)
    assert 0.0 <= gcd.THETA_SATURATION <= 1.0          # optional high backstop gate
    assert gcd.NOVELTY_PANEL_REDUCER == "median"
    audit = gcd.audit_dict()
    assert audit["saturation_penalty_enabled"] == gcd.SATURATION_PENALTY_ENABLED
    assert audit["theta_saturation"] == gcd.THETA_SATURATION
    assert audit["novelty_panel_reducer"] == gcd.NOVELTY_PANEL_REDUCER


# --- Confirmation hook ----------------------------------------------------------------
def test_run_hypothesis_cycle_confirm_fn_selects_from_the_ranking():
    from src.cycles.hypothesis import run_hypothesis_cycle

    store = CausalClaimGraphStore()
    committed = run_hypothesis_cycle(
        [_two_node_candidate("h1")], store=store, enabled=True,
        confirm_fn=lambda surfaced: [sc.candidate.candidate_id for sc in surfaced],
    )
    assert len(committed.transactions) == 1  # confirm_fn confirmed it -> commit


def test_run_hypothesis_cycle_confirm_fn_overrides_static_confirmed_ids():  # callback wins
    from src.cycles.hypothesis import run_hypothesis_cycle

    store = CausalClaimGraphStore()
    result = run_hypothesis_cycle(
        [_two_node_candidate("h1")], store=store, enabled=True,
        confirmed_ids=["h1"], confirm_fn=lambda surfaced: [],  # confirm nothing despite the superset
    )
    assert result.transactions == ()
    assert len(result.surfaced) == 1  # still surfaced for reporting


def _descending_pool(n=6):
    """Return ``n`` eligible candidates with strictly descending ranking signals."""
    return [
        _candidate(f"c{i + 1}", novelty=0.9 - i * 0.05, plausibility=0.9 - i * 0.05,
                   testability=0.9 - i * 0.05, expected_yield=0.9 - i * 0.05,
                   centrality=0.9 - i * 0.05, mechanism_specificity=0.9 - i * 0.05)
        for i in range(n)
    ]


# --- Default ranking behavior ------------------------------------------------------------
def test_cycle_invariance_surfaced_eligible_match_default_ranking():
    cands = _descending_pool(6)
    result = run_hypothesis_cycle(cands, store=CausalClaimGraphStore(), enabled=True, budget=3)
    ref = eligible_ranked(cands, budget=3)  # direct ranking reference
    assert [sc.candidate.candidate_id for sc in result.surfaced] == [
        sc.candidate.candidate_id for sc in ref
    ]
    ref_full = eligible_ranked(cands, budget=len(cands))
    assert [c.candidate_id for c in result.eligible] == [
        sc.candidate.candidate_id for sc in ref_full
    ]
    # No tournament or evolution artifacts are produced on the default path.
    assert "tournament" not in result.audit
    assert "evolution" not in result.audit and "evolution_rounds" not in result.audit


# Optional three-bar behavior: ``drop_novelty`` removes only the novelty bar. The parameter
# defaults to false and no production caller enables it, so these tests exercise it directly.
def test_gate_h_drops_novelty_bar_when_enabled():  # three-bar parameter
    # Low novelty, but testability/scope/dup all clear: BLOCKED by the 4-bar gate, PASSES the 3-bar.
    assert gate_h(novelty=0.0, testability=0.6, scope_fit=0.6, duplication=0.2) == 0
    assert gate_h(novelty=0.0, testability=0.6, scope_fit=0.6, duplication=0.2, drop_novelty=True) == 1


def test_gate_h_three_bar_still_enforces_other_three_bars():  # only novelty drops
    # dropping novelty must NOT relax testability/scope/duplication — they stay fail-closed.
    assert gate_h(novelty=0.0, testability=0.0, scope_fit=0.6, duplication=0.2, drop_novelty=True) == 0
    assert gate_h(novelty=0.0, testability=0.6, scope_fit=0.0, duplication=0.2, drop_novelty=True) == 0
    assert gate_h(novelty=0.0, testability=0.6, scope_fit=0.6, duplication=0.9, drop_novelty=True) == 0


def test_cycle_default_off_does_not_stamp_proposer_provenance():
    cands = _descending_pool(3)
    result = run_hypothesis_cycle(cands, store=CausalClaimGraphStore(), enabled=True, budget=3)
    assert result.surfaced[0].candidate.provenance == ()  # no provenance stamped by default


# --- Claimless-discovery emphasis reranking ----------------------------------------------
def test_run_cycle_reranks_surfaced_winners_by_emphasis():
    store = CausalClaimGraphStore()
    c1 = _two_node_candidate("h1", src_label="alpha", tgt_label="beta")
    c2 = _two_node_candidate("h2", src_label="gamma", tgt_label="delta")

    def reverse_reranker(scored):
        return sorted(scored, key=lambda sc: sc.candidate.candidate_id, reverse=True)

    # Quality (the RankScore top-k) picks the surfaced SET; the emphasis reranker then only
    # RE-ORDERS those winners and never changes membership. budget=2 surfaces both h1, h2; the
    # reverse reranker orders them h2, h1.
    result = run_hypothesis_cycle(
        [c1, c2], store=store, enabled=True, budget=2, reranker=reverse_reranker,
    )
    assert [sc.candidate.candidate_id for sc in result.surfaced] == ["h2", "h1"]


def test_run_cycle_without_reranker_keeps_spec_rankscore_order():  # regression: spec order preserved
    store = CausalClaimGraphStore()
    c1 = _two_node_candidate("h1", src_label="alpha", tgt_label="beta")
    c2 = _two_node_candidate("h2", src_label="gamma", tgt_label="delta")
    result = run_hypothesis_cycle([c1, c2], store=store, enabled=True, budget=1)
    assert [sc.candidate.candidate_id for sc in result.surfaced] == ["h1"]  # candidate_id tie-break
