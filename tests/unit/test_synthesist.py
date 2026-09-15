"""Research Synthesist thread — mine, propose, and revise.

Hermetic: a fake CAMEL backend returns fixtured STRICT-JSON per turn and records the
messages it sees, so we can assert both the deterministic parse and that the thread is a
single persistent conversation (system prompt sent once at turn 1; later turns append).
Real-LLM behavior stays behind the `live` markers.
"""

from __future__ import annotations

import json

import pytest


class _FakeStore:
    """Duck-typed read surface the propose turn needs: committed nodes + edges."""

    def __init__(self, nodes, edges=()):
        self.nodes = {n.node_id: n for n in nodes}
        self.edges = {e.edge_id: e for e in edges}


class _ThreadBackend:
    """A fake persistent-thread backend: returns one canned content per ``.run`` call
    (mine, then propose, then revise) and records the full messages list seen each turn."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls: list[list[dict]] = []

    def run(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        content = self._contents.pop(0) if self._contents else ""
        return {"choices": [{"message": {"content": content}}]}


def _stub_embedder(texts):
    # deterministic unit vectors -> cosine is well-defined; det sub-signals stay stable.
    return [[1.0, 0.0] for _ in texts]


_MINE_JSON = (
    '{"concepts": ['
    '{"label": "rate-distortion floor", "type": "mechanism", "definition": "min achievable loss",'
    ' "source_context": "the paper argues the floor caps factuality under compression",'
    ' "bears_on_outcome": true, "evidence_id": "ev_001", "paper_id": "p1",'
    ' "matched_quote_span": "rate-distortion floor", "rationale": "drives factuality"},'
    '{"label": "adapter rank", "type": "moderator", "definition": "an effect modifier",'
    ' "source_context": "modulates the compression-to-factuality link",'
    ' "bears_on_outcome": true, "evidence_id": "ev_002", "paper_id": "p2",'
    ' "matched_quote_span": "adapter rank", "rationale": "boundary condition"}],'
    ' "merge_decisions": [{"a": "rate-distortion floor", "b": "compression loss",'
    ' "same_concept": true, "reason": "synonyms"}]}'
)


def test_mine_turn_parses_source_context_and_merge_decisions():  # mining turn
    from src.cycles.synthesist import SYNTHESIST_SYSTEM_PROMPT, ResearchSynthesist

    backend = _ThreadBackend([_MINE_JSON])
    synth = ResearchSynthesist(backend, embedder=_stub_embedder, claim="LLM compression", proposal_count=6)
    result = synth.mine(passages=("### Passage 1\nbody",), merge_candidates=())

    assert [c.label for c in result.concepts] == ["rate-distortion floor", "adapter rank"]
    first = result.concepts[0]
    # source_context is the NEW per-concept field (distinct from the universal definition).
    assert first.source_context == "the paper argues the floor caps factuality under compression"
    assert first.type == "mechanism" and first.evidence_id == "ev_001" and first.paper_id == "p1"
    # The moderator type flows through unchanged and is reconciled by type compatibility.
    assert result.concepts[1].type == "moderator"
    # the batched merge adjudication (folds  into turn 1) is parsed verbatim.
    assert list(result.merge_decisions) == [
        {"a": "rate-distortion floor", "b": "compression loss", "same_concept": True, "reason": "synonyms"}
    ]
    # the thread is seeded ONCE with the Research Synthesist system prompt, then a user turn.
    seen = backend.calls[0]
    assert seen[0]["role"] == "system" and seen[0]["content"] == SYNTHESIST_SYSTEM_PROMPT
    assert seen[1]["role"] == "user"
    assert len(backend.calls) == 1


def test_committed_graph_view_renders_mined_source_context():  # mining-to-proposal data flow
    # source_context is carried on the mined node's provenance (enrichment_provenance) so the
    # PROPOSE turn's <committed_graph> view renders what the source argues — the end-to-end flow
    # that was inert while the field never reached the node.
    from src.cycles.enrichment import MinedConcept, mined_to_node
    from src.cycles.synthesist import _synthesist_committed_graph_view

    node = mined_to_node(
        MinedConcept(
            label="rate-distortion floor", type="mechanism", bears_on_outcome=True,
            evidence_id="ev_001", matched_quote_span="rate-distortion floor",
            source_context="the paper argues the floor caps factuality under compression",
        )
    )
    view = _synthesist_committed_graph_view(_FakeStore([node]), {node.node_id})
    assert "source_context: the paper argues the floor caps factuality under compression" in view
    assert '(ev_001: "rate-distortion floor")' in view


def test_mine_appends_miner_focus_to_system_prompt():  # DESIGN §4 inherited hook (miner_focus->)
    from src.cycles.synthesist import SYNTHESIST_SYSTEM_PROMPT, ResearchSynthesist

    backend = _ThreadBackend([_MINE_JSON])
    ResearchSynthesist(
        backend, embedder=_stub_embedder, claim="c", proposal_count=6,
        miner_focus="Focus on tensor-train compression mechanisms.",
    ).mine(passages=("### Passage 1\nbody",))
    system_msg = backend.calls[0][0]
    assert system_msg["role"] == "system"
    assert system_msg["content"].startswith(SYNTHESIST_SYSTEM_PROMPT)      # verbatim prompt preserved
    assert system_msg["content"].endswith("Focus on tensor-train compression mechanisms.")
    # default (no miner_focus) -> the system prompt is byte-identical to the verbatim constant
    plain = _ThreadBackend([_MINE_JSON])
    ResearchSynthesist(plain, embedder=_stub_embedder, claim="c", proposal_count=6).mine(passages=("p",))
    assert plain.calls[0][0]["content"] == SYNTHESIST_SYSTEM_PROMPT


def test_propose_appends_profile_steering_to_user_prompt():  # DESIGN §4 (judgeability->)
    from src.cycles.synthesist import ResearchSynthesist
    from src.delta import build_node

    backend = _ThreadBackend([_MINE_JSON, _PROPOSE_JSON])
    store = _FakeStore([build_node(label="factuality", type="outcome")])
    synth = ResearchSynthesist(
        backend, embedder=_stub_embedder, claim="compression", proposal_count=6,
        propose_steering="Prefer testable, ICML-suitable hypotheses.",
    )
    synth.mine(passages=("### Passage 1\nbody",))
    synth.propose(store)
    propose_user = backend.calls[1][-1]  # turn-2 last message = the propose user turn
    assert propose_user["role"] == "user"
    assert "Prefer testable, ICML-suitable hypotheses." in propose_user["content"]


_PROPOSE_JSON = (
    '{"candidates": [{'
    '"candidate_id": "h1", "rationale": "the floor bounds recoverable factuality",'
    ' "experiment": "SVD sweep; TruthfulQA slope",'
    ' "idea_scaffold": {"claim_anchor": "compression cuts factuality",'
    ' "problem": "find when compressed models lose factual answers"},'
    ' "new_nodes": [{"label": "rate-distortion floor", "type": "mechanism",'
    ' "definition": "min achievable distortion", "aliases": []}],'
    ' "new_edges": [{"source": "rate-distortion floor", "target": "factuality",'
    ' "relation_type": "bounds", "direction": "directed", "mechanism": "information bottleneck"}],'
    ' "mechanism_chain": [{"from": "rate-distortion floor", "relation": "bounds",'
    ' "to": "factuality", "mechanism": "the floor caps recoverable signal"}],'
    ' "source_quotes": [{"evidence_id": "ev_001", "quote_span": "rate-distortion floor",'
    ' "role_in_hypothesis": "names the bounding mechanism"}],'
    ' "assumptions": ["fixed calibration set"], "cross_concept": true, "common_sense": false,'
    ' "llm_signals": {"novelty": 0.8, "testability": 0.6, "scope_fit": 0.7, "duplication": 0.2,'
    ' "plausibility": 0.75, "expected_yield": 0.7, "centrality": 0.65, "mechanism_specificity": 0.8}'
    '}]}'
)


def test_propose_turn_carries_mechanism_chain_and_source_quotes():  # proposal turn
    from src.cycles.hypothesis import det_novelty, fuse_subsignal
    from src.cycles.synthesist import ResearchSynthesist
    from src.delta import build_node
    from src import graph_config_defaults as gcd

    backend = _ThreadBackend([_MINE_JSON, _PROPOSE_JSON])
    store = _FakeStore([build_node(label="factuality", type="outcome")])
    synth = ResearchSynthesist(backend, embedder=_stub_embedder, claim="compression", proposal_count=6)
    synth.mine(passages=("### Passage 1\nbody",))
    candidates = synth.propose(store)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.candidate_id == "h1"
    assert [n.label for n in cand.new_nodes] == ["rate-distortion floor"]
    assert len(cand.new_edges) == 1  # new node -> committed "factuality" resolves
    # Required fields feed the Critic Panel and Experiment Designer.
    assert len(cand.mechanism_chain) == 1
    step = cand.mechanism_chain[0]
    assert (step["from"], step["relation"], step["to"]) == ("rate-distortion floor", "bounds", "factuality")
    assert len(cand.source_quotes) == 1
    assert cand.source_quotes[0]["evidence_id"] == "ev_001"
    assert cand.source_quotes[0]["quote_span"] == "rate-distortion floor"
    # Option-C fusion PRESERVED: testability is LLM-only passthrough; novelty is det-fused (not raw 0.8).
    assert cand.testability == pytest.approx(0.6, abs=1e-6)
    det = det_novelty("rate-distortion floor", ["factuality"], embedder=_stub_embedder)
    w = gcd.HYP_FUSION_WEIGHTS["novelty"]
    assert cand.novelty == pytest.approx(
        fuse_subsignal(llm_ordinal=0.8, det_metric=det, w_llm=w["llm"], w_det=w["det"]), abs=1e-6
    )
    # the thread is ONE persistent conversation: propose runs over system + mine(user+assistant) + propose(user)
    propose_msgs = backend.calls[1]
    assert propose_msgs[0]["role"] == "system"
    assert [m["role"] for m in propose_msgs] == ["system", "user", "assistant", "user"]


_REVISE_JSON = (
    '{"revised": [{'
    '"candidate_id": "h1", "rationale": "repaired: the floor bounds factuality via lost signal",'
    ' "new_nodes": [{"label": "rate-distortion floor", "type": "mechanism",'
    ' "definition": "min distortion", "aliases": []}],'
    ' "new_edges": [{"source": "rate-distortion floor", "target": "factuality",'
    ' "relation_type": "bounds", "direction": "directed", "mechanism": "caps recoverable signal"}],'
    ' "mechanism_chain": [{"from": "rate-distortion floor", "relation": "bounds", "to": "factuality",'
    ' "mechanism": "caps recoverable signal so facts drop"}],'
    ' "source_quotes": [{"evidence_id": "ev_001", "quote_span": "rate-distortion floor",'
    ' "role_in_hypothesis": "the bounding mechanism"}],'
    ' "llm_signals": {"novelty": 0.7, "testability": 0.6, "scope_fit": 0.7, "duplication": 0.2,'
    ' "plausibility": 0.7, "expected_yield": 0.6, "centrality": 0.6, "mechanism_specificity": 0.75}'
    '}],'
    ' "derived": [{'
    '"candidate_id": "h2", "strategy": "divergent", "wasDerivedFrom": ["h1"],'
    ' "rationale": "recombination: adapter rank flips the floor effect",'
    ' "new_nodes": [{"label": "rank-flip regime", "type": "moderator",'
    ' "definition": "regime where rank inverts the effect", "aliases": []}],'
    ' "new_edges": [{"source": "rank-flip regime", "target": "factuality",'
    ' "relation_type": "moderates", "direction": "directed", "mechanism": "inverts the sign"}],'
    ' "mechanism_chain": [{"from": "rank-flip regime", "relation": "moderates", "to": "factuality",'
    ' "mechanism": "high rank reverses the bound"}],'
    ' "source_quotes": [{"evidence_id": null, "quote_span": "min distortion",'
    ' "role_in_hypothesis": "reuses the floor node definition"}],'
    ' "llm_signals": {"novelty": 0.9, "testability": 0.5, "scope_fit": 0.6, "duplication": 0.1,'
    ' "plausibility": 0.6, "expected_yield": 0.7, "centrality": 0.5, "mechanism_specificity": 0.7}'
    '}]}'
)


def test_revise_turn_repairs_survivors_and_derives_with_lineage():  # revision turn
    from src.cycles.synthesist import ResearchSynthesist
    from src.delta import build_node

    backend = _ThreadBackend([_MINE_JSON, _PROPOSE_JSON, _REVISE_JSON])
    store = _FakeStore([build_node(label="factuality", type="outcome")])
    synth = ResearchSynthesist(backend, embedder=_stub_embedder, claim="compression", proposal_count=6)
    synth.mine(passages=("### Passage 1\nbody",))
    synth.propose(store)
    panel_result = {
        "ranking": ["h1"],
        "critique": "sharpen the mechanism; probe the adapter-rank boundary",
        "audit_flags": [
            {
                "candidate_id": "h1",
                "mechanism_steps": [
                    {"from": "rate-distortion floor", "relation": "bounds", "to": "factuality",
                     "verdict": "vague", "note": "no stated mechanism"}
                ],
                "term_issues": [],
            }
        ],
    }
    result = synth.revise(store, panel_result)

    # revised = the flagged survivor, same id, repaired mechanism_chain carried through
    assert [c.candidate_id for c in result.revised] == ["h1"]
    assert len(result.revised[0].mechanism_chain) == 1
    # Derived candidates are genuinely new, include the mandatory divergent candidate, and record lineage.
    assert [c.candidate_id for c in result.derived] == ["h2"]
    d = result.derived[0]
    assert d.provenance and d.provenance[0]["strategy"] == "divergent"
    assert d.provenance[0]["wasDerivedFrom"] == ["h1"]
    assert len(d.mechanism_chain) == 1 and len(d.source_quotes) == 1
    # the divergent candidate may carry a null evidence_id (recombination-only, rule 8)
    assert d.source_quotes[0]["evidence_id"] is None
    # thread persistence: revise runs over the FULL conversation (3 turns)
    revise_msgs = backend.calls[2]
    assert [m["role"] for m in revise_msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert "adapter-rank boundary" in revise_msgs[-1]["content"]  # the panel critique reached the model


# --- Revision contract: cardinality clamp and mandatory divergent candidate ------------
def _derived_raw(cid, *, strategy="inspire", grounded=True):
    return {
        "candidate_id": cid, "strategy": strategy, "wasDerivedFrom": ["h1"],
        "rationale": f"derived {cid}",
        "new_nodes": [{"label": f"node-{cid}", "type": "mechanism", "definition": "d", "aliases": []}],
        "new_edges": [{"source": f"node-{cid}", "target": "factuality",
                       "relation_type": "influences", "direction": "directed", "mechanism": "m"}],
        "mechanism_chain": [{"from": f"node-{cid}", "relation": "influences", "to": "factuality",
                             "mechanism": "m"}],
        "source_quotes": (
            [{"evidence_id": "ev_001", "quote_span": "q", "role_in_hypothesis": "r"}] if grounded else []
        ),
        "llm_signals": {"novelty": 0.5, "testability": 0.5, "scope_fit": 0.5, "duplication": 0.5,
                        "plausibility": 0.5, "expected_yield": 0.5, "centrality": 0.5,
                        "mechanism_specificity": 0.5},
    }


def _revise_json(derived, revised=()):
    return json.dumps({"revised": list(revised), "derived": derived})


def test_revise_clamps_derived_candidates_to_four(caplog):  # maximum derived-candidate count
    from src.cycles.synthesist import ResearchSynthesist
    from src.delta import build_node

    caplog.set_level("WARNING")
    derived = [_derived_raw(f"h{i}", strategy="divergent" if i == 2 else "inspire") for i in range(2, 7)]
    backend = _ThreadBackend([_revise_json(derived)])
    store = _FakeStore([build_node(label="factuality", type="outcome")])
    synth = ResearchSynthesist(backend, embedder=_stub_embedder, claim="compression", proposal_count=6)
    result = synth.revise(store, {})

    assert [c.candidate_id for c in result.derived] == ["h2", "h3", "h4", "h5"]  # first 4 kept, in order
    assert "h6" in caplog.text  # the dropped 5th candidate's reason is recorded


def test_revise_never_fabricates_candidates_when_fewer_come_back():  # no invention
    from src.cycles.synthesist import ResearchSynthesist
    from src.delta import build_node

    derived = [_derived_raw("h2", strategy="divergent")]  # only one candidate comes back
    backend = _ThreadBackend([_revise_json(derived)])
    store = _FakeStore([build_node(label="factuality", type="outcome")])
    synth = ResearchSynthesist(backend, embedder=_stub_embedder, claim="compression", proposal_count=6)
    result = synth.revise(store, {})

    assert [c.candidate_id for c in result.derived] == ["h2"]  # exactly what came back; nothing invented


def test_revise_logs_shortfall_when_fewer_than_two_derived_candidates(caplog):  # minimum target
    from src.cycles.synthesist import ResearchSynthesist
    from src.delta import build_node

    caplog.set_level("WARNING")
    # a single divergent-only candidate would otherwise pass the missing-divergent check silently
    derived = [_derived_raw("h2", strategy="divergent")]
    backend = _ThreadBackend([_revise_json(derived)])
    store = _FakeStore([build_node(label="factuality", type="outcome")])
    synth = ResearchSynthesist(backend, embedder=_stub_embedder, claim="compression", proposal_count=6)
    result = synth.revise(store, {})

    assert [c.candidate_id for c in result.derived] == ["h2"]  # nothing invented
    assert "minimum" in caplog.text.lower()  # the below-minimum cardinality shortfall is recorded


def test_revise_flags_missing_mandatory_divergent_candidate(caplog):  # divergent-candidate requirement
    from src.cycles.synthesist import ResearchSynthesist
    from src.delta import build_node

    caplog.set_level("WARNING")
    derived = [_derived_raw("h2", strategy="inspire"), _derived_raw("h3", strategy="combine")]
    backend = _ThreadBackend([_revise_json(derived)])
    store = _FakeStore([build_node(label="factuality", type="outcome")])
    synth = ResearchSynthesist(backend, embedder=_stub_embedder, claim="compression", proposal_count=6)
    result = synth.revise(store, {})

    assert [c.candidate_id for c in result.derived] == ["h2", "h3"]  # both kept (under the cap)
    assert "divergent" in caplog.text.lower()  # the missing-mandatory-strategy note is recorded


def test_revise_derived_lineage_stays_aligned_when_a_middle_candidate_is_dropped():  # alignment safety
    from src.cycles.synthesist import ResearchSynthesist
    from src.delta import build_node

    derived = [
        _derived_raw("h2", strategy="combine"),
        _derived_raw("h3", strategy="inspire", grounded=False),  # empty source_quotes -> dropped
        _derived_raw("h4", strategy="divergent"),
    ]
    backend = _ThreadBackend([_revise_json(derived)])
    store = _FakeStore([build_node(label="factuality", type="outcome")])
    synth = ResearchSynthesist(backend, embedder=_stub_embedder, claim="compression", proposal_count=6)
    result = synth.revise(store, {})

    assert [c.candidate_id for c in result.derived] == ["h2", "h4"]  # h3 dropped for empty grounding
    h4 = next(c for c in result.derived if c.candidate_id == "h4")
    assert h4.provenance[0]["strategy"] == "divergent"  # lineage stayed bound to h4, not shifted from h3
