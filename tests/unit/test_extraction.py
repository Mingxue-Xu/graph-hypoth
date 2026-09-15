"""Deterministic extraction-cycle and extraction-delta behavior.

The suite covers scope gaps, extraction confidence, merge safety, and delta construction.
Every coefficient comes from ``graph_config_defaults`` and floating-point results are checked
to within 1e-6. The default suite uses injected embedders and fake model backends.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src import graph_config_defaults as gcd
from src.cycles.extraction import (
    ClaimExtraction,
    LLMExtractor,
    build_extract_delta,
    canonicalize_concepts,
    concept_embed_text,
    default_merge_embedder,
    extract_conf,
    merge_decision,
    merge_score,
    run_extraction_cycle,
    scope_gap,
    type_conflict,
)
from src.delta import (
    AddEdgeOp,
    AddNodeOp,
    DeltaFamily,
    MergeNodesOp,
    build_edge,
    build_node,
)
from src.graph_store import CausalClaimGraphStore, EdgeStatus


# --- fixtures (the canonical LLM/embedder-free idiom) -----------------------------------
def _stub_embedder(mapping):
    """Deterministic stub: maps each expected embed-text to a fixed vector (no model load)."""

    def embed(texts):
        return [mapping[t] for t in texts]

    return embed


def _constant_embedder(vector):
    """Every text embeds to the same vector -> cos == 1 for any pair."""

    def embed(texts):
        return [list(vector) for _ in texts]

    return embed


# --- _clip01 / _jaccard re-exports from retrieval.similarity (consolidation regression) --
def test_clip01_still_clamps_to_unit_interval():
    import src.cycles.extraction as extraction_module

    assert extraction_module._clip01(-0.2) == 0.0
    assert extraction_module._clip01(1.5) == 1.0
    assert extraction_module._clip01(0.3) == pytest.approx(0.3, abs=1e-6)


def test_jaccard_still_matches_expected_values():
    import src.cycles.extraction as extraction_module

    assert extraction_module._jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3, abs=1e-6)
    assert extraction_module._jaccard(set(), {"a"}) == 0.0
    assert extraction_module._jaccard({"a"}, {"a"}) == pytest.approx(1.0, abs=1e-6)


# --- scope-gap formula ScopeGap ------------------------------------------------------------------
def test_scope_gap_is_zero_for_a_fully_specified_claim():
    score = scope_gap(
        present_facets=tuple(gcd.SCOPE_GAP_FACET_WEIGHTS),  # all 6 facets present
        salient_terms=("a", "b", "c", "d"),
        defined_terms=("a", "b", "c", "d"),
    )
    assert score == pytest.approx(0.0, abs=1e-6)


def test_scope_gap_rises_with_missing_facets_and_undefined_terms():
    # missing population (1.0) + comparator (0.5) -> M_q = 1.5/4.5 = 1/3;
    # 2 of 4 salient terms defined -> U_terms = 0.5; no conflict -> B = 0.
    # ScopeGap = 0.5*(1/3) + 0.2*0.5 + 0.3*0 = 1/6 + 0.1 = 0.26666667.
    present = tuple(f for f in gcd.SCOPE_GAP_FACET_WEIGHTS if f not in {"population", "comparator"})
    score = scope_gap(
        present_facets=present,
        salient_terms=("a", "b", "c", "d"),
        defined_terms=("a", "b"),
    )
    assert score == pytest.approx(0.26666667, abs=1e-6)


def test_scope_gap_strictly_greater_with_missing_than_without():
    present = tuple(gcd.SCOPE_GAP_FACET_WEIGHTS)
    terms = ("a", "b", "c", "d")
    full = scope_gap(present_facets=present, salient_terms=terms, defined_terms=terms)
    missing = scope_gap(
        present_facets=present[:-1], salient_terms=terms, defined_terms=terms
    )
    assert missing > full


def test_scope_gap_blocker_term_raises_the_score():
    present = tuple(f for f in gcd.SCOPE_GAP_FACET_WEIGHTS if f not in {"population", "comparator"})
    kwargs = dict(present_facets=present, salient_terms=("a", "b", "c", "d"), defined_terms=("a", "b"))
    without = scope_gap(**kwargs)
    with_conflict = scope_gap(**kwargs, scope_conflict=True)
    assert with_conflict == pytest.approx(without + gcd.SCOPE_GAP_WEIGHTS["blockers"], abs=1e-6)
    assert with_conflict == pytest.approx(0.56666667, abs=1e-6)


def test_scope_gap_is_clipped_to_unit_interval():  # scope-gap formula clip01
    score = scope_gap(
        present_facets=(),  # everything missing -> M_q = 1
        salient_terms=("a", "b"),
        defined_terms=(),  # nothing defined -> U_terms = 1
        scope_conflict=True,
        granularity_conflict=True,  # B = 1
    )
    assert score == pytest.approx(1.0, abs=1e-6)


# --- Extraction-confidence formula -----------------------------------------------------
def test_extract_conf_full_support_no_assumption():
    # (0.45*1 + 0.35*1 - 0.40*0) / (0.45+0.35+0.40) = 0.8/1.2 = 0.66666667.
    score = extract_conf(span_support=1.0, qualifier_coverage=1.0, assumption_penalty=0.0)
    assert score == pytest.approx(0.66666667, abs=1e-6)


def test_extract_conf_assumption_penalty_lowers_score():
    # (0.45*0.5 + 0.35*0.5 - 0.40*0.25) / 1.2 = (0.225+0.175-0.10)/1.2 = 0.25.
    score = extract_conf(span_support=0.5, qualifier_coverage=0.5, assumption_penalty=0.25)
    assert score == pytest.approx(0.25, abs=1e-6)


def test_extract_conf_is_clipped_at_zero():  # extraction-confidence formula clip01
    score = extract_conf(span_support=0.0, qualifier_coverage=0.0, assumption_penalty=1.0)
    assert score == pytest.approx(0.0, abs=1e-6)


# --- Merge-safety formula and type compatibility ---------------------------------------
def test_type_conflict_grades_over_the_closed_enum():
    assert type_conflict("outcome", "outcome") == pytest.approx(0.0, abs=1e-6)  # same type
    # same compatibility class {mechanism, mediator, confounder}
    assert type_conflict("mechanism", "mediator") == pytest.approx(0.3, abs=1e-6)
    assert type_conflict("construct", "variable") == pytest.approx(0.3, abs=1e-6)
    # different class -> full penalty
    assert type_conflict("population", "outcome") == pytest.approx(1.0, abs=1e-6)


def test_type_conflict_reconciles_synthesist_mine_moderator_and_condition():
    # The Research Synthesist emits ``moderator`` and ``condition`` beyond the extraction
    # vocabulary. They map to the causal-structure compatibility class, so differences among
    # third-variable roles receive a soft 0.3 penalty instead of a spurious 1.0.
    assert type_conflict("moderator", "moderator") == pytest.approx(0.0, abs=1e-6)
    assert type_conflict("condition", "condition") == pytest.approx(0.0, abs=1e-6)
    assert type_conflict("moderator", "mediator") == pytest.approx(0.3, abs=1e-6)
    assert type_conflict("moderator", "confounder") == pytest.approx(0.3, abs=1e-6)
    assert type_conflict("condition", "mechanism") == pytest.approx(0.3, abs=1e-6)
    assert type_conflict("moderator", "condition") == pytest.approx(0.3, abs=1e-6)
    # still a hard 1.0 against a different class (operational or PICO-core)
    assert type_conflict("moderator", "variable") == pytest.approx(1.0, abs=1e-6)
    assert type_conflict("condition", "population") == pytest.approx(1.0, abs=1e-6)


def test_merge_score_high_for_identical_compatible_concepts():
    a = build_node(label="diabetes mellitus", type="outcome", definition="elevated blood glucose")
    b = build_node(label="diabetes mellitus", type="outcome", definition="elevated blood glucose")
    embedder = _constant_embedder([1.0, 0.0])  # identical text -> cos = 1
    # (0.273*1 + 0.318*1 + 0.182*1 - 0.227*0) / 1.0 = 0.773.
    score = merge_score(a, b, embedder=embedder)
    assert score == pytest.approx(0.773, abs=1e-6)
    assert score >= gcd.THETA_MERGE


def test_merge_score_type_conflict_drops_below_threshold():
    a = build_node(label="diabetes mellitus", type="outcome", definition="elevated blood glucose")
    b = build_node(label="diabetes mellitus", type="population", definition="elevated blood glucose")
    embedder = _constant_embedder([1.0, 0.0])
    compatible = merge_score(a, build_node(label="diabetes mellitus", type="outcome",
                                           definition="elevated blood glucose"), embedder=embedder)
    incompatible = merge_score(a, b, embedder=embedder)
    # (0.273*1 + 0.318*1 + 0.182*1 - 0.227*1) / 1.0 = 0.546.
    assert incompatible == pytest.approx(0.546, abs=1e-6)
    assert incompatible < compatible            # the type conflict strictly lowers the score
    assert incompatible < gcd.THETA_MERGE       # below the merge decision -> keep distinct


def test_merge_score_uses_label_aliases_for_embed_text():
    node = build_node(label="heart attack", type="outcome", aliases=["myocardial infarction", "mi"])
    assert concept_embed_text(node) == "heart attack myocardial infarction mi"


def test_default_merge_embedder_wires_the_pinned_specter2_model(monkeypatch):
    # Production callers get the pinned embedder from scoring_defaults WITHOUT loading the
    # real model: the loader call is captured via monkeypatch (the default suite never loads).
    from src.retrieval import scoring_defaults as sd

    captured = {}

    def _fake_loader(model_name, revision=None):
        captured["model"] = model_name
        captured["revision"] = revision
        return lambda texts: [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        "src.cycles.extraction.load_specter2_embedder", _fake_loader
    )
    embedder = default_merge_embedder()
    assert captured["model"] == sd.EMBEDDING_MODEL == "allenai/specter2_base"
    assert captured["revision"] == sd.EMBEDDING_REVISION  # pinned model revision
    assert embedder(["x"]) == [[1.0, 0.0]]  # the loader's embedder is returned unchanged


def test_default_merge_embedder_returns_none_when_extra_absent(monkeypatch):  # loud fallback
    monkeypatch.setattr(
        "src.cycles.extraction.load_specter2_embedder",
        lambda _model, _revision=None: None,
    )
    assert default_merge_embedder() is None  # callers then supply a stub / handle fallback


# --- extraction-delta builder Δ^extract builder ---------------------------------------------------------
def _commit(delta, store, log=None, validator=None):
    from src.transaction_log import GraphTransactionLog
    from src.validator import GraphDeltaValidator

    log = log or GraphTransactionLog()
    validator = validator or GraphDeltaValidator()
    return log.commit(store, delta, validator, author="atom_relation_extractor",
                      timestamp="2026-06-14T00:00:00+00:00")


def test_builder_emits_add_node_and_add_edge_ops_landing_unverified():
    store = CausalClaimGraphStore()
    n1 = build_node(label="smoking", type="exposure/intervention")
    n2 = build_node(label="lung cancer", type="outcome")
    edge = build_edge(source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
                      direction="causal", relation_type="increases")
    delta = build_extract_delta(
        base_graph_hash=store.base_hash, nodes=[n1, n2], edges=[edge],
        embedder=_constant_embedder([1.0, 0.0]),
    )
    assert delta.family == DeltaFamily.EXTRACT
    op_types = [type(op) for op in delta.payload.operations]
    assert AddNodeOp in op_types and AddEdgeOp in op_types
    result = _commit(delta, store)
    assert result.accepted is True
    assert store.version == 1
    assert all(e.status == EdgeStatus.UNVERIFIED for e in store.edges.values())
    assert len(store.nodes) == 2 and len(store.edges) == 1


def test_builder_rejects_a_non_unverified_edge_at_commit():
    store = CausalClaimGraphStore()
    n1 = build_node(label="a", type="construct")
    n2 = build_node(label="b", type="outcome")
    edge = build_edge(source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
                      direction="causal", relation_type="increases")
    edge.status = EdgeStatus.SUPPORTED  # illegal landing status for Δ^extract
    delta = build_extract_delta(base_graph_hash=store.base_hash, nodes=[n1, n2], edges=[edge],
                                embedder=_constant_embedder([1.0, 0.0]))
    result = _commit(delta, store)
    assert result.accepted is False
    assert result.gate_results.failing_gate == "allowed_transition"
    assert store.version == 0  # the graph remains unchanged after rejection


def test_builder_thresholded_merge_emits_merge_nodes_op_that_redirects_and_dedups():
    store = CausalClaimGraphStore()
    # a/b are near-synonyms (distinct node_ids: different norm(label)); cross-aliased so the
    # label Jaccard + identical definition + cos=1 push Merge >= theta_merge.
    a = build_node(label="myocardial infarction", type="outcome", definition="d",
                   aliases=["heart attack"])
    b = build_node(label="heart attack", type="outcome", definition="d")
    c = build_node(label="aspirin", type="exposure/intervention")
    edge_ca = build_edge(source_node_ids=[c.node_id], target_node_ids=[a.node_id],
                         direction="causal", relation_type="reduces")
    edge_cb = build_edge(source_node_ids=[c.node_id], target_node_ids=[b.node_id],
                         direction="causal", relation_type="reduces")
    edge_ca.open_risks = ["risk-a"]
    edge_cb.open_risks = ["risk-b"]
    embedder = _stub_embedder({
        concept_embed_text(a): [1.0, 0.0],
        concept_embed_text(b): [1.0, 0.0],  # identical vector -> cos = 1
    })
    # confirm the decision fires
    assert merge_score(a, b, embedder=embedder) >= gcd.THETA_MERGE

    delta = build_extract_delta(
        base_graph_hash=store.base_hash, nodes=[a, b, c], edges=[edge_ca, edge_cb],
        merge_pairs=[(a, b)], embedder=embedder,
    )
    assert any(isinstance(op, MergeNodesOp) for op in delta.payload.operations)
    result = _commit(delta, store)
    assert result.accepted is True

    survivor_id = min(a.node_id, b.node_id)
    merged_id = max(a.node_id, b.node_id)
    assert merged_id not in store.nodes and survivor_id in store.nodes   # retired + survives
    assert len(store.edges) == 1                                          # c->a and c->b collapse
    survived = next(iter(store.edges.values()))
    assert survived.target_node_ids == [survivor_id]
    assert set(survived.open_risks) == {"risk-a", "risk-b"}               # list union


def test_builder_no_merge_when_below_threshold_keeps_both_nodes():
    store = CausalClaimGraphStore()
    a = build_node(label="apple", type="construct")
    b = build_node(label="unrelated outcome", type="outcome")
    embedder = _stub_embedder({concept_embed_text(a): [1.0, 0.0], concept_embed_text(b): [0.0, 1.0]})
    assert merge_score(a, b, embedder=embedder) < gcd.THETA_MERGE
    delta = build_extract_delta(base_graph_hash=store.base_hash, nodes=[a, b],
                                merge_pairs=[(a, b)], embedder=embedder)
    assert not any(isinstance(op, MergeNodesOp) for op in delta.payload.operations)
    result = _commit(delta, store)
    assert result.accepted is True
    assert len(store.nodes) == 2  # both preserved as distinct constructs


# --- run_extraction_cycle (orchestration) -----------------------------------------------
class _FakeExtractor:
    def __init__(self, extraction):
        self._extraction = extraction

    def extract(self, claim):
        return self._extraction


def test_run_extraction_cycle_commits_a_normalized_subgraph():
    n1 = build_node(label="exercise", type="exposure/intervention",
                    provenance=[{"source": "claim", "span": [0, 8]}])
    n2 = build_node(label="mortality", type="outcome")
    edge = build_edge(source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
                      direction="causal", relation_type="reduces")
    extraction = ClaimExtraction(
        nodes=(n1, n2), edges=(edge,),
        present_facets=("population", "exposure/intervention", "outcome"),
        salient_terms=("exercise", "mortality"), defined_terms=("exercise",),
    )
    result = run_extraction_cycle(
        "exercise reduces mortality", extractor=_FakeExtractor(extraction),
        embedder=_constant_embedder([1.0, 0.0]),
    )
    assert result.transaction.accepted is True
    assert result.store.version == 1
    assert len(result.store.nodes) == 2 and len(result.store.edges) == 1
    assert all(e.status == EdgeStatus.UNVERIFIED for e in result.store.edges.values())
    assert result.scope_gap is not None and 0.0 <= result.scope_gap <= 1.0  #  audit


# --- LLM Extractor (hermetic: fake model backend; the real path is the live test) -------
class _FakeBackend:
    def __init__(self, content):
        self._content = content
        self.calls = 0

    def run(self, messages, tools=None):
        self.calls += 1
        return {"choices": [{"message": {"content": self._content}}]}


def test_llm_extractor_parses_nodes_and_edges_from_json():
    content = (
        '{"nodes": ['
        '{"label": "smoking", "type": "exposure/intervention", "definition": "tobacco use"},'
        '{"label": "lung cancer", "type": "outcome", "definition": "malignant lung tumor"}],'
        '"edges": [{"source": "smoking", "target": "lung cancer",'
        ' "direction": "causal", "relation_type": "increases"}],'
        '"assumptions": ["adult population"]}'
    )
    extractor = LLMExtractor(_FakeBackend(content))
    extraction = extractor.extract("smoking causes lung cancer")
    assert {n.label for n in extraction.nodes} == {"smoking", "lung cancer"}
    assert all(n.node_id for n in extraction.nodes)  # content-addressed ids assigned
    assert len(extraction.edges) == 1
    edge = extraction.edges[0]
    labels = {n.node_id: n.label for n in extraction.nodes}
    assert labels[edge.source_node_ids[0]] == "smoking"
    assert labels[edge.target_node_ids[0]] == "lung cancer"
    assert edge.status == EdgeStatus.UNVERIFIED
    assert extraction.assumptions == ("adult population",)


def test_llm_extractor_drops_edges_with_unknown_endpoints():
    content = (
        '{"nodes": [{"label": "smoking", "type": "exposure/intervention"}],'
        '"edges": [{"source": "smoking", "target": "ghost", '
        '"direction": "causal", "relation_type": "increases"}]}'
    )
    extraction = LLMExtractor(_FakeBackend(content)).extract("smoking")
    assert len(extraction.nodes) == 1
    assert extraction.edges == ()  # the dangling edge is dropped, not committed as a bad ref


def test_llm_extractor_feeds_the_claim_verbatim_to_the_backend():
    class _Capturing(_FakeBackend):
        def __init__(self):
            super().__init__('{"nodes": []}')
            self.seen = None

        def run(self, messages, tools=None):
            self.seen = messages
            return super().run(messages, tools=tools)

    cap = _Capturing()
    LLMExtractor(cap).extract("aspirin reduces myocardial infarction")
    joined = " ".join(str(m.get("content", "")) for m in cap.seen)
    assert "aspirin reduces myocardial infarction" in joined  # claim fed verbatim


def test_llm_extractor_prompt_accepts_research_goal_seed():  # Extraction input scoping
    class _Capturing(_FakeBackend):
        def __init__(self):
            super().__init__('{"nodes": []}')
            self.seen = None

        def run(self, messages, tools=None):
            self.seen = messages
            return super().run(messages, tools=tools)

    cap = _Capturing()
    seed = "Find mechanisms that could make low-bit quantization reduce factual consistency."
    LLMExtractor(cap).extract(seed)

    system = cap.seen[0]["content"]
    user = cap.seen[1]["content"]
    assert "single claim, research goal, question, or raw message" in system
    assert "seed's structure" in system
    assert "Claim:\n" not in user
    assert "Seed input:" in user
    assert seed in user


# --- Live LLM-backed extraction ---------------------------------------------------------
@pytest.mark.live
def test_live_real_extractor_commits_an_unverified_subgraph():
    # Provider-agnostic gate: the shipped config may point `builder` at a key-free CLI
    # subagent or at an API provider, so skip on whatever that backend actually needs
    # (missing CLI binary, missing API key) rather than on one hard-coded variable.
    from src.camel_adapter import _create_direct_model_backend
    from src.config import load_config

    config = load_config(Path("config/evidence-evaluation.yaml"))
    try:
        backend = _create_direct_model_backend(config.agents.builder, role_name="builder")
    except RuntimeError as exc:  # pragma: no cover - depends on local credentials
        pytest.skip(f"builder backend unavailable for the live Extractor path: {exc}")

    result = run_extraction_cycle(
        "Regular physical activity reduces all-cause mortality in older adults.",
        extractor=LLMExtractor(backend),
        embedder=_constant_embedder([1.0, 0.0]),  # no merge candidates in the MVP extractor
    )

    assert result.transaction.accepted is True
    assert result.store.version == 1
    assert len(result.store.nodes) >= 2  # a real claim yields >= 2 concepts
    # Every committed edge lands unverified without evidence-derived confidence.
    assert all(e.status == EdgeStatus.UNVERIFIED for e in result.store.edges.values())
    assert all(e.confidence is None for e in result.store.edges.values())
    assert result.store.evidence_links == []


@pytest.mark.live
def test_live_pinned_specter2_merge_is_reproducible():
    embedder = default_merge_embedder()
    if embedder is None:
        pytest.skip("specter2_base unavailable (retrieval-coherence extra / model load failed)")
    a = build_node(label="myocardial infarction", type="outcome", definition="death of heart muscle")
    b = build_node(label="heart attack", type="outcome", definition="death of heart muscle")
    first = merge_score(a, b, embedder=embedder)
    second = merge_score(a, b, embedder=embedder)
    assert first == pytest.approx(second, abs=1e-6)  # identical inputs -> identical score
    assert 0.0 <= first <= 1.0


# --- Canonicalization and merge hardening ----------------------------------------------
# Synonyms with different surface forms can land below the merge threshold, so an LLM
# adjudicator decides pairs in the configured borderline band. Decisions outside the band
# stay deterministic; the default suite uses a fixed adjudicator.
class _FakeAdjudicator:
    """A fixtured 'same concept?' judge: True iff both labels are in one declared same-group.
    Records its calls so tests can assert the deterministic legs never reach it. The real
    LLM-backed adjudicator is the `live` path."""

    def __init__(self, same_groups):
        self.same_groups = [set(g) for g in same_groups]
        self.calls = []

    def same_concept(self, c_a, c_b):
        self.calls.append((c_a.label, c_b.label))
        return any(c_a.label in g and c_b.label in g for g in self.same_groups)


def _orthobasis_embedder(label_to_axis, dims=4):
    """Embed each node's label+aliases anchor onto a chosen axis (cos=1 same axis, 0 across)."""

    def embed(texts):
        vectors = []
        for text in texts:
            axis = next((ax for lbl, ax in label_to_axis.items() if lbl in text), 0)
            vec = [0.0] * dims
            vec[axis] = 1.0
            vectors.append(vec)
        return vectors

    return embed


# Synonym family (the motivating case): same axis -> cos high; distinct definitions/labels
# so the deterministic merge_score lands in the adjudication band, not above theta_merge.
def _synonym_family():
    return [
        build_node(label="reconstruction error", type="mechanism",
                   definition="discarded spectral energy magnitude"),
        build_node(label="truncation error", type="mechanism",
                   definition="frobenius norm of the removed singular components"),
        build_node(label="compression loss", type="mechanism",
                   definition="accuracy drop after low rank approximation"),
    ]


# --- merge_decision: band + adjudicator -------------------------------------------------
def test_merge_decision_clearly_distinct_skips_adjudicator():  # deterministically below floor
    a = build_node(label="activation outlier severity", type="mechanism")
    b = build_node(label="fisher information weighting", type="mechanism")
    emb = _orthobasis_embedder({"activation outlier severity": 1, "fisher information weighting": 2})
    adj = _FakeAdjudicator([])
    assert merge_decision(a, b, embedder=emb, adjudicator=adj) is False
    assert adj.calls == []  # below the floor -> confidently distinct, LLM never consulted


def test_merge_decision_clearly_same_skips_adjudicator():  # deterministically above ceiling
    a = build_node(label="diabetes mellitus", type="outcome", definition="elevated blood glucose")
    b = build_node(label="diabetes mellitus", type="outcome", definition="elevated blood glucose")
    adj = _FakeAdjudicator([])
    assert merge_decision(a, b, embedder=_constant_embedder([1.0, 0.0]), adjudicator=adj) is True
    assert adj.calls == []  # identical -> merge_score above the ceil, LLM never consulted


def test_merge_decision_borderline_consults_adjudicator():  # LLM decides within the band
    a, b, _ = _synonym_family()
    emb = _orthobasis_embedder({"reconstruction error": 1, "truncation error": 1})
    score = merge_score(a, b, embedder=emb)
    assert gcd.THETA_MERGE - gcd.CANONICALIZATION_BAND_LOW <= score < gcd.THETA_MERGE  # in band, below theta
    yes = _FakeAdjudicator([{"reconstruction error", "truncation error"}])
    no = _FakeAdjudicator([])
    assert merge_decision(a, b, embedder=emb, adjudicator=yes) is True
    assert merge_decision(a, b, embedder=emb, adjudicator=no) is False
    assert yes.calls and no.calls  # the band was delegated to the adjudicator


def test_merge_decision_without_adjudicator_falls_back_to_theta_merge():  # backward-compat
    a, b, _ = _synonym_family()
    emb = _orthobasis_embedder({"reconstruction error": 1, "truncation error": 1})
    score = merge_score(a, b, embedder=emb)
    # Without an adjudicator, use the ordinary threshold decision (score >= theta_merge).
    assert merge_decision(a, b, embedder=emb, adjudicator=None) == (score >= gcd.THETA_MERGE)


def test_bare_threshold_misses_compressed_cosine_synonym_that_llm_recovers():
    a, b, _ = _synonym_family()
    emb = _orthobasis_embedder({"reconstruction error": 1, "truncation error": 1})
    assert merge_score(a, b, embedder=emb) < gcd.THETA_MERGE          # bare threshold: NO merge
    llm = _FakeAdjudicator([{"reconstruction error", "truncation error"}])
    assert merge_decision(a, b, embedder=emb, adjudicator=llm) is True  # LLM recovers the merge


# --- canonicalize_concepts: cluster a concept set ---------------------------------------
def test_canonicalize_collapses_synonym_family_without_collapsing_distinct():
    family = _synonym_family()
    distinct = [
        build_node(label="activation outlier severity", type="mechanism"),
        build_node(label="fisher information weighting", type="mechanism"),
    ]
    nodes = family + distinct
    emb = _orthobasis_embedder({
        "reconstruction error": 1, "truncation error": 1, "compression loss": 1,
        "activation outlier severity": 2, "fisher information weighting": 3,
    })
    adj = _FakeAdjudicator([{"reconstruction error", "truncation error", "compression loss"}])
    clusters = canonicalize_concepts(nodes, embedder=emb, adjudicator=adj)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 1, 3]                                  # family -> one cluster; 2 singletons
    family_ids = {n.node_id for n in family}
    assert any(set(c) == family_ids for c in clusters)         # the 3 synonyms canonicalize together


def test_canonicalize_keeps_distinct_works_distinct_via_adjudicator():
    # Two related methods have high label and embedding overlap, but remain distinct concepts.
    first_method = build_node(label="svd llm baseline", type="construct", definition="baseline method")
    second_method = build_node(label="svd llm variant", type="construct", definition="distinct variant")
    emb = _stub_embedder({"svd llm baseline": [1.0, 0.0], "svd llm variant": [0.8, 0.6]})
    score = merge_score(first_method, second_method, embedder=emb)
    assert gcd.THETA_MERGE - gcd.CANONICALIZATION_BAND_LOW <= score < gcd.THETA_MERGE + gcd.CANONICALIZATION_BAND_HIGH
    adj = _FakeAdjudicator([])  # the LLM says "not the same concept"
    clusters = canonicalize_concepts([first_method, second_method], embedder=emb, adjudicator=adj)
    assert sorted(len(c) for c in clusters) == [1, 1]  # kept distinct despite high similarity


def test_canonicalize_is_deterministic_given_fixed_adjudications():
    nodes = _synonym_family()
    emb = _orthobasis_embedder({"reconstruction error": 1, "truncation error": 1, "compression loss": 1})
    adj_a = _FakeAdjudicator([{"reconstruction error", "truncation error", "compression loss"}])
    adj_b = _FakeAdjudicator([{"reconstruction error", "truncation error", "compression loss"}])
    first = sorted(sorted(c) for c in canonicalize_concepts(nodes, embedder=emb, adjudicator=adj_a))
    second = sorted(sorted(c) for c in canonicalize_concepts(nodes, embedder=emb, adjudicator=adj_b))
    assert first == second


# --- wiring into build_extract_delta ----------------------------------------------------
def test_build_extract_delta_uses_adjudicator_for_borderline_merges():  #  wiring
    a, b, _ = _synonym_family()
    emb = _orthobasis_embedder({"reconstruction error": 1, "truncation error": 1})
    llm = _FakeAdjudicator([{"reconstruction error", "truncation error"}])
    # Without the adjudicator the borderline synonym pair is NOT merged (bare threshold);
    # with it, a merge_nodes op appears.
    no_adj = build_extract_delta(base_graph_hash="h0", nodes=[], merge_pairs=[(a, b)], embedder=emb)
    with_adj = build_extract_delta(base_graph_hash="h0", nodes=[], merge_pairs=[(a, b)],
                                   embedder=emb, adjudicator=llm)
    assert not any(op.op_type == "merge_nodes" for op in no_adj.payload.operations)
    assert any(op.op_type == "merge_nodes" for op in with_adj.payload.operations)


# --- LLM canonicalization adjudicator with a fake backend -------------------------------
def _concept(label, *, type="variable", definition=""):
    return build_node(label=label, definition=definition, type=type)


def test_llm_adjudicator_returns_true_when_backend_says_same():
    from src.cycles.extraction import LLMCanonicalizationAdjudicator

    backend = _FakeBackend('{"same_concept": true, "reason": "synonyms"}')
    adj = LLMCanonicalizationAdjudicator(backend)
    assert adj.same_concept(_concept("reconstruction error"), _concept("truncation error")) is True


def test_llm_adjudicator_returns_false_when_backend_says_distinct():
    from src.cycles.extraction import LLMCanonicalizationAdjudicator

    backend = _FakeBackend('{"same_concept": false, "reason": "adjacent not same"}')
    adj = LLMCanonicalizationAdjudicator(backend)
    assert adj.same_concept(_concept("Fisher importance"), _concept("off-diagonal Fisher")) is False


def test_llm_adjudicator_degrades_to_false_on_malformed_response():
    from src.cycles.extraction import LLMCanonicalizationAdjudicator

    assert LLMCanonicalizationAdjudicator(_FakeBackend("not json at all")).same_concept(
        _concept("a"), _concept("b")
    ) is False
    assert LLMCanonicalizationAdjudicator(_FakeBackend("")).same_concept(
        _concept("a"), _concept("b")
    ) is False


def test_llm_adjudicator_feeds_both_concepts_to_the_backend():
    class _Capturing(_FakeBackend):
        def __init__(self):
            super().__init__('{"same_concept": false}')
            self.seen = None

        def run(self, messages, tools=None):
            self.seen = messages
            return super().run(messages, tools=tools)

    from src.cycles.extraction import LLMCanonicalizationAdjudicator

    cap = _Capturing()
    LLMCanonicalizationAdjudicator(cap).same_concept(
        _concept("reconstruction error"), _concept("truncation error")
    )
    joined = " ".join(str(m.get("content", "")) for m in cap.seen)
    assert "reconstruction error" in joined and "truncation error" in joined


def test_llm_adjudicator_drives_merge_decision_in_the_band():
    a, b, _ = _synonym_family()
    emb = _orthobasis_embedder({"reconstruction error": 1, "truncation error": 1})
    from src.cycles.extraction import LLMCanonicalizationAdjudicator

    yes = LLMCanonicalizationAdjudicator(_FakeBackend('{"same_concept": true}'))
    no = LLMCanonicalizationAdjudicator(_FakeBackend('{"same_concept": false}'))
    assert merge_decision(a, b, embedder=emb, adjudicator=yes) is True
    assert merge_decision(a, b, embedder=emb, adjudicator=no) is False


# --- Audited canonicalization configuration -------------------------------------------
def test_canonicalization_band_present_and_audited():
    # Asymmetric band: wide BELOW theta_merge (catches compressed-cosine synonyms), narrow above.
    assert gcd.CANONICALIZATION_BAND_LOW > gcd.CANONICALIZATION_BAND_HIGH
    assert 0.0 <= gcd.THETA_MERGE - gcd.CANONICALIZATION_BAND_LOW
    audit = gcd.audit_dict()
    assert audit["canonicalization_band_low"] == gcd.CANONICALIZATION_BAND_LOW
    assert audit["canonicalization_band_high"] == gcd.CANONICALIZATION_BAND_HIGH
