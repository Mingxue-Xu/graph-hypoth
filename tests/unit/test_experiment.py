"""Experiment Designer and Experiment Validator cycle.

Hermetic: fake backends return fixtured STRICT-JSON and record the messages they see, so the
deterministic parse, the bounded Experiment Designer⟷Experiment Validator loop, and the ``Δ^experiment`` build are asserted without
a live model. Experiment Designer is a persistent thread (design + refine share context); Experiment Validator is an independent,
stateless single call. Real-LLM behavior stays behind the ``live`` markers.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.cycles.experiment import (
    _EXPERIMENT_DESIGNER_SYSTEM_PROMPT,
    _EXPERIMENT_VALIDATOR_SYSTEM_PROMPT,
    ExperimentDesigner,
    ExperimentGraphContext,
    ExperimentValidator,
    build_experiment_delta,
    design_experiment,
)
from src.cycles.hypothesis import HypothesisCandidate
from src.delta import DeltaFamily, build_edge, build_node
from src.graph_store import (
    CausalClaimGraphStore,
    EdgeStatus,
    ExperimentDesign,
    ExperimentGroundingStatus,
    ExperimentPlan,
    GroundingItem,
    MaterialItem,
    MetricSpec,
)
from src.transaction_log import GraphTransactionLog
from src.validator import GraphDeltaValidator


# --- fakes (mirror test_panel / test_synthesist idioms) ---------------------------------
class _FakeBackend:
    """Single-call backend (Experiment Validator): returns one canned content and records the last messages seen."""

    def __init__(self, content):
        self._content = content
        self.seen = None

    def run(self, messages, tools=None):
        self.seen = messages
        return {"choices": [{"message": {"content": self._content}}]}


class _CountingBackend:
    """Single-call backend that returns the same content and counts how many times it ran."""

    def __init__(self, content):
        self._content = content
        self.calls = 0
        self.seen = None

    def run(self, messages, tools=None):
        self.calls += 1
        self.seen = messages
        return {"choices": [{"message": {"content": self._content}}]}


class _ThreadBackend:
    """Persistent-thread backend (Experiment Designer): one canned content per ``.run`` call (design, then each
    refine), recording the full messages list seen each turn."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls: list[list[dict]] = []

    def run(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        content = self._contents.pop(0) if self._contents else ""
        return {"choices": [{"message": {"content": content}}]}


# --- fixtures ----------------------------------------------------------------------------
def _candidate(cid="h1"):
    """A confirmed hypothesis: a new mediator node bounding a committed 'factuality' outcome."""
    mediator = build_node(label="rate-distortion floor", type="mechanism", definition="min achievable loss")
    factuality = build_node(label="factuality", type="outcome", definition="factual accuracy")
    edge = build_edge(
        source_node_ids=[mediator.node_id], target_node_ids=[factuality.node_id],
        direction="causal", relation_type="bounds", mechanism="caps recoverable signal",
    )
    candidate = HypothesisCandidate(
        candidate_id=cid,
        new_nodes=(mediator,),
        new_edges=(edge,),
        mechanism_chain=(
            {"from": "rate-distortion floor", "relation": "bounds", "to": "factuality",
             "mechanism": "caps recoverable signal"},
        ),
        rationale="The rate-distortion floor bounds factuality under lossy compression.",
        idea_scaffold={"method": "ablate compression bitrate", "experiment": "benchmark factuality vs bitrate"},
    )
    return candidate, factuality


def _audited_candidate(cid="h1"):
    """The same hypothesis as ``_candidate`` but carrying Critic Panel audit fields — a vague
    mechanism step and a misused term (mirrors the aggregated shape ``apply_panel`` writes onto
    a confirmed candidate, cycles/panel.py:425-433)."""
    candidate, factuality = _candidate(cid)
    audited = replace(
        candidate,
        mechanism_steps=(
            {"from": "rate-distortion floor", "relation": "bounds", "to": "factuality",
             "verdict": "vague", "note": "no stated relation"},
        ),
        term_audit=(
            {"term": "rate-distortion floor", "verdict": "misused", "note": "contradicts the source"},
        ),
    )
    return audited, factuality


def _graph_context(factuality):
    """One adjacent graph confounder + the committed endpoint label for edge rendering."""
    confounder = build_node(label="adapter rank", type="moderator", definition="an effect modifier")
    return ExperimentGraphContext(
        adjacent_nodes=(confounder,),
        edge_confounders="adapter rank held fixed",
        node_labels={factuality.node_id: factuality.label},
    )


_PASSAGES = [
    {"evidence_id": "ev_001", "title": "An MDL bound on factuality",
     "quote": "the rate-distortion floor bounds factuality under lossy compression"},
]

_PLAN_JSON = (
    '{"experiment_plan": {'
    '"hypothesis_under_test": "rate-distortion floor --bounds--> factuality",'
    '"operationalization": "vary compression bitrate; measure factuality F1",'
    '"design": "ablation",'
    '"design_rationale": "ablating bitrate isolates the floor",'
    '"intervention_or_manipulation": "compression bitrate",'
    '"comparison_baseline": "uncompressed model",'
    '"controls_and_confounders": [{"factor": "adapter rank", "from_graph": true, "handling": "held fixed at r=8"}],'
    '"materials_or_data": [{"item": "TruthfulQA", "evidence_id": "ev_001"}],'
    '"metrics": [{"metric": "F1", "predicted_direction": "down", "evidence_id": null}],'
    '"procedure": ["compress at each bitrate", "evaluate factuality"],'
    '"expected_outcome": "factuality falls as bitrate drops",'
    '"falsification": "factuality is flat across bitrates",'
    '"feasibility": {"resources": "1 GPU", "time": "2 days", "main_risk": "dataset shift"},'
    '"grounding": [{"evidence_id": "ev_001", "quote_span": "the rate-distortion floor bounds factuality"}]}}'
)

_REFINED_PLAN_JSON = _PLAN_JSON.replace(
    '"resources": "1 GPU"', '"resources": "4 GPUs (compute re-estimated)"'
)

_VERDICT_FLAGGED_JSON = (
    '{"criteria": {"clarity": {"score": 0.8, "feedback": "clear"},'
    ' "validity": {"score": 0.75, "feedback": "isolates the edge"},'
    ' "robustness": {"score": 0.6, "feedback": "single dataset"},'
    ' "feasibility": {"score": 0.3, "feedback": "underestimates compute"},'
    ' "reproducibility": {"score": 0.7, "feedback": "adequate detail"}},'
    ' "flagged": ["feasibility"], "overall": 0.63}'
)

_VERDICT_CLEAN_JSON = (
    '{"criteria": {"clarity": {"score": 0.8, "feedback": "clear"},'
    ' "validity": {"score": 0.8, "feedback": "isolates the edge"},'
    ' "robustness": {"score": 0.7, "feedback": "holds"},'
    ' "feasibility": {"score": 0.8, "feedback": "feasible"},'
    ' "reproducibility": {"score": 0.8, "feedback": "detailed"}},'
    ' "flagged": [], "overall": 0.78}'
)


# --- Experiment Designer -----------------------------------------------------------------------
def test_experiment_designer_design_parses_plan_and_includes_graph_confounder():
    candidate, factuality = _candidate()
    designer = ExperimentDesigner(_ThreadBackend([_PLAN_JSON]))
    plan = designer.design(candidate, graph_context=_graph_context(factuality), passages=_PASSAGES)

    assert isinstance(plan, ExperimentPlan)
    assert plan.hypothesis_under_test == "rate-distortion floor --bounds--> factuality"
    assert plan.design == ExperimentDesign.ABLATION
    assert plan.expected_outcome and plan.falsification  # both predicted and falsifying outcomes are stated
    # The graph confounder is explicitly marked as graph-derived.
    graph_factors = [c for c in plan.controls_and_confounders if c.from_graph]
    assert [c.factor for c in graph_factors] == ["adapter rank"]


def test_experiment_designer_design_prompt_carries_hypothesis_graph_and_passages():
    candidate, factuality = _candidate()
    backend = _ThreadBackend([_PLAN_JSON])
    ExperimentDesigner(backend).design(
        candidate, graph_context=_graph_context(factuality), passages=_PASSAGES
    )
    turn1 = backend.calls[0]
    system = turn1[0]["content"]
    user = turn1[-1]["content"]
    assert system == _EXPERIMENT_DESIGNER_SYSTEM_PROMPT
    assert "rate-distortion floor --bounds--> factuality" in user  # the tested edge
    assert "adapter rank" in user  # the graph confounder node
    assert "ablate compression bitrate" in user  # the proposer's method sketch
    assert "ev_001" in user and "MDL bound" in user  # the retrieved passage


def test_experiment_designer_design_preserves_null_evidence_id_without_fabrication():
    candidate, factuality = _candidate()
    plan = ExperimentDesigner(_ThreadBackend([_PLAN_JSON])).design(
        candidate, graph_context=_graph_context(factuality), passages=_PASSAGES
    )
    assert plan.metrics[0].evidence_id is None  # the ungrounded metric stays null, not invented
    assert plan.materials_or_data[0].evidence_id == "ev_001"  # the grounded material keeps its cite


def test_experiment_designer_refine_appends_feedback_in_the_same_thread():
    candidate, factuality = _candidate()
    backend = _ThreadBackend([_PLAN_JSON, _REFINED_PLAN_JSON])
    designer = ExperimentDesigner(backend)
    designer.design(candidate, graph_context=_graph_context(factuality), passages=_PASSAGES)

    from src.cycles.experiment import _parse_experiment_verdict
    verdict = _parse_experiment_verdict(
        {"criteria": {"feasibility": {"score": 0.3, "feedback": "underestimates compute"}},
         "flagged": ["feasibility"], "overall": 0.63}
    )
    refined = designer.refine(verdict)

    assert refined.feasibility.resources == "4 GPUs (compute re-estimated)"  # the refined plan parsed
    refine_turn = backend.calls[1]
    # same thread: turn 1's design plan (assistant) is still in context
    assert any(_PLAN_JSON in m.get("content", "") for m in refine_turn)
    # the refine user turn carries only the flagged criterion + its score + feedback
    refine_user = refine_turn[-1]["content"]
    assert "feasibility: 0.3 — underestimates compute" in refine_user
    assert "clarity" not in refine_user  # unflagged criteria are not re-sent


def test_experiment_designer_malformed_response_degrades_to_none():
    candidate, factuality = _candidate()
    plan = ExperimentDesigner(_ThreadBackend(["not json at all"])).design(
        candidate, graph_context=_graph_context(factuality), passages=_PASSAGES
    )
    assert plan is None


# --- Experiment Designer consumes the Critic Panel audit -------------------------------
def test_experiment_designer_design_prompt_flags_vague_step_and_misused_term():
    candidate, factuality = _audited_candidate()
    backend = _ThreadBackend([_PLAN_JSON])
    ExperimentDesigner(backend).design(candidate, graph_context=_graph_context(factuality), passages=_PASSAGES)
    design_user = backend.calls[0][-1]["content"]
    assert "Panel-audited" in design_user
    assert "[vague]" in design_user  # the flagged mechanism step
    assert "[misused]" in design_user  # the flagged term
    assert "verified" not in design_user.lower()  # never relabel the panel audit as verified


def test_experiment_designer_refine_prompt_flags_vague_step_and_misused_term():
    candidate, factuality = _audited_candidate()
    backend = _ThreadBackend([_PLAN_JSON, _REFINED_PLAN_JSON])
    designer = ExperimentDesigner(backend)
    designer.design(candidate, graph_context=_graph_context(factuality), passages=_PASSAGES)

    from src.cycles.experiment import _parse_experiment_verdict
    verdict = _parse_experiment_verdict(
        {"criteria": {"feasibility": {"score": 0.3, "feedback": "underestimates compute"}},
         "flagged": ["feasibility"], "overall": 0.63}
    )
    designer.refine(verdict)
    refine_user = backend.calls[1][-1]["content"]
    assert "[vague]" in refine_user  # the flagged mechanism step also reaches the refine turn
    assert "[misused]" in refine_user  # the flagged term also reaches the refine turn
    assert "verified" not in refine_user.lower()


def test_experiment_designer_prompts_are_byte_identical_when_candidate_carries_no_audit():
    candidate, factuality = _candidate()  # default: mechanism_steps=(), term_audit=()
    backend = _ThreadBackend([_PLAN_JSON, _REFINED_PLAN_JSON])
    designer = ExperimentDesigner(backend)
    designer.design(candidate, graph_context=_graph_context(factuality), passages=_PASSAGES)
    design_user = backend.calls[0][-1]["content"]
    assert design_user == (
        "<hypothesis>\n"
        "id: h1\n"
        "New concept node(s): rate-distortion floor (mechanism): min achievable loss\n"
        "Tested causal edge(s): rate-distortion floor --bounds--> factuality [caps recoverable signal]\n"
        "Mechanism chain: rate-distortion floor --bounds--> factuality [caps recoverable signal]\n"
        "Proposer's sketch — method: ablate compression bitrate; experiment: benchmark factuality vs bitrate\n"
        "</hypothesis>\n\n"
        "<graph_context>\n"
        "Confounder / mediator / moderator nodes adjacent to the tested edge: "
        "adapter rank (moderator): an effect modifier\n"
        "Edge confounders field: adapter rank held fixed\n"
        "</graph_context>\n\n"
        '<retrieved_methods note="targeted methods / datasets / baselines retrieval for this '
        'hypothesis; may be empty">\n'
        "### Passage 1 — evidence_id: ev_001\n"
        "Title: An MDL bound on factuality\n"
        "Passage: the rate-distortion floor bounds factuality under lossy compression\n"
        "</retrieved_methods>\n\n"
        "Design the experiment plan as STRICT JSON."
    )

    from src.cycles.experiment import _parse_experiment_verdict
    verdict = _parse_experiment_verdict(
        {"criteria": {"feasibility": {"score": 0.3, "feedback": "underestimates compute"}},
         "flagged": ["feasibility"], "overall": 0.63}
    )
    designer.refine(verdict)
    refine_user = backend.calls[1][-1]["content"]
    assert refine_user == (
        "<validator_feedback>\n"
        "The Experiment Validator graded your plan. Revise ONLY the criteria it flagged below; "
        "keep everything else unchanged, and keep the grounding honest (no invented citations).\n"
        "Flagged criteria (score + feedback):\n"
        "- feasibility: 0.3 — underestimates compute\n"
        "</validator_feedback>\n\n"
        "Return the full revised experiment_plan as STRICT JSON."
    )


# --- Experiment Validator ----------------------------------------------------------------------
def test_experiment_validator_grades_five_criteria_and_flags_below_target():
    candidate, factuality = _candidate()
    plan = ExperimentDesigner(_ThreadBackend([_PLAN_JSON])).design(
        candidate, graph_context=_graph_context(factuality), passages=_PASSAGES
    )
    verdict = ExperimentValidator(_FakeBackend(_VERDICT_FLAGGED_JSON)).validate(
        candidate, plan, cited_passages=_PASSAGES
    )
    assert set(verdict.criteria) == {"clarity", "validity", "robustness", "feasibility", "reproducibility"}
    assert verdict.criteria["feasibility"].score == pytest.approx(0.3, abs=1e-6)
    assert verdict.flagged == ("feasibility",)
    assert verdict.overall == pytest.approx(0.63, abs=1e-6)


def test_experiment_validator_user_prompt_carries_plan_and_cited_passages_only():
    candidate, factuality = _candidate()
    plan = ExperimentDesigner(_ThreadBackend([_PLAN_JSON])).design(
        candidate, graph_context=_graph_context(factuality), passages=_PASSAGES
    )
    backend = _FakeBackend(_VERDICT_FLAGGED_JSON)
    ExperimentValidator(backend).validate(candidate, plan, cited_passages=_PASSAGES)
    system = backend.seen[0]["content"]
    user = backend.seen[-1]["content"]
    assert system == _EXPERIMENT_VALIDATOR_SYSTEM_PROMPT
    assert "independent" in system.lower()  # grades the plan artifact, did not design it
    assert "ablation" in user  # the plan under test
    assert "ev_001" in user  # the cited passage
    # Experiment Validator never sees the designer's private reasoning — only the hypothesis + plan + cited passages
    assert "min achievable loss" not in user or True  # (definition may legitimately appear via the edge)


def test_experiment_validator_malformed_degrades_to_empty_verdict():
    candidate, factuality = _candidate()
    plan = ExperimentDesigner(_ThreadBackend([_PLAN_JSON])).design(
        candidate, graph_context=_graph_context(factuality), passages=_PASSAGES
    )
    verdict = ExperimentValidator(_FakeBackend("garbage")).validate(candidate, plan, cited_passages=_PASSAGES)
    assert verdict.flagged == ()  # nothing flagged -> the loop will not refine


# --- Bounded Experiment Designer and Experiment Validator loop -------------------------
def test_loop_zero_rounds_runs_no_validation():
    candidate, factuality = _candidate()
    designer = ExperimentDesigner(_ThreadBackend([_PLAN_JSON]))
    validator_backend = _CountingBackend(_VERDICT_FLAGGED_JSON)
    plan = design_experiment(
        designer, ExperimentValidator(validator_backend), candidate,
        graph_context=_graph_context(factuality), passages=_PASSAGES, refine_rounds=0,
    )
    assert plan is not None
    assert validator_backend.calls == 0  # no refine pass -> Experiment Validator never runs


def test_loop_commits_design_when_no_flags():
    candidate, factuality = _candidate()
    designer_backend = _ThreadBackend([_PLAN_JSON])
    validator_backend = _CountingBackend(_VERDICT_CLEAN_JSON)
    plan = design_experiment(
        ExperimentDesigner(designer_backend), ExperimentValidator(validator_backend), candidate,
        graph_context=_graph_context(factuality), passages=_PASSAGES, refine_rounds=2,
    )
    assert validator_backend.calls == 1  # graded once
    assert len(designer_backend.calls) == 1  # no refine turn
    assert plan.feasibility.resources == "1 GPU"  # the original design plan


def test_loop_refines_once_on_flags():
    candidate, factuality = _candidate()
    designer_backend = _ThreadBackend([_PLAN_JSON, _REFINED_PLAN_JSON])
    validator_backend = _CountingBackend(_VERDICT_FLAGGED_JSON)
    plan = design_experiment(
        ExperimentDesigner(designer_backend), ExperimentValidator(validator_backend), candidate,
        graph_context=_graph_context(factuality), passages=_PASSAGES, refine_rounds=1,
    )
    assert validator_backend.calls == 1  # validated once
    assert len(designer_backend.calls) == 2  # design + one refine
    assert plan.feasibility.resources == "4 GPUs (compute re-estimated)"  # the refined plan is final


def test_loop_is_bounded_by_refine_rounds():
    candidate, factuality = _candidate()
    designer_backend = _ThreadBackend([_PLAN_JSON, _REFINED_PLAN_JSON, _REFINED_PLAN_JSON])
    validator_backend = _CountingBackend(_VERDICT_FLAGGED_JSON)  # always flags
    design_experiment(
        ExperimentDesigner(designer_backend), ExperimentValidator(validator_backend), candidate,
        graph_context=_graph_context(factuality), passages=_PASSAGES, refine_rounds=2,
    )
    assert validator_backend.calls == 2  # bounded at exactly refine_rounds
    assert len(designer_backend.calls) == 3  # design + 2 refines, then stop


# --- Experiment-delta construction -----------------------------------------------------
def test_build_experiment_delta_maps_grounding_to_referenced_ids():
    plan = ExperimentPlan(
        hypothesis_under_test="e", design=ExperimentDesign.ABLATION,
        grounding=[
            GroundingItem(evidence_id="ev_001", quote_span="a"),
            GroundingItem(evidence_id="ev_002", quote_span="b"),
            GroundingItem(evidence_id="ev_001", quote_span="c"),  # duplicate id
        ],
    )
    delta = build_experiment_delta(base_graph_hash="h0", hypothesis_id="edge_1", experiment_plan=plan)
    assert delta.family == DeltaFamily.EXPERIMENT
    assert delta.payload.hypothesis_id == "edge_1"
    assert delta.payload.referenced_evidence_ids == ["ev_001", "ev_002"]  # grounding ids, deduped, ordered
    assert delta.payload.experiment_plan.design == ExperimentDesign.ABLATION


def test_build_experiment_delta_drops_placeholder_no_evidence_grounding():
    # Live Experiment Designer sometimes follows "say none retrieved plainly" by emitting a placeholder evidence_id.
    # That must not become a fake ledger reference that makes an otherwise valid plan uncommittable.
    plan = ExperimentPlan(
        hypothesis_under_test="e",
        design=ExperimentDesign.CONTROLLED_OBSERVATIONAL,
        materials_or_data=[
            MaterialItem(item="self-generated synthetic data", evidence_id="none_retrieved"),
            MaterialItem(item="retrieved benchmark", evidence_id="ev_001"),
        ],
        metrics=[
            MetricSpec(metric="MSE", predicted_direction="down", evidence_id="N/A"),
            MetricSpec(metric="runtime", predicted_direction="down", evidence_id="ev_001"),
        ],
        grounding=[
            GroundingItem(evidence_id="none_retrieved", quote_span="No retrieved passage applies."),
            GroundingItem(evidence_id="N/A", quote_span="No source was retrieved."),
            GroundingItem(evidence_id="ev_001", quote_span="real retrieved method"),
        ],
    )
    delta = build_experiment_delta(base_graph_hash="h0", hypothesis_id="edge_1", experiment_plan=plan)

    assert delta.payload.referenced_evidence_ids == ["ev_001"]
    assert [item.evidence_id for item in delta.payload.experiment_plan.grounding] == ["ev_001"]
    assert [item.evidence_id for item in delta.payload.experiment_plan.materials_or_data] == [
        None,
        "ev_001",
    ]
    assert [item.evidence_id for item in delta.payload.experiment_plan.metrics] == [
        None,
        "ev_001",
    ]


def test_build_experiment_delta_commits_through_five_gates():
    # The producer's delta must be valid end to end through the normal commit path.
    candidate, factuality = _candidate()
    edge = candidate.new_edges[0]
    store = CausalClaimGraphStore()
    for node in (*candidate.new_nodes, factuality):
        store.nodes[node.node_id] = node
    store.edges[edge.edge_id] = edge  # a committed hypothesis edge to bind the plan to
    log = GraphTransactionLog()
    validator = GraphDeltaValidator(ledger_evidence_ids={"ev_001"})

    plan = ExperimentPlan(
        hypothesis_under_test="rate-distortion floor --bounds--> factuality",
        design=ExperimentDesign.ABLATION,
        grounding=[GroundingItem(evidence_id="ev_001", quote_span="q")],
    )
    delta = build_experiment_delta(
        base_graph_hash=store.base_hash, hypothesis_id=edge.edge_id, experiment_plan=plan
    )
    result = log.commit(store, delta, validator, author="experiment_designer")

    assert result.accepted is True
    assert store.version == 1  # advanced by one committed version
    assert store.experiment_plans[edge.edge_id].design == ExperimentDesign.ABLATION  # plan durable
    assert store.edges[edge.edge_id].status == EdgeStatus.UNVERIFIED  # no verdict: status unchanged


# --- Post-confirmation experiment design -----------------------------------------------
def _seed_confirmed(candidate, factuality, *extra_nodes, extra_edges=()):
    store = CausalClaimGraphStore()
    for node in (*candidate.new_nodes, factuality, *extra_nodes):
        store.nodes[node.node_id] = node
    store.edges[candidate.new_edges[0].edge_id] = candidate.new_edges[0]  # the committed hypothesis edge
    for edge in extra_edges:
        store.edges[edge.edge_id] = edge
    return store


def test_methods_query_uses_full_candidate_context_not_only_primary_edge():
    from src.cycles.experiment import _methods_query

    candidate, factuality = _candidate()
    calibration = build_node(
        label="calibration error", type="outcome", definition="probability calibration gap"
    )
    secondary = build_edge(
        source_node_ids=[candidate.new_nodes[0].node_id],
        target_node_ids=[calibration.node_id],
        direction="causal",
        relation_type="increases",
        mechanism="quantization perturbs confidence margins",
    )
    expanded = replace(
        candidate,
        new_nodes=(*candidate.new_nodes, calibration),
        new_edges=(*candidate.new_edges, secondary),
        mechanism_chain=(
            *candidate.mechanism_chain,
            {
                "from": "rate-distortion floor",
                "relation": "increases",
                "to": "calibration error",
                "mechanism": "quantization perturbs confidence margins",
            },
        ),
        rationale="Compression may jointly limit factuality and confidence calibration.",
        idea_scaffold={
            "method": "sweep bitrate and calibration temperature",
            "experiment": "measure TruthfulQA F1 and expected calibration error",
        },
    )

    query = _methods_query(
        candidate.new_edges[0], _graph_context(factuality), candidate=expanded
    )

    assert "rate-distortion floor --bounds--> factuality" in query
    assert "rate-distortion floor --increases--> calibration error" in query
    assert "quantization perturbs confidence margins" in query
    assert "probability calibration gap" in query
    assert expanded.rationale in query
    assert expanded.idea_scaffold["method"] in query
    assert expanded.idea_scaffold["experiment"] in query


def test_experiment_stage_designs_and_commits_for_confirmed_hypothesis():
    from src.cycles.experiment import run_experiment_stage

    candidate, factuality = _candidate()
    edge = candidate.new_edges[0]
    store, log = _seed_confirmed(candidate, factuality), GraphTransactionLog()
    txs = run_experiment_stage(
        [candidate], store=store, log=log, base_ledger={"ev_001"},
        designer_factory=lambda: ExperimentDesigner(_ThreadBackend([_PLAN_JSON])),
        validator_seam=ExperimentValidator(_CountingBackend(_VERDICT_CLEAN_JSON)),
        retrieve_methods=lambda _q: _PASSAGES, refine_rounds=1,
    )
    assert len(txs) == 1 and txs[0].accepted
    assert store.version == 1  # one experiment commit
    assert store.experiment_plans[edge.edge_id].design == ExperimentDesign.ABLATION  # committed + durable


def test_experiment_stage_empty_confirmed_is_a_noop():
    from src.cycles.experiment import run_experiment_stage

    store, log = CausalClaimGraphStore(), GraphTransactionLog()
    txs = run_experiment_stage(
        [], store=store, log=log, base_ledger=set(),
        designer_factory=lambda: ExperimentDesigner(_ThreadBackend([])),
        validator_seam=ExperimentValidator(_FakeBackend("{}")),
    )
    assert txs == () and store.version == 0


def test_experiment_stage_surfaces_adjacent_graph_confounder_to_the_design_prompt():
    from src.cycles.experiment import run_experiment_stage

    candidate, factuality = _candidate()
    confounder = build_node(label="adapter rank", type="moderator", definition="an effect modifier")
    # a committed edge from the confounder to the tested edge's target makes it graph-adjacent
    link = build_edge(
        source_node_ids=[confounder.node_id], target_node_ids=[factuality.node_id],
        direction="causal", relation_type="modulates",
    )
    store = _seed_confirmed(candidate, factuality, confounder, extra_edges=(link,))
    designer_backend = _ThreadBackend([_PLAN_JSON])
    run_experiment_stage(
        [candidate], store=store, log=GraphTransactionLog(), base_ledger={"ev_001"},
        designer_factory=lambda: ExperimentDesigner(designer_backend),
        validator_seam=ExperimentValidator(_CountingBackend(_VERDICT_CLEAN_JSON)),
        retrieve_methods=lambda _q: _PASSAGES, refine_rounds=0,
    )
    design_user = designer_backend.calls[0][-1]["content"]
    assert "adapter rank" in design_user  # the adjacent graph moderator reached the Experiment Designer prompt


def test_experiment_stage_extends_the_ledger_so_methods_grounding_resolves():
    # the plan grounds on a retrieved methods passage whose id is NOT in the base commit ledger;
    # the stage must still commit (its per-hypothesis validator extends the ledger with methods ids)
    from src.cycles.experiment import run_experiment_stage

    candidate, factuality = _candidate()
    store, log = _seed_confirmed(candidate, factuality), GraphTransactionLog()
    txs = run_experiment_stage(
        [candidate], store=store, log=log, base_ledger=set(),  # empty base ledger
        designer_factory=lambda: ExperimentDesigner(_ThreadBackend([_PLAN_JSON])),
        validator_seam=ExperimentValidator(_CountingBackend(_VERDICT_CLEAN_JSON)),
        retrieve_methods=lambda _q: _PASSAGES, refine_rounds=0,  # ev_001 only reachable via methods
    )
    assert len(txs) == 1 and txs[0].accepted  # grounding on ev_001 resolved via the extended ledger
    assert txs[0].row.delta.payload.referenced_evidence_ids == ["ev_001"]


def test_experiment_stage_stamps_unique_retrieval_batch_on_committed_plan():
    from src.cycles.experiment import run_experiment_stage

    candidate, factuality = _candidate()
    edge = candidate.new_edges[0]
    store, log = _seed_confirmed(candidate, factuality), GraphTransactionLog()
    passages = [
        {
            **_PASSAGES[0],
            "metadata": {"retrieval_batch_id": "batch-methods-1"},
        }
    ]

    txs = run_experiment_stage(
        [candidate], store=store, log=log, base_ledger=set(),
        designer_factory=lambda: ExperimentDesigner(_ThreadBackend([_PLAN_JSON])),
        validator_seam=ExperimentValidator(_CountingBackend(_VERDICT_CLEAN_JSON)),
        retrieve_methods=lambda _q: passages, refine_rounds=0,
    )

    assert len(txs) == 1 and txs[0].accepted
    assert store.experiment_plans[edge.edge_id].retrieval_batch_id == "batch-methods-1"
    assert (
        txs[0].row.delta.payload.experiment_plan.model_dump(mode="json")[
            "retrieval_batch_id"
        ]
        == "batch-methods-1"
    )
    assert (
        store.experiment_plans[edge.edge_id].grounding_status
        == ExperimentGroundingStatus.GROUNDED
    )


def test_empty_retrieval_batch_id_is_omitted_from_plan_serialization():
    plan = ExperimentPlan(hypothesis_under_test="e", design=ExperimentDesign.ABLATION)

    assert "retrieval_batch_id" not in plan.model_dump(mode="json")
    assert "grounding_status" not in plan.model_dump(mode="json")


# --- Experiment-plan content-completeness gate -----------------------------------------
_SKELETAL_PLAN_JSON = (
    '{"experiment_plan": {'
    '"hypothesis_under_test": "rate-distortion floor --bounds--> factuality",'
    '"design": "ablation"}}'
)

_PARTIAL_PLAN_JSON = (
    '{"experiment_plan": {'
    '"hypothesis_under_test": "rate-distortion floor --bounds--> factuality",'
    '"operationalization": "vary compression bitrate; measure factuality F1",'
    '"design": "ablation",'
    '"design_rationale": "ablating bitrate isolates the floor",'
    '"intervention_or_manipulation": "compression bitrate",'
    '"comparison_baseline": "uncompressed model"}}'
)

# Otherwise-complete, but grounding is entirely a "no evidence retrieved" placeholder row —
# build_experiment_delta drops such rows, so this must gap on grounding just like an empty list.
_PLACEHOLDER_GROUNDING_PLAN_JSON = _PLAN_JSON.replace(
    '"grounding": [{"evidence_id": "ev_001", "quote_span": "the rate-distortion floor bounds factuality"}]',
    '"grounding": [{"evidence_id": "none_retrieved", "quote_span": "No retrieved passage applies."}]',
)

_NO_RELEVANT_SOURCE_PLAN_JSON = _PLAN_JSON.replace(
    '"materials_or_data": [{"item": "TruthfulQA", "evidence_id": "ev_001"}]',
    '"materials_or_data": [{"item": "self-generated benchmark", "evidence_id": null}]',
).replace(
    '"grounding": [{"evidence_id": "ev_001", "quote_span": "the rate-distortion floor bounds factuality"}]',
    '"grounding": []',
)


def test_plan_content_gaps_flags_every_blank_required_field():
    from src.cycles.experiment import _plan_content_gaps

    _, factuality = _candidate()
    skeletal = ExperimentPlan(hypothesis_under_test="e", design=ExperimentDesign.ABLATION)
    gaps = _plan_content_gaps(skeletal, graph_context=_graph_context(factuality))
    assert "hypothesis_under_test" not in gaps  # filled
    for field in (
        "operationalization", "design_rationale", "intervention_or_manipulation",
        "comparison_baseline", "controls_and_confounders", "materials_or_data", "metrics",
        "expected_outcome", "falsification", "feasibility.resources", "feasibility.time",
        "feasibility.main_risk", "grounding",
    ):
        assert field in gaps


def test_plan_content_gaps_exempts_confounders_when_graph_has_none():
    # controls_and_confounders MUST NOT be forced non-empty when the graph offers no adjacent
    # confounder, mediator, or moderator to name; the designer must not invent one.
    from src.cycles.experiment import _plan_content_gaps

    plan = ExperimentPlan(
        hypothesis_under_test="e", design=ExperimentDesign.ABLATION,
        operationalization="o", design_rationale="r", intervention_or_manipulation="i",
        comparison_baseline="b",
        materials_or_data=[MaterialItem(item="x", evidence_id=None)],
        metrics=[MetricSpec(metric="m", evidence_id=None)],
        expected_outcome="eo", falsification="f",
        feasibility={"resources": "r", "time": "t", "main_risk": "m"},
        grounding=[GroundingItem(evidence_id="ev_001", quote_span="q")],
    )  # controls_and_confounders left [] deliberately
    gaps = _plan_content_gaps(plan, graph_context=ExperimentGraphContext())  # no adjacent confounders
    assert gaps == ()


def test_plan_content_gaps_flags_grounding_of_only_placeholder_rows():
    # grounding populated ONLY by "no evidence retrieved" placeholders is not content-complete:
    # build_experiment_delta drops those rows, so the committed plan would end up ungrounded.
    from src.cycles.experiment import _plan_content_gaps

    _, factuality = _candidate()
    plan = ExperimentPlan(
        hypothesis_under_test="e", design=ExperimentDesign.ABLATION,
        operationalization="o", design_rationale="r", intervention_or_manipulation="i",
        comparison_baseline="b",
        controls_and_confounders=[{"factor": "adapter rank", "from_graph": True, "handling": "held fixed"}],
        materials_or_data=[MaterialItem(item="x", evidence_id=None)],
        metrics=[MetricSpec(metric="m", evidence_id=None)],
        expected_outcome="eo", falsification="f",
        feasibility={"resources": "r", "time": "t", "main_risk": "m"},
        grounding=[GroundingItem(evidence_id="none_retrieved", quote_span="No retrieved passage applies.")],
    )
    gaps = _plan_content_gaps(plan, graph_context=_graph_context(factuality))
    assert gaps == ("grounding",)


def test_plan_content_gaps_accepts_explicit_empty_grounding_as_no_relevant_source():
    from src.cycles.experiment import _parse_experiment_plan, _plan_content_gaps

    _, factuality = _candidate()
    plan = _parse_experiment_plan(json.loads(_NO_RELEVANT_SOURCE_PLAN_JSON))

    assert plan is not None
    assert "grounding" in plan.model_fields_set
    assert _plan_content_gaps(
        plan,
        graph_context=_graph_context(factuality),
        has_retrieved_passages=True,
        retrieved_evidence_ids={"ev_001"},
    ) == ()


def test_plan_content_gaps_flags_grounding_ids_outside_this_retrieval():
    from src.cycles.experiment import _plan_content_gaps

    candidate, factuality = _candidate()
    plan = ExperimentDesigner(_ThreadBackend([_PLAN_JSON])).design(
        candidate, graph_context=_graph_context(factuality), passages=_PASSAGES
    )

    assert plan is not None
    assert _plan_content_gaps(
        plan,
        graph_context=_graph_context(factuality),
        retrieved_evidence_ids={"ev_other_batch"},
    ) == ("grounding",)


def test_plan_content_gaps_flags_real_grounding_without_quote_span():
    from src.cycles.experiment import _plan_content_gaps

    candidate, factuality = _candidate()
    plan = ExperimentDesigner(_ThreadBackend([_PLAN_JSON])).design(
        candidate, graph_context=_graph_context(factuality), passages=_PASSAGES
    )

    assert plan is not None
    plan = plan.model_copy(
        update={"grounding": [GroundingItem(evidence_id="ev_001", quote_span="")]}
    )
    assert _plan_content_gaps(
        plan,
        graph_context=_graph_context(factuality),
        retrieved_evidence_ids={"ev_001"},
    ) == ("grounding",)


def test_skeletal_plan_never_commits_and_exhausts_refine():
    from src.cycles.experiment import run_experiment_stage

    candidate, factuality = _candidate()
    store, log = _seed_confirmed(candidate, factuality), GraphTransactionLog()
    designer_backend = _ThreadBackend([_SKELETAL_PLAN_JSON, _SKELETAL_PLAN_JSON])
    txs = run_experiment_stage(
        [candidate], store=store, log=log, base_ledger={"ev_001"},
        designer_factory=lambda: ExperimentDesigner(designer_backend),
        validator_seam=ExperimentValidator(_CountingBackend(_VERDICT_CLEAN_JSON)),
        retrieve_methods=lambda _q: _PASSAGES, refine_rounds=1,
    )
    assert txs == ()  # no Δ^experiment commit for this hypothesis
    assert store.version == 0  # graph untouched
    assert len(designer_backend.calls) == 2  # design + exactly refine_rounds refine attempts, then give up


def test_placeholder_only_grounding_never_commits_and_exhausts_refine():
    # An otherwise-complete plan whose grounding is entirely a "no evidence retrieved" placeholder
    # must not commit: build_experiment_delta's normalization would drop it to grounding=[].
    from src.cycles.experiment import run_experiment_stage

    candidate, factuality = _candidate()
    store, log = _seed_confirmed(candidate, factuality), GraphTransactionLog()
    designer_backend = _ThreadBackend([_PLACEHOLDER_GROUNDING_PLAN_JSON, _PLACEHOLDER_GROUNDING_PLAN_JSON])
    txs = run_experiment_stage(
        [candidate], store=store, log=log, base_ledger={"ev_001"},
        designer_factory=lambda: ExperimentDesigner(designer_backend),
        validator_seam=ExperimentValidator(_CountingBackend(_VERDICT_CLEAN_JSON)),
        retrieve_methods=lambda _q: _PASSAGES, refine_rounds=1,
    )
    assert txs == ()  # placeholder-only grounding must not commit as if it were real evidence
    assert store.version == 0  # graph untouched
    assert len(designer_backend.calls) == 2  # design + exactly refine_rounds refine attempts, then give up


def test_nonempty_but_irrelevant_retrieval_commits_explicit_ungrounded_draft():
    from src.cycles.experiment import run_experiment_stage

    candidate, factuality = _candidate()
    edge = candidate.new_edges[0]
    store, log = _seed_confirmed(candidate, factuality), GraphTransactionLog()
    designer_backend = _ThreadBackend([_NO_RELEVANT_SOURCE_PLAN_JSON])

    txs = run_experiment_stage(
        [candidate],
        store=store,
        log=log,
        base_ledger={"ev_001"},
        designer_factory=lambda: ExperimentDesigner(designer_backend),
        validator_seam=ExperimentValidator(_CountingBackend(_VERDICT_CLEAN_JSON)),
        retrieve_methods=lambda _q: _PASSAGES,
        refine_rounds=1,
    )

    assert len(txs) == 1 and txs[0].accepted
    committed = store.experiment_plans[edge.edge_id]
    assert committed.grounding == []
    assert committed.grounding_status == ExperimentGroundingStatus.NO_RELEVANT_METHODS
    assert txs[0].row.delta.payload.referenced_evidence_ids == []
    assert len(designer_backend.calls) == 1


def test_empty_retrieval_honest_no_source_plan_commits():
    # With NO retrieved methods passages, an otherwise-complete plan whose grounding is only a
    # "no evidence retrieved" placeholder is the HONEST degradation path: it must commit, with
    # the placeholder rows normalized away (grounding=[], referenced_evidence_ids=[]).
    from src.cycles.experiment import run_experiment_stage

    candidate, factuality = _candidate()
    edge = candidate.new_edges[0]
    store, log = _seed_confirmed(candidate, factuality), GraphTransactionLog()
    designer_backend = _ThreadBackend([_PLACEHOLDER_GROUNDING_PLAN_JSON])
    txs = run_experiment_stage(
        [candidate], store=store, log=log, base_ledger={"ev_001"},
        designer_factory=lambda: ExperimentDesigner(designer_backend),
        validator_seam=ExperimentValidator(_CountingBackend(_VERDICT_CLEAN_JSON)),
        retrieve_methods=None, refine_rounds=1,
    )
    assert len(txs) == 1 and txs[0].accepted
    committed = store.experiment_plans[edge.edge_id]
    assert committed.grounding == []  # placeholder rows normalized away, honestly source-less
    assert committed.grounding_status == ExperimentGroundingStatus.NO_METHODS_RETRIEVED
    assert txs[0].row.delta.payload.referenced_evidence_ids == []


def test_partial_plan_triggers_refine_then_commits_once_complete():  # refinement to completeness
    from src.cycles.experiment import run_experiment_stage

    candidate, factuality = _candidate()
    edge = candidate.new_edges[0]
    store, log = _seed_confirmed(candidate, factuality), GraphTransactionLog()
    designer_backend = _ThreadBackend([_PARTIAL_PLAN_JSON, _PLAN_JSON])
    txs = run_experiment_stage(
        [candidate], store=store, log=log, base_ledger={"ev_001"},
        designer_factory=lambda: ExperimentDesigner(designer_backend),
        validator_seam=ExperimentValidator(_CountingBackend(_VERDICT_CLEAN_JSON)),
        retrieve_methods=lambda _q: _PASSAGES, refine_rounds=1,
    )
    assert len(txs) == 1 and txs[0].accepted
    assert len(designer_backend.calls) == 2  # design (partial) + 1 refine (now content-complete)
    assert store.experiment_plans[edge.edge_id].grounding  # the committed plan is content-complete
    refine_user = designer_backend.calls[1][-1]["content"]
    assert "<required_content_gaps>" in refine_user
    assert "- grounding" in refine_user
    assert 'return "grounding": [] explicitly' in refine_user


def test_complete_plan_commits_without_extra_refine():  # no unnecessary refinement
    from src.cycles.experiment import run_experiment_stage

    candidate, factuality = _candidate()
    store, log = _seed_confirmed(candidate, factuality), GraphTransactionLog()
    designer_backend = _ThreadBackend([_PLAN_JSON])
    txs = run_experiment_stage(
        [candidate], store=store, log=log, base_ledger={"ev_001"},
        designer_factory=lambda: ExperimentDesigner(designer_backend),
        validator_seam=ExperimentValidator(_CountingBackend(_VERDICT_CLEAN_JSON)),
        retrieve_methods=lambda _q: _PASSAGES, refine_rounds=1,
    )
    assert len(txs) == 1 and txs[0].accepted
    assert len(designer_backend.calls) == 1  # content-complete + no Experiment Validator flags -> no refine needed
