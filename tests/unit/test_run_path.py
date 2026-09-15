"""The graph-state run path auto-schedules verification from the extracted graph.

This includes transferring scored evidence to the Evidence Reviewer.

In a real run the LLM extractor mints content-addressed edge ids that are unknown until
extraction commits, so the path cannot be handed pre-built ``VerificationTask``s. The auto
mode resolves that: after extraction it builds one task per (edge, role), pulls that target's
evidence via an injected ``retrieve_for_target`` callable, scores + triages it, and verifies
while deriving the validator ledger from retrieved evidence. Tests are LLM-free and deterministic.
"""

from __future__ import annotations

import pytest

from src.cycles import verification as v
from src.cycles.extraction import ClaimExtraction
from src.delta import build_edge, build_node
from src.graph_store import EdgeStatus
from src.run_path import GraphRunResult, run_graph_state_workflow
from src.state import RetrievedEvidence

pytestmark = pytest.mark.smoke


_CAUSE = build_node(label="smoking", type="exposure/intervention")
_EFFECT = build_node(label="lung cancer", type="outcome")
_EDGE = build_edge(
    source_node_ids=[_CAUSE.node_id], target_node_ids=[_EFFECT.node_id],
    direction="causal", relation_type="increases",
)


class _FakeExtractor:
    def extract(self, claim):
        return ClaimExtraction(nodes=(_CAUSE, _EFFECT), edges=(_EDGE,))


def _embedder(texts):
    return [[1.0, 0.0] for _ in texts]


class _FakeReviewer:
    """Evidence Reviewer stub: strong, clean SUPPORT for every passage; no edge-level risks."""

    def review(self, task, evidences):
        coherence = v.CoherenceSignal(
            entailment={"supports": 1.0}, contradiction_risk={},
            context_fit=1.0, construct_match=1.0,
        )
        methods = v.MethodsSignal(1.0, 1.0, 1.0, 1.0, 0.0)
        return v.EdgeReview(
            coherence={ev.evidence_id: coherence for ev in evidences},
            methods={ev.evidence_id: methods for ev in evidences},
            appraisal=v.VerifierAppraisal(open_risks=(), qualifiers=()),
        )


def _evidence(evidence_id: str) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=evidence_id, source="openalex", source_id="s1",
        title="A cohort study of smoking and lung cancer",
        quote="smoking increases lung cancer risk", relevance="high",
        retrieved_by="retriever", tool_call_id=f"tool-{evidence_id}", rank=1, trust_tier="green",
    )


def test_auto_scheduling_verifies_every_extracted_edge():
    seen_tasks = []

    def retrieve_for_target(task):
        # the path built a VerificationTask for the committed edge and asked us for evidence
        seen_tasks.append(task)
        return [_evidence("ev-1")]

    result = run_graph_state_workflow(
        "smoking increases lung cancer",
        extractor=_FakeExtractor(),
        embedder=_embedder,
        retrieve_for_target=retrieve_for_target,        # AUTO mode (no explicit verification_work)
        evidence_reviewer=_FakeReviewer(),
    )

    assert isinstance(result, GraphRunResult)
    # the path scheduled the extracted edge itself and drove it to a committed verdict.
    assert [t.edge_id for t in seen_tasks] == [_EDGE.edge_id]
    assert result.statuses[_EDGE.edge_id] == EdgeStatus.SUPPORTED.value
    # The extraction and verification commits use evidence derived from retrieval (the
    # verify refs-gate only passes if "ev-1" was registered) — no ledger_evidence_ids passed.
    assert result.version == 2
    assert len(result.receipts) == 2
    assert all(r.status == "accepted" for r in result.receipts)
    # the support-only MVP schedule is surfaced as an open risk (refute-side not scheduled).
    assert any("not scheduled" in risk and "contradiction" in risk for risk in result.open_risks)


def test_initial_retrieval_risks_are_exported() -> None:
    risk = "retrieval skipped source exa: missing EXA_API_KEY"
    result = run_graph_state_workflow(
        "smoking increases lung cancer",
        extractor=_FakeExtractor(),
        embedder=_embedder,
        retrieve_for_target=lambda _task: [_evidence("ev-1")],
        evidence_reviewer=_FakeReviewer(),
        initial_open_risks=(risk,),
    )

    assert risk in result.open_risks
    assert f"- {risk}" in result.audit_memo


def test_late_retrieval_risks_are_exported() -> None:
    risk = "retrieval failed for all selected sources"
    result = run_graph_state_workflow(
        "smoking increases lung cancer",
        extractor=_FakeExtractor(),
        embedder=_embedder,
        retrieve_for_target=lambda _task: [_evidence("ev-1")],
        evidence_reviewer=_FakeReviewer(),
        open_risks_provider=lambda: (risk,),
    )

    assert risk in result.open_risks
    assert f"- {risk}" in result.audit_memo


def test_auto_scheduling_carries_relationship_into_appraiser():
    seen: dict[str, str] = {}

    class _CapturingReviewer:
        def review(self, task, evidences):
            seen["question"] = task.question
            seen["criteria"] = task.criteria
            coherence = v.CoherenceSignal(
                entailment={"supports": 1.0}, contradiction_risk={},
                context_fit=1.0, construct_match=1.0,
            )
            return v.EdgeReview(
                coherence={ev.evidence_id: coherence for ev in evidences},
                methods={ev.evidence_id: v.MethodsSignal(1.0, 1.0, 1.0, 1.0, 0.0) for ev in evidences},
                appraisal=v.VerifierAppraisal(open_risks=(), qualifiers=()),
            )

    run_graph_state_workflow(
        "smoking increases lung cancer",
        extractor=_FakeExtractor(), embedder=_embedder,
        retrieve_for_target=lambda task: [_evidence("ev-1")],
        evidence_reviewer=_CapturingReviewer(),
    )

    # The Evidence Reviewer sees the cause-to-effect relationship it scores against,
    # not just the role + passage.
    assert "smoking" in seen["question"] and "lung cancer" in seen["question"]
    assert "increases" in seen["question"]


def test_multi_role_accumulates_into_one_verdict_per_edge():
    def retrieve_for_target(task):
        # each role pulls its own evidence; ALL of it must land in ONE verification.
        return [_evidence(f"ev-{task.evidence_role}")]

    result = run_graph_state_workflow(
        "smoking increases lung cancer",
        extractor=_FakeExtractor(), embedder=_embedder,
        retrieve_for_target=retrieve_for_target,
        verify_roles=(v.EvidenceRole.SUPPORT, v.EvidenceRole.CONTRADICTION),
        evidence_reviewer=_FakeReviewer(),
    )

    # ONE verdict/commit for the edge — not one-per-role with the later roles rejected.
    assert result.statuses[_EDGE.edge_id] == EdgeStatus.SUPPORTED.value
    assert len(result.receipts) == 2                              # extract + exactly one verify
    assert all(r.status == "accepted" for r in result.receipts)  # zero silent rejected dups
    # the single Δ^verify carries BOTH roles' evidence links (support + counterevidence).
    roles = {str(link["evidence_role"]) for link in result.graph_json["evidence_links"]}
    assert {"support", "contradiction"} <= roles


# --- redundancy-fix: passage-cap (shared passages_from_quotes helper) -----------------------
def test_passages_from_quotes_dedupes_truncates_and_caps():  # shared helper, direct unit test
    from src.run_path import passages_from_quotes

    long_quote = "x" * 4500
    quotes = ["alpha", "alpha", "", long_quote, "beta", "gamma"]
    passages = passages_from_quotes(quotes, max_chars=4000, max_passages=3)
    assert passages == ["alpha", long_quote[:4000], "beta"]
    assert len(passages[1]) == 4000


def test_passages_from_quotes_uncapped_when_max_chars_none():
    from src.run_path import passages_from_quotes

    long_quote = "y" * 4500
    passages = passages_from_quotes([long_quote], max_chars=None, max_passages=20)
    assert passages == [long_quote]
    assert len(passages[0]) == 4500


def test_passages_from_work_truncates_long_quotes_to_the_shared_cap():  # byte-identical regression
    from src.run_path import _passages_from_work

    long_quote = "z" * 5000
    work = [(None, [_evidence("e1")]), (None, [
        RetrievedEvidence(
            evidence_id="e2", source="openalex", source_id="s2", title="t", quote=long_quote,
            relevance="high", retrieved_by="r", tool_call_id="tool-e2", rank=1, trust_tier="green",
        ),
    ])]
    passages = _passages_from_work(work)
    assert passages == ["smoking increases lung cancer risk", long_quote[:4000]]
    assert len(passages[1]) == 4000


# Research Synthesist and Critic Panel path.
class _ThreadBackend:
    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0

    def run(self, messages, tools=None):
        self.calls += 1
        return {"choices": [{"message": {"content": self._contents.pop(0) if self._contents else ""}}]}


class _JudgeBackend:
    def __init__(self, content):
        self._content = content

    def run(self, messages, tools=None):
        return {"choices": [{"message": {"content": self._content}}]}


_SYNTHESIST_MINE_RESPONSE = (
    '{"concepts": [{"label": "tar exposure", "type": "mediator", "definition": "carcinogen load",'
    ' "source_context": "tar drives risk", "bears_on_outcome": true, "evidence_id": "ev-1",'
    ' "paper_id": "p1", "matched_quote_span": "tar", "rationale": "r"}], "merge_decisions": []}'
)
_SYNTHESIST_PROPOSE_RESPONSE = (
    '{"candidates": [{"candidate_id": "h1",'
    ' "new_nodes": [{"label": "tar clearance rate", "type": "moderator", "definition": "d", "aliases": []}],'
    ' "new_edges": [{"source": "tar clearance rate", "target": "lung cancer",'
    ' "relation_type": "confounds", "direction": "directed", "mechanism": "m"}],'
    ' "mechanism_chain": [{"from": "tar clearance rate", "relation": "confounds",'
    ' "to": "lung cancer", "mechanism": "m"}],'
    ' "source_quotes": [{"evidence_id": "ev-1", "quote_span": "tar", "role_in_hypothesis": "x"}],'
    ' "assumptions": [], "cross_concept": true, "common_sense": false,'
    ' "llm_signals": {"novelty": 0.9, "testability": 0.9, "scope_fit": 0.9, "duplication": 0.05,'
    ' "plausibility": 0.8, "expected_yield": 0.8, "centrality": 0.8, "mechanism_specificity": 0.8}}]}'
)
_SYNTHESIST_REVISE_RESPONSE = '{"revised": [], "derived": []}'
_CRITIC_PANEL_REVIEW_RESPONSE = (
    '{"candidates": [{"candidate_id": "h1", "field_novelty": 0.8, "saturation": 0.2,'
    ' "already_established": false, "not_judgeable_by_field": false,'
    ' "mechanism_steps": [{"from": "tar clearance rate", "relation": "confounds",'
    ' "to": "lung cancer", "verdict": "sound", "note": ""}], "term_verdicts": [],'
    ' "justification": "novel per P1"}], "ranking": ["h1"], "critique": "ok"}'
)


def test_synthesist_path_runs_when_synthesist_injected():
    from src.cycles.panel import LLMCriticPanelJudge
    from src.cycles.synthesist import ResearchSynthesist

    thread = _ThreadBackend([
        _SYNTHESIST_MINE_RESPONSE,
        _SYNTHESIST_PROPOSE_RESPONSE,
        _SYNTHESIST_REVISE_RESPONSE,
    ])
    panel = (
        LLMCriticPanelJudge(
            _JudgeBackend(_CRITIC_PANEL_REVIEW_RESPONSE),
            judge_id=1,
            reference_field="oncology",
        ),
    )
    result = run_graph_state_workflow(
        "smoking increases lung cancer", extractor=_FakeExtractor(), embedder=_embedder,
        retrieve_for_target=lambda t: [_evidence("ev-1")], evidence_reviewer=_FakeReviewer(),
        enable_expansion=True,
        research_synthesist=lambda: ResearchSynthesist(
            thread, embedder=_embedder, claim="smoking increases lung cancer", proposal_count=6
        ),
        critic_panel=panel,
        enrichment_passages=("### Passage 1\ntar drives risk (ev-1)",),
        expansion_confirmed_ids=["h1"],
    )
    # the seven-seam Research Synthesist thread ran all three turns (mine→propose→revise) — NOT the legacy proposer
    assert thread.calls == 3
    # the panel-graded, confirmed hypothesis committed as UNVERIFIED via the seven-seam path
    assert [r.candidate_id for r in result.surfaced] == ["h1"]
    cross = [e for e in result.graph_json["edges"].values() if e["relation_type"] == "confounds"]
    assert len(cross) == 1 and cross[0]["status"] == EdgeStatus.UNVERIFIED.value


def test_default_on_expansion_without_synthesist_is_evidence_core_noop():
    # With no Research Synthesist injected, the run still executes the evidence core
    # through verification, priority, and export rather than silently skipping verification;
    # it also emits one explicit diagnostic.
    result = run_graph_state_workflow(
        "smoking increases lung cancer",
        extractor=_FakeExtractor(), embedder=_embedder,
        retrieve_for_target=lambda t: [_evidence("ev-1")], evidence_reviewer=_FakeReviewer(),
    )
    assert result.surfaced == ()  # no seams wired -> nothing expanded
    assert result.statuses[_EDGE.edge_id] == EdgeStatus.SUPPORTED.value
    signal = "expansion enabled but no Research Synthesist injected; running evidence core only"
    assert signal in result.open_risks
    assert signal in result.audit_memo


# Surface the Synthesist ranking and confirmation hook from the one-call workflow.
def _synthesist_run(**overrides):
    from src.cycles.panel import LLMCriticPanelJudge
    from src.cycles.synthesist import ResearchSynthesist

    kwargs = dict(
        extractor=_FakeExtractor(), embedder=_embedder,
        retrieve_for_target=lambda t: [_evidence("ev-1")],
        evidence_reviewer=_FakeReviewer(),
        enable_expansion=True,
        research_synthesist=lambda: ResearchSynthesist(
            _ThreadBackend([
                _SYNTHESIST_MINE_RESPONSE,
                _SYNTHESIST_PROPOSE_RESPONSE,
                _SYNTHESIST_REVISE_RESPONSE,
            ]),
            embedder=_embedder, claim="smoking increases lung cancer", proposal_count=6,
        ),
        critic_panel=(
            LLMCriticPanelJudge(
                _JudgeBackend(_CRITIC_PANEL_REVIEW_RESPONSE),
                judge_id=1,
                reference_field="oncology",
            ),
        ),
    )
    kwargs.update(overrides)
    return run_graph_state_workflow("smoking increases lung cancer", **kwargs)


def test_progress_follows_execution_order_and_reports_real_counts(tmp_path):
    import json

    from src.cycles.experiment import ExperimentDesigner, ExperimentValidator
    from src.progress import ProgressReporter

    path = tmp_path / "progress.jsonl"
    with ProgressReporter("run", quiet=True, jsonl_path=path):
        result = _synthesist_run(
            expansion_confirmed_ids=["h1"],
            export_dir=tmp_path,
            experiment_designer=lambda: ExperimentDesigner(
                _ThreadBackend([_EXPERIMENT_DESIGNER_PLAN_RESPONSE])
            ),
            experiment_validator=ExperimentValidator(
                _JudgeBackend(_EXPERIMENT_VALIDATOR_VERDICT_RESPONSE)
            ),
            experiment_refine_rounds=1,
            retrieve_experiment_methods=lambda _: [
                {"evidence_id": "ev-m1", "quote": "cohort design for tar clearance"}
            ],
        )
    events = [json.loads(line) for line in path.read_text().splitlines()]
    stages = list(dict.fromkeys(row["stage"] for row in events))
    assert stages == [
        "Starting run", "Mapping claim", "Checking evidence", "Applying priorities",
        "Propose / critique", "Confirming hypotheses", "Designing experiments", "Exporting graph",
    ]
    review = next(row for row in events if row["detail"] == "reviewing link")
    assert (review["current"], review["total"]) == (1, 1)
    judges = [row for row in events if "critic judge" in row["detail"]]
    assert [row["detail"] for row in judges] == ["round 1 · critic judge", "round 2 · critic judge"]
    plan = next(row for row in events if row["detail"] == "plan committed")
    assert (plan["current"], plan["total"]) == (1, 1)
    assert plan["status"] == "completed"
    assert len(result.graph_json["experiment_plans"]) == 1
    assert (tmp_path / "graph.json").is_file()


def test_run_result_surfaces_the_synthesist_ranking():
    result = _synthesist_run(expansion_confirmed_ids=["h1"])
    assert len(result.surfaced) == 1
    row = result.surfaced[0]
    assert row.candidate_id == "h1" and row.rank == 1
    assert 0.0 <= row.field_novelty <= 1.0
    # RankScore = clip01(HypScore·(1−Saturation)); the Critic Panel graded saturation 0.2.
    assert row.saturation == pytest.approx(0.2)
    assert row.rank_score == pytest.approx(row.hyp_score * (1 - row.saturation), abs=1e-6)


def test_run_result_surfaces_critic_audits_for_elaborator():  # Elaborator context-in from Critic Panel
    from src.cycles.panel import LLMCriticPanelJudge

    critic_review = (
        '{"candidates": [{"candidate_id": "h1", "field_novelty": 0.8, "saturation": 0.2,'
        ' "already_established": false, "not_judgeable_by_field": false,'
        ' "mechanism_steps": [{"from": "tar clearance rate", "relation": "confounds",'
        ' "to": "lung cancer", "verdict": "sound", "note": "supported by tar passage"}],'
        ' "term_verdicts": [{"term": "tar clearance rate", "verdict": "consistent",'
        ' "note": "same construct as passage"}],'
        ' "justification": "novel per P1"}], "ranking": ["h1"], "critique": "ok"}'
    )
    result = _synthesist_run(
        expansion_confirmed_ids=["h1"],
        critic_panel=(
            LLMCriticPanelJudge(_JudgeBackend(critic_review), judge_id=1, reference_field="oncology"),
        ),
    )

    row = result.surfaced[0]
    assert row.mechanism_steps == (
        {
            "from": "tar clearance rate",
            "relation": "confounds",
            "to": "lung cancer",
            "verdict": "sound",
            "note": "supported by tar passage",
        },
    )
    assert row.term_audit == (
        {"term": "tar clearance rate", "verdict": "consistent", "note": "same construct as passage"},
    )


def test_expansion_confirm_fn_decides_from_the_ranking():  # confirmation callback
    seen = {}

    def confirm_fn(surfaced):
        seen["ids"] = [sc.candidate.candidate_id for sc in surfaced]
        return []  # confirm NOTHING, overriding the static superset below

    result = _synthesist_run(expansion_confirmed_ids=["h1"], expansion_confirm_fn=confirm_fn)
    assert seen["ids"] == ["h1"]  # the hook SAW the ranking the cycle produced
    cross = [e for e in result.graph_json["edges"].values() if e["relation_type"] == "confounds"]
    assert cross == []  # confirm_fn returned [] -> nothing committed
    assert len(result.surfaced) == 1  # the ranking is still surfaced for reporting


# --- Experiment design and validation end to end -----------------------------------------
_EXPERIMENT_DESIGNER_PLAN_RESPONSE = (
    '{"experiment_plan": {"hypothesis_under_test": "tar clearance rate --confounds--> lung cancer",'
    ' "operationalization": "measure clearance rate and cancer incidence", "design": "controlled_observational",'
    ' "design_rationale": "isolates the confound", "intervention_or_manipulation": "stratify by clearance",'
    ' "comparison_baseline": "unstratified cohort", "controls_and_confounders": [],'
    ' "materials_or_data": [{"item": "cohort", "evidence_id": "ev-m1"}],'
    ' "metrics": [{"metric": "incidence", "predicted_direction": "up", "evidence_id": null}],'
    ' "procedure": ["stratify", "follow up"], "expected_outcome": "higher incidence at low clearance",'
    ' "falsification": "no association", "feasibility": {"resources": "cohort DB", "time": "1y", "main_risk": "attrition"},'
    ' "grounding": [{"evidence_id": "ev-m1", "quote_span": "cohort design for tar clearance"}]}}'
)
_EXPERIMENT_VALIDATOR_VERDICT_RESPONSE = (
    '{"criteria": {"clarity": {"score": 0.8, "feedback": "c"}, "validity": {"score": 0.8, "feedback": "v"},'
    ' "robustness": {"score": 0.7, "feedback": "r"}, "feasibility": {"score": 0.8, "feedback": "f"},'
    ' "reproducibility": {"score": 0.8, "feedback": "rep"}}, "flagged": [], "overall": 0.8}'
)


def test_experiment_stage_commits_and_exports_for_confirmed_hypothesis():
    from src.cycles.experiment import ExperimentDesigner, ExperimentValidator

    result = _synthesist_run(
        expansion_confirmed_ids=["h1"],
        experiment_designer=lambda: ExperimentDesigner(
            _ThreadBackend([_EXPERIMENT_DESIGNER_PLAN_RESPONSE])
        ),
        experiment_validator=ExperimentValidator(
            _JudgeBackend(_EXPERIMENT_VALIDATOR_VERDICT_RESPONSE)
        ),
        experiment_refine_rounds=1,
        retrieve_experiment_methods=lambda _q: [
            {"evidence_id": "ev-m1", "title": "Cohort methods", "quote": "cohort design for tar clearance"}
        ],
    )
    # the confirmed hypothesis's edge received exactly one committed experiment plan (graph v5), and
    # it exports into graph.json (the methods id ev-m1 grounded the plan via the extended ledger)
    plans = result.graph_json["experiment_plans"]
    assert len(plans) == 1
    (plan,) = plans.values()
    assert plan["design"] == "controlled_observational"
    # The plan is attached to the surfaced row so the Elaborator card can render from it.
    assert result.surfaced[0].experiment_plan["design"] == "controlled_observational"
    assert result.surfaced[0].experiment_attempt == {
        "candidate_id": "h1",
        "primary_edge_id": next(iter(plans)),
        "status": "committed",
        "grounding_status": "grounded",
    }
    # The refine-rounds setting is surfaced in the run's audit output.
    assert "experiment_refine_rounds: 1" in result.audit_memo
    assert "attempted hypotheses: 1" in result.audit_memo
    assert "h1: committed (grounded)" in result.audit_memo
    assert "committed experiment plans: 1" in result.audit_memo


def test_experiment_stage_is_a_noop_when_designer_not_injected():  # disabled-path byte identity
    result = _synthesist_run(expansion_confirmed_ids=["h1"])  # no experiment_* deps
    assert result.graph_json["experiment_plans"] == {}  # no Δ^experiment; the graph stays at v4
    assert "Experiment design (Experiment Designer/Experiment Validator)" not in result.audit_memo  # audit memo byte-identical (stage off)


def test_experiment_stage_attaches_the_candidates_committed_edge_ids():  # exact edge-ID binding
    from src.cycles.experiment import ExperimentDesigner, ExperimentValidator

    result = _synthesist_run(
        expansion_confirmed_ids=["h1"],
        experiment_designer=lambda: ExperimentDesigner(
            _ThreadBackend([_EXPERIMENT_DESIGNER_PLAN_RESPONSE])
        ),
        experiment_validator=ExperimentValidator(
            _JudgeBackend(_EXPERIMENT_VALIDATOR_VERDICT_RESPONSE)
        ),
        experiment_refine_rounds=1,
        retrieve_experiment_methods=lambda _q: [
            {"evidence_id": "ev-m1", "title": "Cohort methods", "quote": "cohort design for tar clearance"}
        ],
    )
    # the surfaced row carries the SAME edge id its committed plan is keyed by in graph.json, so a
    # renderer can resolve the plan exactly instead of guessing by focus-node adjacency.
    (edge_id,) = result.graph_json["experiment_plans"].keys()
    assert result.surfaced[0].hypothesis_edge_ids == (edge_id,)


def test_eager_verification_covers_every_candidate_bearing_edge_with_expansion_on():
    # With expansion enabled, Evidence Reviewer reviews every candidate-bearing
    # committed edge up front — the same eager pass as the expansion-off path. A 2-edge claim
    # (edge A smoking->lung cancer + a DISJOINT edge B): BOTH edges are reviewed, whether or not
    # a confirmed hypothesis touches them.
    from src.cycles.panel import LLMCriticPanelJudge
    from src.cycles.synthesist import ResearchSynthesist

    n_x = build_node(label="unrelated exposure", type="exposure/intervention")
    n_y = build_node(label="unrelated outcome", type="outcome")
    edge_b = build_edge(source_node_ids=[n_x.node_id], target_node_ids=[n_y.node_id],
                        direction="causal", relation_type="increases")

    class _TwoEdgeExtractor:
        def extract(self, claim):
            return ClaimExtraction(nodes=(_CAUSE, _EFFECT, n_x, n_y), edges=(_EDGE, edge_b))

    reviewed_edges: list[str] = []

    class _CapturingReviewer:
        def review(self, task, evidences):
            reviewed_edges.append(task.edge_id)
            coherence = v.CoherenceSignal(
                entailment={"supports": 1.0}, contradiction_risk={},
                context_fit=1.0, construct_match=1.0,
            )
            return v.EdgeReview(
                coherence={ev.evidence_id: coherence for ev in evidences},
                methods={ev.evidence_id: v.MethodsSignal(1.0, 1.0, 1.0, 1.0, 0.0) for ev in evidences},
                appraisal=v.VerifierAppraisal(open_risks=(), qualifiers=()),
            )

    # empty mine -> the ONLY edge committed during expansion is the confirmed hypothesis edge
    # (h1: tar clearance rate -> lung cancer), which touches the CLAIM node `lung cancer`.
    mine_empty = '{"concepts": [], "merge_decisions": []}'
    result = run_graph_state_workflow(
        "smoking increases lung cancer", extractor=_TwoEdgeExtractor(), embedder=_embedder,
        retrieve_for_target=lambda t: [_evidence("ev-1")],
        evidence_reviewer=_CapturingReviewer(),
        enable_expansion=True,
        research_synthesist=lambda: ResearchSynthesist(
            _ThreadBackend([
                mine_empty,
                _SYNTHESIST_PROPOSE_RESPONSE,
                _SYNTHESIST_REVISE_RESPONSE,
            ]),
            embedder=_embedder, claim="smoking increases lung cancer", proposal_count=6,
        ),
        critic_panel=(
            LLMCriticPanelJudge(
                _JudgeBackend(_CRITIC_PANEL_REVIEW_RESPONSE),
                judge_id=1,
                reference_field="oncology",
            ),
        ),
        expansion_confirmed_ids=["h1"],
    )

    # Evidence Reviewer ran on both candidate-bearing committed edges, including disjoint edge B.
    assert sorted(reviewed_edges) == sorted([_EDGE.edge_id, edge_b.edge_id])
    # expansion still ran: the confirmed hypothesis's new node is committed in the graph.
    assert "tar clearance rate" in {n["label"] for n in result.graph_json["nodes"].values()}
    # Both edges were verified as supported; neither was skipped.
    assert result.statuses[_EDGE.edge_id] == EdgeStatus.SUPPORTED.value
    assert result.statuses[edge_b.edge_id] == EdgeStatus.SUPPORTED.value


def test_authored_concepts_build_annotation_from_extracted_labels():
    # Authored concept weights are applied to extracted node labels inside the
    # run (no two-phase probe needed); the matched node gets the authored priority.
    result = run_graph_state_workflow(
        "smoking increases lung cancer", extractor=_FakeExtractor(), embedder=_embedder,
        retrieve_for_target=lambda t: [_evidence("ev-1")],
        evidence_reviewer=_FakeReviewer(),
        authored_concepts=[("smoking", 0.9)], priority_author="Example Researcher",
    )
    nodes = result.graph_json["nodes"]
    assert any(node.get("user_priority") == 0.9 for node in nodes.values())


# --- Claimless discovery: literature-first and anchored on the lens ----------------------
class _RecordingExtractor:
    def __init__(self):
        self.called = False

    def extract(self, claim):
        self.called = True
        return ClaimExtraction(nodes=(_CAUSE, _EFFECT), edges=(_EDGE,))


def test_claimless_run_skips_extraction_and_mines_toward_the_lens():
    from src.cycles.panel import LLMCriticPanelJudge
    from src.cycles.synthesist import ResearchSynthesist

    extractor = _RecordingExtractor()
    # The Research Synthesist mining turn seeds the graph from retrieved passages.
    mine = (
        '{"concepts": [{"label": "tensor-train rank", "type": "mediator", "definition": "d",'
        ' "source_context": "c", "bears_on_outcome": true, "evidence_id": "ev-m", "paper_id": "p1",'
        ' "matched_quote_span": "tt", "rationale": "r"}], "merge_decisions": []}'
    )
    result = run_graph_state_workflow(
        None,  # no claim: skip extraction and mine literature-first
        anchor="tensor factorization for LLM compression",  # the lens anchors scope + mining
        extractor=extractor, embedder=_embedder,
        retrieve_for_target=lambda t: [],  # no claim edges => no verification targets
        evidence_reviewer=_FakeReviewer(),
        enable_expansion=True,
        research_synthesist=lambda: ResearchSynthesist(
            _ThreadBackend([mine, '{"candidates": []}', '{"revised": [], "derived": []}']),
            embedder=_embedder, claim="tensor factorization for LLM compression", proposal_count=6,
        ),
        critic_panel=(
            LLMCriticPanelJudge(_JudgeBackend('{"candidates": [], "ranking": [], "critique": ""}'),
                                judge_id=1, reference_field="LLM compression"),
        ),
        enrichment_passages=("p1", "p2"),
    )
    assert extractor.called is False  # no claim => Extraction extraction never runs
    labels = {n["label"] for n in result.graph_json["nodes"].values()}
    assert "tensor-train rank" in labels  # literature-first mining seeded the graph
    assert "smoking" not in labels and "lung cancer" not in labels  # no claim-extracted nodes


def test_run_path_forwards_expansion_reranker_to_the_cycle():  # claimless emphasis reranking
    called = []

    def spy_reranker(scored):
        called.append(len(list(scored)))  # the cycle handed us the ranked set to re-order
        return scored

    _synthesist_run(expansion_confirmed_ids=["h1"], expansion_reranker=spy_reranker)
    assert called  # the injected reranker reached the hypothesis cycle


def test_surfaced_view_attaches_synthesist_lineage_from_provenance():  # derivation panel
    from src.cycles.hypothesis import HypothesisCandidate, ScoredCandidate
    from src.run_path import _surfaced_view

    # the Research Synthesist revise turn records a FLAT {strategy, wasDerivedFrom} provenance on a derived candidate
    # (synthesist._attach_lineage); _surfaced_view surfaces it as the lineage panel.
    prov = ({"strategy": "divergent", "wasDerivedFrom": ["h1"]},)
    derived = ScoredCandidate(candidate=HypothesisCandidate(candidate_id="derived", provenance=prov),
                              hyp_score=0.5, rank_score=0.5)
    h1 = ScoredCandidate(candidate=HypothesisCandidate(candidate_id="h1"), hyp_score=0.6,
                         rank_score=0.6)
    views = _surfaced_view((derived, h1))
    derived_view = next(x for x in views if x.candidate_id == "derived")
    assert derived_view.lineage["wasDerivedFrom"] == ["h1"]
    assert derived_view.lineage["strategy"] == "divergent"
    # a propose-turn candidate (no provenance) gets no lineage DAG.
    assert next(x for x in views if x.candidate_id == "h1").lineage is None


def test_surfaced_view_carries_public_idea_scaffold_for_trace_reporting():  # plain-language scaffold
    from src.cycles.hypothesis import HypothesisCandidate, ScoredCandidate
    from src.run_path import _surfaced_view

    candidate = HypothesisCandidate(
        candidate_id="h1",
        idea_scaffold={
            "claim_anchor": "compression can reduce factuality",
            "lever": "rate-distortion floor",
            "fail_safe": "smooth decline would weaken it",
        },
    )
    views = _surfaced_view((ScoredCandidate(candidate=candidate, hyp_score=0.6, rank_score=0.6),))
    assert views[0].idea_scaffold == candidate.idea_scaffold


def test_claimless_run_emphasis_rerank_promotes_authored_priority():  # end-to-end reranking
    from src.cycles.panel import LLMCriticPanelJudge
    from src.cycles.synthesist import ResearchSynthesist
    from src.emphasis import emphasis_reranker

    # the Research Synthesist mine seeds "seed concept"; the propose turn offers two gate-eligible candidates (one
    # label matches the authored 'tensor' priority, one does not), each building on the mined concept.
    mine = (
        '{"concepts": [{"label": "seed concept", "type": "mediator", "definition": "d",'
        ' "source_context": "c", "bears_on_outcome": true, "evidence_id": "e", "paper_id": "p1",'
        ' "matched_quote_span": "s", "rationale": "r"}], "merge_decisions": []}'
    )
    _sig = ('"llm_signals": {"novelty": 0.9, "testability": 0.9, "scope_fit": 0.9, "duplication": 0.05,'
            ' "plausibility": 0.8, "expected_yield": 0.8, "centrality": 0.8, "mechanism_specificity": 0.8}')

    def _cand(cid, label):
        return (
            f'{{"candidate_id": "{cid}", "new_nodes": [{{"label": "{label}", "type": "mechanism",'
            ' "definition": "d", "aliases": []}],'
            f' "new_edges": [{{"source": "{label}", "target": "seed concept",'
            ' "relation_type": "increases", "direction": "directed", "mechanism": "m"}],'
            f' "mechanism_chain": [{{"from": "{label}", "relation": "increases", "to": "seed concept",'
            ' "mechanism": "m"}], "source_quotes": [{"evidence_id": "e", "quote_span": "s",'
            ' "role_in_hypothesis": "r"}], "assumptions": [], "cross_concept": true,'
            f' "common_sense": false, {_sig}}}'
        )

    propose = f'{{"candidates": [{_cand("h1", "unrelated widget")}, {_cand("h2", "tensor train decomposition")}]}}'

    def _grade(cid):
        return (f'{{"candidate_id": "{cid}", "field_novelty": 0.8, "saturation": 0.1,'
                ' "already_established": false, "not_judgeable_by_field": false,'
                ' "mechanism_steps": [], "term_verdicts": [], "justification": "j"}')

    review = f'{{"candidates": [{_grade("h1")}, {_grade("h2")}], "ranking": ["h1", "h2"], "critique": "ok"}}'

    result = run_graph_state_workflow(
        None, anchor="tensor factorization",  # claimless: lens anchors the run
        extractor=_RecordingExtractor(), embedder=_embedder,
        retrieve_for_target=lambda t: [],
        evidence_reviewer=_FakeReviewer(),
        enable_expansion=True,
        research_synthesist=lambda: ResearchSynthesist(
            _ThreadBackend([mine, propose, '{"revised": [], "derived": []}']),
            embedder=_embedder, claim="tensor factorization", proposal_count=6,
        ),
        critic_panel=(
            LLMCriticPanelJudge(_JudgeBackend(review), judge_id=1, reference_field="LLM compression"),
        ),
        enrichment_passages=("p",),
        expansion_reranker=emphasis_reranker([("tensor", 0.9)], "author_directed"),
    )
    # Quality (RankScore) surfaces both h1, h2; the authored 'tensor' priority then RE-ORDERS the
    # surfaced winners, promoting h2 (tensor train decomposition) first.
    assert [s.candidate_id for s in result.surfaced] == ["h2", "h1"]
