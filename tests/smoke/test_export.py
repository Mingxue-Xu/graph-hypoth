"""Graph-state workflow smoke — the run path yields a structured
``GraphRunResult`` and exported artifacts.

Drives ``prepare -> extract -> score evidence -> review evidence -> priority`` end to
end with LLM-free fakes / fixtured sub-signals: a fake extractor + stub embedder, RAW
retrieved evidence that the run path SCORES with the real ``AssociationScorer`` (the evidence scoring
step), and fixtured appraiser sub-signals into the verifier. It then asserts the run output
replaces ``final_summary`` with structured statuses + evidence receipts + open risks, and
exports the graph JSON + edge table + audit memo. Deterministic; no network/real-embedder/LLM.
"""

from __future__ import annotations

import json

import pytest

from src.cycles import verification as v
from src.cycles.extraction import ClaimExtraction
from src.cycles.priority import UserPriorityAnnotation
from src.delta import VerificationVerdict, build_edge, build_node
from src.graph_store import EdgeStatus, EvidenceRole
from src.retrieval import scoring_defaults as sd
from src.run_path import GraphRunResult, run_graph_state_workflow
from src.state import RetrievedEvidence

pytestmark = pytest.mark.smoke


# --- the seed claim becomes a smoking -> lung-cancer subgraph (built with the real id helpers
# so the verification task references the SAME content-addressed edge_id the extraction cycle commits).
_CAUSE = build_node(label="smoking", type="exposure/intervention")
_EFFECT = build_node(label="lung cancer", type="outcome")
_EDGE = build_edge(
    source_node_ids=[_CAUSE.node_id], target_node_ids=[_EFFECT.node_id],
    direction="causal", relation_type="increases",
)


class _FakeExtractor:
    def extract(self, claim):  # the extraction role (LLM-free)
        return ClaimExtraction(nodes=(_CAUSE, _EFFECT), edges=(_EDGE,))


def _embedder(texts):
    return [[1.0, 0.0] for _ in texts]


class _FakeReviewer:  # strong, clean SUPPORT signal
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
    # RAW retrieved evidence (NOT pre-scored): the run path's evidence scoring step scores it.
    return RetrievedEvidence(
        evidence_id=evidence_id, source="openalex", source_id="s1",
        title="A cohort study of smoking and lung cancer",
        quote="smoking increases lung cancer risk", relevance="high",
        retrieved_by="retriever", tool_call_id=f"tool-{evidence_id}", rank=1, trust_tier="green",
    )


def _run(export_dir=None) -> GraphRunResult:
    task = v.VerificationTask(
        verification_task_id="vt-1", edge_id=_EDGE.edge_id, evidence_role=EvidenceRole.SUPPORT,
    )
    annotation = UserPriorityAnnotation(
        target_ids=[_CAUSE.node_id], priority_values=[0.8], author="alice",
    )
    return run_graph_state_workflow(
        "smoking increases lung cancer",
        extractor=_FakeExtractor(),
        embedder=_embedder,
        verification_work=[(task, [_evidence("ev-1")])],
        evidence_reviewer=_FakeReviewer(),
        priority_annotation=annotation,
        ledger_evidence_ids={"ev-1"},
        timestamp="2026-06-14T00:00:00+00:00",
        export_dir=export_dir,
    )


# Research Synthesist and Critic Panel expansion path.
# The panel expands the graph only when the expansion flag is enabled. This LLM-free thread
# backend stands in for the Research Synthesist turns; the flag-OFF default path never invokes it.
class _ThreadBackend:
    def __init__(self, contents):
        self._contents = list(contents)

    def run(self, messages, tools=None):
        content = self._contents.pop(0) if self._contents else ""
        return {"choices": [{"message": {"content": content}}]}


def _run_with_expansion(*, enable_expansion=None, export_dir=None):
    from src.cycles.synthesist import ResearchSynthesist

    task = v.VerificationTask(
        verification_task_id="vt-1", edge_id=_EDGE.edge_id, evidence_role=EvidenceRole.SUPPORT,
    )
    annotation = UserPriorityAnnotation(
        target_ids=[_CAUSE.node_id], priority_values=[0.8], author="alice",
    )
    return run_graph_state_workflow(
        "smoking increases lung cancer",
        extractor=_FakeExtractor(), embedder=_embedder,
        verification_work=[(task, [_evidence("ev-1")])],
        evidence_reviewer=_FakeReviewer(),
        priority_annotation=annotation, ledger_evidence_ids={"ev-1"},
        timestamp="2026-06-14T00:00:00+00:00", export_dir=export_dir,
        # A Research Synthesist is always supplied; only the feature flag decides whether synthesis
        # propose→panel→revise→panel loop is entered.
        research_synthesist=lambda: ResearchSynthesist(
            _ThreadBackend([]), embedder=_embedder,
            claim="smoking increases lung cancer", proposal_count=6,
        ),
        critic_panel=(),
        enable_expansion=enable_expansion,
    )


def test_run_path_expansion_disabled_reaches_export_with_zero_hypothesis_proposals():
    # The flag is explicitly disabled here even though the config default is enabled. With the Research Synthesist
    # synthesist injected, the run reaches export with NO expansion: enabling the feature is the
    # only thing that can introduce expansion behavior. (Expansion-enabled pipeline behavior is
    # covered by ``test_synthesist_path_runs_when_synthesist_injected``.)
    result = _run_with_expansion(enable_expansion=False)  # explicit opt-out
    assert isinstance(result, GraphRunResult)  # reached export
    assert not result.surfaced                                   # no hypothesis surfaced
    # The three accepted commits are extraction, verification, and authored priority.
    assert result.version == 3
    assert len(result.receipts) == 3 and all(r.status == "accepted" for r in result.receipts)


def test_graph_state_run_yields_graphrunresult_with_statuses_receipts_open_risks():
    result = _run()
    assert isinstance(result, GraphRunResult)

    # The graph-state run path is the default and it actually verified the edge (off unverified).
    assert result.statuses[_EDGE.edge_id] == EdgeStatus.SUPPORTED.value
    # Extraction, verification, and authored priority produce three auditable commits.
    assert result.version == 3
    assert len(result.receipts) == 3
    assert all(r.status == "accepted" for r in result.receipts)

    # open_risks is a structured collection (empty for clean support is fine).
    assert isinstance(result.open_risks, tuple)

    # the evidence scoring SCORE step actually ran: the committed EvidenceLink carries a bundle the
    # AssociationScorer produced (version-stamped), NOT a hand-built fixture.
    links = result.graph_json["evidence_links"]
    assert links and links[0]["signal"]["scoring_defaults_version"] == sd.SCORING_DEFAULTS_VERSION


def test_graphrunresult_replaces_final_summary_with_an_audit_memo():
    result = _run()
    assert isinstance(result.audit_memo, str) and result.audit_memo.strip()
    # the memo summarises the committed verdict (replaces the legacy free-text final_summary)
    assert EdgeStatus.SUPPORTED.value in result.audit_memo.lower()


def test_export_emits_graph_json_edge_table_audit_memo(tmp_path):
    result = _run(export_dir=tmp_path)

    # in-memory artifacts
    assert _EDGE.edge_id in result.graph_json["edges"]
    json.dumps(result.graph_json)  # graph JSON round-trips
    assert len(result.edge_table) == 1
    row = result.edge_table[0]
    assert row.edge_id == _EDGE.edge_id
    assert row.status == EdgeStatus.SUPPORTED.value
    assert row.verdict == VerificationVerdict.SUPPORT.value

    # written to disk
    assert (tmp_path / "graph.json").is_file()
    assert (tmp_path / "edge_table.csv").is_file()
    assert (tmp_path / "audit_memo.md").is_file()
