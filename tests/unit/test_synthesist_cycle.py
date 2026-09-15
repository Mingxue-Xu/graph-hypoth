"""Research Synthesist loop — mine→propose→panel→revise→panel.

Hermetic end-to-end: a fake Research Synthesist backend plus fake Critic Panel backends
drive the whole loop over a real store; we assert the enriched node + confirmed hypothesis commit
and that both rounds ran. The deterministic tail is the shipped run_hypothesis_cycle (unchanged).
"""

from __future__ import annotations

from src.cycles.synthesist_cycle import run_synthesist_cycle
from src.cycles.panel import LLMCriticPanelJudge
from src.cycles.synthesist import ResearchSynthesist
from src.delta import AddNodeOp, MergeNodesOp, build_node
from src.graph_store import CausalClaimGraphStore
from src.transaction_log import GraphTransactionLog


class _ThreadBackend:
    def __init__(self, contents):
        self._contents = list(contents)
        self.calls: list[list] = []

    def run(self, messages, tools=None):
        self.calls.append(list(messages))
        content = self._contents.pop(0) if self._contents else ""
        return {"choices": [{"message": {"content": content}}]}


class _JudgeBackend:
    def __init__(self, content):
        self._content = content
        self.calls = 0

    def run(self, messages, tools=None):
        self.calls += 1
        return {"choices": [{"message": {"content": self._content}}]}


def _const_embed(texts):
    return [[1.0, 0.0] for _ in texts]


_MINE = (
    '{"concepts": [{"label": "rate-distortion floor", "type": "mechanism", "definition": "min loss",'
    ' "source_context": "caps factuality", "bears_on_outcome": true, "evidence_id": "ev1",'
    ' "paper_id": "p1", "matched_quote_span": "floor", "rationale": "r"}], "merge_decisions": []}'
)
_PROPOSE = (
    '{"candidates": [{"candidate_id": "h1",'
    ' "new_nodes": [{"label": "bridge", "type": "mediator", "definition": "d", "aliases": []}],'
    ' "new_edges": [{"source": "bridge", "target": "rate-distortion floor",'
    ' "relation_type": "mediates", "direction": "directed", "mechanism": "m"}],'
    ' "mechanism_chain": [{"from": "bridge", "relation": "mediates", "to": "rate-distortion floor",'
    ' "mechanism": "m"}],'
    ' "source_quotes": [{"evidence_id": "ev1", "quote_span": "floor", "role_in_hypothesis": "x"}],'
    ' "assumptions": [], "cross_concept": true, "common_sense": false,'
    ' "llm_signals": {"novelty": 0.9, "testability": 0.9, "scope_fit": 0.9, "duplication": 0.05,'
    ' "plausibility": 0.8, "expected_yield": 0.8, "centrality": 0.8, "mechanism_specificity": 0.8}}]}'
)
_REVISE = '{"revised": [], "derived": []}'  # nothing flagged -> h1 carries through unchanged
_REVIEW = (
    '{"candidates": [{"candidate_id": "h1", "field_novelty": 0.8, "saturation": 0.2,'
    ' "already_established": false, "not_judgeable_by_field": false,'
    ' "mechanism_steps": [{"from": "bridge", "relation": "mediates", "to": "rate-distortion floor",'
    ' "verdict": "sound", "note": ""}], "term_verdicts": [], "justification": "novel per P1"}],'
    ' "ranking": ["h1"], "critique": "ok"}'
)


def _synth(thread):
    return ResearchSynthesist(thread, embedder=_const_embed, claim="LLM compression", proposal_count=6)


def test_synthesist_cycle_inert_when_disabled():
    store = CausalClaimGraphStore()
    thread = _ThreadBackend([_MINE, _PROPOSE, _REVISE])
    result = run_synthesist_cycle(_synth(thread), [], store=store, passages=("p",), enabled=False)
    assert result.enabled is False and result.transactions == () and store.version == 0
    assert thread.calls == []  # the Research Synthesist thread never ran


def test_synthesist_cycle_runs_rounds_mines_and_commits_confirmed():
    store = CausalClaimGraphStore()
    thread = _ThreadBackend([_MINE, _PROPOSE, _REVISE])
    judge_backend = _JudgeBackend(_REVIEW)
    panel = [LLMCriticPanelJudge(judge_backend, judge_id=1, reference_field="LLM compression")]
    result = run_synthesist_cycle(
        _synth(thread), panel, store=store, passages=("### Passage 1\nbody",),
        corpus_list="P1: MDL", cited_passages="ev1: body",
        enabled=True, confirmed_ids=["h1"], embedder=_const_embed,
        timestamp="2026-07-05T00:00:00+00:00",
    )
    # the mine turn committed the enriched node (Δ^extract) AND the confirmed hypothesis (Δ^hypothesis)
    assert store.version >= 2
    assert any(tx.accepted for tx in result.transactions)
    assert [sc.candidate.candidate_id for sc in result.surfaced] == ["h1"]
    # the panel's decisive grade rode through onto the surfaced candidate (apply_panel)
    assert result.surfaced[0].candidate.novelty_graded is not None
    # The Research Synthesist ran all three turns; the Critic Panel ran both review rounds.
    assert len(thread.calls) == 3
    assert judge_backend.calls == 2


_REVIEW_ESTAB = (  # judge-2 disagrees with judge-1 on the decisive boolean -> defined disagreement
    '{"candidates": [{"candidate_id": "h1", "field_novelty": 0.6, "saturation": 0.3,'
    ' "already_established": true, "not_judgeable_by_field": false,'
    ' "mechanism_steps": [{"from": "bridge", "relation": "mediates", "to": "rate-distortion floor",'
    ' "verdict": "sound", "note": ""}], "term_verdicts": [], "justification": "seen in P1"}],'
    ' "ranking": ["h1"], "critique": "established"}'
)


def _run_with_panel(panel, store):
    thread = _ThreadBackend([_MINE, _PROPOSE, _REVISE])
    return run_synthesist_cycle(
        _synth(thread), panel, store=store, passages=("### Passage 1\nbody",),
        corpus_list="P1: MDL", cited_passages="ev1: body",
        enabled=True, confirmed_ids=["h1"], embedder=_const_embed,
        timestamp="2026-07-05T00:00:00+00:00",
    )


def test_synthesist_cycle_escalates_to_third_judge_only_on_disagreement():
    j1, j2, j3 = _JudgeBackend(_REVIEW), _JudgeBackend(_REVIEW_ESTAB), _JudgeBackend(_REVIEW)
    panel = [
        LLMCriticPanelJudge(j1, judge_id=1, reference_field="f"),
        LLMCriticPanelJudge(j2, judge_id=2, reference_field="f"),
        LLMCriticPanelJudge(j3, judge_id=3, reference_field="f"),  # tie-breaker
    ]
    _run_with_panel(panel, CausalClaimGraphStore())
    # base judges split 1-1 on already_established -> the 3rd judge breaks the tie, BOTH rounds
    assert j1.calls == 2 and j2.calls == 2
    assert j3.calls == 2  # +<=2 escalation (DESIGN §6), one 3rd-judge call per round


def test_synthesist_cycle_skips_third_judge_when_base_judges_concur():
    j1, j2, j3 = _JudgeBackend(_REVIEW), _JudgeBackend(_REVIEW), _JudgeBackend(_REVIEW)
    panel = [
        LLMCriticPanelJudge(j1, judge_id=1, reference_field="f"),
        LLMCriticPanelJudge(j2, judge_id=2, reference_field="f"),
        LLMCriticPanelJudge(j3, judge_id=3, reference_field="f"),
    ]
    _run_with_panel(panel, CausalClaimGraphStore())
    assert j1.calls == 2 and j2.calls == 2
    assert j3.calls == 0  # base judges concur -> the 3rd judge never runs


# The mine turn's batched same-concept decisions fold duplicate concepts.
_MINE_TWO_DUPES = (
    '{"concepts": ['
    '{"label": "reconstruction error", "type": "mediator", "definition": "frobenius error",'
    ' "bears_on_outcome": true, "evidence_id": "ev-a", "paper_id": "A"},'
    '{"label": "truncation error", "type": "mediator", "definition": "svd truncation loss",'
    ' "bears_on_outcome": true, "evidence_id": "ev-b", "paper_id": "B"}],'
    ' "merge_decisions": [{"a": "reconstruction error", "b": "truncation error",'
    ' "same_concept": true, "reason": "synonyms"}]}'
)
_MINE_MERGES_INTO_STORE = (
    '{"concepts": [{"label": "reconstruction error", "type": "mediator", "definition": "d",'
    ' "bears_on_outcome": true, "evidence_id": "ev-a", "paper_id": "A"}],'
    ' "merge_decisions": [{"a": "compression loss", "b": "reconstruction error",'
    ' "same_concept": true, "reason": "synonym"}]}'
)
_PROPOSE_EMPTY = '{"candidates": []}'


def test_synthesist_cycle_collapses_mined_duplicates_from_merge_decision():
    store = CausalClaimGraphStore()
    log = GraphTransactionLog()
    thread = _ThreadBackend([_MINE_TWO_DUPES, _PROPOSE_EMPTY])
    run_synthesist_cycle(_synth(thread), [], store=store, passages=("p",), enabled=True, log=log)
    ops = log.rows[0].delta.payload.operations
    assert sum(isinstance(op, MergeNodesOp) for op in ops) == 1
    assert len(store.nodes) == 1  # the two near-duplicates collapsed to one committed node
    node = next(iter(store.nodes.values()))
    evidence_ids = {p.get("evidence_id") for p in node.provenance if p.get("evidence_id")}
    assert evidence_ids == {"ev-a", "ev-b"}  # provenance union of both merged sources


def test_synthesist_cycle_merges_mined_duplicate_into_existing_store_node():
    store = CausalClaimGraphStore()
    existing = build_node(label="compression loss", type="mediator", definition="canonical")
    store.nodes[existing.node_id] = existing
    log = GraphTransactionLog()
    thread = _ThreadBackend([_MINE_MERGES_INTO_STORE, _PROPOSE_EMPTY])
    run_synthesist_cycle(_synth(thread), [], store=store, passages=("p",), enabled=True, log=log)
    ops = log.rows[0].delta.payload.operations
    assert sum(isinstance(op, MergeNodesOp) for op in ops) == 1
    assert len(store.nodes) == 1
    assert existing.node_id in store.nodes  # the existing node survives; the mined dup folds in
    evidence_ids = {
        p.get("evidence_id") for p in store.nodes[existing.node_id].provenance if p.get("evidence_id")
    }
    assert evidence_ids == {"ev-a"}


def test_synthesist_cycle_empty_merge_decisions_commit_unchanged():
    store = CausalClaimGraphStore()
    log = GraphTransactionLog()
    thread = _ThreadBackend([_MINE, _PROPOSE_EMPTY])  # _MINE carries "merge_decisions": []
    run_synthesist_cycle(_synth(thread), [], store=store, passages=("p",), enabled=True, log=log)
    ops = log.rows[0].delta.payload.operations
    assert len(ops) == 1
    assert all(isinstance(op, AddNodeOp) for op in ops)  # no MergeNodesOp: commit is unchanged
