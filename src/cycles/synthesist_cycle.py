"""Research Synthesist round loop — propose → panel → revise → panel.

Orchestrates the Research Synthesist cycle: the Research Synthesist thread (mine → propose →
revise) interleaved with the Critic Panel (two grounded rounds), feeding the unchanged
deterministic tail (``run_hypothesis_cycle`` = gate → score → rank → surface → confirm → commit).
The panel only grades (median field_novelty/saturation + Borda rank fusion); ``apply_panel`` writes
those onto the candidates so ``Gate_h`` / ``HypScore`` / ``RankScore`` consume the same shape they
always did. Runs only when expansion is enabled and the seams are injected.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src import graph_config_defaults as gcd
from src.cycles.enrichment import (
    build_enrichment_delta,
    concepts_to_nodes,
    outcome_anchored,
)
from src.cycles.hypothesis import (
    HypothesisCandidate,
    HypothesisCycleResult,
    run_hypothesis_cycle,
)
from src.cycles.panel import (
    AggregatedPanel,
    aggregate_panel,
    apply_panel,
    panel_disagreement,
    panel_result_for_revise,
)
from src.delta import MergeNodesOp
from src.graph_store import CausalClaimGraphStore, ConceptNode
from src.progress import report_progress
from src.transaction_log import GraphTransactionLog
from src.validator import GraphDeltaValidator


def _mined_merge_ops(
    nodes: Sequence[ConceptNode],
    decisions: Sequence[dict[str, Any]],
    *,
    store: CausalClaimGraphStore,
) -> list[MergeNodesOp]:
    """Resolve the Research Synthesist mine turn's label-keyed ``same_concept`` decisions
    (``synthesist._parse_merge_decisions``) into ``MergeNodesOp``s: a mined-vs-mined pair
    survives on the smaller content-addressed id (the canonical ordering convention in
    ``extraction.canonicalize_concepts``); a mined-vs-store pair survives on the already-committed
    store node, so the mined duplicate folds into it. Decisions that don't resolve to a node, or
    resolve to two already-committed store nodes (neither side freshly mined), are skipped — the
    mine turn's merge_decisions come back empty on the single-pass fresh-run path, so this is a
    no-op there."""
    mined_by_label = {node.label: node.node_id for node in nodes}
    store_by_label = {node.label: node.node_id for node in store.nodes.values()}

    def _resolve(label: str) -> tuple[str, bool] | None:
        if label in store_by_label:
            return store_by_label[label], True
        if label in mined_by_label:
            return mined_by_label[label], False
        return None

    ops: list[MergeNodesOp] = []
    for decision in decisions:
        if not decision.get("same_concept"):
            continue
        resolved_a = _resolve(str(decision.get("a", "")))
        resolved_b = _resolve(str(decision.get("b", "")))
        if resolved_a is None or resolved_b is None:
            continue
        a_id, a_in_store = resolved_a
        b_id, b_in_store = resolved_b
        if a_id == b_id or (a_in_store and b_in_store):
            continue
        if a_in_store or b_in_store:
            survivor_id, merged_id = (a_id, b_id) if a_in_store else (b_id, a_id)
        else:
            survivor_id, merged_id = sorted((a_id, b_id))
        ops.append(MergeNodesOp(survivor_node_id=survivor_id, merged_node_id=merged_id))
    return ops


def _commit_mined_concepts(
    mine_result: Any,
    *,
    store: CausalClaimGraphStore,
    log: GraphTransactionLog,
    validator: GraphDeltaValidator,
    max_concepts: int,
    timestamp: str | None,
) -> None:
    """Commit the Research Synthesist mine turn's outcome-anchored concepts as unverified
    extraction-delta nodes,
    reusing the deterministic enrichment path (anchor filter → per-cycle cap → content-address dedup
    → single Δ^extract). Mirrors run_enrichment_cycle's commit for miner output. The mine turn's
    same-concept merge decisions fold duplicates into ``MergeNodesOp`` objects in the same delta,
    so a decision-marked duplicate never lands as a separate node."""
    admitted = outcome_anchored(list(mine_result.concepts))[:max_concepts]
    nodes = concepts_to_nodes(admitted)
    if not nodes:
        return
    merge_ops = _mined_merge_ops(nodes, mine_result.merge_decisions, store=store)
    delta = build_enrichment_delta(
        base_graph_hash=store.base_hash, nodes=nodes, merge_ops=merge_ops
    )
    log.commit(store, delta, validator, author="research_synthesist", timestamp=timestamp)


def _panel_round(
    critic_panel: Sequence[Any],
    candidates: Sequence[HypothesisCandidate],
    *,
    store: CausalClaimGraphStore,
    corpus_list: str,
    cited_passages: str,
    titles: dict[str, str] | None,
    round_index: int = 1,
) -> AggregatedPanel:
    """Run one Critic Panel round over the pool and reduce it to deterministic consensus.

    Two base judges on different models review; a 3rd judge (index 2) is a tie-breaker that runs ONLY on
    defined disagreement, so the escalation cost is bounded
    at one extra call per round. A panel of 1 or 2 has no tie-breaker and never escalates."""
    node_labels = {nid: node.label for nid, node in store.nodes.items()}

    def _review(judge: Any) -> Any:
        return judge.review(
            candidates,
            corpus_list=corpus_list,
            cited_passages=cited_passages,
            node_labels=node_labels,
            titles=titles,
        )

    reviews = []
    for index, judge in enumerate(critic_panel[:2], 1):
        report_progress(
            "Propose / critique", f"round {round_index} · critic judge",
            current=index, total=min(2, len(critic_panel)),
        )
        reviews.append(_review(judge))
    if len(critic_panel) > 2 and panel_disagreement(reviews):
        report_progress(None, f"round {round_index} · disagreement — calling third judge")
        reviews.append(_review(critic_panel[2]))  # 3rd judge only on disagreement
    elif len(critic_panel) > 2:
        report_progress(None, f"round {round_index} · third judge not needed", status="skipped")
    return aggregate_panel(reviews)


def _merge_revised_pool(
    proposed: Sequence[HypothesisCandidate],
    revised: Sequence[HypothesisCandidate],
    derived: Sequence[HypothesisCandidate],
) -> list[HypothesisCandidate]:
    """The post-revise pool: unflagged proposer candidates stay as-is; a flagged survivor is
    replaced in place by its repaired ``revised`` version (same candidate_id); ``derived`` are
    appended (new ids). Preserves proposer order, then derived order."""
    revised_by_id = {c.candidate_id: c for c in revised}
    pool = [revised_by_id.get(c.candidate_id, c) for c in proposed]
    seen = {c.candidate_id for c in pool}
    pool.extend(c for c in derived if c.candidate_id not in seen)
    return pool


def run_synthesist_cycle(
    synthesist: Any,
    critic_panel: Sequence[Any] = (),
    *,
    store: CausalClaimGraphStore,
    passages: Sequence[str] = (),
    merge_candidates: Sequence[Any] = (),
    corpus_list: str = "",
    cited_passages: str = "",
    titles: dict[str, str] | None = None,
    enabled: bool = False,
    precondition: bool = True,
    enrich: bool = True,
    max_concepts: int | None = None,
    extract_validator: GraphDeltaValidator | None = None,
    # deterministic-tail params (forwarded verbatim to run_hypothesis_cycle) ------------------
    confirmed_ids: Sequence[str] = (),
    confirm_fn: Any = None,
    reranker: Any = None,
    embedder: Any = None,
    log: GraphTransactionLog | None = None,
    validator: GraphDeltaValidator | None = None,
    thresholds: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
    budget: int | None = None,
    author: str = "hypothesis_expander",
    timestamp: str | None = None,
) -> HypothesisCycleResult:
    """Run the Research Synthesist round loop and commit the confirmed survivors.

    Order: Research Synthesist mine (+ enrich-commit) → propose → Critic Panel round 1 → apply →
    Research Synthesist revise → Critic Panel
    round 2 → apply → the deterministic tail (``run_hypothesis_cycle`` with no seams: gate → score →
    rank → surface → confirm → commit). Inert when disabled / precondition fails / no synthesist
    (returns the same empty ``HypothesisCycleResult`` shape). The deterministic core is reused
    unchanged; the panel's median/Borda consensus is written onto candidates via ``apply_panel``.
    """
    log = log if log is not None else GraphTransactionLog()
    validator = validator if validator is not None else GraphDeltaValidator()
    extract_validator = extract_validator if extract_validator is not None else validator
    max_concepts = max_concepts if max_concepts is not None else gcd.ENRICHMENT_MAX_CONCEPTS

    def _tail(candidates: Sequence[HypothesisCandidate], *, enabled_tail: bool) -> HypothesisCycleResult:
        return run_hypothesis_cycle(
            candidates, store=store, enabled=enabled_tail, precondition=precondition,
            confirmed_ids=confirmed_ids, confirm_fn=confirm_fn, reranker=reranker,
            embedder=embedder, log=log, validator=validator, thresholds=thresholds,
            weights=weights, budget=budget, author=author, timestamp=timestamp,
        )

    if not enabled or not precondition or synthesist is None:
        report_progress("Propose / critique", "disabled or precondition unmet", status="skipped")
        return _tail((), enabled_tail=False)

    # Mine concepts, adjudicate the precomputed borderline merge pairs, then commit the
    # outcome-anchored concepts. Fresh single-pass runs provide no merge candidates.
    report_progress("Propose / critique", "mining concepts from literature")
    mine_result = synthesist.mine(passages, merge_candidates)
    if enrich:
        _commit_mined_concepts(
            mine_result, store=store, log=log, validator=extract_validator,
            max_concepts=max_concepts, timestamp=timestamp,
        )

    # Propose candidate hypotheses over the enriched store.
    report_progress("Propose / critique", "proposing hypotheses")
    candidates: list[HypothesisCandidate] = list(synthesist.propose(store))
    if not candidates:
        report_progress("Propose / critique", "no candidate hypotheses", status="skipped")
        return _tail((), enabled_tail=True)

    # Critic Panel round 1 writes the decisive grades onto the pool.
    if critic_panel:
        report_progress("Propose / critique", f"round 1 · reviewing {len(candidates)} candidates")
        agg1 = _panel_round(
            critic_panel, candidates, store=store,
            corpus_list=corpus_list, cited_passages=cited_passages, titles=titles,
        )
        candidates = apply_panel(candidates, agg1)

        # Repair panel-flagged survivors and derive new candidates.
        report_progress("Propose / critique", "revising hypotheses after round 1")
        revise = synthesist.revise(store, panel_result_for_revise(agg1))
        candidates = _merge_revised_pool(candidates, revise.revised, revise.derived)

        # Critic Panel round 2 produces final grades over the revised pool.
        report_progress("Propose / critique", f"round 2 · reviewing {len(candidates)} candidates")
        agg2 = _panel_round(
            critic_panel, candidates, store=store,
            corpus_list=corpus_list, cited_passages=cited_passages, titles=titles,
            round_index=2,
        )
        candidates = apply_panel(candidates, agg2)
    else:
        report_progress("Propose / critique", "critic panel unavailable", status="skipped")

    # deterministic tail: gate → score → rank → surface → confirm → commit (core UNCHANGED)
    report_progress("Propose / critique", f"{len(candidates)} candidates ready for ranking", status="completed")
    return _tail(candidates, enabled_tail=True)
