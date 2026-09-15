"""Graph-state workflow and artifact export.

The workflow shares one graph store, transaction log, and validator across extraction,
evidence scoring and review, priority annotation, Research Synthesist expansion, experiment
design, and export. It returns structured statuses, transaction receipts, open risks, graph
JSON, an edge table, and an audit memo.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from src import graph_config_defaults as gcd
from src.cycles.experiment import run_experiment_stage
from src.cycles.extraction import (
    Embedder,
    Extractor,
    run_extraction_cycle,
)
from src.cycles.priority import (
    UserPriorityAnnotation,
    build_priority_from_labels,
    record_user_priority_annotations,
)
from src.cycles.synthesist_cycle import run_synthesist_cycle
from src.cycles.verification import (
    EvidenceReviewer,
    VerificationTask,
    run_verification_cycle,
)
from src.graph_store import CausalClaimGraphStore, EvidenceRole
from src.progress import report_progress
from src.retrieval.association import (
    AssociationScorer,
    AssociationTarget,
    CandidateEvidence,
)
from src.state import RetrievedEvidence
from src.transaction_log import GraphTransactionLog, Receipt, TransactionRow
from src.validator import GraphDeltaValidator


@dataclass(frozen=True)
class EdgeStatusRow:
    """One row of the exported edge table: the edge's committed verification status."""

    edge_id: str
    relation_type: str
    status: str
    verdict: str | None
    confidence: float | None
    open_risks: tuple[str, ...]


@dataclass(frozen=True)
class SurfacedHypothesis:
    """One ranked Synthesist candidate prepared for confirmation and reporting.

    The row carries field-relative scores and proposer flags so consumers need not derive them.
    """

    rank: int
    candidate_id: str
    field_novelty: float
    saturation: float
    hyp_score: float
    rank_score: float
    cross_concept: bool
    common_sense: bool
    rationale: str
    new_node_labels: tuple[str, ...]
    new_node_details: tuple[dict[str, str], ...] = ()
    idea_scaffold: dict[str, str] = field(default_factory=dict)
    # Parent candidates and derivation strategy for a derived candidate.
    lineage: dict[str, Any] | None = None
    # Critic Panel mechanism-step and terminology grades used by the report and Elaborator.
    mechanism_steps: tuple[dict[str, Any], ...] = ()  # concept->relation->concept chain verdicts
    term_audit: tuple[dict[str, Any], ...] = ()       # per-term verdicts
    scaffold_gaps: tuple[str, ...] = ()               # e.g. ("problem:missing",)
    # Committed experiment plan attached after the experiment stage.
    experiment_plan: dict[str, Any] | None = None
    # Report-side outcome for the post-confirmation experiment attempt.  This is
    # deliberately separate from durable graph state: failed attempts remain
    # inspectable without masquerading as committed plans.
    experiment_attempt: dict[str, Any] | None = None
    # Confirmed candidate edge ids let renderers bind a plan exactly rather than by adjacency.
    hypothesis_edge_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphRunResult:
    """Run-level graph statuses, receipts, risks, and exported artifact data."""

    graph_id: str
    version: int
    statuses: dict[str, str]               # edge_id -> EdgeStatus value (status is edge-only)
    receipts: tuple[Receipt, ...]          # every transaction receipt, in commit order
    open_risks: tuple[str, ...]
    graph_json: dict[str, Any]             # export 1
    edge_table: tuple[EdgeStatusRow, ...]  # export 2
    audit_memo: str                        # export 3 (replaces final_summary)
    surfaced: tuple[SurfacedHypothesis, ...] = ()  # Synthesist ranking; empty unless expansion ran


def _lineage_of(candidate: Any) -> dict[str, Any] | None:
    """Return a derived candidate's parents and strategy, or ``None`` for a proposal."""
    prov = candidate.provenance[0] if candidate.provenance else None
    if not prov or "wasDerivedFrom" not in prov:
        return None
    return {
        "wasDerivedFrom": prov.get("wasDerivedFrom"),
        "strategy": prov.get("strategy"),
    }


def _surfaced_view(scored: Sequence[Any]) -> tuple[SurfacedHypothesis, ...]:
    """Project ranked candidates into surfaced report rows with derivation lineage."""
    views: list[SurfacedHypothesis] = []
    for rank, candidate_scored in enumerate(scored, start=1):
        candidate = candidate_scored.candidate
        views.append(
            SurfacedHypothesis(
                rank=rank,
                candidate_id=candidate.candidate_id,
                field_novelty=candidate.ranking_novelty,
                saturation=candidate.saturation,
                hyp_score=candidate_scored.hyp_score,
                rank_score=candidate_scored.rank_score,
                cross_concept=candidate.cross_concept,
                common_sense=candidate.common_sense,
                rationale=candidate.rationale,
                new_node_labels=tuple(node.label for node in candidate.new_nodes),
                new_node_details=tuple(
                    {
                        "label": str(node.label),
                        "type": str(node.type),
                        "definition": str(node.definition),
                    }
                    for node in candidate.new_nodes
                ),
                idea_scaffold=dict(candidate.idea_scaffold),
                lineage=_lineage_of(candidate),
                mechanism_steps=tuple(candidate.mechanism_steps),
                term_audit=tuple(candidate.term_audit),
            )
        )
    return tuple(views)


def _association_target(store: CausalClaimGraphStore, task: VerificationTask) -> AssociationTarget:
    """Project a committed graph edge into the scorer's ``AssociationTarget``.

    The scoring anchor is the edge's cause->effect phrase built from its node labels; the
    role and identifiers come from the task.
    """
    edge = store.edges[task.edge_id]

    def _label(node_id: str) -> str:
        node = store.nodes.get(node_id)
        return node.label if node is not None else node_id

    source = " & ".join(_label(n) for n in edge.source_node_ids)
    effect = " & ".join(_label(n) for n in edge.target_node_ids)
    return AssociationTarget(
        target_id=task.target_id,
        verification_task_id=task.verification_task_id,
        target_kind="causal_edge",
        label=f"{source} {edge.relation_type} {effect}".strip(),
        evidence_role=str(task.evidence_role),
        definition=edge.mechanism,
        question=task.question,
        criteria=task.criteria,
    )


def _edge_relationship(store: CausalClaimGraphStore, edge_id: str) -> tuple[str, str]:
    """Render an edge relationship as the Evidence Reviewer's question and criteria."""
    edge = store.edges[edge_id]

    def _label(node_id: str) -> str:
        node = store.nodes.get(node_id)
        return node.label if node is not None else node_id

    source = " & ".join(_label(n) for n in edge.source_node_ids)
    effect = " & ".join(_label(n) for n in edge.target_node_ids)
    question = (
        f"Does this passage bear on the causal relationship: "
        f"{source} {edge.relation_type} {effect}?"
    )
    criteria = edge.mechanism or f"mechanism by which {source} {edge.relation_type} {effect}"
    return question, criteria


def _receipt_of(row: TransactionRow) -> Receipt:
    return Receipt(
        tx_id=row.tx_id, base_hash=row.base_graph_hash, delta_hash=row.delta_hash,
        status=row.validation_status, author=row.author, timestamp=row.timestamp,
    )


def passages_from_quotes(
    quotes: Iterable[str], *, max_chars: int | None, max_passages: int,
) -> list[str]:
    """Deduplicate ``quotes`` in first-seen order, cap the count, and optionally truncate each
    passage truncated to ``max_chars`` unless it is ``None`` (the claimless discovery path uses
    uncapped source passages)."""
    seen: set[str] = set()
    passages: list[str] = []
    for quote in quotes:
        if quote and quote not in seen:
            seen.add(quote)
            passages.append(quote[:max_chars] if max_chars is not None else quote)
            if len(passages) >= max_passages:
                break
    return passages


def _passages_from_work(
    work: Sequence[tuple[VerificationTask, Sequence[RetrievedEvidence]]],
) -> list[str]:
    """Derive capped Synthesist passages from retrieved evidence quotes."""
    quotes = (
        getattr(evidence, "quote", "") or "" for _, raw in work for evidence in raw
    )
    return passages_from_quotes(
        quotes,
        max_chars=gcd.PASSAGE_PROMPT_MAX_CHARS,
        max_passages=gcd.ENRICHMENT_MAX_PASSAGES,
    )


def _audit_memo(
    store: CausalClaimGraphStore,
    verdicts: dict[str, str],
    open_risks: Sequence[str],
    receipts: Sequence[Receipt],
    experiment_refine_rounds: int | None = None,
    experiment_attempts: Sequence[Mapping[str, Any]] = (),
) -> str:
    lines = [f"# Run audit — graph {store.graph_id}, version {store.version}", ""]
    lines.append("## Edge verification statuses")
    if store.edges:
        for edge in store.edges.values():
            verdict = verdicts.get(edge.edge_id)
            suffix = f" (verdict={verdict})" if verdict else ""
            lines.append(f"- {edge.relation_type}: status={str(edge.status)}{suffix}")
    else:
        lines.append("- (no edges committed)")
    lines += ["", "## Open risks"]
    lines += [f"- {risk}" for risk in open_risks] if open_risks else ["- none"]
    accepted = sum(1 for r in receipts if r.status == "accepted")
    lines += ["", f"## Receipts: {len(receipts)} transactions, {accepted} accepted"]
    # Include experiment configuration and outcome only when the stage ran.
    if experiment_refine_rounds is not None:
        attempts = list(experiment_attempts)
        committed_attempts = sum(1 for attempt in attempts if attempt.get("status") == "committed")
        lines += [
            "", "## Experiment design (Experiment Designer/Experiment Validator)",
            f"- experiment_refine_rounds: {experiment_refine_rounds}",
            f"- attempted hypotheses: {len(attempts)}",
            f"- committed attempts: {committed_attempts}",
            f"- committed experiment plans: {len(store.experiment_plans)}",
        ]
        for attempt in attempts:
            detail = str(attempt.get("grounding_status") or attempt.get("reason") or "")
            suffix = f" ({detail})" if detail else ""
            lines.append(
                f"- {attempt.get('candidate_id', '(unknown)')}: "
                f"{attempt.get('status', 'unknown')}{suffix}"
            )
    return "\n".join(lines)


def _export_to_disk(result: GraphRunResult, export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "graph.json").write_text(
        json.dumps(result.graph_json, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (export_dir / "edge_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["edge_id", "relation_type", "status", "verdict", "confidence", "open_risks"])
        for row in result.edge_table:
            writer.writerow([
                row.edge_id, row.relation_type, row.status, row.verdict or "",
                "" if row.confidence is None else row.confidence, "; ".join(row.open_risks),
            ])
    (export_dir / "audit_memo.md").write_text(result.audit_memo, encoding="utf-8")


def _plan_verification_work(
    store: CausalClaimGraphStore,
    verify_roles: Sequence[EvidenceRole],
    retrieve_for_target: Callable[[VerificationTask], Sequence[RetrievedEvidence]],
) -> list[tuple[VerificationTask, list[RetrievedEvidence]]]:
    """Create one ``VerificationTask`` per committed edge and evidence role.

    Edge ids are content-addressed and only exist AFTER extraction, so the schedule is built
    here from ``store.edges`` rather than supplied up front.
    """
    work: list[tuple[VerificationTask, list[RetrievedEvidence]]] = []
    for index, edge_id in enumerate(store.edges, 1):
        report_progress(
            "Checking evidence", "retrieving for link",
            current=index, total=len(store.edges),
        )
        question, criteria = _edge_relationship(store, edge_id)
        for role in verify_roles:
            task = VerificationTask(
                verification_task_id=f"vt-{edge_id[:12]}-{str(role)}",
                edge_id=edge_id, evidence_role=role,
                question=question, criteria=criteria,
            )
            work.append((task, list(retrieve_for_target(task))))
    return work


def _attach_experiment_plans(
    surfaced: Sequence[SurfacedHypothesis],
    confirmed: Sequence[Any],
    experiment_plans: Mapping[str, Any],
) -> tuple[SurfacedHypothesis, ...]:
    """Attach each confirmed hypothesis's committed edge ids and experiment plan,
    matching a candidate to
    its bound edge (its primary new edge). Rows with no confirmed edges pass through unchanged."""
    edge_ids_by_candidate = {
        c.candidate_id: tuple(e.edge_id for e in c.new_edges) for c in confirmed if c.new_edges
    }
    out: list[SurfacedHypothesis] = []
    for row in surfaced:
        edge_ids = edge_ids_by_candidate.get(row.candidate_id)
        if not edge_ids:
            out.append(row)
            continue
        plan = experiment_plans.get(edge_ids[0])
        if plan is not None:
            dumped = plan.model_dump(mode="json")
            attempt = {
                "candidate_id": row.candidate_id,
                "primary_edge_id": edge_ids[0],
                "status": "committed",
                "grounding_status": str(dumped.get("grounding_status") or "legacy_unspecified"),
            }
        else:
            dumped = row.experiment_plan
            attempt = {
                "candidate_id": row.candidate_id,
                "primary_edge_id": edge_ids[0],
                "status": "not_committed",
                "reason": "no parseable, content-complete plan after bounded design/refinement",
            }
        out.append(replace(
            row, hypothesis_edge_ids=edge_ids,
            experiment_plan=dumped,
            experiment_attempt=attempt,
        ))
    return tuple(out)


def run_graph_state_workflow(
    claim: str | None,
    *,
    anchor: str | None = None,
    extractor: Extractor,
    embedder: Embedder,
    verification_work: Sequence[tuple[VerificationTask, Sequence[RetrievedEvidence]]] = (),
    retrieve_for_target: Callable[[VerificationTask], Sequence[RetrievedEvidence]] | None = None,
    verify_roles: Sequence[EvidenceRole] = (EvidenceRole.SUPPORT,),
    evidence_reviewer: EvidenceReviewer,
    priority_annotation: UserPriorityAnnotation | None = None,
    authored_concepts: Sequence[tuple[str, float]] = (),
    priority_author: str = "",
    priority_focus: str = "",
    ledger_evidence_ids: Collection[str] = (),
    timestamp: str | None = None,
    export_dir: Path | None = None,
    enrichment_passages: Sequence[str] = (),
    enable_expansion: bool | None = None,
    # The Research Synthesist thread factory and different-model Critic Panel run the
    # propose→panel→revise→panel loop only when a
    # Research Synthesist is injected; otherwise expansion is a no-op with no surfaced candidates.
    research_synthesist: Callable[[], Any] | None = None,
    critic_panel: Sequence[Any] = (),
    expansion_confirmed_ids: Collection[str] = (),
    expansion_confirm_fn: Callable[[Sequence[Any]], Collection[str]] | None = None,
    expansion_reranker: Callable[[Sequence[Any]], Sequence[Any]] | None = None,
    expansion_precondition: bool = True,
    # Post-confirmation experiment stage. It runs only when the Experiment Designer is injected.
    experiment_designer: Callable[[], Any] | None = None,
    experiment_validator: Any | None = None,
    experiment_refine_rounds: int = 0,
    retrieve_experiment_methods: Callable[[str], Sequence[RetrievedEvidence]] | None = None,
    initial_open_risks: Sequence[str] = (),
    open_risks_provider: Callable[[], Sequence[str]] | None = None,
) -> GraphRunResult:
    """Run the default graph-state path and return a :class:`GraphRunResult`.

    Every cycle commits through the shared ``store``/``log``; the log is the authoritative
    receipt trail. Verification work is either explicit (``verification_work`` pairs
    each ``VerificationTask`` with its raw evidence — used by tests) or AUTO (pass
    ``retrieve_for_target``; the path schedules one task per (extracted edge, role) and pulls
    evidence per target). In auto mode the validator ledger is derived from the retrieved
    evidence. Evidence scoring turns raw evidence into triaged candidates here.
    """
    # 1. prepare — one transaction spine for the whole run. ``anchor`` is the scope / mining /
    # proposer focus: the claim itself, or the caller-provided lens in claimless discovery.
    # Extraction is the only step that needs the literal claim.
    anchor = anchor if anchor is not None else (claim or "")
    store = CausalClaimGraphStore(scope_context={"seed_claim": anchor})
    log = GraphTransactionLog()
    extract_validator = GraphDeltaValidator(ledger_evidence_ids=ledger_evidence_ids)
    scorer = AssociationScorer(embedder=embedder)

    # 2. extract — seed claim -> committed unverified subgraph (the extraction delta has no evidence).
    # Skipped in claimless discovery mode (no claim to extract): the graph is seeded literature-first
    # by the enrichment miner instead.
    if claim:
        report_progress("Mapping claim", "extracting concepts and links")
        run_extraction_cycle(
            claim, extractor=extractor, embedder=embedder,
            store=store, log=log, validator=extract_validator, timestamp=timestamp,
        )
        report_progress(
            "Mapping claim", f"{len(store.nodes)} concepts, {len(store.edges)} links",
            status="completed",
        )
    else:
        report_progress("Mapping claim", "claimless discovery", status="skipped")

    # 3. plan the verification work + the verify-phase ledger (auto = derived from retrieval).
    report_progress("Checking evidence", "collecting and scoring passages")
    verdicts: dict[str, str] = {}
    revision_risks: list[str] = list(initial_open_risks)
    if verification_work:
        work: list[tuple[VerificationTask, list[RetrievedEvidence]]] = [
            (task, list(raw)) for task, raw in verification_work
        ]
    elif retrieve_for_target is not None:
        work = _plan_verification_work(store, verify_roles, retrieve_for_target)
        # surface the auto-scheduling coverage so support-only runs are not read as complete.
        unscheduled = [str(r) for r in EvidenceRole if r not in set(verify_roles)]
        if unscheduled:
            revision_risks.append(
                "auto-scheduling verified roles "
                f"{{{', '.join(str(r) for r in verify_roles)}}} per edge; "
                f"roles not scheduled: {{{', '.join(unscheduled)}}}"
            )
    else:
        work = []
    verify_ledger = set(ledger_evidence_ids)
    for _, raw in work:
        verify_ledger.update(ev.evidence_id for ev in raw)
    validator = GraphDeltaValidator(ledger_evidence_ids=verify_ledger)

    # 3'. score evidence -> triage per (edge, role), then accumulate every role's triaged
    # candidates into one bucket per edge. Edge status is edge-level, so verifying one
    # (edge, role) at a time settles the edge on the first role and the validator then rejects
    # every later role as a redundant insufficient->insufficient transition; instead one
    # Δ^verify per edge sees support + counterevidence together and commits exactly once.
    candidates_by_edge: dict[str, list[CandidateEvidence]] = {}
    rep_task_by_edge: dict[str, VerificationTask] = {}
    for index, (task, raw_evidence) in enumerate(work, 1):
        report_progress(
            "Checking evidence", "scoring evidence task", current=index, total=len(work),
        )
        target = _association_target(store, task)
        scored = scorer.score_candidates(target, list(raw_evidence))
        triaged = scorer.triage(scored)  # top candidates handed to Evidence Reviewer by score
        candidates_by_edge.setdefault(task.edge_id, []).extend(triaged)
        # Retain one representative task, including its evidence role, for the combined edge review.
        rep_task_by_edge.setdefault(task.edge_id, task)
        if not triaged:  # this role cleared nothing — a per-role coverage gap, noted as a risk
            revision_risks.append(
                f"edge {task.edge_id} ({str(task.evidence_role)}): "
                "no retrieved evidence cleared the relevance gate"
            )

    # 4. verify evidence: one Evidence Reviewer cycle per edge over accumulated candidates gives
    # one verdict and one commit. Review is eager:
    # every candidate-bearing committed edge is verified up front, whether or not expansion
    # follows. Hypothesis edges are still never reviewed here — they carry no retrieved
    # candidates (grounding them is deferred to the experiment stage).
    def _verify(edge_ids: Iterable[str]) -> None:
        reviewable = [edge_id for edge_id in edge_ids if candidates_by_edge.get(edge_id)]
        for index, edge_id in enumerate(reviewable, 1):
            candidates = candidates_by_edge.get(edge_id)
            if not candidates:  # no role cleared the gate for this edge — nothing to verify
                continue
            report_progress(
                "Checking evidence", "reviewing link",
                current=index, total=len(reviewable),
            )
            cycle = run_verification_cycle(
                rep_task_by_edge[edge_id], candidates, evidence_reviewer=evidence_reviewer,
                store=store, log=log, validator=validator, timestamp=timestamp,
            )
            verdicts[edge_id] = str(cycle.selection.verdict)
            if cycle.gap:  # an overclaim is routed to revision as a risk, not committed
                revision_risks.append(
                    f"edge {edge_id} verification routed to revision "
                    f"(overclaim: v*={str(cycle.selection.verdict)})"
                )

    enable_expansion = gcd.EXPANSION_ENABLED if enable_expansion is None else enable_expansion
    expansion_active = enable_expansion and research_synthesist is not None
    _verify(candidates_by_edge)  # eagerly review every candidate-bearing committed edge
    report_progress(
        "Checking evidence",
        f"{len(verdicts)} links reviewed" if verdicts else "no shortlisted evidence",
        status="completed" if verdicts else "skipped",
    )

    # 4. priority — record the user's authored priorities (node-only, identity projection).
    # A pre-built annotation wins; otherwise authored concepts are matched to extracted node labels.
    report_progress("Applying priorities")
    annotation = priority_annotation
    if annotation is None and authored_concepts:
        annotation = build_priority_from_labels(
            {nid: node.label for nid, node in store.nodes.items()},
            authored_concepts, author=priority_author, focus_notes=priority_focus,
        )
    if annotation is not None:
        record_user_priority_annotations(store, log, validator, annotation, timestamp=timestamp)
    report_progress(
        "Applying priorities", status="completed" if annotation is not None else "skipped",
    )

    # 4a. Literature snippets cited by the Research Synthesist and Critic Panel. The Synthesist
    # performs its own enrich-commit inside ``run_synthesist_cycle``.
    passages = list(enrichment_passages) or _passages_from_work(work)

    # 4b. expand — optional and enabled by default. Expansion is the
    # propose→panel→revise→panel loop; it runs only when a Research Synthesist is
    # injected; otherwise expansion is a no-op with no surfaced candidates.
    surfaced: tuple[SurfacedHypothesis, ...] = ()
    experiment_audit_rounds: int | None = None
    if expansion_active:
        report_progress("Propose / critique", "starting synthesis")
        # The Research Synthesist and Critic Panel alternate before the deterministic commit tail.
        cited = "\n\n".join(passages)
        hyp_result = run_synthesist_cycle(
            research_synthesist(), critic_panel, store=store,
            passages=passages, corpus_list=cited, cited_passages=cited,
            enabled=True, precondition=expansion_precondition,
            confirmed_ids=expansion_confirmed_ids, confirm_fn=expansion_confirm_fn,
            reranker=expansion_reranker, embedder=embedder,
            log=log, validator=validator, extract_validator=extract_validator,
            timestamp=timestamp,
        )
        surfaced = _surfaced_view(hyp_result.surfaced)

        # 4e. Experiment Designer and Experiment Validator create a plan for each confirmed
        # hypothesis and commit its experiment delta. The per-hypothesis
        # methods passages' ids extend the commit ledger inside the stage so grounding resolves.
        if experiment_designer is not None and experiment_validator is not None:
            run_experiment_stage(
                hyp_result.confirmed, store=store, log=log, base_ledger=verify_ledger,
                designer_factory=experiment_designer, validator_seam=experiment_validator,
                retrieve_methods=retrieve_experiment_methods,
                refine_rounds=experiment_refine_rounds, timestamp=timestamp,
            )
            # Attach each committed plan so the Elaborator card can render it.
            surfaced = _attach_experiment_plans(surfaced, hyp_result.confirmed, store.experiment_plans)
            experiment_audit_rounds = experiment_refine_rounds
        else:
            report_progress(
                "Designing experiments", "designer or validator unavailable",
                status="skipped",
            )
    elif enable_expansion:  # enabled but no Research Synthesist wired: evidence-only no-op
        revision_risks.append(
            "expansion enabled but no Research Synthesist injected; running evidence core only"
        )
    if not expansion_active:
        report_progress(
            "Propose / critique", "expansion disabled or synthesist unavailable",
            status="skipped",
        )
        report_progress("Designing experiments", "no confirmed hypotheses", status="skipped")

    # 5. export — collect statuses, receipts, open risks; build the three artifacts.
    report_progress("Exporting graph", "assembling results")
    statuses = {edge_id: str(edge.status) for edge_id, edge in store.edges.items()}
    receipts = tuple(_receipt_of(row) for row in log.rows)
    late_open_risks = open_risks_provider() if open_risks_provider is not None else ()
    open_risks = tuple(
        sorted(
            {risk for edge in store.edges.values() for risk in edge.open_risks}
            | set(revision_risks)
            | set(late_open_risks)
        )
    )
    edge_table = tuple(
        EdgeStatusRow(
            edge_id=edge.edge_id, relation_type=edge.relation_type, status=str(edge.status),
            verdict=verdicts.get(edge.edge_id), confidence=edge.confidence,
            open_risks=tuple(edge.open_risks),
        )
        for edge in store.edges.values()
    )
    result = GraphRunResult(
        graph_id=store.graph_id,
        version=store.version,
        statuses=statuses,
        receipts=receipts,
        open_risks=open_risks,
        graph_json=store.model_dump(mode="json"),
        edge_table=edge_table,
        audit_memo=_audit_memo(
            store, verdicts, open_risks, receipts, experiment_audit_rounds,
            tuple(
                row.experiment_attempt
                for row in surfaced
                if row.experiment_attempt is not None
            ),
        ),
        surfaced=surfaced,
    )
    if export_dir is not None:
        _export_to_disk(result, Path(export_dir))
    report_progress(
        "Exporting graph",
        f"saved to {export_dir}" if export_dir is not None else "results ready",
        status="completed",
    )
    return result
