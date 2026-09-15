"""Production assembly for the graph-state run path.

Builds the real, credentialed dependencies the default graph-state CLI run needs — the LLM
agents (extractor and Evidence Reviewer), the production embedder, and a claim-level
retriever — bundled as :class:`GraphStateDeps` so they can be INJECTED for deterministic
tests (the default suite passes fakes; this real builder is exercised by the gated live e2e).

Retrieval performs one claim-level ``search_papers`` pass through the
``RetrievalService``/``RetrievalToolRegistry``/
``EvidenceLedger`` stack, then the same pool is offered to every target — the per-target
``AssociationScorer`` + triage (in ``run_path``) select what is relevant to each.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.progress import report_progress

if TYPE_CHECKING:
    from src.research_profile import ResearchProfile

from src import graph_config_defaults as gcd
from src.config import OrchestrationConfig
from src.cycles.experiment import ExperimentDesigner, ExperimentValidator
from src.cycles.extraction import (
    Embedder,
    Extractor,
    LLMExtractor,
    default_merge_embedder,
)
from src.cycles.panel import CriticPanelJudge, LLMCriticPanelJudge
from src.cycles.synthesist import ResearchSynthesist
from src.cycles.verification import (
    EvidenceReviewer,
    LLMEvidenceReviewer,
    VerificationTask,
)
from src.log_store import SQLiteLogStore
from src.state import RetrievedEvidence


@dataclass(frozen=True)
class GraphStateDeps:
    """The injectable dependency set for ``run_graph_state_workflow``."""

    extractor: Extractor
    embedder: Embedder | None
    evidence_reviewer: EvidenceReviewer
    retrieve_for_target: Callable[[VerificationTask], Sequence[RetrievedEvidence]]
    # Optional synthesis seams. The Research Synthesist is a per-run stateful thread, so it is
    # injected as a factory; the Critic Panel is a tuple of judges on different models.
    research_synthesist: Callable[[], ResearchSynthesist] | None = None
    critic_panel: tuple[CriticPanelJudge, ...] = ()
    # Post-confirmation experiment stage. The Experiment Designer is a per-hypothesis stateful
    # thread factory; the Experiment Validator is stateless and reusable. ``experiment_refine_rounds``
    # bounds their loop, and ``retrieve_experiment_methods`` is the cached per-hypothesis retriever.
    experiment_designer: Callable[[], ExperimentDesigner] | None = None
    experiment_validator: ExperimentValidator | None = None
    experiment_refine_rounds: int = 0
    retrieve_experiment_methods: Callable[[str], Sequence[RetrievedEvidence]] | None = None
    retrieval_open_risks: tuple[str, ...] = ()
    # Experiment-method retrieval happens after dependency construction. The provider exposes
    # those late diagnostics to the final run result without making the frozen dependency bundle
    # itself mutable.
    retrieval_open_risks_provider: Callable[[], Sequence[str]] | None = None


@dataclass(frozen=True)
class SynthesistBuildSpec:
    """Run-specific Research Synthesist steering derived from the researcher profile, not model
    config. The model-per-role settings live in ``config.agents``; this carries the prompt steering
    (miner focus, proposer judgeability constraint, judge reference-field + saturation corpus, panel
    size) the researcher's expertise/research-interest produce."""

    miner_focus: str = ""
    judgeability: str = ""
    judge_reference_field: str = ""
    judge_corpus: str = ""
    k_judges: int = 3
    venue_preference: str = ""  # soft hint: frame hypotheses for these venues (never excludes)
    proposal_count: int = 6  # Research Synthesist candidate-pool size.
    experiment_refine_rounds: int = 1  # Zero commits the design without validator refinement.


def _persist_retrieval_event_payloads(
    payloads: Sequence[dict[str, Any]],
    *,
    log_store: SQLiteLogStore,
    run_id: str,
    thread_id: str,
    checkpoint_id: str,
    node_name: str,
    round_index: int,
) -> list[dict[str, Any]]:
    """Persist every host-side retrieval call as a current graph-state audit event."""
    from src.events import EventType, build_event

    persisted = list(payloads)
    for payload in persisted:
        called_by = str(payload.get("called_by") or node_name)
        tool_call_id = str(payload.get("tool_call_id") or "unknown")
        log_store.append(
            build_event(
                run_id=run_id,
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                node_name=node_name,
                round_index=round_index,
                event_type=EventType.TOOL_EVENT,
                sender_role=called_by,
                receiver_role="orchestrator",
                content="search_papers retrieval attempt",
                previous_hash=None,
                structured_payload=payload,
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": "search_papers",
                        "arguments": {
                            "query": payload.get("query"),
                            "sources": payload.get("sources", []),
                            "filters": payload.get("filters", {}),
                        },
                    }
                ],
                idempotency_key=(
                    f"{run_id}:{checkpoint_id}:{node_name}:"
                    f"{round_index}:tool_event:{called_by}:{tool_call_id}"
                ),
            )
        )
    return persisted


def _persist_retrieval_tool_events(
    registry: Any,
    *,
    log_store: SQLiteLogStore,
    run_id: str,
    thread_id: str,
    node_name: str = "retrieve_evidence",
) -> list[dict[str, Any]]:
    return _persist_retrieval_event_payloads(
        registry.drain_pending_event_payloads(),
        log_store=log_store,
        run_id=run_id,
        thread_id=thread_id,
        checkpoint_id=str(registry.checkpoint_id),
        node_name=node_name,
        round_index=int(registry.round_index),
    )


def _retrieval_open_risks(payloads: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    """Project retrieval diagnostics into durable, user-visible run risks."""
    risks: list[str] = []
    for payload in payloads:
        statuses = [
            status
            for status in payload.get("source_statuses", [])
            if isinstance(status, dict)
        ]
        evidence = [
            item for item in payload.get("evidence", []) if isinstance(item, dict)
        ]
        if not statuses:
            risks.extend(
                str(item)
                for item in [
                    *payload.get("warnings", []),
                    *payload.get("errors", []),
                ]
                if str(item).strip()
            )
            continue

        failed = [status for status in statuses if status.get("status") == "failed"]
        partial = [
            status for status in statuses if status.get("status") == "partial_failure"
        ]
        skipped = [status for status in statuses if status.get("status") == "skipped"]
        if len(failed) == len(statuses):
            risks.append("retrieval failed for all selected sources")
        elif not evidence:
            risks.append("retrieval returned no evidence for selected sources")
        for label, affected in (
            ("failed", failed if len(failed) != len(statuses) else []),
            ("partial failure", partial),
            ("skipped", skipped),
        ):
            for status in affected:
                details = status.get("errors") or status.get("warnings") or []
                suffix = f": {'; '.join(str(item) for item in details)}" if details else ""
                risks.append(
                    f"retrieval {label} source {status.get('source', 'unknown')}{suffix}"
                )
        for item in evidence:
            metadata = item.get("metadata", {})
            selection = (
                metadata.get("quote_selection", {})
                if isinstance(metadata, dict)
                else {}
            )
            if not isinstance(selection, dict):
                continue
            candidate_status = selection.get("candidate_verification_status")
            if candidate_status in {"rejected", "unverified"}:
                risks.append(
                    "retrieval candidate excerpt "
                    f"{candidate_status} for {item.get('evidence_id', 'unknown')}"
                )
            verification_status = selection.get("verification_status")
            if verification_status in {"rejected", "unverified", "summary_fallback"}:
                risks.append(
                    "retrieval quote "
                    f"{verification_status} for {item.get('evidence_id', 'unknown')}"
                )
    return tuple(dict.fromkeys(risks))


def _record_retrieval_risks(
    retrieval_risks: list[str] | None,
    payloads: Sequence[dict[str, Any]],
) -> None:
    if retrieval_risks is None:
        return
    retrieval_risks[:] = list(
        dict.fromkeys([*retrieval_risks, *_retrieval_open_risks(payloads)])
    )


def _proposer_steering(judgeability: str, venue_preference: str) -> str | None:
    """Combine the proposer's judgeability constraint with the optional venue soft hint.

    The venue line is framed as a soft preference (never a hard constraint), so it biases
    how hypotheses are framed without excluding anything. Returns None when both are empty
    so no steering is injected (backward-compatible)."""
    parts = [judgeability.strip()] if judgeability.strip() else []
    if venue_preference.strip():
        parts.append(
            "Frame each hypothesis to suit these target venues (a soft preference, "
            f"not a hard constraint): {venue_preference.strip()}."
        )
    return "\n\n".join(parts) or None


def build_synthesist_seams(
    config: OrchestrationConfig,
    *,
    claim: str,
    anchor: str | None = None,
    embedder: Embedder,
    spec: SynthesistBuildSpec,
    backend_factory: Callable[..., Any],
) -> dict[str, Any]:
    """Build the Research Synthesist factory and Critic Panel LLM seams on their configured
    role backends (``config.agents``' per-role tier fallback),
    threading the profile ``spec`` into the prompts. Returns a kwargs dict for ``GraphStateDeps`` /
    ``run_graph_state_workflow``. ``backend_factory`` is injected so the wiring is testable without
    real credentials. ``anchor`` is the Synthesist scope-fit reference: the claim or, in claimless
    discovery, the lens; defaults to ``claim``."""

    def backend(role: str) -> Any:
        return backend_factory(config.agents.for_role(role), role_name=role)

    # Critic Panel size is capped at the three configured backends, each on its own model. The default uses
    # two base judges plus the disagreement tie-breaker invoked by ``_panel_round``.
    judge_roles = [("builder", 1), ("critic_panel", 2), ("skeptical_verifier", 3)]
    judge_roles = judge_roles[: min(spec.k_judges, len(judge_roles))]
    if len(judge_roles) < 2:
        raise ValueError(
            f"Critic Panel needs at least two distinct judge models; k_judges={spec.k_judges} "
            f"would build only {len(judge_roles)}. Set the profile's `k_judges` (or "
            "SynthesistBuildSpec.k_judges) to 2 or more."
        )

    return {
        # Research Synthesist factory: one fresh stateful thread per run.
        # miner_focus / judgeability+venue steering are threaded from the profile (inherited hooks).
        "research_synthesist": lambda: ResearchSynthesist(
            backend("research_synthesist"),
            embedder=embedder,
            claim=(anchor if anchor is not None else claim),
            proposal_count=spec.proposal_count,
            miner_focus=spec.miner_focus,
            propose_steering=_proposer_steering(spec.judgeability, spec.venue_preference),
        ),
        "critic_panel": tuple(
            LLMCriticPanelJudge(
                backend(role), judge_id=jid, reference_field=spec.judge_reference_field
            )
            for role, jid in judge_roles
        ),
        # Experiment Designer factory: design and refine share a fresh per-hypothesis thread.
        "experiment_designer": lambda: ExperimentDesigner(backend("experiment_designer")),
        # Experiment Validator: eager, stateless, and reused across the confirmed set.
        "experiment_validator": ExperimentValidator(backend("experiment_validator")),
    }


def retrieve_claim_evidence(
    config: OrchestrationConfig,
    *,
    log_store: SQLiteLogStore,
    run_id: str,
    thread_id: str,
    claim: str,
    embedder: Embedder | None = None,
    retrieval_risks: list[str] | None = None,
) -> list[RetrievedEvidence]:
    """One claim-level retrieval pass (mirrors the ``retrieve_evidence`` node).

    When ``embedder`` is supplied, the pool is re-ranked relevance-first (specter2
    cosine of the claim against each paper's title+quote) and dead ``no_source_text``
    stubs are dropped, so on-topic full-text papers are not demoted below abstract-less
    metadata. With no embedder the ledger's source ordering is retained."""
    if not config.retrieval.enabled:
        report_progress("Retrieving literature", "retrieval disabled", status="skipped")
        return []
    report_progress("Retrieving literature", "searching configured sources")
    from src.retrieval.models import SearchPaperFilters
    from src.retrieval.tool_registry import build_retrieval_stack

    _service, ledger, registry = build_retrieval_stack(
        config.retrieval, log_store,
        run_id=run_id, thread_id=thread_id, checkpoint_id="retrieve_evidence",
    )
    registry.search_papers(
        query=claim, sources=None, limit=config.retrieval.final_top_k,
        filters=SearchPaperFilters(), called_by="retrieve_evidence",
    )
    payloads = _persist_retrieval_tool_events(
        registry,
        log_store=log_store,
        run_id=run_id,
        thread_id=thread_id,
    )
    _record_retrieval_risks(retrieval_risks, payloads)
    if embedder is not None:
        report_progress("Retrieving literature", "ranking evidence")
        ledger.drop_empty_quote_stubs()
        ledger.score_relevance(embedder, claim)
    evidence = ledger.all_evidence()
    report_progress(
        "Retrieving literature", f"{len(evidence)} evidence records", status="completed",
    )
    return evidence


def retrieve_claim_evidence_planned(
    config: OrchestrationConfig,
    profile: ResearchProfile,
    *,
    log_store: SQLiteLogStore,
    run_id: str,
    thread_id: str,
    planner: Any,
    embedder: Embedder | None = None,
    retrieval_risks: list[str] | None = None,
) -> list[RetrievedEvidence]:
    """Planned multi-query retrieval. The planner derives sub-queries from the profile
    (each with its own sources + recency/venue filters); we fan them out into ONE shared
    ledger (identity dedup merges duplicates across queries, preserving evidence_ids),
    fuse the per-query ranks via RRF, score claim-relevance, drop empty stubs, and return
    the relevance-first top-K pool. Degrades gracefully: a planner that yields its T1
    fallback still runs."""
    if not config.retrieval.enabled:
        report_progress("Retrieving literature", "retrieval disabled", status="skipped")
        return []
    report_progress("Retrieving literature", "planning searches")
    from src.retrieval.planner import rrf_scores
    from src.retrieval.tool_registry import build_retrieval_stack
    from src.runtime_trace import record_runtime_event

    _service, ledger, registry = build_retrieval_stack(
        config.retrieval, log_store,
        run_id=run_id, thread_id=thread_id, checkpoint_id="retrieve_evidence",
    )

    plans = planner.plan(profile)
    record_runtime_event(
        "retrieval_plan",
        run_id=run_id, thread_id=thread_id, actor="retrieval_planner",
        target="search_papers", direction="request",
        payload={
            "plans": [
                {
                    "query": plan.query,
                    "sources": plan.sources,
                    "filters": plan.filters.model_dump(exclude_none=True),
                }
                for plan in plans
            ]
        },
    )

    rankings: list[list[str]] = []
    for index, plan in enumerate(plans, 1):
        report_progress(
            "Retrieving literature", "search query", current=index, total=len(plans),
        )
        payload = registry.search_papers(
            query=plan.query, sources=plan.sources,
            limit=config.retrieval.final_top_k, filters=plan.filters,
            called_by="retrieve_evidence",
        )
        rankings.append(
            [ev["evidence_id"] for ev in payload.get("evidence", []) if ev.get("evidence_id")]
        )
    payloads = _persist_retrieval_tool_events(
        registry,
        log_store=log_store,
        run_id=run_id,
        thread_id=thread_id,
    )
    _record_retrieval_risks(retrieval_risks, payloads)

    scores = rrf_scores(rankings)
    for item in ledger.all_evidence():
        if item.evidence_id in scores:
            item.metadata["rrf_score"] = scores[item.evidence_id]

    report_progress("Retrieving literature", "ranking evidence")
    ledger.drop_empty_quote_stubs()
    if embedder is not None:
        # Claimless discovery has no claim to score against, so use the profile lens. In claim mode,
        # ``anchor()`` returns the claim.
        ledger.score_relevance(embedder, profile.anchor())
    evidence = ledger.all_evidence()[: config.retrieval.final_top_k]
    report_progress(
        "Retrieving literature", f"{len(evidence)} evidence records", status="completed",
    )
    return evidence


def make_experiment_methods_retriever(
    config: OrchestrationConfig,
    log_store: SQLiteLogStore,
    run_id: str,
    thread_id: str,
    *,
    embedder: Embedder | None = None,
    retrieval_risks: list[str] | None = None,
) -> Callable[[str], list[RetrievedEvidence]]:
    """The Experiment Designer's targeted methods/datasets/baselines retriever: a
    ``retrieve_methods(query)`` closure over the SAME ``RetrievalService`` stack the claim-level
    retrieval uses, so per-hypothesis re-queries hit the run's ``retrieval_cache`` and replay
    deterministically (near-free on a cache hit; no unaudited retrieval path). Results feed the
    Experiment Designer
    prompt only — NEVER committed to the evidence ledger — and are attributed ``called_by=
    "experiment_design"`` in the cache/audit trail. Retrieval disabled -> always []."""
    from src.retrieval.models import SearchPaperFilters
    from src.retrieval.service import RetrievalService
    from src.retrieval.tool_registry import build_retrieval_event_payload

    if not config.retrieval.enabled:
        return lambda _query: []
    service = RetrievalService(
        config=config.retrieval,
        log_store=log_store,
        rerank_embedder=embedder,
    )
    counter = [0]

    def retrieve_methods(query: str) -> list[RetrievedEvidence]:
        counter[0] += 1
        result = service.search_papers(
            run_id=run_id,
            query=query,
            sources=None,
            limit=gcd.EXPERIMENT_METHODS_QUOTES,  # small per-hypothesis methods pool (audit-surfaced)
            filters=SearchPaperFilters(),
            called_by="experiment_design",
            tool_call_id=f"{thread_id}-exp-methods-{counter[0]:03d}",
        )
        event_payload = build_retrieval_event_payload(
            result=service.result_to_dict(result),
            called_by="experiment_design",
        )
        payloads = _persist_retrieval_event_payloads(
            [event_payload],
            log_store=log_store,
            run_id=run_id,
            thread_id=thread_id,
            checkpoint_id="experiment_methods",
            node_name="experiment_design",
            round_index=counter[0] - 1,
        )
        _record_retrieval_risks(retrieval_risks, payloads)
        return [RetrievedEvidence.model_validate(item) for item in result.evidence]

    return retrieve_methods


def _llm_call_logger(
    log_store: SQLiteLogStore, run_id: str, thread_id: str
) -> Callable[[str, Any, Any], None]:
    """A backend-hook sink that appends each LLM seam call — role, raw request messages, raw
    response text — to the audit log as one ``AGENT_MESSAGE`` event (the intermediate artifact the
    subagent setting surfaces). A per-run counter keeps the idempotency key unique across same-role
    calls (e.g. a multi-judge panel), so ``append`` never raises a conflict."""
    from src.camel_adapter import (
        _first_response_message,
        _message_content,
        _response_info,
        _response_token_usage,
    )
    from src.events import EventType, build_event

    counter = [0]

    def sink(role: str, messages: Any, response: Any) -> None:
        seq = counter[0]
        counter[0] += 1
        content = _message_content(_first_response_message(response))
        info = _response_info(response)
        provider = info.get("provider")
        model = info.get("model")
        latency_ms = info.get("latency_ms")
        termination_reason = info.get("termination_reason")
        token_usage = _response_token_usage(response)
        response_metadata: dict[str, object] = {}
        for key in ("provider", "model", "role", "termination_reason"):
            if isinstance(info.get(key), str):
                response_metadata[key] = info[key]
        if isinstance(latency_ms, int) and not isinstance(latency_ms, bool):
            response_metadata["latency_ms"] = latency_ms
        if token_usage:
            response_metadata["usage"] = token_usage
        if isinstance(info.get("codex_thread_id"), str):
            response_metadata["codex_thread_id"] = info["codex_thread_id"]
        log_store.append(
            build_event(
                run_id=run_id,
                thread_id=thread_id,
                checkpoint_id="llm_call",
                node_name=role,
                round_index=seq,
                event_type=EventType.AGENT_MESSAGE,
                sender_role=role,
                receiver_role="orchestrator",
                content=content if isinstance(content, str) else str(content),
                previous_hash=None,
                structured_payload={
                    "role": role,
                    "seq": seq,
                    "request_messages": messages,
                    "raw_response": content,
                    "response_metadata": response_metadata,
                },
                provider=provider if isinstance(provider, str) else None,
                model=model if isinstance(model, str) else None,
                token_usage=token_usage,
                latency_ms=(
                    latency_ms
                    if isinstance(latency_ms, int) and not isinstance(latency_ms, bool)
                    else None
                ),
                termination_reason=(
                    termination_reason
                    if isinstance(termination_reason, str)
                    else None
                ),
            )
        )

    return sink


def _logging_make_backend(
    base_factory: Callable[..., Any],
    log_store: SQLiteLogStore,
    run_id: str,
    thread_id: str,
) -> Callable[..., Any]:
    """Wrap a backend factory so every backend it builds logs its raw LLM I/O to ``log_store`` —
    one chokepoint that instruments the extractor, Evidence Reviewer, and synthesis seams."""
    from src.camel_adapter import _install_logging_backend_hook

    sink = _llm_call_logger(log_store, run_id, thread_id)

    def make_backend(agent_config: Any, *, role_name: str) -> Any:
        backend = base_factory(agent_config, role_name=role_name)
        model = getattr(agent_config, "model", None)
        return _install_logging_backend_hook(
            backend, sink, role=role_name, provider=getattr(model, "provider", None)
        )

    return make_backend


def logged_backend(
    config: OrchestrationConfig,
    role: str,
    log_store: SQLiteLogStore,
    run_id: str,
    thread_id: str,
    *,
    agent_role: str | None = None,
) -> Any:
    """Build the configured direct backend for ``role`` and audit-log the call.

    This covers seams built outside :func:`build_graph_state_deps` (the elaboration
    writer, reader-translation pair, and retrieval planner). ``agent_role`` can
    select the model tier independently from the audit/backend role name. Each call
    gets its own logger/counter. The direct resolver delegates ordinary providers
    to CAMEL and the local CLI providers (``claude-cli``, ``codex-cli``) to a
    constrained process-per-call backend.
    """
    from src.camel_adapter import _create_direct_model_backend

    make_backend = _logging_make_backend(
        _create_direct_model_backend, log_store, run_id, thread_id
    )
    return make_backend(
        config.agents.for_role(agent_role or role),
        role_name=role,
    )


def build_graph_state_deps(
    config: OrchestrationConfig,
    *,
    claim: str,
    anchor: str | None = None,
    log_store: SQLiteLogStore,
    run_id: str,
    thread_id: str,
    synthesist: SynthesistBuildSpec | None = None,
    backend_factory: Callable[..., Any] | None = None,
    evidence: Sequence[RetrievedEvidence] | None = None,
    retrieval_open_risks: Sequence[str] = (),
    log_llm_calls: bool = False,
) -> GraphStateDeps:
    """Assemble the real LLM agents + embedder + claim-level retriever (live path).

    The Atom/Relation Extractor runs on the ``builder`` backend; the Evidence Reviewer runs
    on the ``evidence_reviewer`` role backend (skeptical tier — it folds in the former coherence
    judge, methods appraiser, and causal verifier). When ``synthesist`` is provided, the Research
    Synthesist factory and Critic Panel are additionally built
    (each on its own configured role backend) and injected, so the standalone expand path runs live.
    ``backend_factory`` is injected only by tests; production uses the configured
    direct-backend resolver (CAMEL/OpenRouter providers, ``claude-cli``, or ``codex-cli``).
    """
    from src.camel_adapter import _create_direct_model_backend

    make_backend = backend_factory or _create_direct_model_backend
    if log_llm_calls:
        # One chokepoint logs request and response data for every LLM-backed role.
        make_backend = _logging_make_backend(make_backend, log_store, run_id, thread_id)
    extract_backend = make_backend(config.agents.builder, role_name="builder")
    reviewer_backend = make_backend(
        config.agents.for_role("evidence_reviewer"), role_name="evidence_reviewer"
    )
    embedder = default_merge_embedder()
    # ``evidence`` lets the caller pass the claim-level pool it already retrieved (e.g. the standalone
    # driver, which needs it to curate the judge corpus) so the search runs once, not twice.
    collected_retrieval_risks = list(retrieval_open_risks)
    pool = list(evidence) if evidence is not None else retrieve_claim_evidence(
        config,
        log_store=log_store,
        run_id=run_id,
        thread_id=thread_id,
        claim=claim,
        retrieval_risks=collected_retrieval_risks,
    )
    deps_kwargs: dict[str, Any] = {
        "extractor": LLMExtractor(extract_backend),
        "embedder": embedder,
        "evidence_reviewer": LLMEvidenceReviewer(reviewer_backend),
        "retrieve_for_target": lambda _task: pool,
        "retrieval_open_risks": tuple(dict.fromkeys(collected_retrieval_risks)),
        "retrieval_open_risks_provider": lambda: tuple(
            dict.fromkeys(collected_retrieval_risks)
        ),
    }
    if synthesist is not None:
        deps_kwargs.update(
            build_synthesist_seams(
                config, claim=claim, anchor=anchor, embedder=embedder, spec=synthesist,
                backend_factory=make_backend,
            )
        )
        # Configure the bounded experiment-review loop and the cached, ledgered per-hypothesis
        # methods retriever here, where the run context is available.
        deps_kwargs["experiment_refine_rounds"] = synthesist.experiment_refine_rounds
        deps_kwargs["retrieve_experiment_methods"] = make_experiment_methods_retriever(
            config,
            log_store,
            run_id,
            thread_id,
            embedder=embedder,
            retrieval_risks=collected_retrieval_risks,
        )
    return GraphStateDeps(**deps_kwargs)
