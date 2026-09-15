"""Standalone Research Synthesist run driver — ``python -m src.synthesist_run``.

Ties the two input files together (the researcher ``profile.yaml`` + the system
``config/evidence-evaluation.yaml``) and drives the shipped graph-state pipeline with live LLM
backends: retrieve -> curate the judge corpus -> build the Research Synthesist and Critic Panel
seams on their configured role backends -> run extraction, evidence review, priority annotation,
and synthesis in one call -> apply the confirmation decision from the surfaced ranking.
Runs the seams on whatever backend each role is configured with -- coding-agent CLI
subagents (``claude-cli``, ``codex-cli``) or OpenRouter -- with no record/replay harness.

The confirmation hooks (the part that decides which surfaced hypotheses commit) are pure and
hermetically tested here; the live assembly + ``main`` are exercised by the gated live e2e.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from src import graph_config_defaults as gcd
from src.camel_adapter import _preflight_model_credentials
from src.config import DEFAULT_CONFIG_PATH, load_config
from src.connected_render import render_connected_after_trace
from src.corpus_curation import corpus_paper_list
from src.cycles.extraction import default_merge_embedder
from src.emphasis import emphasis_reranker
from src.graph_state_runtime import (
    build_graph_state_deps,
    retrieve_claim_evidence_planned,
)
from src.log_store import SQLiteLogStore
from src.progress import (
    ProgressReporter,
    add_progress_arguments,
    report_progress,
)
from src.research_profile import (
    ConfirmPolicy,
    load_research_profile,
    resolve_confirmed_ids,
)
from src.retrieval.planner import LLMRetrievalPlanner
from src.retrieval.preflight import preflight_required_retrieval
from src.run_path import (
    GraphRunResult,
    passages_from_quotes,
    run_graph_state_workflow,
)
from src.state import RetrievedEvidence
from src.trace_render import (
    LLMHypothesisElaborator,
    elaborate_all,
    write_trace,
)
from src.wiki_links import DEFAULT_WIKI_CACHE

ConfirmFn = Callable[[Sequence[Any]], Collection[str]]


# --- corpus pool adaptation --------------------------------------------------------------
def node_definitions(graph_json: dict[str, Any]) -> dict[str, str]:
    """Committed-node label -> universal definition (the miner's), so the elaborator can expand any
    paper-coined term inline from its definition rather than guessing (direction B)."""
    return {
        str(node.get("label", "")): str(node.get("definition", ""))
        for node in (graph_json.get("nodes") or {}).values()
        if isinstance(node, dict) and node.get("label")
    }


def evidence_to_records(evidence: Sequence[RetrievedEvidence]) -> list[dict[str, str]]:
    """Adapt the retrieved pool into the dicts ``corpus_curation`` consumes (the retrieved snippet
    is the body; the ledger holds full text)."""
    return [
        {
            "evidence_id": item.evidence_id,
            "paper_id": item.source_id or "",
            "title": item.title,
            "body": item.quote,
        }
        for item in evidence
    ]


def sources_from_evidence(evidence: Sequence[RetrievedEvidence]) -> list[dict[str, str]]:
    """The retrieved references for the trace ``Sources`` list: deduped ``{title, url, source}`` in
    retrieval (trust/rank) order, so the trace report links every paper the run drew on."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in evidence:
        title = (item.title or "").strip()
        if not title:
            continue
        url = (item.url or "").strip()
        key = url or title
        if key in seen:
            continue
        seen.add(key)
        rows.append({"title": title, "url": url, "source": item.source})
    return rows


def passages_from_evidence(evidence: Sequence[RetrievedEvidence]) -> list[str]:
    """The retrieved quotes as the miner's passages — deduped, empties dropped, capped at
    ``ENRICHMENT_MAX_PASSAGES``, retrieval order preserved. Used for claimless discovery,
    where no verification work is scheduled to derive passages from (the claim-mode analogue is
    ``run_path._passages_from_work``).

    Unlike the claim-mode path, this does not truncate each quote to
    ``PASSAGE_PROMPT_MAX_CHARS``. A source can therefore contribute a full-length quote to the
    claimless miner prompt."""
    quotes = (item.quote or "" for item in evidence)
    return passages_from_quotes(quotes, max_chars=None, max_passages=gcd.ENRICHMENT_MAX_PASSAGES)


# --- confirmation hooks (decide commits from the surfaced ranking) -----------------------
def _scored_to_row(scored: Any) -> dict[str, Any]:
    candidate = scored.candidate
    return {
        "candidate_id": candidate.candidate_id,
        "field_novelty": candidate.ranking_novelty,
        "saturation": candidate.saturation,
        "rank_score": scored.rank_score,
        "cross_concept": candidate.cross_concept,
        "common_sense": candidate.common_sense,
    }


def _format_surfaced(surfaced: Sequence[Any]) -> str:
    lines = [f"surfaced hypotheses ({len(surfaced)}):"]
    for rank, scored in enumerate(surfaced, start=1):
        candidate = scored.candidate
        labels = " + ".join(node.label for node in candidate.new_nodes) or candidate.candidate_id
        tag = "cross-concept" if candidate.cross_concept else "within-concept"
        lines.append(
            f"  #{rank} {candidate.candidate_id}  f_nov {candidate.ranking_novelty:.2f}  "
            f"sat {candidate.saturation:.2f}  RankScore {scored.rank_score:.3f}  [{tag}]"
        )
        lines.append(f"       {labels}")
        if candidate.rationale:
            lines.append(f"       {candidate.rationale.splitlines()[0][:120]}")
    return "\n".join(lines)


def interactive_confirm_fn(
    *, input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print
) -> ConfirmFn:
    """Print the surfaced ranking and read the owner's confirmation selection. Accepts
    ``all``, ``none``/empty, or a comma-separated id list. stdin/stdout are injected for testing."""

    def confirm(surfaced: Sequence[Any]) -> list[str]:
        if not surfaced:
            return []
        report_progress("Confirming hypotheses", "waiting for user selection", status="waiting")
        output_fn(_format_surfaced(surfaced))
        raw = input_fn("confirm ids to commit [e.g. h5,h6 | all | none]: ").strip()
        report_progress("Confirming hypotheses", "applying user selection")
        lowered = raw.lower()
        if lowered in ("", "none"):
            return []
        if lowered == "all":
            return [scored.candidate.candidate_id for scored in surfaced]
        wanted = {token.strip() for token in raw.split(",") if token.strip()}
        return [
            scored.candidate.candidate_id
            for scored in surfaced
            if scored.candidate.candidate_id in wanted
        ]

    return confirm


def policy_confirm_fn(policy: ConfirmPolicy) -> ConfirmFn:
    """A non-interactive confirm hook: apply an auto ConfirmPolicy to the surfaced ranking."""

    def confirm(surfaced: Sequence[Any]) -> list[str]:
        resolved = resolve_confirmed_ids([_scored_to_row(scored) for scored in surfaced], policy)
        return resolved if resolved is not None else []

    return confirm


def confirm_fn_for(
    policy: ConfirmPolicy,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> ConfirmFn:
    """Pick the confirm hook for a policy: interactive prompt vs auto-policy."""
    if policy.mode == "interactive":
        return interactive_confirm_fn(input_fn=input_fn, output_fn=output_fn)
    return policy_confirm_fn(policy)


def _preflight_authored_priority(profile: Any) -> None:
    """Reject weighted ``concepts`` with no ``priority_author``, before retrieval.

    The priority stage builds a ``UserPriorityAnnotation`` from the profile's
    concepts as soon as one term matches a committed node label, and that model
    requires a non-empty ``author`` -- so the combination aborts the run. Left to
    the builder the abort lands at the priority stage, on the far side of planned
    retrieval and evidence review: a quarter of an hour of live model and network
    work for a profile that was unrunnable when it was read.
    """
    if profile.concepts and not profile.priority_author.strip():
        terms = ", ".join(repr(concept.term) for concept in profile.concepts[:3])
        raise ValueError(
            f"profile sets weighted `concepts` ({terms}) but no `priority_author`. "
            "The priority annotation those weights build records who authored the "
            "preference and its author may not be empty, so the run would abort at "
            "the priority stage, after retrieval and evidence review. Set "
            "`priority_author` to the researcher or team, or remove `concepts` to "
            "run without authored priority."
        )


# --- live run assembly -------------------------------------------------------------------
def run_synthesist(
    *,
    profile_path: str | Path,
    config_path: str | Path,
    log_store: SQLiteLogStore,
    run_id: str,
    thread_id: str,
    export_dir: Path | None = None,
    trace_dir: Path | None = None,
    elaborate: bool = True,
    wiki_links: bool = False,
    wiki_cache: Path = DEFAULT_WIKI_CACHE,
    connected_render: bool = True,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> GraphRunResult:
    """Drive the full standalone Research Synthesist run (live LLM). Retrieves once, curates the
    judge corpus, builds the seams on their configured role backends, and runs the one-call
    workflow with the
    confirm hook resolved from the profile's confirm policy, and (when ``trace_dir`` is set) writes
    the deterministic trace report plus per-hypothesis prose by default. Once the trace report is
    written, the default subpipeline renders the per-hypothesis connected pages unless
    ``connected_render`` is disabled."""
    report_progress("Preparing run", "loading configuration and checking credentials")
    log_store.setup()  # create audit_events + retrieval_cache tables (mirrors cli.py); without
    # it the first retrieval cache-read raises ``no such table`` and retrieval silently returns 0.
    config = load_config(config_path)
    preflight_required_retrieval(config)
    profile = load_research_profile(profile_path)
    _preflight_authored_priority(profile)
    model_roles = [
        "builder",
        "evidence_reviewer",
        "research_synthesist",
        "critic_panel",
        "experiment_designer",
        "experiment_validator",
    ]
    if profile.k_judges >= 3:
        model_roles.append("skeptical_verifier")
    if trace_dir is not None and elaborate:
        model_roles.append("elaboration_writer")
    if trace_dir is not None and elaborate and profile.reader_lexicon is not None:
        model_roles.extend(("reader_translator", "translation_verifier"))
    _preflight_model_credentials(config, model_roles)

    anchor = profile.anchor()  # Claim or claimless lens used for scope, mining, and proposals.
    # Plan -> fan out -> relevance-first pool. The planner is a live LLM seam on the
    # configured direct backend; the embedder is shared (lru-cached) with the seams. The planner
    # derives its sub-queries from the profile, so claimless profiles plan from the lens. Ordinary
    # providers resolve through CAMEL; ``provider: codex-cli`` resolves to a fresh Codex process.
    # The shared logged-backend chokepoint records the planner's generated queries too.
    from src.graph_state_runtime import logged_backend

    report_progress("Preparing run", "loading evidence embedder")
    embedder = default_merge_embedder()
    planner = LLMRetrievalPlanner(
        logged_backend(
            config,
            "retrieval_planner",
            log_store,
            run_id,
            thread_id,
            agent_role="research_synthesist",
        )
    )
    retrieval_open_risks: list[str] = []
    evidence = retrieve_claim_evidence_planned(
        config, profile, log_store=log_store, run_id=run_id, thread_id=thread_id,
        planner=planner, embedder=embedder, retrieval_risks=retrieval_open_risks,
    )
    judge_corpus = corpus_paper_list(evidence_to_records(evidence))
    spec = profile.synthesist_spec(judge_corpus=judge_corpus)
    deps = build_graph_state_deps(
        config, claim=profile.claim or anchor, anchor=anchor, log_store=log_store, run_id=run_id,
        thread_id=thread_id, synthesist=spec, evidence=evidence, log_llm_calls=True,
        retrieval_open_risks=retrieval_open_risks,
    )
    confirm = confirm_fn_for(profile.confirm, input_fn=input_fn, output_fn=output_fn)
    # Tier B emphasis re-rank: bind the authored priority + policy into the cycle's reranker hook
    # (identity when no priority is authored, so claim-mode / no-priority runs are unaffected).
    authored_concepts = [(concept.term, concept.weight) for concept in profile.concepts]
    reranker = emphasis_reranker(authored_concepts, profile.emphasis_policy)
    # Claimless runs schedule no verification work, so feed the miner the retrieved pool directly.
    # Claim-based runs derive passages from evidence scoring inside ``run_path``.
    enrichment_passages = () if profile.claim else passages_from_evidence(evidence)
    result = run_graph_state_workflow(
        profile.claim,
        anchor=anchor,
        extractor=deps.extractor,
        embedder=deps.embedder,
        retrieve_for_target=deps.retrieve_for_target,
        evidence_reviewer=deps.evidence_reviewer,
        authored_concepts=authored_concepts,
        priority_author=profile.priority_author,
        priority_focus=profile.priority_focus,
        enrichment_passages=enrichment_passages,
        enable_expansion=True,
        # The Research Synthesist factory and different-model Critic Panel run the
        # propose→panel→revise→panel loop.
        research_synthesist=deps.research_synthesist,
        critic_panel=deps.critic_panel,
        expansion_confirm_fn=confirm,
        expansion_reranker=reranker,
        # The Experiment Designer and Experiment Validator create and review a grounded experiment
        # for each confirmed hypothesis before committing its experiment delta.
        experiment_designer=deps.experiment_designer,
        experiment_validator=deps.experiment_validator,
        experiment_refine_rounds=deps.experiment_refine_rounds,
        retrieve_experiment_methods=deps.retrieve_experiment_methods,
        initial_open_risks=deps.retrieval_open_risks,
        open_risks_provider=deps.retrieval_open_risks_provider,
        export_dir=export_dir,
    )

    if trace_dir is not None:
        report_progress("Writing reports", "preparing hypothesis pages")
        elaborations = None
        reader_context = None
        if elaborate and result.surfaced:
            report_progress("Writing reports", "elaborating hypotheses")
            from src.graph_state_runtime import logged_backend

            # The writer is built outside build_graph_state_deps, so route it through the SAME
            # backend-logging hook here too — otherwise its LLM completion would escape the audit log.
            writer = logged_backend(config, "elaboration_writer", log_store, run_id, thread_id)
            elaborations = elaborate_all(
                LLMHypothesisElaborator(writer), result.surfaced,
                concept_definitions=node_definitions(result.graph_json),
                reader_profile=profile,
            )
        # Reader translation runs in the driver when the profile carries a reader lexicon, keeping
        # renderers offline. Both seams are built outside
        # build_graph_state_deps, so they route through the same backend-logging hook.
        if elaborations and profile.reader_lexicon is not None:
            report_progress("Writing reports", "translating and verifying reader cards")
            from src.graph_state_runtime import logged_backend
            from src.reader_translation import (
                LLMReaderTranslator,
                source_quote_index,
                translate_all,
            )

            translator = logged_backend(config, "reader_translator", log_store, run_id, thread_id)
            verifier = logged_backend(config, "translation_verifier", log_store, run_id, thread_id)
            elaborations = translate_all(
                LLMReaderTranslator(translator, verifier_backend=verifier),
                elaborations,
                lexicon=profile.reader_lexicon,
                concept_definitions=node_definitions(result.graph_json),
                source_quotes=source_quote_index(evidence),
            )
            reader_context = {
                "home_field": profile.reader_lexicon.home_field,
                "lexicon_source": profile.reader_lexicon.source,
            }
        # Verified deeper-dive links are opt-in, cache-first, and safe for offline rendering.
        if elaborations and wiki_links:
            report_progress("Writing reports", "checking reference links")
            from src.wiki_links import attach_key_term_links

            elaborations = attach_key_term_links(elaborations, cache_path=wiki_cache)
        report_progress("Writing reports", "rendering trace pages")
        write_trace(
            result, trace_dir, claim=profile.claim, elaborations=elaborations,
            sources=sources_from_evidence(evidence), reader_context=reader_context,
            seed_kind=profile.seed_kind,
        )
        # Render per-hypothesis connected pages by default once the trace report is on disk.
        # ``export_dir`` is the run root; if ``trace_dir`` is not literally
        # ``export_dir / "trace"`` the layout check inside the driver finds it incomplete and
        # warns-and-skips rather than guessing across unrelated directories.
        if export_dir is not None:
            report_progress("Writing reports", "rendering connected hypothesis pages")
            render_connected_after_trace(
                export_dir,
                enabled=connected_render,
                run_id=run_id,
                events_db=log_store.path,
            )
        report_progress("Writing reports", f"saved to {trace_dir}", status="completed")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the standalone Research Synthesist hypothesis pipeline."
    )
    parser.add_argument("--profile", required=True, type=Path, help="researcher profile.yaml")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="system config (per-role LLM models, retrieval)",
    )
    parser.add_argument("--events-db", type=Path, default=Path("synthesist-run-events.sqlite"))
    parser.add_argument("--export-dir", type=Path, default=None, help="graph.json + edge table")
    parser.add_argument("--trace-dir", type=Path, default=None, help="deterministic HTML trace report")
    parser.add_argument(
        "--elaborate", action=argparse.BooleanOptionalAction, default=True,
        help="write plain-language prose when --trace-dir is set "
             "(default: enabled; one LLM completion per hypothesis)",
    )
    parser.add_argument(
        "--wiki-links", action="store_true",
        help="resolve verified deeper-dive links for reader-card key terms "
             "(wikipedia -> wikidata -> scholar-search; cache-first)",
    )
    parser.add_argument(
        "--wiki-cache", type=Path, default=DEFAULT_WIKI_CACHE,
        help="JSON cache for resolved key-term links",
    )
    parser.add_argument(
        "--no-connected-render", dest="connected_render", action="store_false",
        help="skip the default post-trace connected-hypothesis HTML pages "
             "(default: render when graph.json + trace/ are present)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="logical run id (default: a fresh UUID, avoiding stale cache replay)",
    )
    add_progress_arguments(parser)
    args = parser.parse_args(argv)

    run_id = args.run_id or str(uuid4())
    log_store = SQLiteLogStore(args.events_db)
    try:
        with ProgressReporter(run_id, quiet=args.quiet, jsonl_path=args.progress_jsonl):
            result = run_synthesist(
                profile_path=args.profile, config_path=args.config, log_store=log_store,
                run_id=run_id, thread_id=run_id, export_dir=args.export_dir,
                trace_dir=args.trace_dir, elaborate=args.elaborate,
                wiki_links=args.wiki_links, wiki_cache=args.wiki_cache,
                connected_render=args.connected_render,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    committed = sum(1 for edge in result.edge_table if edge.status)
    print(
        f"done: graph v{result.version}, {len(result.surfaced)} hypotheses surfaced, "
        f"{committed} edges in graph"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
