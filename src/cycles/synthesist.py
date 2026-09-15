"""A persistent Research Synthesist thread for mining, proposing, and revision.

The first turn mines outcome-anchored concepts and adjudicates borderline merge pairs. The
second proposes grounded candidate hypotheses over the committed graph. The third repairs
panel-flagged survivors and derives new candidates, including at least one divergent candidate.
Retrieved passages and prior responses remain in context throughout the three turns.

The Critic Panel, rather than this builder, assigns decisive field-relative novelty and
saturation grades. The builder's permissive ``llm_signals`` feed only the deterministic
eligibility gate. Parsing reuses ``camel_adapter`` so the deterministic core remains LLM-free.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from src.cycles._prompt import passage_block
from src.cycles.enrichment import MinedConcept, _parse_mined_concepts
from src.graph_store import ConceptNode

if TYPE_CHECKING:
    from src.cycles.hypothesis import HypothesisCandidate

logger = logging.getLogger(__name__)

# The revision turn accepts two to four derived candidates. Excess candidates are dropped; a
# shortfall is logged without fabricating candidates or retrying the model.
_REVISE_DERIVED_MIN = 2
_REVISE_DERIVED_MAX = 4

# The thread's only system prompt; later turns supply their instructions as user messages.
SYNTHESIST_SYSTEM_PROMPT = """<role>
You are the Research Synthesist for a causal-claim graph. This conversation is a persistent thread: after this first mine turn you stay in the same thread to propose candidate hypotheses and then revise them after an independent panel's critique, with the retrieved passages remaining in your context throughout. Read the passages as the material you will build hypotheses from, not just text to index — the concepts you mine now become the [mined] nodes your later candidates will link. Each later turn's task and output shape arrive in its user message; every turn returns STRICT JSON only.
</role>

<task>
This turn has two sub-tasks:
1. Mine — from the retrieved full-text passages, extract the distinct impact-concepts (variables, mechanisms, mediators, confounders, moderators, conditions, constructs) that causally bear on the claim's outcome or its decision target.
2. Merge — for each borderline candidate pair listed in the user message (from the deterministic cosine band), decide whether the two concept nodes denote the same underlying concept, for graph canonicalization / de-duplication.
</task>

<rules>
Mining:
1. Set bears_on_outcome=true when a concept bears on the outcome, false for off-outcome incidentals.
2. evidence_id and paper_id must come from the passage markers; matched_quote_span is the verbatim phrase the concept was read from.
3. Write each concept's definition in universal, field-standard terms — decompose any paper-coined construct into the widely-shared concepts it operationalizes — so the definition is self-contained and a non-author understands it without the source paper.
4. Keep label as the paper's own term (so its literature provenance stays intact); the universal expansion lives in definition.
5. Write source_context as one sentence stating what the source actually argues about this concept — the specific relationship, mechanism, or condition the source claims it participates in — grounded in the passage it was read from. definition says what the concept is in universal terms; source_context preserves what this source argues it does.

Merging:
6. Merge true synonyms (e.g. 'reconstruction error' vs 'truncation error' vs 'compression loss'); do NOT merge concepts that are merely related or adjacent (e.g. 'Fisher importance weighting' vs 'off-diagonal Fisher correlations' are DISTINCT).
7. Return exactly one decision per listed pair, in the order listed, with "a" and "b" repeating the pair's labels exactly as given; if no pairs are listed, return "merge_decisions": [].
</rules>

<output_format>
Return STRICT JSON only (no markdown):
{"concepts": [{"label","type","definition","source_context","bears_on_outcome","evidence_id","paper_id","matched_quote_span","rationale"}],
 "merge_decisions": [{"a","b","same_concept": true|false,"reason"}]}
type is one of: variable, mechanism, mediator, confounder, moderator, condition, construct.
</output_format>"""


def _merge_candidates_block(merge_candidates: Sequence[tuple[ConceptNode, ConceptNode]]) -> str:
    """Render the deterministic cosine-band borderline pairs for the mine turn (may be empty)."""
    if not merge_candidates:
        return "(none)"
    blocks: list[str] = []
    for i, (c_a, c_b) in enumerate(merge_candidates, start=1):
        blocks.append(
            f"### Pair {i}\n"
            f"Concept A:\nlabel: {c_a.label}\ntype: {c_a.type}\ndefinition: {c_a.definition}\n\n"
            f"Concept B:\nlabel: {c_b.label}\ntype: {c_b.type}\ndefinition: {c_b.definition}"
        )
    return "\n\n".join(blocks)


def _synthesist_mine_user(
    claim: str,
    passages: Sequence[str],
    merge_candidates: Sequence[tuple[ConceptNode, ConceptNode]],
) -> str:
    return (
        f"<claim>\n{claim}\n</claim>\n\n"
        f"<passages>\n{passage_block(passages)}\n</passages>\n\n"
        '<merge_candidates note="deterministic cosine band; may be empty">\n'
        f"{_merge_candidates_block(merge_candidates)}\n</merge_candidates>\n\n"
        "Mine the outcome-anchored impact-concepts and decide every listed merge pair as STRICT JSON."
    )


def _parse_merge_decisions(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse the batched cosine-band merge adjudications (folds  into turn 1)."""
    out: list[dict[str, Any]] = []
    for raw in data.get("merge_decisions") or []:
        if not isinstance(raw, dict):
            continue
        out.append(
            {
                "a": str(raw.get("a", "")),
                "b": str(raw.get("b", "")),
                "same_concept": bool(raw.get("same_concept", False)),
                "reason": str(raw.get("reason", "")),
            }
        )
    return out


@dataclass(frozen=True)
class MineResult:
    """Turn-1 output: the mined outcome-anchored concepts (each with ``source_context``) and the
    batched borderline merge adjudications. Both replay as pure functions of the recorded JSON."""

    concepts: tuple[MinedConcept, ...]
    merge_decisions: tuple[dict[str, Any], ...] = ()


# Proposal instructions are folded into the user message; no new system prompt is added.
# `{proposal_count}` is filled at runtime via str.replace (NOT str.format — the block contains
# literal JSON braces). The <committed_graph> is rendered per run and prepended.
_SYNTHESIST_PROPOSE_INSTRUCTIONS = """<task>
You are the Research Synthesist, in the propose turn of your thread — a research provocateur and same-field teammate, not a node-generator. The retrieved full-text passages you mined in the previous turn are still in context above; they are your only quotable sources. Surface the surprising, field-challenging links a practitioner has not yet drawn — new mediators / confounders / moderators / mechanisms — that exploit the mined concepts. Propose {proposal_count} candidates.
</task>

<rules>
1. Prefer edges that link two mined concepts, or a mined concept to a claim concept, via a relationship the seed claim never implies (surprising, cross-concept, research-grade).
2. For each candidate, name the unstated assumption in the field that it challenges or overturns (idea_scaffold.challenged_assumption).
3. If a candidate merely restates a textbook / established / common-sense relationship, do not submit it as-is — sharpen it into a more specific version (a precise mediator or boundary condition) or an opposing hypothesis (where/when the established relation reverses or fails).
4. Make your candidates span distinct intellectual moves — e.g. a hidden mediator, a confounder, a moderator/boundary condition, a reversal/opposition, a cross-domain transfer — not variations of one idea.
5. Each candidate introduces one new node (the proposed mechanism/mediator) and 1-2 new edges whose endpoints reference existing node labels (claim or mined) and/or the new node's own label.
6. mechanism_chain — state the explicit step chain concept -> relation -> concept the hypothesis asserts: steps in causal order, each "from"/"to" an exact node label (an existing label or the new node's own label), each step's mechanism stating why that link holds. The chain must be consistent with new_edges: every new edge appears as a step with the same source, target and relation_type; steps may additionally traverse already-committed edges.
7. source_quotes — copy spans verbatim, character-for-character, from the passages above in this conversation: at least one span for every mined concept the candidate uses, and at least one span for each bridged relationship, quoting the source text that bridge is built from. evidence_id must come from the passage markers; role_in_hypothesis names in one factual clause what the quote grounds.
8. Hard rule: a candidate whose bridge cannot be quoted from the sources in context must not be submitted — drop it and propose one you can ground. Copy every quote_span from the passages, not from this graph view, not from memory.
9. Do not grade your own field-novelty; an independent panel does that.
10. candidate_id must be h1, h2, h3, ... in order.
11. idea_scaffold is public rationale, not hidden chain-of-thought: keep each field to one concise sentence. idea_scaffold.problem is required: name the specific problem this hypothesis solves — who/what is blocked and why it matters, one concrete sentence, never a generic like 'improves performance'.
12. llm_signals are [0,1]; duplication is low for genuinely new structure. mechanism_chain and source_quotes are required for every candidate and never empty.
</rules>

<output_format>
Return STRICT JSON only (no markdown):
{"candidates": [{"candidate_id","rationale","experiment",
  "idea_scaffold":{"claim_anchor","challenged_assumption","incumbent_limit","lever","synthesis","guarantee","fail_safe","why","problem","method","experiment","critique"},
  "new_nodes":[{"label","type","definition","aliases"}],
  "new_edges":[{"source","target","relation_type","direction","mechanism"}],
  "mechanism_chain":[{"from","relation","to","mechanism"}],
  "source_quotes":[{"evidence_id","quote_span","role_in_hypothesis"}],
  "assumptions":[...],"cross_concept": true|false,"common_sense": true|false,
  "llm_signals":{"novelty","testability","scope_fit","duplication","plausibility","expected_yield","centrality","mechanism_specificity"}}]}
</output_format>

Propose the candidate hypotheses as STRICT JSON."""


def _synthesist_committed_graph_view(store: Any, mined_ids: set[str]) -> str:
    """Render the committed graph for the propose turn: seed-claim + literature-mined nodes
    (tagged [claim]/[mined]), each mined node adding its ``source_context`` when carried on the
    node's provenance, then the typed edges."""
    lines = ["Nodes:"]
    for nid, node in store.nodes.items():
        tag = "mined" if nid in mined_ids else "claim"
        lines.append(f"- {node.label} ({node.type}) [{tag}]: {node.definition}")
        for prov in getattr(node, "provenance", ()) or ():
            source_context = prov.get("source_context") if isinstance(prov, dict) else None
            if source_context:
                evidence_id = prov.get("evidence_id", "")
                quote = prov.get("matched_quote_span", "")
                lines.append(f'  source_context: {source_context} ({evidence_id}: "{quote}")')
                break
    lines.append("\nEdges:")
    for edge in store.edges.values():
        src = " & ".join(store.nodes[s].label for s in edge.source_node_ids if s in store.nodes)
        tgt = " & ".join(store.nodes[t].label for t in edge.target_node_ids if t in store.nodes)
        lines.append(f"- {src} --{edge.relation_type}--> {tgt}")
    return "\n".join(lines)


def _synthesist_propose_user(store: Any, mined_ids: set[str], proposal_count: int) -> str:
    return (
        f"<committed_graph>\n{_synthesist_committed_graph_view(store, mined_ids)}\n</committed_graph>\n\n"
        + _SYNTHESIST_PROPOSE_INSTRUCTIONS.replace("{proposal_count}", str(proposal_count))
    )


# Revision instructions are folded into the user message; no new system prompt is added.
_SYNTHESIST_REVISE_INSTRUCTIONS = """<task>
You are the Research Synthesist, in the revise turn of your thread — a research provocateur, not a rephraser. Your earlier turns are still in this conversation: the retrieved full-text passages, your mined concepts, and the full candidate pool you proposed, each candidate with its mechanism_chain and source_quotes. Those candidates are the parents; the panel result above references them by candidate_id only — work from their full text in this thread, never from summaries. Do both jobs in this one turn: repair the flagged survivors (the flags dictate what you repair), then derive new candidates (the critique steers what you derive).
</task>

<rules>
Repair (job a):
1. Make the minimal rewording that fixes each finding: state the missing relation for a vague step in one clause consistent with the typed edges, in both the flagged mechanism_chain step and the prose sentence that carries it; replace or explicitly qualify a term to match its quoted source meaning, or requote it to the exact source span from the passages in this thread — never invent a quote.
2. Do not add new claims or citations; preserve the hedged tone; keep the same candidate_id; copy every unflagged field through unchanged.
3. Do not change a candidate's new_nodes or new_edges — except when a chain step was graded false: only then repair the affected edge(s), with their mechanism_chain steps and source_quotes, so the stated chain no longer contradicts the typed structure or the quoted evidence.
4. Do not re-emit unflagged candidates; they stay in the pool as-is.

Derive (job b):
5. Derive 2-4 genuinely new candidate hypotheses from the parents + critique. Do not edit, restate, or merely rephrase a parent (repairing parents is job a; a derived candidate never replaces one).
6. Allowed strategies: inspire, combine, simplify, ground_enhance, divergent. When a derived child lands near an established or parent relationship, push it to a more specific form (a precise mediator or boundary condition) or an opposing hypothesis (where/when that relationship reverses or fails) rather than restating it.
7. Continue candidate_id numbering after the highest id in the pool; wasDerivedFrom must list the parent candidate_id(s) it derives from.
8. At least one derived candidate must use strategy "divergent" and be built by recombination alone — from concepts already in the pool or the committed graph, with no reliance on the retrieved passages; its source_quotes may quote only claim-graph node definitions, quote_span carrying the verbatim node definition, with evidence_id null.
9. Every derived candidate carries the full obligations of your propose turn — except the divergent candidate's source_quotes as specified in rule 8: it introduces one new node (the proposed mechanism/mediator) and 1-2 new edges whose endpoints reference existing node labels (claim or mined) and/or the new node's own label; it states its mechanism_chain step by step (concept -> relation -> concept), each step consistent with its typed edges; its source_quotes give the verbatim span(s) each edge bridges; evidence_id must come from the passage markers. A candidate whose chain you cannot state from the material in this thread is a candidate you must not submit.
</rules>

<output_format>
Return STRICT JSON only (no markdown):
{"revised":[{"candidate_id","rationale","experiment",
  "idea_scaffold":{"claim_anchor","challenged_assumption","incumbent_limit","lever","synthesis","guarantee","fail_safe","why","problem","method","experiment","critique"},
  "new_nodes":[{"label","type","definition","aliases"}],
  "new_edges":[{"source","target","relation_type","direction","mechanism"}],
  "mechanism_chain":[{"from","relation","to","mechanism"}],
  "source_quotes":[{"evidence_id","quote_span","role_in_hypothesis"}],
  "assumptions":[...],"cross_concept":true|false,"common_sense":true|false,
  "llm_signals":{"novelty","testability","scope_fit","duplication","plausibility","expected_yield","centrality","mechanism_specificity"}}],
 "derived":[{...same keys as a revised entry...,"strategy":"inspire|combine|simplify|ground_enhance|divergent","wasDerivedFrom":["<parent candidate_id>",...]}]}
"revised" holds only the flagged survivors (same candidate_id); "derived" holds only genuinely new candidates. llm_signals are [0,1]; duplication is low for genuinely new structure.
</output_format>

Repair the flagged survivors and derive the new candidates as STRICT JSON."""


def _synthesist_panel_result_block(panel_result: dict[str, Any]) -> str:
    """Render the Critic Panel result for the revise turn: pool ranking + critique + per-flagged-
    candidate audit flags for vague or false mechanism steps and terminology issues."""
    ranking = ", ".join(str(cid) for cid in (panel_result.get("ranking") or []))
    lines = [
        "<panel_result>",
        "Pool ranking (best first; these surviving parents are your propose-turn candidates, "
        f"referenced by candidate_id only): {ranking}",
        "",
        "Panel critique:",
        str(panel_result.get("critique", "")),
        "",
        '<audit_flags note="one block per flagged candidate; no block = unflagged">',
    ]
    for flag in panel_result.get("audit_flags") or []:
        lines.append(f"### {flag.get('candidate_id', '')}")
        steps = [
            s for s in (flag.get("mechanism_steps") or [])
            if isinstance(s, dict) and s.get("verdict") in ("vague", "false")
        ]
        if steps:
            lines.append("mechanism steps graded vague/false:")
            for s in steps:
                lines.append(
                    f"- {s.get('from', '')} --{s.get('relation', '')}--> {s.get('to', '')}: "
                    f"{s.get('verdict', '')} — {s.get('note', '')}"
                )
        term_issues = [t for t in (flag.get("term_issues") or []) if isinstance(t, dict)]
        if term_issues:
            lines.append("term issues:")
            for t in term_issues:
                quote = t.get("quote")
                cite = (
                    f'; conflicting source quote: [{t.get("evidence_id", "")} | '
                    f'{t.get("title", "")}] "{quote}"'
                    if quote else ""
                )
                lines.append(f"- '{t.get('term', '')}': {t.get('verdict', '')} — {t.get('note', '')}{cite}")
    lines.append("</audit_flags>")
    lines.append("</panel_result>")
    return "\n".join(lines)


def _synthesist_revise_user(panel_result: dict[str, Any]) -> str:
    return f"{_synthesist_panel_result_block(panel_result)}\n\n{_SYNTHESIST_REVISE_INSTRUCTIONS}"


@dataclass(frozen=True)
class ReviseResult:
    """Repaired survivors and new candidates with derivation lineage in provenance."""

    revised: tuple[HypothesisCandidate, ...] = ()
    derived: tuple[HypothesisCandidate, ...] = ()


def _attach_lineage(candidate: HypothesisCandidate, raw: dict[str, Any]) -> HypothesisCandidate:
    """Record a derived candidate's strategy and parent IDs on its provenance."""
    return replace(
        candidate,
        provenance=(
            {
                "strategy": str(raw.get("strategy", "")),
                "wasDerivedFrom": [str(p) for p in (raw.get("wasDerivedFrom") or [])],
            },
        ),
    )


class ResearchSynthesist:
    """The Research Synthesist thread over any CAMEL model backend (``.run(messages)``).

    One persistent conversation: ``mine`` seeds the thread with the single system prompt and the
    mine user turn; ``propose`` / ``revise`` (added alongside) append their user turn to the same
    ``messages`` list so the retrieved passages and prior candidates remain in context. Every
    turn's raw assistant text is appended verbatim. Parsing reuses ``camel_adapter`` so the
    deterministic core stays LLM-free; a malformed/empty response degrades to an empty turn.
    """

    def __init__(
        self,
        model_backend: Any,
        *,
        embedder: Any,
        claim: str,
        proposal_count: int,
        mined_ids: Sequence[str] = (),
        miner_focus: str = "",
        propose_steering: str | None = None,
    ) -> None:
        self._backend = model_backend
        self._embedder = embedder
        self._claim = claim
        self._proposal_count = proposal_count
        self._mined_ids = set(mined_ids)
        # Profile-derived steering is appended to the relevant prompt at runtime. Empty steering
        # leaves the base prompt unchanged.
        self._miner_focus = miner_focus
        self._propose_steering = propose_steering
        self._messages: list[dict[str, str]] = []

    def _run_turn(self, user_prompt: str, *, system_prompt: str | None = None) -> dict[str, Any]:
        """Append the turn's user message (seeding the system prompt on turn 1), call the backend
        over the accumulated thread, append the raw assistant response, and return parsed JSON."""
        from src.camel_adapter import (
            _extract_json_object,
            _first_response_message,
            _message_content,
        )

        if system_prompt is not None:
            self._messages.append({"role": "system", "content": system_prompt})
        self._messages.append({"role": "user", "content": user_prompt})
        response = self._backend.run(self._messages)
        content = _message_content(_first_response_message(response))
        raw = str(content) if content else ""
        self._messages.append({"role": "assistant", "content": raw})
        if not raw:
            return {}
        try:
            return _extract_json_object(raw)
        except ValueError:
            return {}

    def mine(
        self,
        passages: Sequence[str],
        merge_candidates: Sequence[tuple[ConceptNode, ConceptNode]] = (),
    ) -> MineResult:
        """Mine outcome-anchored concepts and adjudicate borderline pairs in one batch.

        Each concept includes ``source_context``. Profile ``miner_focus`` steering is appended to
        the system prompt at runtime.
        """
        system_prompt = SYNTHESIST_SYSTEM_PROMPT
        if self._miner_focus:
            system_prompt = f"{system_prompt}\n\n{self._miner_focus}"
        data = self._run_turn(
            _synthesist_mine_user(self._claim, passages, merge_candidates),
            system_prompt=system_prompt,
        )
        return MineResult(
            concepts=tuple(_parse_mined_concepts(data)),
            merge_decisions=tuple(_parse_merge_decisions(data)),
        )

    def propose(self, store: Any) -> list[HypothesisCandidate]:
        """Propose grounded candidate hypotheses over the committed graph.

        Each candidate has a required ``mechanism_chain`` and verbatim ``source_quotes``. The
        proposal turn reuses the persistent thread and the shared candidate parser.
        """
        from src.cycles.hypothesis import build_hypothesis_candidates

        user = _synthesist_propose_user(store, self._mined_ids, self._proposal_count)
        if self._propose_steering:  # Judgeability and optional venue preference.
            user = f"{user}\n\n{self._propose_steering}"
        data = self._run_turn(user)
        return build_hypothesis_candidates(
            data.get("candidates") or [], store=store, embedder=self._embedder, claim=self._claim
        )

    def revise(self, store: Any, panel_result: dict[str, Any]) -> ReviseResult:
        """Repair panel-flagged survivors and derive two to four new candidates.

        ``panel_result`` carries the pool ranking, critique, and per-candidate audit flags. Derived
        candidates record their strategy and ``wasDerivedFrom`` lineage. Cardinality and the
        required divergent strategy are enforced deterministically: excess candidates are dropped,
        while shortfalls and missing divergent candidates are logged without fabrication or retry.
        """
        from src.cycles.hypothesis import build_hypothesis_candidates

        data = self._run_turn(_synthesist_revise_user(panel_result))
        revised = build_hypothesis_candidates(
            data.get("revised") or [], store=store, embedder=self._embedder, claim=self._claim
        )
        derived_raw = [r for r in (data.get("derived") or []) if isinstance(r, dict)]
        derived = build_hypothesis_candidates(
            derived_raw, store=store, embedder=self._embedder, claim=self._claim
        )
        # candidate_id-keyed (not positional zip): build_hypothesis_candidates may itself drop a
        # raw entry with empty grounding, which would otherwise misalign a later candidate's
        # lineage with an earlier raw entry's strategy/wasDerivedFrom.
        raw_by_id = {str(r.get("candidate_id", "")): r for r in derived_raw}
        derived = [_attach_lineage(cand, raw_by_id.get(cand.candidate_id, {})) for cand in derived]

        if len(derived) > _REVISE_DERIVED_MAX:
            for cand in derived[_REVISE_DERIVED_MAX:]:
                logger.warning(
                    "dropped derived candidate %s: exceeds REVISE cardinality cap (max %d)",
                    cand.candidate_id, _REVISE_DERIVED_MAX,
                )
            derived = derived[:_REVISE_DERIVED_MAX]
        elif len(derived) < _REVISE_DERIVED_MIN:
            logger.warning(
                "Research Synthesist revise contract shortfall: only %d derived candidate(s), below REVISE "
                "minimum %d (accepted as-is, not fabricated)",
                len(derived), _REVISE_DERIVED_MIN,
            )

        if not any(cand.provenance[0].get("strategy") == "divergent" for cand in derived):
            logger.warning(
                "Research Synthesist revise contract violation: no derived candidate carries the mandatory "
                "'divergent' strategy"
            )

        return ReviseResult(revised=tuple(revised), derived=tuple(derived))
