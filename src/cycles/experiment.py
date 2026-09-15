"""Bounded Experiment Designer and Experiment Validator refinement cycle.

After a user confirms a hypothesis, the cycle generates, validates, and optionally refines a full,
retrieval-grounded experiment plan before committing an experiment delta.

  * **Experiment Designer** (builder tier) — a persistent thread: ``design`` expands the
    proposer's one-line sketch into ONE structured ``experiment_plan`` grounded in the retrieved
    methods/datasets/baselines passages; ``refine`` (same thread, so the hypothesis + passages +
    prior plan stay in context) revises only the criteria the validator flagged.
  * **Experiment Validator** (skeptical tier, independent) — a stateless single call that grades
    the plan on five criteria and flags the low-scoring ones, driving one designer refine pass.

The bounded loop runs ``experiment_refine_rounds`` passes; zero skips refinement. A plan must still
pass required-content checks before commit. Determinism boundary: the LLM is injected (fixtured in the default suite; the
real backends are the ``live`` path); parsing reuses ``camel_adapter`` so the deterministic core
(the loop control + the commit gates) stays LLM-free — a malformed/empty response degrades to a
``None`` plan / empty verdict, never a raise.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from src.delta import DeltaFamily, ExperimentPayload, GraphDeltaProposal
from src.experiment_plan_render import (
    optional_evidence_id as _optional_evidence_id,
)
from src.experiment_plan_render import (
    real_evidence_id as _real_evidence_id,
)
from src.graph_store import (
    ConceptNode,
    ExperimentGroundingStatus,
    ExperimentPlan,
    GroundingItem,
    MaterialItem,
    MetricSpec,
)
from src.progress import progress_operation, report_progress

if TYPE_CHECKING:
    from src.cycles.hypothesis import HypothesisCandidate
    from src.graph_store import CausalClaimGraphStore, CausalEdge
    from src.transaction_log import GraphTransactionLog, TransactionResult

# Concept-node types that count as a controllable factor adjacent to the tested edge.
_GRAPH_CONFOUNDER_TYPES = frozenset({"confounder", "mediator", "moderator"})

# Experiment Designer system prompt.
_EXPERIMENT_DESIGNER_SYSTEM_PROMPT = """<role>
You are the Experiment Designer for a causal-claim graph. For ONE surfaced hypothesis you design a detailed, rigorous, runnable experiment that would test its proposed causal link — expanding the proposer's one-line experiment sketch into a full plan grounded in how the field actually tests such claims.
</role>

<task>
You are given the hypothesis (its new causal edge(s) and mechanism chain, its concept definitions, the proposer's one-line method/experiment sketch), the relevant committed-graph context (the tested edge and any confounder/mediator/moderator nodes around it), and retrieved methods/datasets/baselines passages. Produce ONE structured experiment_plan that operationalizes the tested edge, chooses a design that isolates it, controls the graph's named confounders, grounds its baselines/datasets/metrics in the retrieved passages, and states what result would support versus refute the mechanism.
</task>

<rules>
1. Test the specific link. hypothesis_under_test names the exact edge(s) (source --relation--> target) and the mechanism_chain step(s) the experiment probes — not a vague topic.
2. Operationalize both ends. operationalization states how the cause and the effect are each measured (concrete variables, instruments, or metrics), not restated as concepts.
3. Choose a design that fits. design is one of: randomized_controlled, controlled_observational, ablation, benchmark_comparison, simulation — pick the one that can actually isolate this causal link; design_rationale says in one clause why.
4. Name the comparison. intervention_or_manipulation says what is varied; comparison_baseline names the baseline/comparator, grounded in a retrieved baseline where one exists.
5. Control the graph's confounders. controls_and_confounders lists the confounder/mediator/moderator factors adjacent to the tested edge (from the committed graph) and how each is held fixed, measured, or randomized; set from_graph=true for factors taken from the graph. Do not invent confounders absent from the graph unless the evidence names one.
6. Ground materials in retrieval. materials_or_data (datasets/benchmarks/populations) and metrics must cite retrieved passages by evidence_id where a real dataset/benchmark/metric is used; if none was retrieved, say so plainly rather than inventing a citation.
7. State outcomes both ways. expected_outcome is the result that would support the hypothesis via the stated mechanism; falsification is the result that would refute it (the rigorous form of "what would change our mind").
8. Be feasible and honest. feasibility names the resources, time, and main risk; do not promise more than the design supports. Keep every field concrete; no generic filler such as "improves performance".
9. Ground, do not fabricate. grounding lists each evidence_id the plan relied on, with the verbatim quote_span it was read from (this array populates the committed Δ-experiment's referenced_evidence_ids). Retrieved rows are candidates, not automatically relevant sources: if none of them supports this plan, return grounding=[] explicitly rather than inventing a citation.
</rules>

<output_format>
Return STRICT JSON only (no markdown):
{"experiment_plan": {"hypothesis_under_test","operationalization","design":"randomized_controlled|controlled_observational|ablation|benchmark_comparison|simulation","design_rationale","intervention_or_manipulation","comparison_baseline","controls_and_confounders":[{"factor","from_graph": true|false,"handling"}],"materials_or_data":[{"item","evidence_id"}],"metrics":[{"metric","predicted_direction","evidence_id"}],"procedure":["<ordered step>"],"expected_outcome","falsification","feasibility":{"resources","time","main_risk"},"grounding":[{"evidence_id","quote_span"}]}}
</output_format>"""

# Experiment Validator system prompt.
_EXPERIMENT_VALIDATOR_SYSTEM_PROMPT = """<role>
You are the Experiment Validator (independent) for a causal-claim graph. You grade ONE proposed experiment plan on five criteria and give concise, actionable feedback so the designer can improve it. You did not design this plan; judge it skeptically.
</role>

<task>
You are given the hypothesis under test, the experiment_plan, and the retrieved passages the plan cites. Grade the plan on the five criteria below — each a number in [0,1] with one concise feedback sentence — and flag any criterion scoring below the target so the designer can revise it.
</task>

<rules>
Grade each criterion from the plan and the cited passages only:
1. Clarity — is the design, procedure, and measurement stated precisely enough to run and to comprehend?
2. Validity — does the design actually isolate the tested causal edge (right comparison; cause and effect properly operationalized; the confounders the plan lists in controls_and_confounders — including those marked from_graph — adequately controlled)?
3. Robustness — would the result hold across reasonable conditions/datasets, not hinge on one fragile setup?
4. Feasibility — can it realistically be run with the stated resources and time, and are the datasets/baselines/metrics real (grounded in the cited passages) rather than invented?
5. Reproducibility — is there enough detail (materials, metrics, procedure) for an independent team to reproduce it?
Grounding check: for each materials_or_data or metrics entry that cites an evidence_id, confirm the cited passage supports it; a fabricated or unsupported citation caps Feasibility and Validity low. Judge substance, not phrasing.
</rules>

<output_format>
Return STRICT JSON only (no markdown):
{"criteria": {"clarity": {"score": 0..1, "feedback": "…"}, "validity": {"score": 0..1, "feedback": "…"}, "robustness": {"score": 0..1, "feedback": "…"}, "feasibility": {"score": 0..1, "feedback": "…"}, "reproducibility": {"score": 0..1, "feedback": "…"}}, "flagged": ["<criterion name scoring below target>"], "overall": 0..1}
</output_format>"""

# The five verdict criteria in canonical order.
_CRITERIA = ("clarity", "validity", "robustness", "feasibility", "reproducibility")

# Required experiment-plan content fields owned by the Experiment Designer.
# contract; the validator stays thin (graph_store.py's ExperimentPlan docstring assigns this
# contract here, not to the validator). Dotted paths reach into feasibility's subfields
# Empty strings or collections mean the plan is incomplete and must not commit.
_REQUIRED_PLAN_CONTENT_FIELDS: tuple[str, ...] = (
    "hypothesis_under_test",
    "operationalization",
    "design_rationale",
    "intervention_or_manipulation",
    "comparison_baseline",
    "controls_and_confounders",
    "materials_or_data",
    "metrics",
    "expected_outcome",
    "falsification",
    "feasibility.resources",
    "feasibility.time",
    "feasibility.main_risk",
    "grounding",
)


# --- render helpers (mirror cycles/panel.py structural rendering) ------------------------
def _label_of(candidate: HypothesisCandidate, node_labels: Mapping[str, str] | None) -> dict[str, str]:
    """Resolve edge endpoint ids to labels: committed ``node_labels`` + the candidate's new nodes."""
    resolved = dict(node_labels or {})
    for node in candidate.new_nodes:
        resolved.setdefault(node.node_id, node.label)
    return resolved


def _edge_view(edge: Any, label_of: Mapping[str, str]) -> str:
    src = " & ".join(label_of.get(s, s) for s in edge.source_node_ids)
    tgt = " & ".join(label_of.get(t, t) for t in edge.target_node_ids)
    mechanism = f" [{edge.mechanism}]" if edge.mechanism else ""
    return f"{src} --{edge.relation_type}--> {tgt}{mechanism}"


def _edges_view(candidate: HypothesisCandidate, label_of: Mapping[str, str]) -> str:
    return "; ".join(_edge_view(e, label_of) for e in candidate.new_edges) or "(none)"


def _nodes_view(nodes: Sequence[ConceptNode]) -> str:
    return "; ".join(f"{n.label} ({n.type}): {n.definition}" for n in nodes) or "(none)"


def _chain_view(chain: Sequence[Mapping[str, Any]]) -> str:
    steps: list[str] = []
    for step in chain:
        base = f"{step.get('from', '')} --{step.get('relation', '')}--> {step.get('to', '')}"
        mechanism = step.get("mechanism")
        steps.append(f"{base} [{mechanism}]" if mechanism else base)
    return "; ".join(steps) or "(none)"


def _panel_audit_view(candidate: HypothesisCandidate) -> str:
    """Project the Critic Panel's residual ``mechanism_steps`` and ``term_audit`` warnings into the
    Experiment Designer prompt. Only 'vague'/'false' steps and 'stretched'/'misused'/'no_source'
    terms appear, each explicitly flagged with its verdict — 'sound'/'consistent' entries add no
    design consideration beyond the stated ``Mechanism chain`` line above, so they are omitted;
    a panel warning never mixes silently into the plain narrative. Empty when the candidate
    carries no Critic Panel audit (panel off/legacy) — the design/refine prompts then stay byte-identical."""
    steps = [
        f"[{s.get('verdict', '')}] {s.get('from', '')} --{s.get('relation', '')}--> {s.get('to', '')}"
        for s in candidate.mechanism_steps
        if s.get("verdict") in ("vague", "false")
    ]
    steps_line = (
        f"Panel-audited mechanism chain (residual warnings): {'; '.join(steps)}\n" if steps else ""
    )
    terms = [
        f"[{t.get('verdict', '')}] {t.get('term', '')}"
        for t in candidate.term_audit
        if t.get("verdict") in ("stretched", "misused", "no_source")
    ]
    terms_line = (
        f"Panel-audited terminology (residual warnings): {'; '.join(terms)}\n" if terms else ""
    )
    return steps_line + terms_line


def _passage_field(passage: Any, name: str, *alts: str) -> str:
    """Read a field off a retrieved passage (a ``RetrievedEvidence`` or a plain dict)."""
    keys = (name, *alts)
    if isinstance(passage, Mapping):
        for key in keys:
            if passage.get(key):
                return str(passage[key])
        return ""
    for key in keys:
        value = getattr(passage, key, None)
        if value:
            return str(value)
    return ""


def _passage_retrieval_batch_id(passage: Any) -> str:
    """Read deterministic cache-batch identity from retrieved-passage metadata."""
    metadata = (
        passage.get("metadata")
        if isinstance(passage, Mapping)
        else getattr(passage, "metadata", None)
    )
    if not isinstance(metadata, Mapping):
        return ""
    return str(metadata.get("retrieval_batch_id") or "").strip()


def _retrieved_methods_block(passages: Sequence[Any]) -> str:
    if not passages:
        return "(none retrieved)"
    blocks = [
        f"### Passage {i} — evidence_id: {_passage_field(p, 'evidence_id')}\n"
        f"Title: {_passage_field(p, 'title')}\n"
        f"Passage: {_passage_field(p, 'quote', 'passage')}"
        for i, p in enumerate(passages, start=1)
    ]
    return "\n\n".join(blocks)


def _cited_passages_block(passages: Sequence[Any]) -> str:
    if not passages:
        return "(none)"
    blocks = [
        f"### evidence_id: {_passage_field(p, 'evidence_id')}\n{_passage_field(p, 'quote', 'passage')}"
        for p in passages
    ]
    return "\n\n".join(blocks)


def _experiment_designer_user(
    candidate: HypothesisCandidate, graph_context: ExperimentGraphContext, passages: Sequence[Any]
) -> str:
    """Build the Experiment Designer's initial user turn."""
    label_of = _label_of(candidate, graph_context.node_labels)
    scaffold = candidate.idea_scaffold or {}
    return (
        "<hypothesis>\n"
        f"id: {candidate.candidate_id}\n"
        f"New concept node(s): {_nodes_view(candidate.new_nodes)}\n"
        f"Tested causal edge(s): {_edges_view(candidate, label_of)}\n"
        f"Mechanism chain: {_chain_view(candidate.mechanism_chain)}\n"
        f"{_panel_audit_view(candidate)}"
        f"Proposer's sketch — method: {scaffold.get('method', '')}; experiment: {scaffold.get('experiment', '')}\n"
        "</hypothesis>\n\n"
        "<graph_context>\n"
        "Confounder / mediator / moderator nodes adjacent to the tested edge: "
        f"{_nodes_view(graph_context.adjacent_nodes)}\n"
        f"Edge confounders field: {graph_context.edge_confounders or '(none)'}\n"
        "</graph_context>\n\n"
        '<retrieved_methods note="targeted methods / datasets / baselines retrieval for this '
        'hypothesis; may be empty">\n'
        f"{_retrieved_methods_block(passages)}\n"
        "</retrieved_methods>\n\n"
        "Design the experiment plan as STRICT JSON."
    )


def _experiment_designer_refine_user(
    verdict: ExperimentVerdict,
    candidate: HypothesisCandidate | None = None,
    *,
    content_gaps: Sequence[str] = (),
) -> str:
    """Build the Experiment Designer's refinement user turn after validation.

    Only the flagged criteria + their score + feedback are sent; the hypothesis, passages, and the
    prior plan stay in the thread's context (Experiment Designer is a persistent conversation). The Critic Panel's
    residual mechanism and terminology warnings are re-sent so a flagged item stays visible
    across the refine turn; empty when ``candidate`` carries no Critic Panel audit."""
    lines: list[str] = []
    for criterion in verdict.flagged:
        grade = verdict.criteria.get(criterion)
        score = f"{grade.score:g}" if grade else "0"
        feedback = grade.feedback if grade else ""
        lines.append(f"- {criterion}: {score} — {feedback}")
    flagged_block = "\n".join(lines) or "(none)"
    gap_block = ""
    instruction = (
        "Revise ONLY the criteria it flagged below; keep everything else unchanged, and keep "
        "the grounding honest (no invented citations)."
    )
    if content_gaps:
        gaps = "\n".join(f"- {path}" for path in content_gaps)
        instruction = (
            "Revise the criteria it flagged and fill the deterministic required-content gaps "
            "below; keep everything else unchanged, and keep the grounding honest (no invented "
            "citations)."
        )
        gap_block = (
            "\n\n<required_content_gaps>\n"
            "Deterministic checks found these blank or uncitable required fields:\n"
            f"{gaps}\n"
            "For a grounding gap, cite only a retrieved evidence_id with its verbatim quote_span. "
            'If no retrieved passage is relevant, return "grounding": [] explicitly.\n'
            "</required_content_gaps>"
        )
    audit = _panel_audit_view(candidate) if candidate is not None else ""
    return (
        f"{audit}"
        "<validator_feedback>\n"
        f"The Experiment Validator graded your plan. {instruction}\n"
        "Flagged criteria (score + feedback):\n"
        f"{flagged_block}\n"
        f"</validator_feedback>{gap_block}\n\n"
        "Return the full revised experiment_plan as STRICT JSON."
    )


def _experiment_validator_user(
    candidate: HypothesisCandidate,
    plan: ExperimentPlan,
    cited_passages: Sequence[Any],
    *,
    node_labels: Mapping[str, str] | None = None,
) -> str:
    """The Experiment Validator user turn: the hypothesis under test, the
    plan artifact, and the passages its grounding cites; NOT the designer's private reasoning."""
    label_of = _label_of(candidate, node_labels)
    return (
        "<hypothesis>\n"
        f"id: {candidate.candidate_id}\n"
        f"Tested causal edge(s): {_edges_view(candidate, label_of)}\n"
        f"Mechanism chain: {_chain_view(candidate.mechanism_chain)}\n"
        "</hypothesis>\n\n"
        "<experiment_plan>\n"
        f"{plan.model_dump_json(indent=2)}\n"
        "</experiment_plan>\n\n"
        '<cited_passages note="the passages the plan\'s grounding cites">\n'
        f"{_cited_passages_block(cited_passages)}\n"
        "</cited_passages>\n\n"
        "Grade the five criteria and flag the low-scoring ones. Return STRICT JSON."
    )


# --- data carriers -----------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentGraphContext:
    """The committed-graph neighborhood Experiment Designer controls for: the confounder/mediator/moderator nodes
    adjacent to the tested edge, the edge's own ``confounders`` field, and the ID-to-label map used
    to render the tested edges."""

    adjacent_nodes: tuple[ConceptNode, ...] = ()
    edge_confounders: str = ""
    node_labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CriterionGrade:
    """One Experiment Validator criterion grade: a score in [0,1] with one feedback sentence."""

    score: float = 0.0
    feedback: str = ""


@dataclass(frozen=True)
class ExperimentVerdict:
    """The Experiment Validator verdict: five criterion grades, the ``flagged`` criteria
    below target, and the ``overall`` score. Recorded in provenance; drives one Experiment Designer refine pass. It
    never commits state itself."""

    criteria: Mapping[str, CriterionGrade] = field(default_factory=dict)
    flagged: tuple[str, ...] = ()
    overall: float = 0.0


# --- parse (graceful-degrade, LLM-free) --------------------------------------------------
def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_experiment_plan(data: Mapping[str, Any]) -> ExperimentPlan | None:
    """Validate the Experiment Designer ``{"experiment_plan": {...}}`` payload into an ``ExperimentPlan``; a malformed
    or missing plan degrades to ``None`` (nothing commits for that hypothesis)."""
    raw = data.get("experiment_plan") if isinstance(data, Mapping) else None
    if not isinstance(raw, Mapping):
        return None
    try:
        return ExperimentPlan.model_validate(dict(raw))
    except ValidationError:
        return None


def _plan_field_value(plan: ExperimentPlan, path: str) -> Any:
    """Read a (possibly dotted, e.g. ``feasibility.resources``) field off a plan."""
    value: Any = plan
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _plan_content_gaps(
    plan: ExperimentPlan,
    *,
    graph_context: ExperimentGraphContext,
    has_retrieved_passages: bool = True,
    retrieved_evidence_ids: Sequence[str] | set[str] | None = None,
) -> tuple[str, ...]:
    """Return required plan fields that remain blank.

    An empty tuple means the plan is complete. ``controls_and_confounders`` is
    exempt when the committed graph offers no adjacent confounder/mediator/moderator and no
    edge-level confounder to name; Experiment Designer must not invent one. ``grounding`` counts
    only real evidence IDs: a plan whose grounding is entirely "no evidence
    retrieved" placeholders gaps too, matching what :func:`build_experiment_delta` actually
    commits once its placeholder normalization drops those rows.

    Raw retrieval rows are not evidence that the plan has a relevant source. An otherwise
    complete Designer response may explicitly return ``grounding: []`` to declare that none of
    those rows applies. Omitted grounding, placeholder-only grounding, and real IDs that do not
    resolve in ``retrieved_evidence_ids`` remain deterministic content gaps.
    """
    graph_has_confounders = bool(graph_context.adjacent_nodes or graph_context.edge_confounders)
    gaps: list[str] = []
    for path in _REQUIRED_PLAN_CONTENT_FIELDS:
        if path == "controls_and_confounders" and not graph_has_confounders:
            continue
        if path == "grounding":
            # Pydantic retains which fields the Designer actually emitted, allowing an explicit
            # empty list ("no retrieved row is relevant") to differ from an omitted field.
            explicit_no_relevant_source = (
                "grounding" in plan.model_fields_set and not plan.grounding
            )
            grounding_ids = _grounding_evidence_ids(plan)
            if not has_retrieved_passages:
                # Empty retrieval is the honest no-source path; placeholder rows are normalized
                # away when the delta is built, preserving the legacy disabled-retrieval path.
                continue
            if explicit_no_relevant_source:
                continue
            has_blank_real_quote = any(
                _real_evidence_id(item.evidence_id) and not item.quote_span.strip()
                for item in plan.grounding
            )
            empty = not grounding_ids or has_blank_real_quote
            if retrieved_evidence_ids is not None and grounding_ids:
                available = set(retrieved_evidence_ids)
                empty = empty or any(
                    evidence_id not in available for evidence_id in grounding_ids
                )
        else:
            value = _plan_field_value(plan, path)
            empty = not value.strip() if isinstance(value, str) else not value
        if empty:
            gaps.append(path)
    return tuple(gaps)


def _plan_is_content_complete(
    plan: ExperimentPlan,
    *,
    graph_context: ExperimentGraphContext,
    has_retrieved_passages: bool = True,
    retrieved_evidence_ids: Sequence[str] | set[str] | None = None,
) -> bool:
    """Return true when no required field is blank.

    Grounding must either resolve within this retrieval or be an explicitly emitted empty list;
    empty retrieval keeps the honest no-source path open as well.
    """
    return not _plan_content_gaps(
        plan,
        graph_context=graph_context,
        has_retrieved_passages=has_retrieved_passages,
        retrieved_evidence_ids=retrieved_evidence_ids,
    )


def _parse_experiment_verdict(data: Mapping[str, Any]) -> ExperimentVerdict:
    """Parse the Experiment Validator verdict; a malformed/empty response degrades to an all-zero, unflagged verdict
    (so the bounded loop simply stops refining)."""
    if not isinstance(data, Mapping):
        data = {}
    criteria_raw = data.get("criteria")
    criteria_raw = criteria_raw if isinstance(criteria_raw, Mapping) else {}
    criteria = {}
    for name in _CRITERIA:
        grade = criteria_raw.get(name)
        grade = grade if isinstance(grade, Mapping) else {}
        criteria[name] = CriterionGrade(score=_as_float(grade.get("score")), feedback=str(grade.get("feedback", "")))
    flagged = tuple(str(f) for f in (data.get("flagged") or []) if str(f))
    return ExperimentVerdict(criteria=criteria, flagged=flagged, overall=_as_float(data.get("overall")))


def _grounding_evidence_ids(plan: ExperimentPlan) -> list[str]:
    """The plan's grounding evidence ids (distinct, order-preserving) — these populate the committed
    experiment delta's ``referenced_evidence_ids``."""
    seen: list[str] = []
    for item in plan.grounding:
        eid = _real_evidence_id(item.evidence_id)
        if eid and eid not in seen:
            seen.append(eid)
    return seen


def _clean_material_evidence(items: Sequence[MaterialItem]) -> tuple[list[MaterialItem], bool]:
    cleaned: list[MaterialItem] = []
    changed = False
    for item in items:
        eid = _optional_evidence_id(item.evidence_id)
        if eid != item.evidence_id:
            changed = True
            item = item.model_copy(update={"evidence_id": eid})
        cleaned.append(item)
    return cleaned, changed


def _clean_metric_evidence(items: Sequence[MetricSpec]) -> tuple[list[MetricSpec], bool]:
    cleaned: list[MetricSpec] = []
    changed = False
    for item in items:
        eid = _optional_evidence_id(item.evidence_id)
        if eid != item.evidence_id:
            changed = True
            item = item.model_copy(update={"evidence_id": eid})
        cleaned.append(item)
    return cleaned, changed


def _without_placeholder_evidence_ids(plan: ExperimentPlan) -> ExperimentPlan:
    """Remove Experiment Designer's explicit "no retrieved evidence" placeholders from citation fields.

    The Experiment Designer prompt tells the model to say plainly when no methods passage was retrieved. Some live
    completions encode that as ``evidence_id: none_retrieved`` in grounding/material/metric fields;
    that is an honest note, not a ledger reference, so it must not block the Δ-experiment reference
    gate or render as a fake citation.
    """
    grounding: list[GroundingItem] = []
    updates: dict[str, Any] = {}
    for item in plan.grounding:
        eid = _real_evidence_id(item.evidence_id)
        if not eid:
            updates["grounding"] = grounding
            continue
        if eid != item.evidence_id:
            updates["grounding"] = grounding
            item = item.model_copy(update={"evidence_id": eid})
        grounding.append(item)
    if "grounding" in updates:
        updates["grounding"] = grounding

    materials, materials_changed = _clean_material_evidence(plan.materials_or_data)
    if materials_changed:
        updates["materials_or_data"] = materials

    metrics, metrics_changed = _clean_metric_evidence(plan.metrics)
    if metrics_changed:
        updates["metrics"] = metrics

    return plan.model_copy(update=updates) if updates else plan


def _cited_passages(plan: ExperimentPlan, passages: Sequence[Any]) -> list[Any]:
    """The subset of retrieved passages the plan's grounding cites (what Experiment Validator grades against)."""
    ids = set(_grounding_evidence_ids(plan))
    return [p for p in passages if _passage_field(p, "evidence_id") in ids]


# --- the two seams -----------------------------------------------------------------------
class ExperimentDesigner:
    """A persistent Experiment Designer thread over any CAMEL backend (``.run(messages)``).

    ``design`` seeds the thread with the single system prompt and the design user turn; ``refine``
    appends only the validator feedback to the SAME ``messages`` list, so the hypothesis, passages,
    and prior plan stay in context. Every turn's raw assistant text is appended verbatim; parsing
    reuses ``camel_adapter`` so the deterministic core stays LLM-free (malformed → ``None``)."""

    def __init__(self, model_backend: Any) -> None:
        self._backend = model_backend
        self._messages: list[dict[str, str]] = []
        self._candidate: HypothesisCandidate | None = None

    def _run_turn(self, user_prompt: str, *, system_prompt: str | None = None) -> dict[str, Any]:
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

    def design(
        self,
        candidate: HypothesisCandidate,
        *,
        graph_context: ExperimentGraphContext,
        passages: Sequence[Any],
    ) -> ExperimentPlan | None:
        """Expand the proposer's sketch into one grounded experiment plan."""
        self._candidate = candidate
        data = self._run_turn(
            _experiment_designer_user(candidate, graph_context, passages),
            system_prompt=_EXPERIMENT_DESIGNER_SYSTEM_PROMPT,
        )
        return _parse_experiment_plan(data)

    def refine(
        self, verdict: ExperimentVerdict, *, content_gaps: Sequence[str] = ()
    ) -> ExperimentPlan | None:
        """Revise validator-flagged criteria and deterministic content gaps in the same thread."""
        return _parse_experiment_plan(
            self._run_turn(
                _experiment_designer_refine_user(
                    verdict, self._candidate, content_gaps=content_gaps
                )
            )
        )


class ExperimentValidator:
    """An independent, stateless Experiment Validator over any CAMEL backend.

    It did not design the plan; each ``validate`` is a fresh call that sees only the hypothesis under
    test, the plan artifact, and the passages its grounding cites. A malformed or empty
    response degrades to an unflagged verdict."""

    def __init__(self, model_backend: Any, *, system_prompt: str | None = None) -> None:
        self._backend = model_backend
        self._system_prompt = system_prompt or _EXPERIMENT_VALIDATOR_SYSTEM_PROMPT

    def validate(
        self,
        candidate: HypothesisCandidate,
        plan: ExperimentPlan,
        *,
        cited_passages: Sequence[Any],
        node_labels: Mapping[str, str] | None = None,
    ) -> ExperimentVerdict:
        from src.camel_adapter import backend_json

        user = _experiment_validator_user(
            candidate, plan, cited_passages, node_labels=node_labels
        )
        return _parse_experiment_verdict(backend_json(self._backend, self._system_prompt, user))


# --- Experiment-delta build and bounded loop ----------------------------------------
def build_experiment_delta(
    *,
    base_graph_hash: str,
    hypothesis_id: str,
    experiment_plan: ExperimentPlan,
    author_role: str = "experiment_designer",
    provenance: list[dict[str, Any]] | None = None,
) -> GraphDeltaProposal:
    """Assemble an experiment delta for a confirmed hypothesis.

    ``referenced_evidence_ids`` are the plan's distinct grounding IDs,
    checked at ``1_refs`` against the ledger; author/provenance ride on the envelope, as with the
    other families. The delta commits through the shared validation gates."""
    clean_plan = _without_placeholder_evidence_ids(experiment_plan)
    return GraphDeltaProposal(
        family=DeltaFamily.EXPERIMENT,
        base_graph_hash=base_graph_hash,
        payload=ExperimentPayload(
            hypothesis_id=hypothesis_id,
            experiment_plan=clean_plan,
            referenced_evidence_ids=_grounding_evidence_ids(clean_plan),
        ),
        author_role=author_role,
        provenance=provenance or [],
    )


def design_experiment(
    designer: ExperimentDesigner,
    validator: ExperimentValidator | None,
    candidate: HypothesisCandidate,
    *,
    graph_context: ExperimentGraphContext,
    passages: Sequence[Any],
    refine_rounds: int,
) -> ExperimentPlan | None:
    """Design once, then run up to ``refine_rounds`` validator-guided refinements.

    Stop early when the validator flags nothing and the plan is complete. With zero rounds the
    validator does not run. Return ``None`` when the designer cannot produce a complete plan.
    """
    with progress_operation("drafting experiment plan"):
        plan = designer.design(candidate, graph_context=graph_context, passages=passages)
    if plan is None:
        return None
    has_retrieved = bool(passages)
    retrieved_evidence_ids = {
        evidence_id
        for passage in passages
        if (evidence_id := _passage_field(passage, "evidence_id"))
    }
    for round_index in range(max(0, refine_rounds)):
        if validator is None:
            break
        content_gaps = _plan_content_gaps(
            plan,
            graph_context=graph_context,
            has_retrieved_passages=has_retrieved,
            retrieved_evidence_ids=retrieved_evidence_ids,
        )
        with progress_operation(f"validating plan — round {round_index + 1}/{refine_rounds}"):
            verdict = validator.validate(
                candidate, plan,
                cited_passages=_cited_passages(plan, passages),
                node_labels=graph_context.node_labels,
            )
        if not verdict.flagged and not content_gaps:
            break
        with progress_operation(f"revising flagged plan — round {round_index + 1}/{refine_rounds}"):
            refined = designer.refine(verdict, content_gaps=content_gaps)
        if refined is None:
            break
        plan = refined
    return (
        plan
        if _plan_is_content_complete(
            plan,
            graph_context=graph_context,
            has_retrieved_passages=has_retrieved,
            retrieved_evidence_ids=retrieved_evidence_ids,
        )
        else None
    )


# Post-confirmation experiment stage.
def _experiment_graph_context(
    store: CausalClaimGraphStore, tested_edge: CausalEdge
) -> ExperimentGraphContext:
    """Read the committed-graph neighborhood the Experiment Designer controls for: the
    confounder/mediator/moderator-typed nodes graph-adjacent to the tested edge (they share a
    committed edge with one of its endpoints) plus the edge's own ``confounders`` field. Deterministic
    (nodes in sorted ID order) so graph-derived attribution is reproducible."""
    endpoints = set(tested_edge.source_node_ids) | set(tested_edge.target_node_ids)
    neighbors: set[str] = set()
    for edge in store.edges.values():
        ends = set(edge.source_node_ids) | set(edge.target_node_ids)
        if ends & endpoints:
            neighbors |= ends
    neighbors -= endpoints
    adjacent = tuple(
        store.nodes[nid]
        for nid in sorted(neighbors)
        if nid in store.nodes and store.nodes[nid].type in _GRAPH_CONFOUNDER_TYPES
    )
    edge_confounders = ", ".join(tested_edge.confounders) if tested_edge.confounders else ""
    return ExperimentGraphContext(
        adjacent_nodes=adjacent,
        edge_confounders=edge_confounders,
        node_labels={nid: node.label for nid, node in store.nodes.items()},
    )


def _methods_query(
    tested_edge: CausalEdge,
    graph_context: ExperimentGraphContext,
    *,
    candidate: HypothesisCandidate | None = None,
) -> str:
    """Build the targeted methods, datasets, and baselines query for one hypothesis.

    The optional ``candidate`` keeps legacy direct callers working while allowing production
    retrieval to use the complete proposal rather than only its first edge.
    """
    label_of = (
        _label_of(candidate, graph_context.node_labels)
        if candidate is not None
        else dict(graph_context.node_labels)
    )
    src = " & ".join(label_of.get(s, s) for s in tested_edge.source_node_ids)
    tgt = " & ".join(label_of.get(t, t) for t in tested_edge.target_node_ids)
    primary = (
        f"experimental methods, datasets, and baselines to test whether "
        f"{src} {tested_edge.relation_type} {tgt}"
    )
    if candidate is None:
        return primary
    scaffold = candidate.idea_scaffold or {}
    return (
        f"{primary}. "
        f"All proposed causal edges: {_edges_view(candidate, label_of)}. "
        f"Mechanism chain: {_chain_view(candidate.mechanism_chain)}. "
        f"New concepts: {_nodes_view(candidate.new_nodes)}. "
        f"Hypothesis rationale: {candidate.rationale or '(none)'}. "
        f"Proposed method: {scaffold.get('method', '') or '(none)'}. "
        f"Proposed experiment: {scaffold.get('experiment', '') or '(none)'}"
    )


def run_experiment_stage(
    confirmed: Sequence[HypothesisCandidate],
    *,
    store: CausalClaimGraphStore,
    log: GraphTransactionLog,
    base_ledger: Sequence[str] | set[str],
    designer_factory: Callable[[], ExperimentDesigner],
    validator_seam: ExperimentValidator,
    retrieve_methods: Callable[[str], Sequence[Any]] | None = None,
    refine_rounds: int = 1,
    author: str = "experiment_designer",
    timestamp: str | None = None,
) -> tuple[TransactionResult, ...]:
    """Run post-confirmation experiment design for each confirmed hypothesis.

    Read graph context, retrieve cached methods passages, run the bounded designer-validator loop,
    and commit the experiment delta bound to the hypothesis's committed edge. An empty confirmed
    collection is a no-op.

    Each commit uses a per-hypothesis validator whose ledger is the base commit ledger extended with
    the retrieved methods passages' IDs, so the plan's grounding references can be resolved
    without polluting the main claim-evidence ledger. Returns the commit transactions."""
    from src.validator import GraphDeltaValidator

    base = set(base_ledger)
    transactions: list[TransactionResult] = []
    if not confirmed:
        report_progress("Designing experiments", "no confirmed hypotheses", status="skipped")
    for index, candidate in enumerate(confirmed, 1):
        report_progress("Designing experiments", "hypothesis", current=index, total=len(confirmed))
        if not candidate.new_edges:
            report_progress(None, "hypothesis has no edge to test", status="skipped")
            continue  # nothing to bind a plan to
        tested_edge = candidate.new_edges[0]  # the hypothesis's primary proposed causal edge
        if tested_edge.edge_id not in store.edges:
            report_progress(None, "hypothesis edge was not committed", status="skipped")
            continue  # its Δ^hypothesis did not commit — skip
        graph_context = _experiment_graph_context(store, tested_edge)
        passages: list[Any] = []
        if retrieve_methods is not None:
            with progress_operation("retrieving methods literature"):
                passages = list(
                    retrieve_methods(
                        _methods_query(tested_edge, graph_context, candidate=candidate)
                    )
                )
        plan = design_experiment(
            designer_factory(), validator_seam, candidate,
            graph_context=graph_context, passages=passages, refine_rounds=refine_rounds,
        )
        if plan is None:
            report_progress(None, "no complete experiment plan after review", status="skipped")
            continue  # No parseable, content-complete plan after the bounded review rounds.
        batch_ids = {
            batch_id
            for passage in passages
            if (batch_id := _passage_retrieval_batch_id(passage))
        }
        # The production stage performs one targeted methods retrieval per plan. Stamp only a
        # unique batch and derive grounding status from the final real citations plus raw result
        # presence; overwrite any model-supplied values so provenance remains deterministic.
        grounding_ids = _grounding_evidence_ids(plan)
        grounding_status = (
            ExperimentGroundingStatus.GROUNDED
            if grounding_ids
            else (
                ExperimentGroundingStatus.NO_RELEVANT_METHODS
                if passages
                else ExperimentGroundingStatus.NO_METHODS_RETRIEVED
            )
        )
        plan = plan.model_copy(
            update={
                "retrieval_batch_id": next(iter(batch_ids)) if len(batch_ids) == 1 else "",
                "grounding_status": grounding_status,
            }
        )
        methods_ids = {eid for p in passages if (eid := _passage_field(p, "evidence_id"))}
        validator = GraphDeltaValidator(ledger_evidence_ids=base | methods_ids)
        delta = build_experiment_delta(
            base_graph_hash=store.base_hash,
            hypothesis_id=tested_edge.edge_id,
            experiment_plan=plan,
            author_role=author,
        )
        transactions.append(log.commit(store, delta, validator, author=author, timestamp=timestamp))
        report_progress(
            "Designing experiments", "plan committed" if transactions[-1].accepted else "plan rejected",
            current=index, total=len(confirmed),
            status="completed" if transactions[-1].accepted else "failed",
        )
    return tuple(transactions)
