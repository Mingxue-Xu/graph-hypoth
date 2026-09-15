"""Evidence Reviewer cycle from scored evidence bundles to committed verification deltas.

This module owns deterministic scheduling, signal fusion, confidence calculation, verdict
selection, and overclaim checks. LLM-produced passage signals enter through injected seams, while
all coefficients come from scoring or graph defaults. Shared delta, validation, and transaction
modules own commits and receipts.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from src import graph_config_defaults as gcd
from src.delta import (
    DeltaFamily,
    GraphDeltaProposal,
    VerificationVerdict,
    VerifyPayload,
)
from src.graph_store import (
    CausalClaimGraphStore,
    CausalEdge,
    EdgeStatus,
    EvidenceLink,
    EvidenceRole,
    compute_target_id,
)
from src.retrieval import scoring_defaults as sd
from src.retrieval.association import CandidateEvidence, CandidateEvidenceSet
from src.retrieval.similarity import clip01
from src.state import RetrievedEvidence
from src.transaction_log import GraphTransactionLog, TransactionResult
from src.validator import GraphDeltaValidator


# =====================================================================================
# Scheduler
# =====================================================================================
def u_target(edge: CausalEdge) -> float:
    """Return full uncertainty for unverified edges, otherwise ``1 - confidence``.

    A settled edge with no recorded confidence contributes its full uncertainty (treat a
    missing confidence as 0.0). Clamped so a stray out-of-range confidence cannot escape.
    """
    if edge.status == EdgeStatus.UNVERIFIED:
        return 1.0
    return clip01(1.0 - (edge.confidence if edge.confidence is not None else 0.0))


def gap_evidence(required_roles: Collection[str], observed_roles: Collection[str]) -> float:
    """Return the clamped fraction of required evidence roles that are missing.

    An empty required
    set means no gap (0.0) and avoids a divide-by-zero.
    """
    required = set(required_roles)
    if not required:
        return 0.0
    missing = required - set(observed_roles)
    return clip01(len(missing) / len(required))


def selection_score(
    *,
    p_sched: float,
    u_target: float,
    gap_evidence: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Return the weighted, clipped scheduling score.

    Exactly three weights are used. Cost is a hard budget guard, not a weighted term.
    """
    weights = weights if weights is not None else gcd.SEL_WEIGHTS
    return clip01(
        weights["p"] * p_sched + weights["u"] * u_target + weights["g"] * gap_evidence
    )


@dataclass(frozen=True)
class SchedulerCandidate:
    """A target with its precomputed selection score, cost, and selectability."""

    target_id: str
    selection_score: float
    cost: float
    selectable: bool = True


def select_work_set(
    candidates: Sequence[SchedulerCandidate],
    *,
    budget: float,
    theta_sel: float | None = None,
) -> tuple[SchedulerCandidate, ...]:
    """Select candidates that clear selectability, score, and per-target cost guards.

    The budget is a per-target hard guard, not a cumulative knapsack. Input order is preserved.
    """
    theta_sel = theta_sel if theta_sel is not None else gcd.THETA_SEL
    return tuple(
        c
        for c in candidates
        if c.selectable and c.selection_score >= theta_sel and c.cost <= budget
    )


def continue_loop(
    *,
    selected_count: int,
    iteration: int,
    stall: int,
    clarification_required: bool = False,
    n_max: int | None = None,
    n_stall: int | None = None,
) -> bool:
    """``Continue_i = 1[|P_i|>0]·1[i<N_max]·1[¬clarificationRequired]·1[stall<N_stall]``.

    The nonempty-selectable-set ``|P_i|>0`` is the SUBSTANTIVE stop; ``N_max`` is only a
    safety stop. An empty selected set stops the loop regardless of iteration, stall, or
    clarification. The caller maintains the stall counter.
    """
    n_max = n_max if n_max is not None else gcd.N_MAX
    n_stall = n_stall if n_stall is not None else gcd.N_STALL
    return (
        selected_count > 0
        and iteration < n_max
        and not clarification_required
        and stall < n_stall
    )


# =====================================================================================
# Evidence-review signal fusion
# =====================================================================================
class RelationLabel(StrEnum):
    """Closed relation-label set produced by coherence appraisal.

    It is distinct from committed verification verdicts; the mapping tables below bridge them.
    """

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    QUALIFIES = "qualifies"
    MECHANISM = "mechanism"
    CONFOUNDER = "confounder"
    NOT_RELEVANT = "not relevant"


@dataclass(frozen=True)
class CoherenceSignal:
    """Agent-produced coherence signals for one evidence item.

    ``entailment`` and ``contradiction_risk`` are per relation
    label ``r`` (keyed by ``RelationLabel`` value); ``context_fit`` and ``construct_match``
    are label-independent. All on ``[0,1]`` with rubric anchors; a missing label reads 0.0."""

    entailment: Mapping[str, float]
    contradiction_risk: Mapping[str, float]
    context_fit: float
    construct_match: float


@dataclass(frozen=True)
class MethodsSignal:
    """Agent-produced method and measurement signals for one evidence item.

    All values lie in ``[0,1]`` and include explicit confounder adjustment.
    """

    design_strength: float
    measurement_validity: float
    population_fit: float
    confounder_adjustment: float
    bias_risk: float


# Neutral fallbacks for a passage the Evidence Reviewer omitted from its output (graceful degrade —
# ``relation_label`` of an all-zero coherence reads ``supports`` at magnitude 0, contributing
# nothing once the methods/quote terms are also 0).
_EMPTY_COHERENCE = CoherenceSignal(
    entailment={}, contradiction_risk={}, context_fit=0.0, construct_match=0.0
)
_EMPTY_METHODS = MethodsSignal(0.0, 0.0, 0.0, 0.0, 0.0)


def rel_score(
    label: RelationLabel, signal: CoherenceSignal, *, weights: dict[str, float] | None = None
) -> float:
    """``RelScore(r,e,t) = clip01(w_entail·Entailment_r + w_ctx·ContextFit
    + w_construct·ConstructMatch − w_contra·ContradictionRisk_r)`` (F-RELSCORE)."""
    weights = weights if weights is not None else sd.RELSCORE_WEIGHTS
    r = label.value
    return clip01(
        weights["entail"] * signal.entailment.get(r, 0.0)
        + weights["ctx"] * signal.context_fit
        + weights["construct"] * signal.construct_match
        - weights["contra"] * signal.contradiction_risk.get(r, 0.0)
    )


def relation_label(
    signal: CoherenceSignal, *, weights: dict[str, float] | None = None
) -> RelationLabel:
    """``Rel(e,t) = argmax_r RelScore(r,e,t)`` over the closed set (F-REL).

    Ties break by fixed ``RelationLabel`` declaration order (only a strictly greater score
    displaces the incumbent), so the argmax is reproducible."""
    best_label = RelationLabel.SUPPORTS
    best_score = float("-inf")
    for label in RelationLabel:
        score = rel_score(label, signal, weights=weights)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def methods_score(
    signal: MethodsSignal, *, weights: dict[str, float] | None = None
) -> float:
    """``M(e,t) = clip01(w_design·Design + w_measure·Measure + w_pop·Pop
    + w_adjust·Adjust − w_bias·Bias)`` (F-M)."""
    weights = weights if weights is not None else sd.METHODS_WEIGHTS
    return clip01(
        weights["design"] * signal.design_strength
        + weights["measure"] * signal.measurement_validity
        + weights["pop"] * signal.population_fit
        + weights["adjust"] * signal.confounder_adjustment
        - weights["bias"] * signal.bias_risk
    )


# =====================================================================================
# Relation-to-verdict support and counterevidence mappings
# =====================================================================================
# ``mechanism``
# ∈ Support(support); `not relevant` maps to ∅ (in no Support/Counter set). `insufficient`
# is the floor fallback and has no support or counterevidence set.
_SUPPORT_SETS: dict[VerificationVerdict, frozenset[RelationLabel]] = {
    VerificationVerdict.SUPPORT: frozenset({RelationLabel.SUPPORTS, RelationLabel.MECHANISM}),
    VerificationVerdict.CONTRADICT: frozenset({RelationLabel.CONTRADICTS}),
    VerificationVerdict.QUALIFY: frozenset({RelationLabel.QUALIFIES}),
    VerificationVerdict.NOT_CAUSAL: frozenset({RelationLabel.CONFOUNDER}),
}
_COUNTER_SETS: dict[VerificationVerdict, frozenset[RelationLabel]] = {
    VerificationVerdict.SUPPORT: frozenset({RelationLabel.CONTRADICTS, RelationLabel.CONFOUNDER}),
    VerificationVerdict.CONTRADICT: frozenset({RelationLabel.SUPPORTS, RelationLabel.MECHANISM}),
    VerificationVerdict.QUALIFY: frozenset(),
    VerificationVerdict.NOT_CAUSAL: frozenset({RelationLabel.SUPPORTS, RelationLabel.MECHANISM}),
}

# Candidate verdicts; ``insufficient`` is the floor rather than an argmax candidate. Tuple order
# provides the deterministic label tie-break.
CANDIDATE_VERDICTS: tuple[VerificationVerdict, ...] = (
    VerificationVerdict.SUPPORT,
    VerificationVerdict.CONTRADICT,
    VerificationVerdict.QUALIFY,
    VerificationVerdict.NOT_CAUSAL,
)


def supports_verdict(rel: RelationLabel, verdict: VerificationVerdict) -> bool:
    """Return whether a relation supports the candidate verdict."""
    return rel in _SUPPORT_SETS.get(verdict, frozenset())


def counters_verdict(rel: RelationLabel, verdict: VerificationVerdict) -> bool:
    """Return whether a relation counters the candidate verdict."""
    return rel in _COUNTER_SETS.get(verdict, frozenset())


# =====================================================================================
# Per-evidence verdict strength
# =====================================================================================
def evidence_strength(
    *,
    rel: RelationLabel,
    methods_score: float,
    quote_quality: float,
    verdict: VerificationVerdict,
    weights: dict[str, float] | None = None,
) -> float:
    """``s_e(v,t) = clip01(w_rel·1[Rel∈Support(v)] + w_method·M(e,t) + w_quote·Q(e,t))``
    The relation indicator comes from the mapping above; quote quality is computed upstream.
    """
    weights = weights if weights is not None else sd.SE_WEIGHTS
    indicator = 1.0 if supports_verdict(rel, verdict) else 0.0
    return clip01(
        weights["rel"] * indicator
        + weights["method"] * methods_score
        + weights["quote"] * quote_quality
    )


def quote_quality(
    *,
    has_span: bool,
    context_coverage: float = 0.0,
    target_mention: float = 0.0,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute deterministic quote quality.

    MVP realization over the QuoteVerification span signal: a present span sets ExactSpan
    and QuoteVerified to 1 (and SummaryFallback to 0); an absent span applies the
    summary-fallback penalty. Context coverage and target mention default conservatively to zero.
    All weights come from ``scoring_defaults.QUOTE_WEIGHTS``.
    """
    weights = weights if weights is not None else sd.QUOTE_WEIGHTS
    exact_span = 1.0 if has_span else 0.0
    quote_verified = 1.0 if has_span else 0.0
    summary_fallback = 0.0 if has_span else 1.0
    return clip01(
        weights["span"] * exact_span
        + weights["context"] * context_coverage
        + weights["target"] * target_mention
        + weights["status"] * quote_verified
        - weights["summary"] * summary_fallback
    )


# =====================================================================================
# Verdict confidence
# =====================================================================================
@dataclass(frozen=True)
class AppraisedEvidence:
    """One evidence item after the deterministic appraiser fusion: its relation label
    relation label, methods score, quote quality, and source identity used for distinct-source
    breadth. Verdict strength is derived separately for each candidate verdict.
    """

    evidence_id: str
    source_id: str
    rel: RelationLabel
    methods_score: float
    quote_quality: float


def _se(evidence: AppraisedEvidence, verdict: VerificationVerdict) -> float:
    return evidence_strength(
        rel=evidence.rel,
        methods_score=evidence.methods_score,
        quote_quality=evidence.quote_quality,
        verdict=verdict,
    )


def supporting_set(
    evidences: Sequence[AppraisedEvidence], verdict: VerificationVerdict
) -> tuple[AppraisedEvidence, ...]:
    """``S_v(t) = { e : Rel(e,t) ∈ Support(v) AND s_e(v,t) > 0 }`` — the by-relation
    supporting set so the relation bonus is meaningful (F-SV,  refinement)."""
    return tuple(
        e for e in evidences if supports_verdict(e.rel, verdict) and _se(e, verdict) > 0.0
    )


def top3(
    evidences: Sequence[AppraisedEvidence],
    verdict: VerificationVerdict,
    *,
    k_conf: int | None = None,
) -> float:
    """``Top3_v(t)`` = mean of the strongest ``min(K_conf,|S_v|)`` ``s_e`` values, else 0
    when ``S_v`` is empty (F-TOP3)."""
    k_conf = k_conf if k_conf is not None else sd.K_CONF
    strengths = sorted((_se(e, verdict) for e in supporting_set(evidences, verdict)), reverse=True)
    if not strengths:
        return 0.0
    top = strengths[:k_conf]
    return sum(top) / len(top)


def support_strength(
    evidences: Sequence[AppraisedEvidence], verdict: VerificationVerdict
) -> float:
    """``support_strength_v(t)`` = sum of ``s_e(v,t)`` over ``S_v(t)`` (F-BALANCE input)."""
    return sum(_se(e, verdict) for e in supporting_set(evidences, verdict))


def counter_strength(
    evidences: Sequence[AppraisedEvidence], verdict: VerificationVerdict
) -> float:
    """``counter_strength_v(t)`` = sum of ``s_e(v,t)`` over ``{ e : Rel(e,t) ∈ Counter(v) }``
    (the relation indicator is 0 there, so each contributes its methods+quote magnitude)."""
    return sum(_se(e, verdict) for e in evidences if counters_verdict(e.rel, verdict))


def balance(
    evidences: Sequence[AppraisedEvidence],
    verdict: VerificationVerdict,
    *,
    epsilon: float | None = None,
) -> float:
    """``Balance_v(t) = clip01(support / (support + counter + epsilon))`` (F-BALANCE)."""
    epsilon = epsilon if epsilon is not None else sd.EPSILON
    support = support_strength(evidences, verdict)
    counter = counter_strength(evidences, verdict)
    return clip01(support / (support + counter + epsilon))


def breadth(
    evidences: Sequence[AppraisedEvidence],
    verdict: VerificationVerdict,
    *,
    k_breadth: int | None = None,
) -> float:
    """``Breadth_v(t) = min(1, |distinct sources in S_v(t)| / K_breadth)`` (F-CONFINPUTS)."""
    k_breadth = k_breadth if k_breadth is not None else sd.K_BREADTH
    sources = {e.source_id for e in supporting_set(evidences, verdict)}
    return min(1.0, len(sources) / k_breadth)


def open_risk_term(open_risks: Collection[str], *, k_risk: int | None = None) -> float:
    """``OpenRisk_v(t) = min(1, |open_risks(t)| / K_risk)`` (F-CONFINPUTS); target-level."""
    k_risk = k_risk if k_risk is not None else sd.K_RISK
    return min(1.0, len(open_risks) / k_risk)


def limits_term(qualifiers: Collection[str], *, k_limits: int | None = None) -> float:
    """``Limits_v(t) = min(1, |qualifiers(t)| / K_limits)`` (F-CONFINPUTS); target-level."""
    k_limits = k_limits if k_limits is not None else sd.K_LIMITS
    return min(1.0, len(qualifiers) / k_limits)


def confidence(
    evidences: Sequence[AppraisedEvidence],
    verdict: VerificationVerdict,
    *,
    open_risks: Collection[str] = (),
    qualifiers: Collection[str] = (),
    weights: dict[str, float] | None = None,
) -> float:
    """``Conf(v,t) = clip01(0.55·Top3 + 0.25·Balance + 0.20·Breadth − 0.25·OpenRisk
    − 0.15·Limits)`` (F-CONF — the only formula with default coefficients, DOC-PINNED)."""
    weights = weights if weights is not None else sd.CONF_WEIGHTS
    return clip01(
        weights["top3"] * top3(evidences, verdict)
        + weights["balance"] * balance(evidences, verdict)
        + weights["breadth"] * breadth(evidences, verdict)
        + weights["open_risk"] * open_risk_term(open_risks)
        + weights["limits"] * limits_term(qualifiers)
    )


# =====================================================================================
# Deterministic verdict selection with an insufficient floor and near-tie hold
# =====================================================================================
@dataclass(frozen=True)
class VerdictSelection:
    """Selected verdict, confidence, per-verdict scores, and abstention metadata."""

    verdict: VerificationVerdict
    confidence: float
    per_verdict_confidence: dict[VerificationVerdict, float]
    held: bool
    hold_reason: str | None


def select_verdict(
    evidences: Sequence[AppraisedEvidence],
    *,
    open_risks: Collection[str] = (),
    qualifiers: Collection[str] = (),
    theta_verdict: float | None = None,
    delta_margin: float | None = None,
) -> VerdictSelection:
    """Select the highest-confidence verdict with an insufficient floor and near-tie hold.

    Order: compute per-candidate ``Conf``; if every ``S_v`` is empty OR ``max Conf <
    theta_verdict`` → ``insufficient`` (one-sided floor); else if ``top1Conf − top2Conf <
    delta_margin`` → ``insufficient`` (two-sided near-tie hold, re-verifiable, no
    new enum). Ties break by ``support_strength`` then fixed label order (CANDIDATE_VERDICTS).
    """
    theta_verdict = theta_verdict if theta_verdict is not None else sd.THETA_VERDICT
    delta_margin = delta_margin if delta_margin is not None else sd.DELTA_MARGIN

    per_conf = {
        verdict: confidence(
            evidences, verdict, open_risks=open_risks, qualifiers=qualifiers
        )
        for verdict in CANDIDATE_VERDICTS
    }
    strengths = {verdict: support_strength(evidences, verdict) for verdict in CANDIDATE_VERDICTS}
    all_empty = all(not supporting_set(evidences, verdict) for verdict in CANDIDATE_VERDICTS)

    # Rank by (Conf desc, support_strength desc, fixed label order).
    ranked = sorted(
        CANDIDATE_VERDICTS,
        key=lambda verdict: (-per_conf[verdict], -strengths[verdict], CANDIDATE_VERDICTS.index(verdict)),
    )
    top1, top2 = ranked[0], ranked[1]
    top1_conf = per_conf[top1]

    held, reason, committed = False, None, top1
    if all_empty or top1_conf < theta_verdict:
        held, reason, committed = True, "floor", VerificationVerdict.INSUFFICIENT
    elif top1_conf - per_conf[top2] < delta_margin:
        held, reason, committed = True, "near_tie", VerificationVerdict.INSUFFICIENT

    return VerdictSelection(
        verdict=committed,
        confidence=top1_conf,
        per_verdict_confidence=per_conf,
        held=held,
        hold_reason=reason,
    )


# =====================================================================================
# Overclaim guard
# =====================================================================================
# A settled verdict
# requires evidence carrying its primary relation-role; `insufficient` is an abstention
# with no required role (never an overclaim). The schema is recalibratable, not doc truth.
_REQUIRED_ROLES: dict[VerificationVerdict, frozenset[EvidenceRole]] = {
    VerificationVerdict.SUPPORT: frozenset({EvidenceRole.SUPPORT}),
    VerificationVerdict.CONTRADICT: frozenset({EvidenceRole.CONTRADICTION}),
    VerificationVerdict.QUALIFY: frozenset({EvidenceRole.QUALIFICATION}),
    VerificationVerdict.NOT_CAUSAL: frozenset({EvidenceRole.CONFOUNDER}),
    VerificationVerdict.INSUFFICIENT: frozenset(),
}


def required_roles_for_verdict(verdict: VerificationVerdict) -> frozenset[EvidenceRole]:
    """Return the evidence-role set a verdict must cover."""
    return _REQUIRED_ROLES[verdict]


def observed_roles(evidence_links: Sequence[EvidenceLink]) -> frozenset[EvidenceRole]:
    """``observed_roles(Δ^verify)`` — the roles present on the proposed evidence links."""
    return frozenset(link.evidence_role for link in evidence_links)


def overclaim_gap(
    required_roles: Collection[str], observed_roles: Collection[str]
) -> bool:
    """Return true when any required evidence role is missing.

    ``True`` (a gap) when any required role is missing from the observed set — the caller
    MUST route such a proposal to revision/semantic review, never hide it as an opaque
    validator rejection."""
    return not (set(required_roles) <= set(observed_roles))


# =====================================================================================
# Verification builders
# =====================================================================================
def build_evidence_link(
    *,
    target_id: str,
    verification_task_id: str,
    evidence_id: str,
    evidence_role: EvidenceRole,
    signal: Any,
    retrieval_event_id: str,
    trust_tier: str,
    committed_transaction_id: str = "tx-uncommitted",
) -> EvidenceLink:
    """Build the committed bridge from a graph target to a ledger record.

    ``signal`` nests the entire deterministic ``EvidenceSignalBundle``; its quote span is read via
    ``link.signal['matched_quote_span']`` rather than copied to the link. A
    dataclass bundle is captured by value as a plain dict so the link stays JSON-hashable for
    receipt replay. ``committed_transaction_id`` is a placeholder until the validator stamps
    the real ``tx_id`` at ``apply`` time (mirroring the validator's default)."""
    signal_dict = asdict(signal) if is_dataclass(signal) and not isinstance(signal, type) else signal
    return EvidenceLink(
        target_id=target_id,
        verification_task_id=verification_task_id,
        evidence_id=evidence_id,
        evidence_role=evidence_role,
        signal=signal_dict,
        retrieval_event_id=retrieval_event_id,
        committed_transaction_id=committed_transaction_id,
        trust_tier=trust_tier,
    )


def build_verify_delta(
    *,
    base_graph_hash: str,
    edge_id: str,
    evidence_role: EvidenceRole,
    verdict: VerificationVerdict,
    confidence: float | None,
    evidence_links: Sequence[EvidenceLink],
    open_risks: Sequence[str] = (),
    author_role: str = "causal_evidence_verifier",
    rationale: str = "",
    provenance: list[dict[str, Any]] | None = None,
) -> GraphDeltaProposal:
    """Package evidence links, verdict, confidence, and risks into a verification delta.

    ``verdict`` is computed by ``select_verdict`` rather than selected by an agent;
    ``referenced_evidence_ids`` mirrors the links so the envelope is audit-complete."""
    return GraphDeltaProposal(
        family=DeltaFamily.VERIFY,
        base_graph_hash=base_graph_hash,
        payload=VerifyPayload(
            edge_id=edge_id,
            evidence_role=evidence_role,
            verdict=verdict,
            confidence=confidence,
            evidence_links=list(evidence_links),
            open_risks=list(open_risks),
        ),
        author_role=author_role,
        rationale=rationale,
        provenance=provenance or [],
        referenced_evidence_ids=[link.evidence_id for link in evidence_links],
    )


class VerificationResult(BaseModel):
    """The richer, uncommitted result wrapper. It carries the delta fields plus
    the derivable-but-uncommitted ``counterevidence_ids`` and agent-emitted ``qualifiers``
    these stay on the wrapper rather than the delta. ``held`` and ``hold_reason`` surface an
    abstention for human review.
    """

    target_id: str
    verdict: VerificationVerdict
    confidence: float | None
    evidence_ids: list[str] = Field(default_factory=list)
    counterevidence_ids: list[str] = Field(default_factory=list)
    qualifiers: list[str] = Field(default_factory=list)
    open_risks: list[str] = Field(default_factory=list)
    proposed_delta: GraphDeltaProposal
    held: bool = False
    hold_reason: str | None = None


def build_verification_result(
    *,
    proposed_delta: GraphDeltaProposal,
    qualifiers: Sequence[str] = (),
    held: bool = False,
    hold_reason: str | None = None,
) -> VerificationResult:
    """Wrap a proposed verification delta in its ``VerificationResult``.

    ``evidence_ids`` / ``counterevidence_ids`` partition the committed links by their
    ``evidence_role``: the ``contradiction`` role is counterevidence,
    every other role is supporting/qualifying evidence. ``qualifiers`` is the verifier's
    agent-emitted list; ``open_risks`` reads off the committed delta."""
    payload = proposed_delta.payload
    if not isinstance(payload, VerifyPayload):
        raise ValueError("build_verification_result requires a Δ^verify proposal")
    evidence_ids = [
        link.evidence_id
        for link in payload.evidence_links
        if link.evidence_role != EvidenceRole.CONTRADICTION
    ]
    counterevidence_ids = [
        link.evidence_id
        for link in payload.evidence_links
        if link.evidence_role == EvidenceRole.CONTRADICTION
    ]
    return VerificationResult(
        target_id=payload.target_id,
        verdict=payload.verdict,
        confidence=payload.confidence,
        evidence_ids=evidence_ids,
        counterevidence_ids=counterevidence_ids,
        qualifiers=list(qualifiers),
        open_risks=list(payload.open_risks),
        proposed_delta=proposed_delta,
        held=held,
        hold_reason=hold_reason,
    )


# =====================================================================================
# Scheduling and verification task
# =====================================================================================
_RESOLVED_STATUSES = frozenset(
    {EdgeStatus.SUPPORTED, EdgeStatus.CONTRADICTED, EdgeStatus.QUALIFIED, EdgeStatus.NOT_CAUSAL}
)


@dataclass(frozen=True)
class VerificationTask:
    """Projection over an edge and evidence role consumed by verification."""

    verification_task_id: str
    edge_id: str
    evidence_role: EvidenceRole
    question: str = ""
    criteria: str = ""

    @property
    def target_id(self) -> str:
        return compute_target_id(self.edge_id, self.evidence_role)


def is_selectable(edge: CausalEdge) -> bool:
    """Return whether an edge remains eligible for verification.

    A settled edge drops out; an
    ``unverified`` or (re-verifiable) ``insufficient`` edge stays selectable. Out-of-scope /
    clarification-blocked exclusions are layered by the caller."""
    return edge.status not in _RESOLVED_STATUSES


# =====================================================================================
# Bounded-judgment agent seams (Evidence Reviewer input contract). The deterministic core is independent of these;
# the default suite injects fakes, the live e2e injects the LLM-backed agents below.
# =====================================================================================
@dataclass(frozen=True)
class VerifierAppraisal:
    """Agent-produced open risks and qualifiers; never a verdict label."""

    open_risks: tuple[str, ...] = ()
    qualifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class EdgeReview:
    """The Evidence Reviewer's per-edge output. One skeptical call over all K triaged
    passages replaces the former separate coherence, methods, and verifier calls. ``coherence`` and
    ``methods`` map each passage's ``evidence_id`` to its per-passage sub-signals, and
    ``appraisal`` carries the edge-level ``open_risks``/``qualifiers`` (never a verdict). The
    deterministic fusion (``relation_label`` / ``methods_score`` / ``select_verdict``)
    consumes these unchanged."""

    coherence: Mapping[str, CoherenceSignal]
    methods: Mapping[str, MethodsSignal]
    appraisal: VerifierAppraisal


class EvidenceReviewer(Protocol):
    """Evidence Reviewer: ONE skeptical-tier call per edge over all triaged passages -> per-passage
    coherence + methods sub-signals + edge-level ``open_risks``/``qualifiers``; NEVER a
    verdict, which is decided deterministically downstream."""

    def review(
        self, task: VerificationTask, evidences: Sequence[RetrievedEvidence]
    ) -> EdgeReview: ...


# =====================================================================================
# Triage hand-off. ``AssociationScorer.triage`` owns scoring and top-K routing; this is the thin
# consumer-side view used by evidence review.
# =====================================================================================
@dataclass(frozen=True)
class TriageSignal:
    """The Paper Triage hand-off this cycle consumes: the top-K gated candidates forwarded
    for appraisal plus the routing metadata the cycle reads."""

    forwarded: tuple[CandidateEvidence, ...] = ()
    triage_decision: str = "forward"
    reason: str = ""
    uncertainty_flag: bool = False


def build_triage_signal(
    candidates: Sequence[CandidateEvidence] | CandidateEvidenceSet,
    *,
    triage_decision: str = "forward",
    reason: str = "",
    uncertainty_flag: bool = False,
) -> TriageSignal:
    """Wrap top-K association output in the cycle's ``TriageSignal`` hand-off."""
    forwarded = candidates.items if isinstance(candidates, CandidateEvidenceSet) else tuple(candidates)
    return TriageSignal(
        forwarded=tuple(forwarded),
        triage_decision=triage_decision,
        reason=reason,
        uncertainty_flag=uncertainty_flag,
    )


# =====================================================================================
# Orchestration — run_verification_cycle (scheduler-fed target -> verdict -> Δ^verify)
# =====================================================================================
@dataclass(frozen=True)
class VerificationCycleResult:
    """The auditable outcome of verifying ONE target: the un-committed ``VerificationResult``,
    the F-VSTAR ``selection``, the overclaim ``gap`` flag, and the committed transaction (or
    ``None`` when the overclaim guard routed the proposal to revision — )."""

    result: VerificationResult
    selection: VerdictSelection
    gap: bool
    transaction: TransactionResult | None
    store: CausalClaimGraphStore
    log: GraphTransactionLog


def _resolve_candidate_role(candidate: CandidateEvidence, default: EvidenceRole) -> EvidenceRole:
    """The candidate's own evidence role (from its scoring bundle), normalized to the enum;
    falls back to ``default`` (the task role) when absent or unrecognized."""
    raw = getattr(candidate.bundle, "evidence_role", None)
    if raw is None:
        return default
    if isinstance(raw, EvidenceRole):
        return raw
    try:
        return EvidenceRole(str(raw))
    except ValueError:
        return default


def run_verification_cycle(
    task: VerificationTask,
    candidates: Sequence[CandidateEvidence] | TriageSignal,
    *,
    evidence_reviewer: EvidenceReviewer,
    store: CausalClaimGraphStore,
    log: GraphTransactionLog,
    validator: GraphDeltaValidator,
    author: str = "causal_evidence_verifier",
    timestamp: str | None = None,
) -> VerificationCycleResult:
    """Verify one target: the Evidence Reviewer appraises all triaged passages in one call,
    the fusion runs deterministically, a verdict is selected, and a verification delta is packaged
    and committed unless the overclaim guard finds a missing required role. Such proposals are
    routed to revision rather than committed.
    The reviewer is injected; the fusion is deterministic.

    ``candidates`` is the triaged top-K, supplied either as a ``TriageSignal`` or directly as a
    ``CandidateEvidence`` sequence.
    """
    if isinstance(candidates, TriageSignal):
        candidates = candidates.forwarded

    # Each candidate keeps its OWN role (from the scoring bundle), so a single per-edge
    # cycle can verify support and counterevidence together: links carry per-candidate
    # roles so ``observed_roles`` covers the verdict's required role. The Evidence Reviewer grades
    # every passage against the edge's single target relationship in ONE call.
    roled = [(c, _resolve_candidate_role(c, task.evidence_role)) for c in candidates]

    # One Evidence Reviewer call covers all triaged passages and returns passage-level coherence
    # and methods signals plus edge-level open risks and qualifiers.
    review = evidence_reviewer.review(task, [candidate.evidence for candidate, _ in roled])
    appraisal = review.appraisal

    appraised: list[AppraisedEvidence] = []
    for candidate, _role in roled:
        evidence = candidate.evidence
        coherence = review.coherence.get(evidence.evidence_id, _EMPTY_COHERENCE)
        methods = review.methods.get(evidence.evidence_id, _EMPTY_METHODS)
        appraised.append(
            AppraisedEvidence(
                evidence_id=evidence.evidence_id,
                source_id=evidence.source_id or evidence.source,
                rel=relation_label(coherence),
                methods_score=methods_score(methods),
                quote_quality=quote_quality(
                    has_span=candidate.bundle.matched_quote_span is not None
                ),
            )
        )

    selection = select_verdict(
        appraised, open_risks=appraisal.open_risks, qualifiers=appraisal.qualifiers
    )

    links = [
        build_evidence_link(
            target_id=compute_target_id(task.edge_id, role),
            verification_task_id=task.verification_task_id,
            evidence_id=candidate.evidence.evidence_id,
            evidence_role=role,
            signal=candidate.bundle,
            retrieval_event_id=candidate.evidence.tool_call_id,
            trust_tier=candidate.evidence.trust_tier,
        )
        for candidate, role in roled
    ]
    delta = build_verify_delta(
        base_graph_hash=store.base_hash,
        edge_id=task.edge_id,
        evidence_role=task.evidence_role,
        verdict=selection.verdict,
        confidence=selection.confidence,
        evidence_links=links,
        open_risks=list(appraisal.open_risks),
        rationale=f"evidence review of target {task.target_id}",
    )
    gap = overclaim_gap(
        required_roles_for_verdict(selection.verdict), observed_roles(links)
    )
    result = build_verification_result(
        proposed_delta=delta,
        qualifiers=appraisal.qualifiers,
        held=selection.held,
        hold_reason=selection.hold_reason,
    )

    transaction: TransactionResult | None = None
    if not gap:
        transaction = log.commit(store, delta, validator, author=author, timestamp=timestamp)

    return VerificationCycleResult(
        result=result, selection=selection, gap=gap,
        transaction=transaction, store=store, log=log,
    )


# =====================================================================================
# Live LLM-backed agents (the `live` path; fixtured with a fake backend in unit tests).
# Each live seam follows the Evidence Reviewer input contract: it sees only its contract's
# content/labels, never the deterministic association floats, other agents' raw outputs, or
# any verdict label, and parses STRICT JSON with graceful degradation.
# =====================================================================================
def _clip_signal(value: Any) -> float:
    """Coerce an agent-emitted scalar to a clipped ``[0,1]`` float; non-numeric -> 0.0."""
    try:
        return clip01(float(value))
    except (TypeError, ValueError):
        return 0.0


def _backend_json(backend: Any, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """Run a CAMEL ``.run(messages)`` backend and parse STRICT JSON; degrade to ``{}``."""
    from src.camel_adapter import backend_json

    return backend_json(backend, system_prompt, user_prompt)


def _target_context(task: VerificationTask) -> str:
    parts = [f"evidence_role: {task.evidence_role.value}"]
    if task.question:
        parts.append(f"question: {task.question}")
    if task.criteria:
        parts.append(f"criteria: {task.criteria}")
    return "\n".join(parts)


def _passage_for(evidence: RetrievedEvidence) -> str:
    """The text the appraiser judges: the fuller ``full_text_excerpt`` body when retrieval
    stored one, else the compact verified ``quote``."""
    excerpt = evidence.metadata.get("full_text_excerpt")
    if isinstance(excerpt, str) and excerpt:
        return excerpt
    return evidence.quote


# One Evidence Reviewer call per edge grades relation and methods signals for every passage and
# lists edge-level open risks and qualifiers, never a verdict. Kept as a plain string because its
# <output_format> shows literal JSON braces.
_EVIDENCE_REVIEWER_SYSTEM_PROMPT = """<role>
You are the evidence reviewer for one causal relationship in a claim graph. In a single call you appraise every retrieved passage for that relationship, then list the unresolved risks and scope qualifiers across them.
</role>

<task>
You are given the target relationship (cause->effect, evidence role, question, criteria) and all triaged passages, numbered, each marked with its evidence_id. Do three things:
1. For each passage, grade how it relates to the relationship: entailment and contradiction_risk are objects mapping each relation label in ["supports", "contradicts", "qualifies", "mechanism", "confounder", "not relevant"] to a number in [0,1]; context_fit and construct_match are numbers in [0,1].
2. For each passage, grade its study design quality: design_strength, measurement_validity, population_fit, confounder_adjustment, bias_risk — each a number in [0,1], using the per-axis anchors in <design_quality_anchors> (higher design/adjustment = stronger; higher bias_risk = weaker).
3. Across the passages, list only (1) unresolved risks and (2) scope qualifiers for the relationship, as open_risks and qualifiers — each a list of short strings.
</task>

<design_quality_anchors>
Anchors for the five study-design-quality axes graded in task step 2. They draw on three appraisal frameworks — RoB 2 (randomized trials), ROBINS-I (non-randomized studies of interventions), and GRADE (certainty of a body of evidence) — but naming them is optional context: each anchor below is self-contained, so grade sensibly from the anchor text alone even if the frameworks are unfamiliar. Rate the study the passage describes, only to the extent the passage reveals relevant information: score what is reported, not what may be true of the underlying study. When a passage omits what an axis needs (e.g., blinding, attrition, adjustment method), treat that axis as uncertain and grade near the midpoint rather than assuming either high or low quality. On each line, near-0 names the 0 end and near-1 names the 1 end.
1. design_strength — near-0: design is weak for causal inference (cross-sectional, case series, uncontrolled before-after), like observational evidence starting at "low" certainty in GRADE | near-1: design is strong for causal inference (well-conducted RCT, or natural experiment with strong quasi-randomization), like RCT evidence starting at "high" certainty.
2. measurement_validity — near-0: outcome and/or exposure measurement is prone to substantial bias (assessors aware of intervention status, subjective measures without blinding, unreliable or unvalidated instruments) | near-1: measurement is valid and unlikely to be biased (blinded assessment, objective measures, validated instruments, or measurement errors non-differential across groups).
3. population_fit — near-0: study population, setting, or outcome definition is poorly matched to the claim's target (substantial differences in demographics, clinical context, intervention delivery, or outcome definitions) | near-1: study population, intervention, comparator, and outcome directly match the claim's target (same or highly similar to what the claim addresses).
4. confounder_adjustment — near-0: confounding is poorly identified or uncontrolled (key prognostic factors unmeasured, no adjustment, or inadequate methods such as stratification missing important confounders) | near-1: confounding is well identified and appropriately controlled (relevant confounders measured and adjusted via multivariable regression, propensity scores, instrumental variables, or randomization).
5. bias_risk (inverted — higher means weaker evidence) — near-0: low overall risk of bias, no serious concerns across bias domains, evidence trustworthy | near-1: high overall risk of bias, serious or critical concerns in one or more domains, evidence may be substantially distorted.
</design_quality_anchors>

<rules>
1. Judge every per-passage grade only from that passage and the relationship described — grade each passage as if it were the only one retrieved; one passage's grades must not be influenced by any other passage.
2. Only open_risks and qualifiers may draw on the cross-passage view: agreement, contradiction, and gaps across the passages.
3. State no conclusion, verdict, or label — return only the graded sub-signals, the risks, and the qualifiers; the final verdict is decided deterministically downstream.
4. Return exactly one passages entry per input passage, in input order, with evidence_id copied verbatim from the passage marker.
</rules>

<output_format>
Return STRICT JSON only (no markdown):
{"passages": [{"evidence_id": "...", "entailment": {...}, "contradiction_risk": {...}, "context_fit": 0..1, "construct_match": 0..1, "design_strength": 0..1, "measurement_validity": 0..1, "population_fit": 0..1, "confounder_adjustment": 0..1, "bias_risk": 0..1}], "open_risks": ["..."], "qualifiers": ["..."]}
</output_format>"""


def _evidence_reviewer_user_prompt(
    task: VerificationTask, evidences: Sequence[RetrievedEvidence]
) -> str:
    """Render the edge target context and all triaged passages, each
    marked with its ``evidence_id`` so the deterministic core can reassociate grades."""
    blocks = [
        f"### Passage {i + 1} — evidence_id: {ev.evidence_id}\n"
        f"Evidence title: {ev.title}\nPassage:\n{_passage_for(ev)}"
        for i, ev in enumerate(evidences)
    ]
    passages = "\n\n".join(blocks)
    return (
        f"<target_relationship>\n{_target_context(task)}\n</target_relationship>\n\n"
        f'<passages count="{len(evidences)}">\n{passages}\n</passages>\n\n'
        "Return the review JSON."
    )


class LLMEvidenceReviewer:
    """The Evidence Reviewer over a CAMEL backend: one skeptical call per edge over all the
    triaged passages. It emits per-passage
    coherence + methods sub-signals (keyed by evidence_id) and edge-level open_risks/qualifiers,
    never a verdict; deterministic selection computes the committed label. The reviewer sees the
    edge target context + the passages; never the association floats, other seams' output, or a
    verdict. Malformed/omitted output degrades per-passage to neutral signals (never crashes)."""

    def __init__(self, model_backend: Any, *, system_prompt: str | None = None) -> None:
        self._backend = model_backend
        self._system_prompt = system_prompt or _EVIDENCE_REVIEWER_SYSTEM_PROMPT

    def review(
        self, task: VerificationTask, evidences: Sequence[RetrievedEvidence]
    ) -> EdgeReview:
        evidences = list(evidences)
        data = _backend_json(
            self._backend, self._system_prompt, _evidence_reviewer_user_prompt(task, evidences)
        )
        coherence: dict[str, CoherenceSignal] = {}
        methods: dict[str, MethodsSignal] = {}
        for raw in data.get("passages") or []:
            if not isinstance(raw, dict):
                continue
            evidence_id = str(raw.get("evidence_id", ""))
            if not evidence_id:
                continue
            entail_raw = raw.get("entailment")
            entail = entail_raw if isinstance(entail_raw, dict) else {}
            contra_raw = raw.get("contradiction_risk")
            contra = contra_raw if isinstance(contra_raw, dict) else {}
            coherence[evidence_id] = CoherenceSignal(
                entailment={str(k): _clip_signal(val) for k, val in entail.items()},
                contradiction_risk={str(k): _clip_signal(val) for k, val in contra.items()},
                context_fit=_clip_signal(raw.get("context_fit")),
                construct_match=_clip_signal(raw.get("construct_match")),
            )
            methods[evidence_id] = MethodsSignal(
                design_strength=_clip_signal(raw.get("design_strength")),
                measurement_validity=_clip_signal(raw.get("measurement_validity")),
                population_fit=_clip_signal(raw.get("population_fit")),
                confounder_adjustment=_clip_signal(raw.get("confounder_adjustment")),
                bias_risk=_clip_signal(raw.get("bias_risk")),
            )
        risks_raw = data.get("open_risks")
        open_risks = risks_raw if isinstance(risks_raw, list) else []
        quals_raw = data.get("qualifiers")
        qualifiers = quals_raw if isinstance(quals_raw, list) else []
        return EdgeReview(
            coherence=coherence,
            methods=methods,
            appraisal=VerifierAppraisal(
                open_risks=tuple(str(r) for r in open_risks),
                qualifiers=tuple(str(q) for q in qualifiers),
            ),
        )
