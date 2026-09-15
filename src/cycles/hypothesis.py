"""Research Synthesist hypothesis-expansion cycle.

The cycle surfaces new, unverified mediators, confounders, mechanisms, and testable predictions
for later evidence review. It owns deterministic eligibility, ranking, and hypothesis-delta
construction; shared validation and transaction modules own commits and receipts. All thresholds
and weights come from ``graph_config_defaults`` and are included in audit output.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src import graph_config_defaults as gcd
from src.delta import (
    DeltaFamily,
    GraphDeltaProposal,
    HypothesisPayload,
)
from src.graph_store import (
    CausalClaimGraphStore,
    CausalEdge,
    ConceptNode,
)
from src.progress import report_progress
from src.relation_labels import normalize_relation_label
from src.retrieval.similarity import Embedder, _cosine
from src.retrieval.similarity import clip01 as _clip01
from src.transaction_log import GraphTransactionLog, TransactionResult
from src.validator import GraphDeltaValidator

logger = logging.getLogger(__name__)


# --- Conjunctive eligibility gate ----------------------------------------------------
def gate_h(
    *,
    novelty: float,
    testability: float,
    scope_fit: float,
    duplication: float,
    thresholds: dict[str, float] | None = None,
    drop_novelty: bool = False,
) -> int:
    """Evaluate the conjunctive novelty, testability, scope, and duplication gate.

    A logical AND of four indicators — fail-closed: a single failing bar blocks the
    candidate regardless of the other three scores. Novelty, testability, and scope use ``>=``;
    duplication uses strict ``<``. The output is binary, and eligibility is required before a
    hypothesis delta is built.

    ``drop_novelty`` makes novelty ranking-only and leaves a three-bar validity gate. It defaults
    to false; the normal path uses all four bars.
    """
    thresholds = thresholds if thresholds is not None else gcd.HYP_GATE_THRESHOLDS
    novelty_ok = drop_novelty or novelty >= thresholds["novelty"]
    return int(
        novelty_ok
        and testability >= thresholds["testability"]
        and scope_fit >= thresholds["scope"]
        and duplication < thresholds["dup"]
    )


# --- Post-gate hypothesis ranking ---------------------------------------------------
def hyp_score(
    *,
    novelty: float,
    plausibility: float,
    testability: float,
    expected_yield: float,
    centrality: float,
    mechanism_specificity: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Return the clipped weighted sum of six graded signals.

    The score orders eligible candidates and is not a truth estimate. Scope fit and duplication
    remain pass/fail gate bars rather than ranking signals.
    """
    weights = weights if weights is not None else gcd.HYPSCORE_WEIGHTS
    total = (
        weights["nov"] * novelty
        + weights["plaus"] * plausibility
        + weights["test"] * testability
        + weights["voi"] * expected_yield
        + weights["imp"] * centrality
        + weights["mech"] * mechanism_specificity
    )
    return _clip01(total)


# --- Literature-saturation demotion -------------------------------------------------
def rank_score(*, hyp_score: float, saturation: float) -> float:
    """Apply a soft literature-saturation demotion to ``HypScore``.

    A soft multiplicative demotion is applied after the weighted sum, which is left
    untouched): a fully-established relationship (``saturation -> 1``) has ~zero marginal
    research value, so its rank collapses regardless of plausibility/centrality. It only
    re-orders the gate-eligible set; it never gates a candidate out. With ``saturation = 0`` it
    is the identity, so an un-judged candidate ranks exactly on its ``HypScore``.
    """
    return _clip01(hyp_score * (1.0 - saturation))


# --- Hybrid sub-signal production ---------------------------------------------------
def fuse_subsignal(
    *, llm_ordinal: float, det_metric: float, w_llm: float, w_det: float
) -> float:
    """Fuse a recorded LLM ordinal with a deterministic metric.

    The LLM ordinal is recorded once and replayed (fixtured in the deterministic suite); the
    ``det_metric`` is deterministic. The fused signal is therefore a pure function of recorded
    values.
    """
    return _clip01(w_llm * llm_ordinal + w_det * det_metric)


def candidate_anchor_text(nodes: Sequence[ConceptNode]) -> str:
    """The deterministic embedding anchor for a candidate: its new nodes' label + aliases.

    Restricted to semantic ``ConceptNode`` surface fields, never
    workflow-state fields. Mirrors the Extraction merge anchor (label + aliases only).
    """
    parts: list[str] = []
    for node in nodes:
        parts.append(node.label)
        parts.extend(node.aliases)
    return " ".join(part for part in parts if part)


def _max_cosine(anchor_text: str, reference_texts: Sequence[str], *, embedder: Embedder) -> float:
    """Max clamped cosine of ``anchor_text`` against a reference text set; 0.0 when empty."""
    references = list(reference_texts)
    if not references:
        return 0.0
    vectors = embedder([anchor_text, *references])
    anchor_vec = vectors[0]
    return max(_cosine(anchor_vec, vec) for vec in vectors[1:])


def det_novelty(anchor_text: str, existing_texts: Sequence[str], *, embedder: Embedder) -> float:
    """``det_metric = 1 - max cosine`` to the COMMITTED GRAPH ( Novelty recipe)."""
    return _clip01(1.0 - _max_cosine(anchor_text, existing_texts, embedder=embedder))


def det_duplication(
    anchor_text: str, reference_texts: Sequence[str], *, embedder: Embedder
) -> float:
    """``det_metric = max cosine`` over committed graph PLUS the candidate pool ( Dup
    recipe — a deliberately broader reference set than Novelty's)."""
    return _max_cosine(anchor_text, reference_texts, embedder=embedder)


def det_scope_fit(anchor_text: str, scope_anchor_text: str, *, embedder: Embedder) -> float:
    """``det_metric = cosine`` to the user-scoped domain anchor ( ScopeFit recipe)."""
    anchor_vec, scope_vec = embedder([anchor_text, scope_anchor_text])
    return _cosine(anchor_vec, scope_vec)


_IDEA_SCAFFOLD_KEYS = (
    "claim_anchor",
    "challenged_assumption",
    "incumbent_limit",
    "lever",
    "synthesis",
    "guarantee",
    "fail_safe",
    "why",
    "problem",
    "method",
    "experiment",
    "critique",
)


def _coerce_idea_scaffold(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    scaffold: dict[str, str] = {}
    for key in _IDEA_SCAFFOLD_KEYS:
        value = raw.get(key)
        text = str(value).strip() if value is not None else ""
        if text:
            scaffold[key] = text
    for raw_key, value in sorted(raw.items(), key=lambda item: str(item[0])):
        key = str(raw_key).strip()
        if not key or key in scaffold:
            continue
        text = str(value).strip() if value is not None else ""
        if text:
            scaffold[key] = text
    return scaffold


# --- Candidate + the typed Δ^hypothesis builder ------------------------------------------
@dataclass(frozen=True)
class HypothesisCandidate:
    """One proposer-produced candidate hypothesis with its recorded, already-fused signals.

    The four gate sub-signals (Novelty/Testability/ScopeFit/Duplication) and the four
    ranking-only graded signals (Plausibility/ExpectedYield/Centrality/MechanismSpecificity)
    enter as fixed [0,1] values per the  record-and-replay contract; Novelty and
    Testability are shared by the gate and the ranking ( ``*_graded``). ``duplication``
    defaults to 1.0 so an unscored candidate fails the gate closed.
    """

    candidate_id: str
    new_nodes: tuple[ConceptNode, ...] = ()
    new_edges: tuple[CausalEdge, ...] = ()
    # Research Synthesist output: the stated causal chain (each step has from/relation/to/mechanism)
    # and the verbatim grounding spans (each {evidence_id,quote_span,role_in_hypothesis}). Required
    # by the Research Synthesist; consumed by the Critic Panel (mechanism_steps / term_verdicts) and the
    # Experiment Designer. Empty when a producer emits neither field.
    mechanism_chain: tuple[dict[str, Any], ...] = ()
    source_quotes: tuple[dict[str, Any], ...] = ()
    assumptions: tuple[str, ...] = ()
    rationale: str = ""
    idea_scaffold: dict[str, str] = field(default_factory=dict)
    provenance: tuple[dict[str, Any], ...] = ()
    # Critic Panel outputs preserved for later report/elaboration surfaces.
    mechanism_steps: tuple[dict[str, Any], ...] = ()
    term_audit: tuple[dict[str, Any], ...] = ()
    # Eligibility-gate signals; ``novelty`` is the permissive novelty bar.
    novelty: float = 0.0
    testability: float = 0.0
    scope_fit: float = 0.0
    duplication: float = 1.0
    # Ranking-only graded signals.
    plausibility: float = 0.0
    expected_yield: float = 0.0
    centrality: float = 0.0
    mechanism_specificity: float = 0.0
    # Field-relative ranking novelty from the independent judge. ``None`` falls back to the gate
    # novelty. ``saturation`` is the recorded literature-saturation demotion signal.
    novelty_graded: float | None = None
    saturation: float = 0.0
    # Proposer-declared audit flags; these are not scoring inputs.
    cross_concept: bool = False
    common_sense: bool = False

    @property
    def ranking_novelty(self) -> float:
        """Use field-relative graded novelty when present, otherwise gate novelty."""
        return self.novelty if self.novelty_graded is None else self.novelty_graded


@dataclass(frozen=True)
class ScoredCandidate:
    """An eligible candidate paired with its base and saturation-adjusted scores."""

    candidate: HypothesisCandidate
    hyp_score: float
    rank_score: float


def build_hypothesis_delta(
    *,
    base_graph_hash: str,
    new_nodes: Sequence[ConceptNode],
    new_edges: Sequence[CausalEdge] = (),
    assumptions: Sequence[str] = (),
    author_role: str = "hypothesis_expander",
    rationale: str = "",
    provenance: list[dict[str, Any]] | None = None,
) -> GraphDeltaProposal:
    """Assemble a typed hypothesis delta.

    The ``new_nodes``/``new_edges`` surface desugars to ``add_node``/``add_edge`` ops at apply
    time (B1, owned by ``delta.py``), so the delta mutates ONLY through the closed Operation
    union; every new edge begins ``unverified``.
    """
    return GraphDeltaProposal(
        family=DeltaFamily.HYPOTHESIS,
        base_graph_hash=base_graph_hash,
        payload=HypothesisPayload(
            new_nodes=list(new_nodes),
            new_edges=list(new_edges),
            assumptions=list(assumptions),
        ),
        author_role=author_role,
        rationale=rationale,
        provenance=provenance or [],
    )


# --- Ranking (eligible -> ranked -> top-k) -----------------------------------------------
def _rank(
    candidates: Sequence[HypothesisCandidate],
    *,
    thresholds: dict[str, float],
    weights: dict[str, float],
) -> list[ScoredCandidate]:
    """Filter to ``Gate_h = 1`` then sort descending by ``RankScore`` (ties -> candidate_id).

    ``RankScore = HypScore * (1 - saturation)`` and HypScore uses field-relative
    ``ranking_novelty``. The saturation demotion is controlled by
    ``SATURATION_PENALTY_ENABLED``. Candidate ID is the deterministic tie-break.
    """
    penalty_on = gcd.SATURATION_PENALTY_ENABLED
    scored = []
    for candidate in candidates:
        if gate_h(
            novelty=candidate.novelty,
            testability=candidate.testability,
            scope_fit=candidate.scope_fit,
            duplication=candidate.duplication,
            thresholds=thresholds,
        ) != 1:
            continue
        hs = hyp_score(
            novelty=candidate.ranking_novelty,
            plausibility=candidate.plausibility,
            testability=candidate.testability,
            expected_yield=candidate.expected_yield,
            centrality=candidate.centrality,
            mechanism_specificity=candidate.mechanism_specificity,
            weights=weights,
        )
        sat = candidate.saturation if penalty_on else 0.0  # flag OFF -> no demotion (rank = HypScore)
        scored.append(
            ScoredCandidate(
                candidate=candidate, hyp_score=hs, rank_score=rank_score(hyp_score=hs, saturation=sat)
            )
        )
    scored.sort(key=lambda sc: (-sc.rank_score, sc.candidate.candidate_id))
    return scored


def eligible_ranked(
    candidates: Sequence[HypothesisCandidate],
    *,
    thresholds: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
    budget: int | None = None,
) -> list[ScoredCandidate]:
    """Filter eligible candidates, rank by adjusted score, and return the top budgeted set."""
    thresholds = thresholds if thresholds is not None else gcd.HYP_GATE_THRESHOLDS
    weights = weights if weights is not None else gcd.HYPSCORE_WEIGHTS
    budget = budget if budget is not None else gcd.EXPANSION_BUDGET_B
    return _rank(candidates, thresholds=thresholds, weights=weights)[:budget]


def build_hypothesis_candidates(
    raw_cands: Sequence[Any],
    *,
    store: CausalClaimGraphStore,
    embedder: Embedder,
    claim: str,
) -> list[HypothesisCandidate]:
    """Parse candidates and perform late fusion for Research Synthesist proposal and revision.

    Builds each candidate's new nodes/edges (resolving edge endpoints against the committed graph
    and other candidates' new nodes; dangling refs dropped), computes the SHIPPED det sub-signals
    with the injected embedder, and late-fuses them with the LLM ordinals. Option-C is preserved:
    the proposer's ``novelty`` ordinal feeds the PERMISSIVE ``Gate_h`` bar via ``fuse_subsignal``;
    ``testability`` has no shipped det recipe so its ordinal passes through. Also records the Research Synthesist
    ``mechanism_chain`` + verbatim ``source_quotes`` (empty for the legacy proposer). A candidate
    with no ``new_nodes``, no ``new_edges``, an empty ``mechanism_chain``, or empty
    ``source_quotes`` fails the grounding requirement and is dropped here, at the parse
    boundary, with the reason logged (a candidate with no new concept has no focus node, and a
    candidate with no new edge has no hypothesis claim to commit or test). Surviving candidate
    ids are normalised to a unique ``h<N>``, because the
    connected renderer enumerates hypotheses from ``trace/h*.html``. The
    HypothesisCandidate dataclass, ``gate_h``, and scoring stay untouched. Deterministic
    given the recorded raw candidates + embedder — the fusion math is NOT reimplemented here.
    """
    from src.delta import build_edge, build_node

    if not raw_cands:
        return []

    fusion = gcd.HYP_FUSION_WEIGHTS

    def anchor_of(node: ConceptNode) -> str:
        return " ".join([node.label, *list(node.aliases)]).strip()

    committed_anchors = [anchor_of(node) for node in store.nodes.values()]
    label_to_id = {node.label.lower(): nid for nid, node in store.nodes.items()}

    # pass 1: build each candidate's new nodes -> a GLOBAL label->node map (an edge may
    # reference a node another candidate introduced; first-definer wins).
    pre: list[tuple[dict[str, Any], list[ConceptNode], str]] = []
    all_anchors: list[str] = []
    global_node_by_label: dict[str, ConceptNode] = {}
    for cand in raw_cands:
        if not isinstance(cand, dict):
            continue
        own: list[ConceptNode] = []
        for raw_node in cand.get("new_nodes") or []:
            if not isinstance(raw_node, dict) or not str(raw_node.get("label", "")).strip():
                continue
            node = build_node(
                label=str(raw_node["label"]).strip(),
                definition=str(raw_node.get("definition", "")),
                type=str(raw_node.get("type", "")),
                aliases=[str(a) for a in (raw_node.get("aliases") or [])],
            )
            own.append(node)
            global_node_by_label.setdefault(node.label.lower(), node)
        anchor = candidate_anchor_text(own) if own else str(cand.get("candidate_id", ""))
        all_anchors.append(anchor)
        pre.append((cand, own, anchor))

    global_label_to_id = dict(label_to_id)
    for label, node in global_node_by_label.items():
        global_label_to_id.setdefault(label, node.node_id)

    candidates: list[HypothesisCandidate] = []
    used_cids: set[str] = set()
    for cand, own_nodes, anchor in pre:
        node_by_id = {node.node_id: node for node in own_nodes}
        new_edges: list[CausalEdge] = []
        for raw_edge in cand.get("new_edges") or []:
            if not isinstance(raw_edge, dict):
                continue
            src = global_label_to_id.get(str(raw_edge.get("source", "")).lower())
            tgt = global_label_to_id.get(str(raw_edge.get("target", "")).lower())
            if src is None or tgt is None:
                continue  # drop dangling refs (1_refs would reject them anyway)
            # pull in a node another candidate introduced so it commits alongside this edge
            for label, nid in (
                (str(raw_edge.get("source", "")).lower(), src),
                (str(raw_edge.get("target", "")).lower(), tgt),
            ):
                if (
                    nid not in label_to_id.values()
                    and nid not in node_by_id
                    and label in global_node_by_label
                ):
                    extra = global_node_by_label[label]
                    node_by_id[extra.node_id] = extra
            new_edges.append(
                build_edge(
                    source_node_ids=[src],
                    target_node_ids=[tgt],
                    direction=str(raw_edge.get("direction", "directed")),
                    relation_type=str(raw_edge.get("relation_type", "influences")),
                    mechanism=str(raw_edge.get("mechanism", "")),
                )
            )
        new_nodes = list(node_by_id.values())

        signals = cand.get("llm_signals") or {}
        ord_nov = float(signals.get("novelty", 0.0))
        ord_test = float(signals.get("testability", 0.0))
        ord_scope = float(signals.get("scope_fit", 0.0))
        ord_dup = float(signals.get("duplication", 1.0))

        #  det_metrics via the SHIPPED recipes (real embedder on the live path).
        d_nov = det_novelty(anchor, committed_anchors, embedder=embedder) if anchor else 0.0
        dup_ref = committed_anchors + [a for a in all_anchors if a and a != anchor]
        d_dup = det_duplication(anchor, dup_ref, embedder=embedder) if anchor else 1.0
        d_scope = det_scope_fit(anchor, claim, embedder=embedder) if anchor else 0.0

        f_nov = fuse_subsignal(
            llm_ordinal=ord_nov, det_metric=d_nov,
            w_llm=fusion["novelty"]["llm"], w_det=fusion["novelty"]["det"],
        )
        f_scope = fuse_subsignal(
            llm_ordinal=ord_scope, det_metric=d_scope,
            w_llm=fusion["scope"]["llm"], w_det=fusion["scope"]["det"],
        )
        f_dup = fuse_subsignal(
            llm_ordinal=ord_dup, det_metric=d_dup,
            w_llm=fusion["duplication"]["llm"], w_det=fusion["duplication"]["det"],
        )

        rationale = str(cand.get("rationale", ""))
        experiment = str(cand.get("experiment", "")).strip()
        if experiment:
            rationale = (
                f"{rationale}\n\nExperiment: {experiment}" if rationale
                else f"Experiment: {experiment}"
            )

        mechanism_chain = tuple(
            {
                "from": str(step.get("from", "")),
                "relation": normalize_relation_label(step.get("relation", "")),
                "to": str(step.get("to", "")),
                "mechanism": str(step.get("mechanism", "")),
            }
            for step in (cand.get("mechanism_chain") or [])
            if isinstance(step, dict)
        )
        source_quotes = tuple(
            {
                "evidence_id": quote.get("evidence_id"),  # may be null (divergent revise candidate)
                "quote_span": str(quote.get("quote_span", "")),
                "role_in_hypothesis": str(quote.get("role_in_hypothesis", "")),
            }
            for quote in (cand.get("source_quotes") or [])
            if isinstance(quote, dict)
        )

        raw_cid = str(cand.get("candidate_id", "")).strip()
        empty_fields = [
            name for name, value in (
                # A hypothesis with no new node has no focus for the connected page.
                ("new_nodes", new_nodes),
                # A hypothesis with no new edge has no graph claim or experiment target.
                ("new_edges", new_edges),
                ("mechanism_chain", mechanism_chain),
                ("source_quotes", source_quotes),
            )
            if not value
        ]
        if empty_fields:  # Reject candidates with incomplete grounding at the parse boundary.
            logger.warning(
                "dropped hypothesis candidate %s: empty required grounding field(s): %s",
                raw_cid, ", ".join(empty_fields),
            )
            continue

        # The connected renderer enumerates hypotheses from trace/h*.html, so a
        # model-authored id that is not a unique h<N> silently loses that page.
        if re.fullmatch(r"h\d+", raw_cid) and raw_cid not in used_cids:
            candidate_id = raw_cid
        else:
            n = 1
            while f"h{n}" in used_cids:
                n += 1
            candidate_id = f"h{n}"
            logger.warning(
                "hypothesis candidate_id %r is not a unique h<N> id "
                "(the connected renderer enumerates trace/h*.html); renamed to %s",
                raw_cid, candidate_id,
            )
        used_cids.add(candidate_id)

        candidates.append(
            HypothesisCandidate(
                candidate_id=candidate_id,
                new_nodes=tuple(new_nodes),
                new_edges=tuple(new_edges),
                mechanism_chain=mechanism_chain,
                source_quotes=source_quotes,
                assumptions=tuple(str(a) for a in (cand.get("assumptions") or [])),
                rationale=rationale,
                idea_scaffold=_coerce_idea_scaffold(cand.get("idea_scaffold")),
                novelty=f_nov,
                testability=ord_test,  # No deterministic recipe; retain the model-provided value.
                scope_fit=f_scope,
                duplication=f_dup,
                plausibility=float(signals.get("plausibility", 0.0)),
                expected_yield=float(signals.get("expected_yield", 0.0)),
                centrality=float(signals.get("centrality", 0.0)),
                mechanism_specificity=float(signals.get("mechanism_specificity", 0.0)),
                cross_concept=bool(cand.get("cross_concept", False)),
                common_sense=bool(cand.get("common_sense", False)),
            )
        )
    return candidates


# --- The cycle ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HypothesisCycleResult:
    """The auditable outcome of one Synthesist cycle. When disabled it is inert: zero
    proposals, zero commits, and a byte-identical store."""

    enabled: bool
    proposed: tuple[HypothesisCandidate, ...]
    eligible: tuple[HypothesisCandidate, ...]
    surfaced: tuple[ScoredCandidate, ...]
    transactions: tuple[TransactionResult, ...]
    store: CausalClaimGraphStore
    log: GraphTransactionLog
    audit: dict[str, object] = field(default_factory=dict)
    # Confirmed candidates whose hypothesis deltas committed; these feed experiment design.
    confirmed: tuple[HypothesisCandidate, ...] = ()


def run_hypothesis_cycle(
    candidates: Sequence[HypothesisCandidate],
    *,
    store: CausalClaimGraphStore,
    enabled: bool = False,
    precondition: bool = True,
    confirmed_ids: Collection[str] = (),
    confirm_fn: Callable[[Sequence[ScoredCandidate]], Collection[str]] | None = None,
    embedder: Embedder | None = None,
    log: GraphTransactionLog | None = None,
    validator: GraphDeltaValidator | None = None,
    thresholds: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
    budget: int | None = None,
    reranker: Callable[[Sequence[ScoredCandidate]], Sequence[ScoredCandidate]] | None = None,
    author: str = "hypothesis_expander",
    timestamp: str | None = None,
) -> HypothesisCycleResult:
    """Run the Synthesist cycle: judge, gate, rank, surface, and commit confirmed candidates.

    ``enabled`` defaults to false, leaving the cycle inert and the store untouched.
    ``precondition`` is the expansion-consideration guard (in scope, safeguards exist,
    no user-blocking scope gap); when it fails the cycle skips with no commit. When both hold,
    only gate-eligible candidates are surfaced (top-k under budget B), and a surfaced candidate
    commits ONLY when the user confirms it
    through ``confirmed_ids``. Every commit routes through the shared validator and lands
    ``unverified``; a rejected delta leaves the graph version unchanged and can be revised.
    """
    candidates = list(candidates)
    thresholds = thresholds if thresholds is not None else gcd.HYP_GATE_THRESHOLDS
    weights = weights if weights is not None else gcd.HYPSCORE_WEIGHTS
    budget = budget if budget is not None else gcd.EXPANSION_BUDGET_B
    log = log if log is not None else GraphTransactionLog()
    validator = validator if validator is not None else GraphDeltaValidator()
    audit: dict[str, object] = {
        "graph_defaults_version": gcd.GRAPH_DEFAULTS_VERSION,
        "hyp_gate_thresholds": thresholds,
        "hypscore_weights": weights,
        "expansion_budget_b": budget,
        "saturation_penalty_enabled": gcd.SATURATION_PENALTY_ENABLED,
        "theta_saturation": gcd.THETA_SATURATION,
        "novelty_panel_reducer": gcd.NOVELTY_PANEL_REDUCER,
    }

    # Disabled or out-of-scope execution behaves as if the feature were absent.
    if not enabled or not precondition:
        return HypothesisCycleResult(
            enabled=enabled, proposed=(), eligible=(), surfaced=(), transactions=(),
            store=store, log=log, audit=audit,
        )

    # Gate, score, rank, and surface the top candidates under the expansion budget. The pairwise
    # Tournament and evolution-loop code is absent: the Critic Panel's
    # listwise ranking + the Research Synthesist revise turn replace them, and the deterministic surfacing order is
    # RankScore (unchanged core). The emphasis reranker below then only RE-ORDERS the surfaced set.
    report_progress("Propose / critique", "ranking and surfacing hypotheses")
    proposed_candidates = list(candidates)
    ranked = _rank(candidates, thresholds=thresholds, weights=weights)
    surfaced = tuple(ranked[:budget])

    # The surfaced set is the RankScore top-k; authored-priority emphasis then
    # authored-priority emphasis re-rank then only RE-ORDERS those surfaced winners. Identity when no
    # priority is authored, so the RankScore order is preserved; the four Gate_h bars and the eligible
    # set (RankScore order) are untouched — emphasis never changes membership.
    if reranker is not None:
        surfaced = tuple(reranker(surfaced))

    eligible = tuple(sc.candidate for sc in ranked)  # Final adjusted-score order.

    # The user confirms which surfaced candidates commit. ``confirm_fn`` (when injected) is
    # called WITH the ranked surfaced set, so an interactive prompt or an auto-policy can decide
    # from the ranking the cycle just produced; otherwise the static ``confirmed_ids`` is honored.
    report_progress(
        "Confirming hypotheses",
        f"{len(surfaced)} hypotheses surfaced; applying confirmation policy",
    )
    confirmed = (
        {str(cid) for cid in confirm_fn(surfaced)} if confirm_fn is not None else set(confirmed_ids)
    )
    report_progress("Confirming hypotheses", "committing selected hypotheses")
    transactions: list[TransactionResult] = []
    confirmed_candidates: list[HypothesisCandidate] = []
    for scored in surfaced:
        if scored.candidate.candidate_id not in confirmed:
            continue  # Commit only candidates confirmed by the user or policy.
        delta = build_hypothesis_delta(
            base_graph_hash=store.base_hash,  # re-read per commit so sequential commits chain
            new_nodes=scored.candidate.new_nodes,
            new_edges=scored.candidate.new_edges,
            assumptions=scored.candidate.assumptions,
            author_role=author,
            rationale=scored.candidate.rationale,
            provenance=list(scored.candidate.provenance),
        )
        tx = log.commit(store, delta, validator, author=author, timestamp=timestamp)
        transactions.append(tx)
        if tx.accepted:  # Only successfully committed hypotheses reach experiment design.
            confirmed_candidates.append(scored.candidate)

    report_progress(
        "Confirming hypotheses", f"{len(confirmed_candidates)} hypotheses committed",
        status="completed",
    )
    return HypothesisCycleResult(
        enabled=True, proposed=tuple(proposed_candidates), eligible=eligible, surfaced=surfaced,
        transactions=tuple(transactions), store=store, log=log, audit=audit,
        confirmed=tuple(confirmed_candidates),
    )
