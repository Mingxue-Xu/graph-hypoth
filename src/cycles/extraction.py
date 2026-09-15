"""Extraction cycle from a seed claim to a committed, unverified extraction delta.

The module owns deterministic scope-gap, extraction-confidence, and merge-safety calculations plus
delta construction. Shared delta, validator, and transaction modules own content-addressed IDs,
commits, and receipts. Coefficients come from ``graph_config_defaults``; agent extraction and
embedding are injected so deterministic logic can be tested with fixed inputs.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src import graph_config_defaults as gcd
from src.delta import (
    AddEdgeOp,
    AddNodeOp,
    DeltaFamily,
    ExtractPayload,
    GraphDeltaProposal,
    MergeNodesOp,
    Operation,
    build_edge,
    build_node,
    norm,
)
from src.graph_store import (
    CausalClaimGraphStore,
    CausalEdge,
    ConceptNode,
)
from src.retrieval import scoring_defaults as sd
from src.retrieval.coherence import (
    Embedder,
    _cosine,
    load_specter2_embedder,
)
from src.retrieval.similarity import clip01 as _clip01
from src.retrieval.similarity import jaccard as _jaccard
from src.transaction_log import GraphTransactionLog, TransactionResult
from src.validator import GraphDeltaValidator

# Type-conflict table used by merge safety; this defines the
# sub-signal over the ConceptNode.type enum, NOT a calibration weight. The enum is the
# extractor vocabulary plus the Research Synthesist types ``moderator`` and ``condition``.
_TYPE_CONFLICT_SAME = 0.0
_TYPE_CONFLICT_SAME_CLASS = 0.3
_TYPE_CONFLICT_DIFF_CLASS = 1.0
# Named multi-member compatibility classes; every core type
# {population, exposure/intervention, outcome, comparator, context} is its own singleton.
# The Research Synthesist mine-turn causal-structure types `moderator`/`condition` join the third-variable
# class, so a role-label difference among them is a soft 0.3 penalty rather than a spurious
# 1.0 that would block an otherwise-good merge of the same underlying concept.
_TYPE_COMPAT_CLASSES: tuple[frozenset[str], ...] = (
    frozenset({"mechanism", "mediator", "confounder", "moderator", "condition"}),
    frozenset({"construct", "variable"}),
)


# --- Scope-gap formula -----------------------------------------------------------------
def scope_gap(
    *,
    present_facets: Collection[str],
    salient_terms: Collection[str],
    defined_terms: Collection[str],
    scope_conflict: bool = False,
    granularity_conflict: bool = False,
    weights: dict[str, float] | None = None,
    facet_weights: dict[str, float] | None = None,
    epsilon: float | None = None,
) -> float:
    """Compute ``ScopeGap = clip01(w_M·M_q + w_U·U_terms + w_B·B_blockers)``.

    A routing score in [0,1] (not a truth claim). All sub-signals are computed
    deterministically over the Scope Clarifier's agent-emitted facet slots / salient /
    defined terms, so the score is reproducible.
    """
    weights = weights if weights is not None else gcd.SCOPE_GAP_WEIGHTS
    facet_weights = facet_weights if facet_weights is not None else gcd.SCOPE_GAP_FACET_WEIGHTS
    epsilon = epsilon if epsilon is not None else gcd.EPSILON

    present = set(present_facets)
    missing_weight = sum(a for facet, a in facet_weights.items() if facet not in present)
    m_q = missing_weight / (sum(facet_weights.values()) + epsilon)

    n_salient = len(set(salient_terms))
    n_defined = len(set(defined_terms))
    u_terms = 1.0 - n_defined / (n_salient + epsilon)

    b_blockers = 1.0 if (scope_conflict or granularity_conflict) else 0.0

    return _clip01(
        weights["missing"] * m_q + weights["terms"] * u_terms + weights["blockers"] * b_blockers
    )


# --- Extraction-confidence formula -----------------------------------------------------
def extract_conf(
    *,
    span_support: float,
    qualifier_coverage: float,
    assumption_penalty: float,
    weights: dict[str, float] | None = None,
    epsilon: float | None = None,
) -> float:
    """``ExtractConf = clip01((w_span·Span + w_qual·Qual − w_assumption·Assume)/Σw)``.

    ``w_assumption`` also appears in the denominator, so the achievable maximum is below one by
    design. The three sub-signals enter as already-computed fractions.
    """
    weights = weights if weights is not None else gcd.EXTRACT_CONF_WEIGHTS
    epsilon = epsilon if epsilon is not None else gcd.EPSILON
    numerator = (
        weights["span"] * span_support
        + weights["qual"] * qualifier_coverage
        - weights["assumption"] * assumption_penalty
    )
    denominator = weights["span"] + weights["qual"] + weights["assumption"] + epsilon
    return _clip01(numerator / denominator)


# Merge-safety formula with four signals plus type conflict.
def type_conflict(type_a: str, type_b: str) -> float:
    """Grade type conflict over the closed enum: same=0.0, same
    compatibility class=0.3, different class=1.0. A penalty, not a hard veto."""
    if type_a == type_b:
        return _TYPE_CONFLICT_SAME
    for compat_class in _TYPE_COMPAT_CLASSES:
        if type_a in compat_class and type_b in compat_class:
            return _TYPE_CONFLICT_SAME_CLASS
    return _TYPE_CONFLICT_DIFF_CLASS


def concept_embed_text(node: ConceptNode) -> str:
    """Build the merge embedding text from label and aliases only.

    The definition signal is carried separately by ``DefinitionSim``.
    """
    return " ".join(part for part in [node.label, *node.aliases] if part)


def _surface_label_tokens(node: ConceptNode) -> set[str]:
    """``norm()`` tokens over the node's surface labels (label ∪ aliases)."""
    tokens = set(norm(node.label))
    for alias in node.aliases:
        tokens |= set(norm(alias))
    return tokens


def default_merge_embedder() -> Embedder | None:
    """Return the pinned production embedder used for cosine-based concept merging.

    Loads the model ID pinned in ``scoring_defaults`` (``allenai/specter2_base``) through the
    shared SPECTER2 loader so production callers need not know the ID; returns
    ``None`` when the optional ``retrieval-coherence`` extra is absent or the model fails to
    load (callers then inject a stub or handle the lexical fallback). The default unit suite
    never calls this; it injects a deterministic stub embedder instead.

    The loader pins both the model ID and ``scoring_defaults.EMBEDDING_REVISION`` so the default
    merge path loads the same frozen revision as retrieval.
    """
    return load_specter2_embedder(sd.EMBEDDING_MODEL, sd.EMBEDDING_REVISION)


def merge_score(
    c_a: ConceptNode,
    c_b: ConceptNode,
    *,
    embedder: Embedder,
    weights: dict[str, float] | None = None,
    epsilon: float | None = None,
) -> float:
    """``Merge = clip01((w_label·Jaccard + w_embed·cos + w_def·DefSim − w_type·TypeConflict)/Σw)``.

    An entity-resolution score in [0,1], NOT an automatic merge command — the thresholded
    decision lives in ``build_extract_delta``. The cosine uses the raw clamp primitive over
    label-and-alias embeddings; definitions are scored lexically.
    """
    weights = weights if weights is not None else gcd.MERGE_WEIGHTS
    epsilon = epsilon if epsilon is not None else gcd.EPSILON

    label_jaccard = _jaccard(_surface_label_tokens(c_a), _surface_label_tokens(c_b))
    vec_a, vec_b = embedder([concept_embed_text(c_a), concept_embed_text(c_b)])
    cos = _cosine(vec_a, vec_b)
    def_sim = _jaccard(set(norm(c_a.definition)), set(norm(c_b.definition)))
    t_conflict = type_conflict(c_a.type, c_b.type)

    numerator = (
        weights["label"] * label_jaccard
        + weights["embed"] * cos
        + weights["def"] * def_sim
        - weights["type"] * t_conflict
    )
    denominator = weights["label"] + weights["embed"] + weights["def"] + weights["type"] + epsilon
    return _clip01(numerator / denominator)


# ---   — borderline canonicalization adjudicator ------------------------------------
class CanonicalizationAdjudicator(Protocol):
    """The 'same concept?' adjudicator (live LLM path; fixtured in the default suite). Decides
    the borderline merge pairs the compressed specter2 cosine cannot (decision-record )."""

    def same_concept(self, c_a: ConceptNode, c_b: ConceptNode) -> bool: ...


_ADJ_SYSTEM_PROMPT = (
    "You decide whether two concept nodes denote the SAME underlying concept, for graph "
    "canonicalization / de-duplication. Merge true synonyms (e.g. 'reconstruction error' vs "
    "'truncation error' vs 'compression loss'); do NOT merge concepts that are merely related or "
    "adjacent (e.g. 'Fisher importance weighting' vs 'off-diagonal Fisher correlations' are "
    "DISTINCT). Return STRICT JSON only (no markdown): "
    '{"same_concept": true|false, "reason": "..."}.'
)


def _adjudicator_user_prompt(c_a: ConceptNode, c_b: ConceptNode) -> str:
    def fmt(node: ConceptNode) -> str:
        return f"label: {node.label}\ntype: {node.type}\ndefinition: {node.definition}"

    return f"Concept A:\n{fmt(c_a)}\n\nConcept B:\n{fmt(c_b)}\n\nSame underlying concept?"


class LLMCanonicalizationAdjudicator:
    """A real ``CanonicalizationAdjudicator`` over any CAMEL model backend (``.run(messages)``).

    The concrete live-path implementation of the   'same concept?' seam (the default suite
    fixtures it). ``merge_decision`` only consults it for the borderline band; a malformed/empty
    response degrades to ``False`` — the conservative default that keeps two distinct concepts
    distinct rather than over-merging (the  failure mode). Parsing reuses ``camel_adapter`` so
    the deterministic core stays LLM-free.
    """

    def __init__(self, model_backend: Any, *, system_prompt: str | None = None) -> None:
        self._backend = model_backend
        self._system_prompt = system_prompt or _ADJ_SYSTEM_PROMPT

    def same_concept(self, c_a: ConceptNode, c_b: ConceptNode) -> bool:
        from src.camel_adapter import backend_json

        data = backend_json(self._backend, self._system_prompt, _adjudicator_user_prompt(c_a, c_b))
        return bool(data.get("same_concept", False))


def merge_decision(
    c_a: ConceptNode,
    c_b: ConceptNode,
    *,
    embedder: Embedder,
    theta_merge: float | None = None,
    band_low: float | None = None,
    band_high: float | None = None,
    adjudicator: CanonicalizationAdjudicator | None = None,
    weights: dict[str, float] | None = None,
    epsilon: float | None = None,
) -> bool:
    """Whether two concept candidates are the same concept (decision-record ).

    WITHOUT an adjudicator this is exactly the pre- thresholded decision
    (``Merge >= theta_merge``) — so every existing caller is unchanged. WITH one, the
    deterministic ``Merge`` score still decides the confident ends, and the LLM adjudicates the
    ASYMMETRIC band ``[theta_merge - band_low, theta_merge + band_high)``: wide BELOW
    ``theta_merge`` because the compressed specter2 cosine sinks true synonyms there (the spec's
    own 0.462 probe), narrow above to catch near-threshold false-positives. Deterministic given
    the recorded adjudication (replay-safe).
    """
    theta_merge = theta_merge if theta_merge is not None else gcd.THETA_MERGE
    score = merge_score(c_a, c_b, embedder=embedder, weights=weights, epsilon=epsilon)
    if adjudicator is None:
        return score >= theta_merge
    band_low = band_low if band_low is not None else gcd.CANONICALIZATION_BAND_LOW
    band_high = band_high if band_high is not None else gcd.CANONICALIZATION_BAND_HIGH
    if score >= theta_merge + band_high:
        return True
    if score < theta_merge - band_low:
        return False
    return adjudicator.same_concept(c_a, c_b)


def canonicalize_concepts(
    nodes: Sequence[ConceptNode],
    *,
    embedder: Embedder,
    theta_merge: float | None = None,
    band_low: float | None = None,
    band_high: float | None = None,
    adjudicator: CanonicalizationAdjudicator | None = None,
) -> list[tuple[str, ...]]:
    """Cluster a concept set into canonical groups via pairwise ``merge_decision`` + union-find
    (decision-record  — the prerequisite for  mining at scale). Returns each cluster as a
    sorted tuple of ``node_id``s; the survivor is the lexicographically-smaller id (matching the
    B1 collision rule). Deterministic given fixed adjudications (replay-safe)."""
    node_list = list(nodes)
    parent = {node.node_id: node.node_id for node in node_list}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)  # deterministic survivor = smaller id

    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            a, b = node_list[i], node_list[j]
            if a.node_id == b.node_id or merge_decision(
                a, b, embedder=embedder, theta_merge=theta_merge,
                band_low=band_low, band_high=band_high, adjudicator=adjudicator,
            ):
                union(a.node_id, b.node_id)

    clusters: dict[str, list[str]] = {}
    for node_id in parent:
        clusters.setdefault(find(node_id), []).append(node_id)
    return [tuple(sorted(members)) for members in clusters.values()]


# --- extraction-delta builder — Δ^extract builder ------------------------------------------------------
def build_extract_delta(
    *,
    base_graph_hash: str,
    nodes: Sequence[ConceptNode],
    edges: Sequence[CausalEdge] = (),
    merge_pairs: Sequence[tuple[ConceptNode, ConceptNode]] = (),
    embedder: Embedder,
    assumptions: Sequence[str] = (),
    theta_merge: float | None = None,
    adjudicator: CanonicalizationAdjudicator | None = None,
    author_role: str = "atom_relation_extractor",
    rationale: str = "",
    provenance: list[dict[str, Any]] | None = None,
) -> GraphDeltaProposal:
    """Assemble the typed ``Δ^extract``: ``add_node`` then ``add_edge`` then ``merge_nodes``.

    A merge candidate pair becomes a ``merge_nodes`` op only when ``merge_decision`` accepts it
    using the Normalizer's thresholded decision: deterministic ``Merge >= theta_merge`` by
    default, or an injected adjudicator's decision for borderline pairs. Below-decision pairs stay
    as two
    distinct nodes. The survivor is the lexicographically-smaller ``node_id`` so the choice is
    reproducible. Edge endpoints and merge references resolve because every node is added first;
    every edge lands ``unverified`` through ``build_edge``.
    """
    theta_merge = theta_merge if theta_merge is not None else gcd.THETA_MERGE

    operations: list[Operation] = []
    seen_node_ids: set[str] = set()
    candidate_nodes = list(nodes) + [node for pair in merge_pairs for node in pair]
    for node in candidate_nodes:
        if node.node_id not in seen_node_ids:
            seen_node_ids.add(node.node_id)
            operations.append(AddNodeOp(node=node))

    for edge in edges:
        operations.append(AddEdgeOp(edge=edge))

    for c_a, c_b in merge_pairs:
        if c_a.node_id == c_b.node_id:
            continue  # identical content-addressed id: nothing to merge
        if merge_decision(
            c_a, c_b, embedder=embedder, theta_merge=theta_merge, adjudicator=adjudicator
        ):
            survivor_id, merged_id = sorted((c_a.node_id, c_b.node_id))
            operations.append(
                MergeNodesOp(survivor_node_id=survivor_id, merged_node_id=merged_id)
            )

    return GraphDeltaProposal(
        family=DeltaFamily.EXTRACT,
        base_graph_hash=base_graph_hash,
        payload=ExtractPayload(operations=operations, assumptions=list(assumptions)),
        author_role=author_role,
        rationale=rationale,
        provenance=provenance or [],
    )


# Extractor agent layer; the deterministic core is independent of it.
@dataclass(frozen=True)
class ClaimExtraction:
    """Structured extractor output plus optional Scope Clarifier facet slots.

    The deterministic core consumes this; it is produced by a real
    LLM extractor on the live path and by a fake in the default suite."""

    nodes: tuple[ConceptNode, ...] = ()
    edges: tuple[CausalEdge, ...] = ()
    merge_pairs: tuple[tuple[ConceptNode, ConceptNode], ...] = ()
    assumptions: tuple[str, ...] = ()
    # scope-gap formula scope slots (optional; supplied by the Scope Clarifier role).
    present_facets: tuple[str, ...] = ()
    salient_terms: tuple[str, ...] = ()
    defined_terms: tuple[str, ...] = ()
    scope_conflict: bool = False
    granularity_conflict: bool = False


class Extractor(Protocol):
    """The extraction role: a claim in, a structured ``ClaimExtraction`` out."""

    def extract(self, claim: str) -> ClaimExtraction: ...


@dataclass(frozen=True)
class ExtractionCycleResult:
    """The auditable outcome of one extraction cycle; never a silent no-op."""

    store: CausalClaimGraphStore
    transaction: TransactionResult
    delta: GraphDeltaProposal
    scope_gap: float | None
    log: GraphTransactionLog


def run_extraction_cycle(
    claim: str,
    *,
    extractor: Extractor,
    embedder: Embedder,
    store: CausalClaimGraphStore | None = None,
    log: GraphTransactionLog | None = None,
    validator: GraphDeltaValidator | None = None,
    author: str = "atom_relation_extractor",
    timestamp: str | None = None,
    theta_merge: float | None = None,
) -> ExtractionCycleResult:
    """Run the extraction cycle: extract -> build ``Δ^extract`` -> commit via the validator.

    Always returns an auditable outcome (committed, or a rejected receipt) — the seed claim
    is never silently dropped. Scope gap is recorded when the extractor supplies scope slots.
    """
    store = store if store is not None else CausalClaimGraphStore()
    log = log if log is not None else GraphTransactionLog()
    validator = validator if validator is not None else GraphDeltaValidator()

    extraction = extractor.extract(claim)
    delta = build_extract_delta(
        base_graph_hash=store.base_hash,
        nodes=extraction.nodes,
        edges=extraction.edges,
        merge_pairs=extraction.merge_pairs,
        embedder=embedder,
        assumptions=extraction.assumptions,
        theta_merge=theta_merge,
        rationale=f"extraction of seed claim: {claim}",
        provenance=[{"source": "seed_claim", "claim": claim}],
    )
    transaction = log.commit(store, delta, validator, author=author, timestamp=timestamp)

    computed_scope_gap: float | None = None
    if extraction.present_facets or extraction.salient_terms:
        computed_scope_gap = scope_gap(
            present_facets=extraction.present_facets,
            salient_terms=extraction.salient_terms,
            defined_terms=extraction.defined_terms,
            scope_conflict=extraction.scope_conflict,
            granularity_conflict=extraction.granularity_conflict,
        )

    return ExtractionCycleResult(
        store=store, transaction=transaction, delta=delta,
        scope_gap=computed_scope_gap, log=log,
    )


# --- Real LLM Extractor (the `live` path; fixtured with a fake backend in unit tests) ---
_EXTRACTOR_SYSTEM_PROMPT = (
    "You extract a causal-claim graph from a single claim, research goal, question, or raw message. Return STRICT JSON only "
    "(no markdown) with keys: nodes, edges, assumptions. Each node: "
    '{"label","type","definition","aliases"} where type is one of population, '
    "exposure/intervention, outcome, comparator, context, construct, variable, mechanism, "
    "mediator, confounder. Each edge: "
    '{"source","target","direction","relation_type","mechanism"} where source/target are '
    "node labels. Do NOT assert truth or verdicts — only the seed's structure."
)


def _extractor_user_prompt(claim: str) -> str:
    return f"Seed input:\n{claim}\n\nReturn the extraction JSON."


def _claim_provenance(label: str, claim: str) -> list[dict[str, Any]]:
    """Anchor a label to a char-offset span in the claim when it was read, else mark it
    inferred-rather-than-read (the AssumptionPenalty signal of extraction-confidence formula)."""
    idx = claim.lower().find(label.lower())
    if idx >= 0:
        return [{"source": "claim", "span": [idx, idx + len(label)], "read": True}]
    return [{"source": "claim", "read": False}]


def _parse_extraction(data: dict[str, Any], claim: str) -> ClaimExtraction:
    nodes: list[ConceptNode] = []
    node_id_by_label: dict[str, str] = {}
    for raw in data.get("nodes") or []:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label", "")).strip()
        if not label:
            continue
        node = build_node(
            label=label,
            definition=str(raw.get("definition", "")),
            type=str(raw.get("type", "")),
            aliases=[str(alias) for alias in (raw.get("aliases") or [])],
            provenance=_claim_provenance(label, claim),
        )
        nodes.append(node)
        node_id_by_label[label.lower()] = node.node_id

    edges: list[CausalEdge] = []
    for raw in data.get("edges") or []:
        if not isinstance(raw, dict):
            continue
        source_id = node_id_by_label.get(str(raw.get("source", "")).strip().lower())
        target_id = node_id_by_label.get(str(raw.get("target", "")).strip().lower())
        if source_id is None or target_id is None:
            continue  # drop dangling refs (1_refs would reject them anyway)
        edges.append(
            build_edge(
                source_node_ids=[source_id],
                target_node_ids=[target_id],
                direction=str(raw.get("direction", "causal")),
                relation_type=str(raw.get("relation_type", "")),
                mechanism=str(raw.get("mechanism", "")),
            )
        )

    assumptions = tuple(str(a) for a in (data.get("assumptions") or []) if str(a).strip())
    return ClaimExtraction(nodes=tuple(nodes), edges=tuple(edges), assumptions=assumptions)


class LLMExtractor:
    """A real Atom/Relation Extractor over any CAMEL model backend (``.run(messages)``).

    The backend is injected (Extractor input contract: the claim is fed verbatim; nothing else). Response
    parsing reuses the project's single source of truth for model-response shape + JSON
    extraction (``camel_adapter``), imported lazily so the deterministic core stays
    LLM-free. Malformed / empty responses degrade to an empty extraction, never a crash.
    """

    def __init__(self, model_backend: Any, *, system_prompt: str | None = None) -> None:
        self._backend = model_backend
        self._system_prompt = system_prompt or _EXTRACTOR_SYSTEM_PROMPT

    def extract(self, claim: str) -> ClaimExtraction:
        from src.camel_adapter import backend_json

        data = backend_json(self._backend, self._system_prompt, _extractor_user_prompt(claim))
        return _parse_extraction(data, claim)
