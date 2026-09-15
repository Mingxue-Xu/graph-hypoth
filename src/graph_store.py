"""Graph object model and durable versioned store.

The transaction *spine* every reasoning cycle commits through. This module owns the
durable entities and the content-addressed store hash; the delta envelope/ops live in
``delta.py`` and the commit invariant in ``validator.py``.

Hashing is the pinned ``stable_hash_payload`` canonicalizer (sorted-key compact JSON +
SHA-256, ``events.py:15``) so semantically identical states hash identically across
runs and processes.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from src.events import stable_hash_payload


class EdgeStatus(StrEnum):
    """Closed verification-status enum carried only by ``CausalEdge``."""

    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    QUALIFIED = "qualified"
    INSUFFICIENT = "insufficient"
    NOT_CAUSAL = "not_causal"


class EvidenceRole(StrEnum):
    """Closed, required relation-role enum with no default.

    Citation-distance ``direct``/``adjacent`` are internal scorer variable names, never
    relation-role values.
    """

    SUPPORT = "support"
    CONTRADICTION = "contradiction"
    QUALIFICATION = "qualification"
    MECHANISM = "mechanism"
    CONFOUNDER = "confounder"


class ExperimentGroundingStatus(StrEnum):
    """Deterministic provenance state for a committed experiment plan.

    The Experiment Designer never authors this value.  Experiment orchestration
    stamps it after the bounded design loop so consumers can distinguish a
    literature-grounded plan from an explicitly ungrounded draft without
    inferring that distinction from retrieval side effects.
    """

    GROUNDED = "grounded"
    NO_RELEVANT_METHODS = "no_relevant_methods"
    NO_METHODS_RETRIEVED = "no_methods_retrieved"


class ConceptNode(BaseModel):
    """An inspectable semantic atom with no status; status belongs to edges."""

    node_id: str
    label: str
    aliases: list[str] = Field(default_factory=list)
    definition: str = ""
    type: str = ""
    scope_qualifiers: list[str] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    user_priority: float | None = None
    uncertainty: float | None = None


class CausalEdge(BaseModel):
    """A typed directed causal assertion. New edges begin ``unverified``."""

    edge_id: str
    source_node_ids: list[str]
    target_node_ids: list[str]
    direction: str
    relation_type: str
    mechanism: str = ""
    conditions: list[str] = Field(default_factory=list)
    confounders: list[str] = Field(default_factory=list)
    status: EdgeStatus = EdgeStatus.UNVERIFIED
    confidence: float | None = None
    open_risks: list[str] = Field(default_factory=list)


class EvidenceLink(BaseModel):
    """The committed bridge from a graph target to a ledger record.

    The evidence body stays in the ledger; the link copies no paper object. ``signal`` stores the
    complete ``EvidenceSignalBundle`` as an opaque mapping.
    """

    target_id: str
    verification_task_id: str
    evidence_id: str
    evidence_role: EvidenceRole
    signal: dict[str, Any] | None = None
    retrieval_event_id: str
    committed_transaction_id: str
    trust_tier: str


def compute_target_id(node_or_edge_id: str, evidence_role: EvidenceRole) -> str:
    """Return ``stable_hash(node_or_edge_id, evidence_role)``.

    An opaque, replay-stable handle; callers keep the node/edge ref and role as separate
    explicit fields and never parse them back out of the id.
    """
    return stable_hash_payload(
        {"node_or_edge_id": node_or_edge_id, "evidence_role": EvidenceRole(evidence_role).value}
    )


class GraphTarget(BaseModel):
    """Scheduling and verification projection over ``(node|edge, role)``.

    This is not stored state; the store holds no ``targets`` collection.
    """

    node_or_edge_id: str
    target_type: str
    evidence_role: EvidenceRole
    question: str | None = None
    criteria: str | None = None
    priority: float | None = None
    evidence_gap: bool | None = None

    @property
    def target_id(self) -> str:
        return compute_target_id(self.node_or_edge_id, self.evidence_role)


class ExperimentDesign(StrEnum):
    """Closed experiment-design enum for a validated plan."""

    RANDOMIZED_CONTROLLED = "randomized_controlled"
    CONTROLLED_OBSERVATIONAL = "controlled_observational"
    ABLATION = "ablation"
    BENCHMARK_COMPARISON = "benchmark_comparison"
    SIMULATION = "simulation"


class ControlFactor(BaseModel):
    """A confounder, mediator, or moderator controlled by an experiment."""

    factor: str
    from_graph: bool = False
    handling: str = ""


class MaterialItem(BaseModel):
    """A dataset, benchmark, or material with an optional ledger citation."""

    item: str
    evidence_id: str | None = None


class MetricSpec(BaseModel):
    """A metric with its predicted direction and optional ledger citation."""

    metric: str
    predicted_direction: str = ""
    evidence_id: str | None = None


class Feasibility(BaseModel):
    """Resources, time, and primary-risk statement."""

    resources: str = ""
    time: str = ""
    main_risk: str = ""


class GroundingItem(BaseModel):
    """One relied-on evidence ID with its verbatim quote span."""

    evidence_id: str
    quote_span: str = ""


class ExperimentPlan(BaseModel):
    """A validated, retrieval-grounded experiment plan bound to a hypothesis edge.

    The durable payload Experiment Designer designs, Experiment Validator grades, and ``Δ^experiment`` commits. The validator
    does NOT inspect its internals — it gates on ``hypothesis_id`` + ``referenced_evidence_ids``
    plus the no-status-change invariant; the Experiment Designer owns the field-content contract.
    """

    hypothesis_under_test: str
    operationalization: str = ""
    design: ExperimentDesign
    design_rationale: str = ""
    intervention_or_manipulation: str = ""
    comparison_baseline: str = ""
    controls_and_confounders: list[ControlFactor] = Field(default_factory=list)
    materials_or_data: list[MaterialItem] = Field(default_factory=list)
    metrics: list[MetricSpec] = Field(default_factory=list)
    procedure: list[str] = Field(default_factory=list)
    expected_outcome: str = ""
    falsification: str = ""
    feasibility: Feasibility = Field(default_factory=Feasibility)
    grounding: list[GroundingItem] = Field(default_factory=list)
    # Deterministic provenance state stamped by orchestration.  ``None`` is the
    # legacy/default value and remains absent from serialization so historical
    # plan and graph hashes stay stable.
    grounding_status: ExperimentGroundingStatus | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # Deterministic provenance scope for the one targeted methods retrieval that
    # supplied this plan. The Designer does not author it; orchestration stamps it
    # after design. Empty legacy/default values stay out of serialization so old
    # graph and delta hashes remain unchanged.
    retrieval_batch_id: str = Field(default="", exclude_if=lambda value: not value)


class CausalClaimGraphStore(BaseModel):
    """Durable, versioned graph state.

    Agents never mutate it directly; deltas are applied/rejected by the validator.
    ``base_hash`` is the content hash of the current state and is the single source of
    truth for the optimistic-concurrency gate.
    """

    graph_id: str = "graph"
    version: int = 0
    nodes: dict[str, ConceptNode] = Field(default_factory=dict)
    edges: dict[str, CausalEdge] = Field(default_factory=dict)
    evidence_links: list[EvidenceLink] = Field(default_factory=list)
    scope_context: dict[str, Any] = Field(default_factory=dict)
    # Δ^experiment durable content: validated plans keyed by the bound hypothesis edge id
    # Empty by default so graphs without plans stay byte-identical.
    experiment_plans: dict[str, ExperimentPlan] = Field(default_factory=dict)

    def content_hash(self) -> str:
        """Hash over the durable content only (nodes/edges/links/scope).

        Excludes ``graph_id`` and ``version`` so the hash is content-addressed: replaying
        the same deltas reproduces the same hash regardless of path or graph id. Nodes and
        edges are keyed dicts (order-independent after sort-keys); evidence links are a
        list, so they are sorted by their own canonical hash for order-independence.
        """
        payload = {
            "nodes": {nid: n.model_dump(mode="json") for nid, n in self.nodes.items()},
            "edges": {eid: e.model_dump(mode="json") for eid, e in self.edges.items()},
            "evidence_links": sorted(
                (link.model_dump(mode="json") for link in self.evidence_links),
                key=stable_hash_payload,
            ),
            "scope_context": self.scope_context,
        }
        # experiment_plans enters the hash ONLY when non-empty, so a plans-free graph (the
        # experiment-flag-OFF path) hashes byte-identically to the pre-Δ^experiment store.
        if self.experiment_plans:
            payload["experiment_plans"] = {
                eid: plan.model_dump(mode="json") for eid, plan in self.experiment_plans.items()
            }
        return stable_hash_payload(payload)

    @property
    def base_hash(self) -> str:
        return self.content_hash()
