"""Typed graph-delta envelope, five mutation families, and content-addressed operations.

Each cycle supplies a family-specific payload. Extraction operations use deterministic node and
edge identities, while every proposal derives its idempotency key and hashes through
``stable_hash_payload``.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from src.events import stable_hash_payload
from src.graph_store import (
    CausalEdge,
    ConceptNode,
    EdgeStatus,
    EvidenceLink,
    EvidenceRole,
    ExperimentPlan,
    compute_target_id,
)
from src.retrieval import scoring_defaults as sd

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def norm(text: str) -> tuple[str, ...]:
    """Return lowercase alphanumeric tokens minus the pinned stopwords.

    Token order is preserved; reproducibility comes from the version-stamped
    ``scoring_defaults.STOPWORDS``.
    """
    return tuple(t for t in _TOKEN.findall(text.lower()) if t not in sd.STOPWORDS)


def compute_node_id(node_type: str, label: str, definition: str) -> str:
    """Derive a content-addressed node ID from type, label, and definition."""
    return stable_hash_payload(
        {"type": node_type, "label": list(norm(label)), "definition": list(norm(definition))}
    )


def compute_edge_id(
    source_node_ids: list[str], target_node_ids: list[str], direction: str, relation_type: str
) -> str:
    """Derive a content-addressed edge ID from its endpoints and relation."""
    return stable_hash_payload(
        {
            "source_node_ids": sorted(source_node_ids),
            "target_node_ids": sorted(target_node_ids),
            "direction": direction,
            "relation_type": relation_type,
        }
    )


def build_node(
    *,
    label: str,
    definition: str = "",
    type: str = "",
    aliases: list[str] | None = None,
    scope_qualifiers: list[str] | None = None,
    provenance: list[dict[str, Any]] | None = None,
) -> ConceptNode:
    """Construct a concept node with a content-addressed ID."""
    return ConceptNode(
        node_id=compute_node_id(type, label, definition),
        label=label,
        definition=definition,
        type=type,
        aliases=aliases or [],
        scope_qualifiers=scope_qualifiers or [],
        provenance=provenance or [],
    )


def build_edge(
    *,
    source_node_ids: list[str],
    target_node_ids: list[str],
    direction: str,
    relation_type: str,
    mechanism: str = "",
    conditions: list[str] | None = None,
    confounders: list[str] | None = None,
    open_risks: list[str] | None = None,
) -> CausalEdge:
    """Construct a causal edge with a content-addressed ID and ``unverified`` status."""
    return CausalEdge(
        edge_id=compute_edge_id(source_node_ids, target_node_ids, direction, relation_type),
        source_node_ids=source_node_ids,
        target_node_ids=target_node_ids,
        direction=direction,
        relation_type=relation_type,
        mechanism=mechanism,
        conditions=conditions or [],
        confounders=confounders or [],
        status=EdgeStatus.UNVERIFIED,
        open_risks=open_risks or [],
    )


class DeltaFamily(StrEnum):
    """Closed set of graph-delta families."""

    EXTRACT = "extract"
    PRIORITY = "priority"
    VERIFY = "verify"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"


class VerificationVerdict(StrEnum):
    """Closed five-label verification-verdict set."""

    SUPPORT = "support"
    CONTRADICT = "contradict"
    QUALIFY = "qualify"
    INSUFFICIENT = "insufficient"
    NOT_CAUSAL = "not_causal"


VERDICT_TO_STATUS: dict[VerificationVerdict, EdgeStatus] = {
    VerificationVerdict.SUPPORT: EdgeStatus.SUPPORTED,
    VerificationVerdict.CONTRADICT: EdgeStatus.CONTRADICTED,
    VerificationVerdict.QUALIFY: EdgeStatus.QUALIFIED,
    VerificationVerdict.INSUFFICIENT: EdgeStatus.INSUFFICIENT,
    VerificationVerdict.NOT_CAUSAL: EdgeStatus.NOT_CAUSAL,
}


# --- Operation union shared by extraction and hypothesis deltas ----------------------
class AddNodeOp(BaseModel):
    op_type: Literal["add_node"] = "add_node"
    node: ConceptNode


class AddEdgeOp(BaseModel):
    op_type: Literal["add_edge"] = "add_edge"
    edge: CausalEdge


class MergeNodesOp(BaseModel):
    op_type: Literal["merge_nodes"] = "merge_nodes"
    survivor_node_id: str
    merged_node_id: str


Operation = Annotated[
    Union[AddNodeOp, AddEdgeOp, MergeNodesOp], Field(discriminator="op_type")
]


# --- Per-family payloads --------------------------------------------------------------
class ExtractPayload(BaseModel):
    kind: Literal["extract"] = "extract"
    operations: list[Operation] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class PriorityPayload(BaseModel):
    # A priority delta may carry only priority and focus, so
    # a smuggled status/confidence field is rejected at 1_schema, never silently dropped.
    model_config = ConfigDict(extra="forbid")

    kind: Literal["priority"] = "priority"
    node_or_edge_ids: list[str] = Field(default_factory=list)
    priority_values: list[float] = Field(default_factory=list)
    focus_notes: str = ""


class VerifyPayload(BaseModel):
    kind: Literal["verify"] = "verify"
    edge_id: str
    evidence_role: EvidenceRole
    verdict: VerificationVerdict
    confidence: float | None = None
    evidence_links: list[EvidenceLink] = Field(default_factory=list)
    open_risks: list[str] = Field(default_factory=list)

    @property
    def target_id(self) -> str:
        return compute_target_id(self.edge_id, self.evidence_role)


class HypothesisPayload(BaseModel):
    kind: Literal["hypothesis"] = "hypothesis"
    new_nodes: list[ConceptNode] = Field(default_factory=list)
    new_edges: list[CausalEdge] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    def to_operations(self) -> list[AddNodeOp | AddEdgeOp]:
        """Desugar ``new_nodes`` and ``new_edges`` into the closed operation union.

        Hypothesis deltas reuse ``add_node`` and ``add_edge`` operations,
        so ``apply`` mutates ONLY via that union (handoff guardrail)."""
        return [AddNodeOp(node=node) for node in self.new_nodes] + [
            AddEdgeOp(edge=edge) for edge in self.new_edges
        ]


class ExperimentPayload(BaseModel):
    # Δ^experiment binds ONE validated experiment_plan to a committed hypothesis edge
    # Like a priority delta, an experiment delta carries no verdict or status change. The
    # grounding evidence ids ride in referenced_evidence_ids (checked at 1_refs); author
    # and provenance ride on the envelope (GraphDeltaProposal), as with the other families.
    kind: Literal["experiment"] = "experiment"
    hypothesis_id: str
    # SerializeAsAny lets ExperimentPlan omit its empty, legacy-compatible
    # retrieval_batch_id without union-serializer warnings.
    experiment_plan: SerializeAsAny[ExperimentPlan]
    referenced_evidence_ids: list[str] = Field(default_factory=list)


DeltaPayload = Annotated[
    Union[ExtractPayload, PriorityPayload, VerifyPayload, HypothesisPayload, ExperimentPayload],
    Field(discriminator="kind"),
]


class GraphDeltaProposal(BaseModel):
    """A typed proposed mutation from one of the five families.

    ``payload`` carries the family's operations; the common envelope fields carry the
    volatile metadata excluded from the idempotency key.
    """

    family: DeltaFamily
    base_graph_hash: str
    payload: DeltaPayload
    author_role: str = "agent"
    rationale: str = ""
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    referenced_evidence_ids: list[str] = Field(default_factory=list)
    validation_status: str = "pending"

    def operations_canonical(self) -> str:
        """``canonical(delta operations)`` = hash over the family payload only."""
        return stable_hash_payload(self.payload.model_dump(mode="json"))

    def idempotency_key(self) -> str:
        """Hash the base graph and canonical payload, excluding volatile metadata."""
        return stable_hash_payload(self.base_graph_hash + self.operations_canonical())

    def delta_hash(self) -> str:
        """``hash(Δ_i)`` of the proposed delta for the receipt (CT-05).

        Excludes only ``validation_status`` (set by the validator after the fact).
        """
        return stable_hash_payload(
            {
                "family": self.family.value,
                "base_graph_hash": self.base_graph_hash,
                "payload": self.payload.model_dump(mode="json"),
                "author_role": self.author_role,
                "rationale": self.rationale,
                "provenance": self.provenance,
                "referenced_evidence_ids": sorted(self.referenced_evidence_ids),
            }
        )
