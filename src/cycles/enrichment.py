"""Literature concept mining and graph enrichment.

The cycle mines outcome-anchored concepts from retrieved papers and commits them as unverified,
content-addressed nodes with literature provenance. Mined concepts reuse extraction operations;
there is no separate enrichment-delta family. The deterministic core filters by the outcome
anchor, applies an admission cap, builds the delta, and commits recorded miner output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.cycles.extraction import (
    CanonicalizationAdjudicator,
    canonicalize_concepts,
)
from src.delta import (
    AddNodeOp,
    DeltaFamily,
    ExtractPayload,
    GraphDeltaProposal,
    MergeNodesOp,
    build_node,
)
from src.graph_store import ConceptNode
from src.retrieval.similarity import Embedder

# This provenance marker distinguishes literature-mined nodes from claim-extracted nodes. It is an
# audit annotation; the evidence itself is reviewed later.
ENRICHMENT_SOURCE = "literature_enrichment"


@dataclass(frozen=True)
class MinedConcept:
    """One impact-concept the miner extracts from the retrieved full-text passages.

    ``bears_on_outcome`` is the miner's outcome-anchor flag: true for variables, mechanisms, or
    conditions that causally bear on the claim's outcome or rank-selection target; False for
    off-outcome items (e.g. KV-cache) the anchor rule drops. The ``evidence_id`` / ``paper_id`` /
    ``matched_quote_span`` become the node's literature provenance. ``source_context`` is one
    sentence describing what this source argues the concept does, distinct from the universal,
    source-independent definition.
    """

    label: str
    type: str = ""
    definition: str = ""
    source_context: str = ""
    bears_on_outcome: bool = True
    evidence_id: str = ""
    paper_id: str = ""
    matched_quote_span: str = ""
    rationale: str = ""


def _parse_mined_concepts(data: dict[str, Any]) -> list[MinedConcept]:
    out: list[MinedConcept] = []
    for raw in data.get("concepts") or []:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label", "")).strip()
        if not label:
            continue
        out.append(
            MinedConcept(
                label=label,
                type=str(raw.get("type", "")),
                definition=str(raw.get("definition", "")),
                source_context=str(raw.get("source_context", "")),
                bears_on_outcome=bool(raw.get("bears_on_outcome", True)),
                evidence_id=str(raw.get("evidence_id", "")),
                paper_id=str(raw.get("paper_id", "")),
                matched_quote_span=str(raw.get("matched_quote_span", "")),
                rationale=str(raw.get("rationale", "")),
            )
        )
    return out


def outcome_anchored(concepts: Sequence[MinedConcept]) -> list[MinedConcept]:
    """Keep concepts that bear on the outcome and drop off-outcome concepts."""
    return [c for c in concepts if c.bears_on_outcome]


def enrichment_provenance(concept: MinedConcept) -> dict[str, Any]:
    """Build the literature-provenance entry for a mined node.

    Carries the Research Synthesist's per-source ``source_context`` so the proposal view can render
    what this source argues the concept does — provenance is per-source, so a concept mined from
    two papers keeps each entry's own ``source_context`` (not in the content-addressed node_id)."""
    return {
        "source": ENRICHMENT_SOURCE,
        "evidence_id": concept.evidence_id,
        "paper_id": concept.paper_id,
        "matched_quote_span": concept.matched_quote_span,
        "source_context": concept.source_context,
    }


def mined_to_node(concept: MinedConcept) -> ConceptNode:
    """Build a content-addressed ``ConceptNode`` from a mined concept, tagged with its
    literature provenance through ``delta.build_node``."""
    return build_node(
        label=concept.label,
        definition=concept.definition,
        type=concept.type,
        provenance=[enrichment_provenance(concept)],
    )


def concepts_to_nodes(concepts: Sequence[MinedConcept]) -> list[ConceptNode]:
    """Build the de-duped ``ConceptNode``s for a concept set (content-addressed ids: a re-mined
    identical concept collapses to one node). When the SAME concept is mined from two papers, the
    node keeps both provenance entries, not just the first."""
    first_concept: dict[str, MinedConcept] = {}
    provenance: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for concept in concepts:
        node = mined_to_node(concept)
        if node.node_id not in first_concept:
            first_concept[node.node_id] = concept
            provenance[node.node_id] = list(node.provenance)
            order.append(node.node_id)
        else:
            provenance[node.node_id].extend(node.provenance)  # merge the other source's provenance
    return [
        build_node(
            label=first_concept[nid].label,
            definition=first_concept[nid].definition,
            type=first_concept[nid].type,
            provenance=provenance[nid],
        )
        for nid in order
    ]


def canonicalize_mined_nodes(
    nodes: Sequence[ConceptNode],
    *,
    embedder: Embedder,
    adjudicator: CanonicalizationAdjudicator | None = None,
) -> list[ConceptNode]:
    """Collapse near-duplicate mined nodes into canonical survivors.

    Each ``canonicalize_concepts`` cluster becomes one survivor
    (the lexicographically-smaller id) carrying the UNION of the cluster's provenance
    (cross-source corroboration). Deterministic given the recorded adjudications."""
    by_id = {node.node_id: node for node in nodes}
    survivors: list[ConceptNode] = []
    for cluster in canonicalize_concepts(nodes, embedder=embedder, adjudicator=adjudicator):
        survivor = by_id[min(cluster)]
        provenance = list(survivor.provenance)
        for node_id in cluster:
            if node_id != survivor.node_id:
                provenance.extend(by_id[node_id].provenance)
        survivors.append(
            build_node(
                label=survivor.label, definition=survivor.definition, type=survivor.type,
                provenance=provenance,
            )
        )
    return survivors


def build_enrichment_delta(
    *,
    base_graph_hash: str,
    nodes: Sequence[ConceptNode],
    merge_ops: Sequence[MergeNodesOp] = (),
    author_role: str = "research_synthesist",
    rationale: str = "",
) -> GraphDeltaProposal:
    """Assemble the typed ``Δ^extract`` for the (already built + de-duped/canonicalized) mined
    nodes: one ``AddNodeOp`` per node, followed by confirmed merge operations. Node additions come
    first so each merge's endpoints already
    exist in the store when ``apply`` walks the operations in order. The delta mutates ONLY via
    the closed Operation union and lands as unverified structure."""
    return GraphDeltaProposal(
        family=DeltaFamily.EXTRACT,
        base_graph_hash=base_graph_hash,
        payload=ExtractPayload(
            operations=[AddNodeOp(node=node) for node in nodes] + list(merge_ops)
        ),
        author_role=author_role,
        rationale=rationale,
    )
