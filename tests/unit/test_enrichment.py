"""Deterministic concept-mining helpers used by the Research Synthesist.

The tests cover outcome-anchor admission, provenance on extraction deltas,
content-addressed node construction, and near-duplicate canonicalization.
"""

from __future__ import annotations

from src.cycles.enrichment import (
    ENRICHMENT_SOURCE,
    MinedConcept,
    build_enrichment_delta,
    canonicalize_mined_nodes,
    concepts_to_nodes,
    enrichment_provenance,
    mined_to_node,
    outcome_anchored,
)
from src.delta import AddNodeOp, DeltaFamily
from src import graph_config_defaults as gcd


def _demo_concepts():
    """Return three outcome-bearing concepts and one concept the anchor rule must drop."""
    return [
        MinedConcept(label="weight reconstruction error", type="mediator",
                     definition="frobenius error from singular value truncation",
                     bears_on_outcome=True, evidence_id="ev-svdllm", paper_id="SVD-LLM",
                     matched_quote_span="truncation error mediates accuracy loss"),
        MinedConcept(label="activation outlier severity", type="confounder",
                     bears_on_outcome=True, evidence_id="ev-asvd", paper_id="ASVD"),
        MinedConcept(label="fisher information weighting", type="mechanism",
                     bears_on_outcome=True, evidence_id="ev-gfwsvd", paper_id="GFWSVD"),
        MinedConcept(label="kv cache memory footprint", type="variable",
                     bears_on_outcome=False, evidence_id="ev-kv", paper_id="SVD-LLM"),  # OFF-outcome
    ]


# --- Outcome-anchored admission --------------------------------------------------------
def test_outcome_anchored_excludes_off_outcome_concepts():
    kept = outcome_anchored(_demo_concepts())
    labels = [c.label for c in kept]
    assert "kv cache memory footprint" not in labels       # off-outcome dropped
    assert len(kept) == 3                                    # the 3 outcome-bearing concepts kept


# --- Provenance contract and extraction-delta reuse -----------------------------------
def test_enrichment_provenance_tags_source_and_evidence():
    c = _demo_concepts()[0]
    prov = enrichment_provenance(c)
    assert prov["source"] == ENRICHMENT_SOURCE              # distinguishes mined from claim-extracted
    assert prov["evidence_id"] == "ev-svdllm"
    assert prov["paper_id"] == "SVD-LLM"


def test_enrichment_provenance_carries_source_context():
    # The Research Synthesist mine turn's per-concept source_context must ride on the node's provenance so the
    # PROPOSE turn's <committed_graph> view can render it (synthesist._synthesist_committed_graph_view);
    # otherwise the new field is inert end-to-end.
    c = MinedConcept(
        label="rate-distortion floor", type="mechanism", bears_on_outcome=True,
        evidence_id="ev-x", paper_id="X",
        source_context="the paper argues the floor caps factuality under compression",
    )
    assert enrichment_provenance(c)["source_context"] == (
        "the paper argues the floor caps factuality under compression"
    )
    assert mined_to_node(c).provenance[0]["source_context"] == (
        "the paper argues the floor caps factuality under compression"
    )


def test_mined_to_node_carries_literature_provenance_and_fields():
    node = mined_to_node(_demo_concepts()[0])
    assert node.label == "weight reconstruction error"
    assert node.type == "mediator"
    assert node.provenance[0]["source"] == ENRICHMENT_SOURCE
    assert node.provenance[0]["evidence_id"] == "ev-svdllm"


def test_build_enrichment_delta_is_extract_family_with_addnode_ops_only():
    nodes = concepts_to_nodes(outcome_anchored(_demo_concepts()))
    delta = build_enrichment_delta(base_graph_hash="h0", nodes=nodes)
    assert delta.family == DeltaFamily.EXTRACT                      # reuse Δ^extract, no Δ^enrich
    assert delta.payload.operations                                 # non-empty
    assert all(isinstance(op, AddNodeOp) for op in delta.payload.operations)  # Operation union only


# --- Near-duplicate canonicalization ---------------------------------------------------
class _AlwaysSameAdjudicator:
    def same_concept(self, c_a, c_b):
        return True


def _const_embedder(texts):
    return [[1.0, 0.0] for _ in texts]  # cos == 1 for any pair -> merge_score lands in the band


def test_canonicalize_mined_nodes_collapses_near_duplicates():
    # Two near-duplicate mined concepts (different surface labels, the same concept). Exact
    # content deduplication leaves them distinct; the injected adjudicator collapses them to one node,
    # carrying the UNION of the cluster's provenance (cross-source corroboration).
    near_dups = concepts_to_nodes([
        MinedConcept(label="reconstruction error", type="mediator",
                     definition="frobenius error from truncation", bears_on_outcome=True,
                     evidence_id="ev-a", paper_id="A"),
        MinedConcept(label="truncation error", type="mediator",
                     definition="error from discarding singular values", bears_on_outcome=True,
                     evidence_id="ev-b", paper_id="B"),
    ])
    survivors = canonicalize_mined_nodes(
        near_dups, embedder=_const_embedder, adjudicator=_AlwaysSameAdjudicator()
    )
    assert len(survivors) == 1                                          # collapsed to one node
    assert {p["evidence_id"] for p in survivors[0].provenance} == {"ev-a", "ev-b"}  # both kept


def test_canonicalize_mined_nodes_without_adjudicator_keeps_distinct():
    # No adjudicator -> only exact content-dedup; the two distinct-label concepts stay two nodes.
    near_dups = concepts_to_nodes([
        MinedConcept(label="reconstruction error", type="mediator", bears_on_outcome=True,
                     evidence_id="ev-a"),
        MinedConcept(label="truncation error", type="mediator", bears_on_outcome=True,
                     evidence_id="ev-b"),
    ])
    survivors = canonicalize_mined_nodes(near_dups, embedder=_const_embedder, adjudicator=None)
    assert len(survivors) == 2


def test_concepts_to_nodes_merges_provenance_across_duplicate_sources():
    # The same concept mined from two papers -> one content-addressed node carrying BOTH provenances.
    c_a = MinedConcept(label="reconstruction error", type="mediator", definition="d",
                       evidence_id="ev-a", paper_id="A")
    c_b = MinedConcept(label="reconstruction error", type="mediator", definition="d",
                       evidence_id="ev-b", paper_id="B")
    nodes = concepts_to_nodes([c_a, c_b])
    assert len(nodes) == 1                                    # deduped by content-addressed id
    assert {p["evidence_id"] for p in nodes[0].provenance} == {"ev-a", "ev-b"}  # both papers recorded


# --- Audited configuration -------------------------------------------------------------
def test_enrichment_limit_is_positive_and_audited():
    assert isinstance(gcd.ENRICHMENT_MAX_CONCEPTS, int) and gcd.ENRICHMENT_MAX_CONCEPTS >= 1
    audit = gcd.audit_dict()
    assert audit["enrichment_max_concepts"] == gcd.ENRICHMENT_MAX_CONCEPTS
