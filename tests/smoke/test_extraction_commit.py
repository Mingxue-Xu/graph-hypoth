"""Extraction-cycle smoke — a seed claim compiles to a committed, normalized UNVERIFIED subgraph.

Exercises the whole extraction happy path through the real validator and transaction log: a
fake (LLM-free) extractor proposes ConceptNodes/CausalEdges, the Δ^extract builder packages
them, and the commit lands every node/edge at ``status = unverified`` with an auditable
receipt. Deterministic; no network or embedder.
"""

from __future__ import annotations

import pytest

from src.cycles.extraction import ClaimExtraction, run_extraction_cycle
from src.delta import build_edge, build_node
from src.graph_store import EdgeStatus

pytestmark = pytest.mark.smoke


class _FakeExtractor:
    def extract(self, claim):
        n1 = build_node(label="physical activity", type="exposure/intervention",
                        definition="bodily movement", provenance=[{"source": "claim", "span": [0, 17]}])
        n2 = build_node(label="cardiovascular disease", type="outcome",
                        definition="disease of heart and vessels")
        edge = build_edge(source_node_ids=[n1.node_id], target_node_ids=[n2.node_id],
                          direction="causal", relation_type="reduces")
        return ClaimExtraction(nodes=(n1, n2), edges=(edge,), assumptions=("adult population",))


def _embedder(texts):
    return [[1.0, 0.0] for _ in texts]


def test_seed_claim_becomes_a_committed_unverified_subgraph():
    result = run_extraction_cycle(
        "physical activity reduces cardiovascular disease",
        extractor=_FakeExtractor(), embedder=_embedder,
    )

    # exactly one committed Δ^extract; graph advanced
    assert result.transaction.accepted is True
    assert result.store.version == 1

    # 100% of committed nodes/edges land unverified and carry provenance; 0 evidence links
    assert len(result.store.nodes) == 2 and len(result.store.edges) == 1
    edge = next(iter(result.store.edges.values()))
    assert edge.status == EdgeStatus.UNVERIFIED
    assert edge.confidence is None                      # no evidence-derived confidence
    assert result.store.evidence_links == []            # extraction does not create evidence links
    assert any(n.provenance for n in result.store.nodes.values())  # provenance links to the input

    # An auditable receipt was recorded; this was not a silent no-op.
    assert result.transaction.receipt.status == "accepted"
    assert result.transaction.receipt.tx_id
