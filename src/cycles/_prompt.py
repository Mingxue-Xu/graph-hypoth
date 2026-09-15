"""Small deterministic prompt fragments shared by cycle seams."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def passage_block(passages: Iterable[str]) -> str:
    """Render passages in stable input order with human-readable one-based labels."""
    return "\n\n".join(
        f"### Passage {index}\n{passage}"
        for index, passage in enumerate(passages, start=1)
    )


def candidate_brief(nodes: Sequence[Any]) -> str:
    """Render the structural fields of candidate nodes as a compact prompt fragment."""
    return "; ".join(
        f"{node.label} ({node.type}): {node.definition}" for node in nodes
    )


def edge_brief(edges: Sequence[Any]) -> str:
    """Render candidate edges without adding rationale or inferred context."""
    rendered: list[str] = []
    for edge in edges:
        sources = ",".join(edge.source_node_ids)
        targets = ",".join(edge.target_node_ids)
        mechanism = f" [{edge.mechanism}]" if edge.mechanism else ""
        rendered.append(f"{sources} --{edge.relation_type}--> {targets}{mechanism}")
    return "; ".join(rendered)
