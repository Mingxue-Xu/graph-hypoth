"""Offline view-model for the claim-graph canvas of the local web app.

Projects one run directory into a single JSON document the browser draws: ``graph.json`` (the
committed store) supplies nodes, edges, statuses, evidence links, and experiment plans;
``edge_table.csv`` supplies the fused verdict per edge, which is recorded even when the commit was
withheld; the run's ``*.sqlite`` retrieval cache supplies the evidence bodies the links point at; and
``trace/plain_language.json`` names each confirmed hypothesis's committed edges. It reads artifacts
only -- no model calls, no graph-state mutation -- so it can never disagree with the pages it links
to.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.connected_render import node_class
from src.graph_store import EvidenceRole, compute_target_id
from src.provenance_links import load_retrieval_batches, node_sources

STATUS_ORDER: tuple[str, ...] = (
    "supported",
    "contradicted",
    "qualified",
    "not_causal",
    "insufficient",
    "unverified",
)


def read_edge_table(path: Path) -> dict[str, dict[str, str]]:
    """``edge_id -> row`` from the exported edge table; ``{}`` when the file is absent."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            str(row["edge_id"]): {key: (value or "") for key, value in row.items() if key}
            for row in csv.DictReader(handle)
            if row.get("edge_id")
        }


def load_evidence_pool(run_dir: Path, run_id: str | None = None) -> list[dict[str, Any]]:
    """Every retrieved evidence item logged under ``run_dir``, stamped with its batch id.

    Filters to ``run_id`` when rows for it exist; an imported directory whose logical run id is
    unknown falls back to every row in the file. A foreign or unreadable sqlite file contributes
    nothing rather than hiding the graph.
    """
    items: list[dict[str, Any]] = []
    run_ids: tuple[str | None, ...] = (run_id, None) if run_id else (None,)
    for db_path in sorted(run_dir.glob("*.sqlite")):
        batches: list[dict[str, Any]] = []
        for candidate_run_id in run_ids:
            try:
                batches = load_retrieval_batches(db_path, run_id=candidate_run_id)
            except Exception:  # noqa: BLE001 -- see the docstring
                batches = []
            if batches:
                break
        for batch in batches:
            for item in batch.get("evidence") or []:
                if isinstance(item, dict):
                    items.append({**item, "_batch_id": str(batch.get("batch_id") or "")})
    return items


def _evidence_by_id(pool: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for item in pool:
        evidence_id = str(item.get("evidence_id") or item.get("id") or "").strip()
        if evidence_id:
            by_id.setdefault(evidence_id, []).append(item)
    return by_id


def _resolve_link(
    link: dict[str, Any], by_id: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """One evidence link as the browser shows it.

    The ledger body (title, url, quote) comes from the pool when it resolves; the signal fields the
    link itself carries (matched span, association score) need no pool. ``evidence_id`` is local to
    one retrieval batch, so a batch that matches the link's retrieval event id wins over a mere id
    collision, and a collision without such a match is flagged ``ambiguous``.
    """
    signal = link.get("signal") or {}
    evidence_id = str(link.get("evidence_id") or "")
    event_id = str(link.get("retrieval_event_id") or "")
    candidates = by_id.get(evidence_id, [])
    exact = [
        item
        for item in candidates
        if event_id
        and event_id
        in (
            str(item.get("_batch_id") or ""),
            str((item.get("metadata") or {}).get("retrieval_batch_id") or ""),
            str(item.get("tool_call_id") or ""),
        )
    ]
    chosen = exact[0] if exact else (candidates[0] if candidates else None)
    view: dict[str, Any] = {
        "evidence_id": evidence_id,
        "role": link.get("evidence_role"),
        "trust_tier": link.get("trust_tier"),
        "matched_quote_span": signal.get("matched_quote_span"),
        "association_score": signal.get("association_score"),
        "resolved": chosen is not None,
        "ambiguous": not exact and len(candidates) > 1,
        "title": "",
        "url": "",
        "quote": "",
        "source": "",
        "authors": [],
        "published_date": None,
    }
    if chosen is not None:
        view.update(
            title=str(chosen.get("title") or ""),
            url=str(chosen.get("url") or ""),
            quote=str(chosen.get("quote") or ""),
            source=str(chosen.get("source") or ""),
            authors=[str(author) for author in (chosen.get("authors") or [])],
            published_date=chosen.get("published_date"),
        )
    return view


def read_sidecar(run_dir: Path) -> dict[str, Any]:
    """The trace sidecar ``trace/plain_language.json`` as a dict; ``{}`` when absent or unreadable."""
    sidecar = run_dir / "trace" / "plain_language.json"
    if not sidecar.is_file():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def hypothesis_edge_ids(run_dir: Path) -> dict[str, list[str]]:
    """``candidate_id -> committed edge ids`` from the trace sidecar; ``{}`` when absent or old."""
    mapping = read_sidecar(run_dir).get("hypothesis_edge_ids")
    if not isinstance(mapping, dict):
        return {}
    return {
        str(candidate_id): [str(edge_id) for edge_id in (edge_ids or [])]
        for candidate_id, edge_ids in mapping.items()
    }


def surfaced_from_sidecar(
    sidecar: dict[str, Any], plans: dict[str, Any]
) -> list[dict[str, Any]]:
    """One row per surfaced hypothesis the trace report wrote, in the report's order.

    The sidecar's ``hypotheses`` cards carry the plain-language headline; ``hypothesis_edge_ids``
    names the committed edges of confirmed candidates; ``experiment_attempts`` records how the
    experiment stage ended for each of those. Older sidecars without cards yield ``[]``.
    """
    cards = sidecar.get("hypotheses")
    if not isinstance(cards, dict):
        return []
    edge_ids = sidecar.get("hypothesis_edge_ids")
    edge_ids = edge_ids if isinstance(edge_ids, dict) else {}
    attempts = sidecar.get("experiment_attempts")
    attempts = attempts if isinstance(attempts, dict) else {}
    rows: list[dict[str, Any]] = []
    for candidate_id, card in cards.items():
        card = card if isinstance(card, dict) else {}
        ids = [str(edge_id) for edge_id in (edge_ids.get(candidate_id) or [])]
        attempt = attempts.get(candidate_id)
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "headline": str(card.get("headline") or ""),
                "hypothesis_edge_ids": ids,
                "committed": bool(ids),
                "experiment_plan": any(edge_id in plans for edge_id in ids),
                "experiment": (
                    {
                        "status": attempt.get("status"),
                        "grounding_status": attempt.get("grounding_status"),
                    }
                    if isinstance(attempt, dict)
                    else None
                ),
            }
        )
    return rows


def build_graph_view(run_dir: Path, *, run_id: str | None = None) -> dict[str, Any]:
    """The canvas document for ``run_dir``: nodes, edges with evidence and plans, and counts.

    ``run_id`` is the logical run id the events database was written under; pass ``None`` for an
    imported directory whose id is unknown. Raises ``FileNotFoundError`` when there is no graph.
    """
    run_dir = Path(run_dir)
    graph_path = run_dir / "graph.json"
    if not graph_path.is_file():
        raise FileNotFoundError(f"no graph.json under {run_dir}")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes_raw: dict[str, dict[str, Any]] = graph.get("nodes") or {}
    edges_raw: dict[str, dict[str, Any]] = graph.get("edges") or {}
    plans: dict[str, Any] = graph.get("experiment_plans") or {}
    verdicts = read_edge_table(run_dir / "edge_table.csv")
    pool = load_evidence_pool(run_dir, run_id)
    by_id = _evidence_by_id(pool)
    links_by_target: dict[str, list[dict[str, Any]]] = {}
    for link in graph.get("evidence_links") or []:
        if isinstance(link, dict):
            links_by_target.setdefault(str(link.get("target_id") or ""), []).append(link)
    sidecar = read_sidecar(run_dir)
    by_candidate = hypothesis_edge_ids(run_dir)
    candidate_of_edge = {
        edge_id: candidate_id
        for candidate_id, edge_ids in by_candidate.items()
        for edge_id in edge_ids
    }

    nodes: list[dict[str, Any]] = []
    coined: set[str] = set()  # proposer-coined concepts carry no provenance at all
    for node_id, node in nodes_raw.items():
        provenance = list(node.get("provenance") or [])
        if not provenance:
            coined.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "label": str(node.get("label") or node_id),
                "type": str(node.get("type") or ""),
                "definition": str(node.get("definition") or ""),
                "aliases": [str(alias) for alias in (node.get("aliases") or [])],
                "scope_qualifiers": [str(q) for q in (node.get("scope_qualifiers") or [])],
                "klass": node_class(node),
                "user_priority": node.get("user_priority"),
                "uncertainty": node.get("uncertainty"),
                "sources": node_sources(node, pool) if pool else [],
                "provenance": provenance,
            }
        )

    edges: list[dict[str, Any]] = []
    for edge_id, edge in edges_raw.items():
        evidence = [
            _resolve_link(link, by_id)
            for role in EvidenceRole
            for link in links_by_target.get(compute_target_id(edge_id, role), [])
        ]
        row = verdicts.get(edge_id, {})
        sources = [s for s in (edge.get("source_node_ids") or []) if s in nodes_raw]
        targets = [t for t in (edge.get("target_node_ids") or []) if t in nodes_raw]
        risks = [str(r) for r in (edge.get("open_risks") or [])] or [
            r for r in (row.get("open_risks") or "").split("; ") if r
        ]
        is_hypothesis = (
            edge_id in candidate_of_edge
            or edge_id in plans
            or any(node_id in coined for node_id in sources + targets)
        )
        edges.append(
            {
                "id": edge_id,
                "sources": sources,
                "targets": targets,
                "relation": str(edge.get("relation_type") or ""),
                "direction": str(edge.get("direction") or ""),
                "mechanism": str(edge.get("mechanism") or ""),
                "conditions": [str(c) for c in (edge.get("conditions") or [])],
                "confounders": [str(c) for c in (edge.get("confounders") or [])],
                "status": str(edge.get("status") or "unverified"),
                "confidence": edge.get("confidence"),
                "verdict": row.get("verdict") or None,
                "open_risks": risks,
                "is_hypothesis": is_hypothesis,
                "candidate_id": candidate_of_edge.get(edge_id),
                "evidence": evidence,
                "plan": plans.get(edge_id),
            }
        )

    status_counts = {status: 0 for status in STATUS_ORDER}
    for view in edges:
        status_counts[view["status"]] = status_counts.get(view["status"], 0) + 1
    return {
        "run_id": run_id,
        "graph_id": graph.get("graph_id"),
        "version": graph.get("version"),
        "seed": str((graph.get("scope_context") or {}).get("seed_claim") or ""),
        "nodes": nodes,
        "edges": edges,
        "hypotheses": by_candidate,
        "surfaced": surfaced_from_sidecar(sidecar, plans),
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "hypothesis_edges": sum(1 for view in edges if view["is_hypothesis"]),
            "experiment_plans": len(plans),
            "evidence_links": sum(len(view["evidence"]) for view in edges),
            "pool_items": len(pool),
            "status_counts": status_counts,
        },
    }
