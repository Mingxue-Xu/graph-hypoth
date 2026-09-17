"""The canvas view-model reads a run directory the way the connected renderer does: the committed
graph, the exported verdicts, the retrieval pool in the events database, and the trace sidecar.
Everything is built from real store models and written to disk as a run would write it."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.delta import build_edge, build_node
from src.graph_store import (
    CausalClaimGraphStore,
    EdgeStatus,
    EvidenceLink,
    EvidenceRole,
    ExperimentDesign,
    ExperimentPlan,
    compute_target_id,
)
from src.log_store import SQLiteLogStore
from src.web.graph_view import (
    build_graph_view,
    hypothesis_edge_ids,
    load_evidence_pool,
    read_edge_table,
)

SPAN = "smoking increases lung cancer risk through chronic inflammation"
CAUSE = build_node(
    label="smoking", type="exposure", definition="tobacco use", provenance=[{"source": "claim"}]
)
EFFECT = build_node(label="lung cancer", type="outcome", provenance=[{"source": "claim"}])
MINED = build_node(
    label="chronic inflammation", type="mechanism",
    provenance=[
        {"source": "literature_enrichment", "matched_quote_span": SPAN, "paper_id": "1"}
    ],
)
COINED = build_node(label="inflammation index", type="measure")  # no provenance: coined
EDGE = build_edge(
    source_node_ids=[CAUSE.node_id], target_node_ids=[EFFECT.node_id],
    direction="causal", relation_type="increases",
)
HYPO = build_edge(
    source_node_ids=[COINED.node_id], target_node_ids=[EFFECT.node_id],
    direction="causal", relation_type="predicts", mechanism="the index tracks tissue damage",
)


def _store() -> CausalClaimGraphStore:
    verified = EDGE.model_copy(update={"status": EdgeStatus.SUPPORTED, "confidence": 0.82})
    hypothesis = HYPO.model_copy(update={"open_risks": ["scope: cohort studies only"]})
    link = EvidenceLink(
        target_id=compute_target_id(EDGE.edge_id, EvidenceRole.SUPPORT),
        verification_task_id="task-1",
        evidence_id="ev-1",
        evidence_role=EvidenceRole.SUPPORT,
        signal={"matched_quote_span": "smoking increases lung cancer risk", "association_score": 0.91},
        retrieval_event_id="batch-x",
        committed_transaction_id="tx-1",
        trust_tier="green",
    )
    plan = ExperimentPlan(
        hypothesis_under_test="the index predicts cancer incidence",
        design=ExperimentDesign.CONTROLLED_OBSERVATIONAL,
        procedure=["measure the index at baseline", "follow the cohort"],
        falsification="no association after adjustment",
    )
    return CausalClaimGraphStore(
        version=4,
        nodes={node.node_id: node for node in (CAUSE, EFFECT, MINED, COINED)},
        edges={verified.edge_id: verified, hypothesis.edge_id: hypothesis},
        evidence_links=[link],
        scope_context={"seed_claim": "smoking increases lung cancer"},
        experiment_plans={HYPO.edge_id: plan},
    )


def _write_run(tmp_path: Path, *, pool: bool = True, sidecar: bool = True) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "graph.json").write_text(
        json.dumps(_store().model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    (run_dir / "edge_table.csv").write_text(
        "edge_id,relation_type,status,verdict,confidence,open_risks\n"
        f"{EDGE.edge_id},increases,supported,supported,0.82,\n"
        f"{HYPO.edge_id},predicts,unverified,qualified,,scope: cohort studies only\n",
        encoding="utf-8",
    )
    if sidecar:
        (run_dir / "trace").mkdir()
        (run_dir / "trace" / "plain_language.json").write_text(
            json.dumps({
                "hypothesis_edge_ids": {"h1": [HYPO.edge_id]},
                "hypotheses": {
                    "h1": {"headline": "The index predicts incidence.", "status": "confirmed"},
                    "h2": {"headline": "A second idea.", "status": "surfaced"},
                },
                "experiment_attempts": {
                    "h1": {"candidate_id": "h1", "status": "committed", "grounding_status": "grounded"}
                },
            }),
            encoding="utf-8",
        )
    if pool:
        SQLiteLogStore(run_dir / "events.sqlite").setup()
        payload = {
            "query": "smoking lung cancer",
            "evidence": [
                {
                    "evidence_id": "ev-1", "title": "A cohort study of smoking",
                    "url": "https://example.org/cohort", "quote": SPAN, "source": "openalex",
                    "tool_call_id": "batch-x", "metadata": {},
                },
                {"evidence_id": "ev-9", "title": "Unrelated", "url": "", "quote": "nothing", "source": "exa"},
            ],
        }
        connection = sqlite3.connect(run_dir / "events.sqlite")
        connection.execute(
            "INSERT INTO retrieval_cache "
            "(run_id, input_hash, retrieval_config_hash, payload_hash, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            ("run-1", "input", "config", "payload", json.dumps(payload)),
        )
        connection.commit()
        connection.close()
    return run_dir


def test_view_projects_nodes_edges_verdicts_evidence_and_plans(tmp_path):
    view = build_graph_view(_write_run(tmp_path), run_id="run-1")

    assert view["seed"] == "smoking increases lung cancer" and view["version"] == 4
    nodes = {node["label"]: node for node in view["nodes"]}
    assert nodes["smoking"]["klass"] == "claim"
    assert nodes["chronic inflammation"]["klass"] == "mined"
    assert nodes["inflammation index"]["klass"] == "hypo"
    assert nodes["smoking"]["definition"] == "tobacco use"
    assert nodes["chronic inflammation"]["sources"][0]["title"] == "A cohort study of smoking"

    edges = {edge["id"]: edge for edge in view["edges"]}
    verified = edges[EDGE.edge_id]
    assert verified["status"] == "supported" and verified["confidence"] == 0.82
    assert verified["verdict"] == "supported" and verified["plan"] is None
    assert not verified["is_hypothesis"]
    assert verified["sources"] == [CAUSE.node_id] and verified["targets"] == [EFFECT.node_id]
    [link] = verified["evidence"]
    assert link["resolved"] and not link["ambiguous"]
    assert link["title"] == "A cohort study of smoking"
    assert link["url"] == "https://example.org/cohort" and link["quote"] == SPAN
    assert link["role"] == "support" and link["trust_tier"] == "green"
    assert link["association_score"] == 0.91
    assert link["matched_quote_span"] == "smoking increases lung cancer risk"

    hypothesis = edges[HYPO.edge_id]
    assert hypothesis["is_hypothesis"] and hypothesis["candidate_id"] == "h1"
    assert hypothesis["status"] == "unverified" and hypothesis["verdict"] == "qualified"
    assert hypothesis["open_risks"] == ["scope: cohort studies only"]
    assert hypothesis["evidence"] == [] and hypothesis["mechanism"] == "the index tracks tissue damage"
    assert hypothesis["plan"]["hypothesis_under_test"] == "the index predicts cancer incidence"
    assert hypothesis["plan"]["design"] == "controlled_observational"

    assert view["hypotheses"] == {"h1": [HYPO.edge_id]}
    assert view["surfaced"] == [
        {
            "candidate_id": "h1", "headline": "The index predicts incidence.",
            "hypothesis_edge_ids": [HYPO.edge_id], "committed": True, "experiment_plan": True,
            "experiment": {"status": "committed", "grounding_status": "grounded"},
        },
        {
            "candidate_id": "h2", "headline": "A second idea.",
            "hypothesis_edge_ids": [], "committed": False, "experiment_plan": False,
            "experiment": None,
        },
    ]
    stats = view["stats"]
    assert stats["nodes"] == 4 and stats["edges"] == 2 and stats["hypothesis_edges"] == 1
    assert stats["experiment_plans"] == 1 and stats["evidence_links"] == 1
    assert stats["pool_items"] == 2
    assert stats["status_counts"]["supported"] == 1 and stats["status_counts"]["unverified"] == 1


def test_view_without_a_pool_or_sidecar_still_flags_the_coined_hypothesis_edge(tmp_path):
    view = build_graph_view(_write_run(tmp_path, pool=False, sidecar=False))

    edges = {edge["id"]: edge for edge in view["edges"]}
    [link] = edges[EDGE.edge_id]["evidence"]
    assert not link["resolved"] and link["title"] == ""
    assert link["matched_quote_span"] == "smoking increases lung cancer risk"
    assert edges[HYPO.edge_id]["is_hypothesis"] and edges[HYPO.edge_id]["candidate_id"] is None
    assert view["stats"]["pool_items"] == 0 and view["hypotheses"] == {}
    assert view["surfaced"] == []
    assert all(node["sources"] == [] for node in view["nodes"])


def test_pool_lookup_prefers_the_run_id_and_falls_back_to_every_row(tmp_path):
    run_dir = _write_run(tmp_path)

    by_run = [item["evidence_id"] for item in load_evidence_pool(run_dir, "run-1")]
    by_other = [item["evidence_id"] for item in load_evidence_pool(run_dir, "another-run")]

    assert by_run == ["ev-1", "ev-9"]
    assert by_other == ["ev-1", "ev-9"]  # an unknown logical id falls back to every row
    assert load_evidence_pool(run_dir, "run-1")[0]["_batch_id"]
    assert load_evidence_pool(tmp_path) == []  # no sqlite file at all


def test_missing_inputs_are_tolerated_or_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_graph_view(tmp_path)
    assert read_edge_table(tmp_path / "edge_table.csv") == {}
    assert hypothesis_edge_ids(tmp_path) == {}
    (tmp_path / "trace").mkdir()
    (tmp_path / "trace" / "plain_language.json").write_text("[]", encoding="utf-8")
    assert hypothesis_edge_ids(tmp_path) == {}
