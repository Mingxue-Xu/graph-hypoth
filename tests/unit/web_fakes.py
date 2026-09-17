"""Hermetic stand-ins for the web app's tests.

``FakePipeline`` plays ``run_synthesist``: it records its keyword arguments, reports progress through
the real ``ProgressReporter`` the manager installs, writes the same export files a real run does,
calls the injected confirm hook with real ``ScoredCandidate`` objects, and can block or fail on
demand. Nothing here touches a model or the network.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.cycles.hypothesis import HypothesisCandidate, ScoredCandidate
from src.delta import build_edge, build_node
from src.progress import report_progress


def scored_candidate(
    candidate_id: str, *, field_novelty: float = 0.7, saturation: float = 0.25,
    cross_concept: bool = True,
) -> ScoredCandidate:
    cause = build_node(
        label=f"{candidate_id} cause", type="mechanism", definition=f"the {candidate_id} lever"
    )
    effect = build_node(label=f"{candidate_id} effect", type="outcome")
    edge = build_edge(
        source_node_ids=[cause.node_id], target_node_ids=[effect.node_id],
        direction="causal", relation_type="increases", mechanism="via a shared pathway",
    )
    candidate = HypothesisCandidate(
        candidate_id=candidate_id,
        new_nodes=(cause, effect),
        new_edges=(edge,),
        mechanism_chain=(
            {
                "from": cause.label, "relation": "increases", "to": effect.label,
                "mechanism": "via a shared pathway",
            },
        ),
        source_quotes=(
            {"evidence_id": "ev-1", "quote_span": "a verbatim span", "role_in_hypothesis": "premise"},
        ),
        assumptions=("the effect is measurable",),
        rationale=f"{candidate_id} closes a gap",
        idea_scaffold={"problem": "latency vs accuracy", "method": "ablation"},
        novelty_graded=field_novelty,
        saturation=saturation,
        cross_concept=cross_concept,
    )
    return ScoredCandidate(candidate=candidate, hyp_score=0.5, rank_score=0.5 * (1 - saturation))


def write_minimal_artifacts(export_dir: Path) -> None:
    """The three export files every run writes, with an empty committed graph."""
    export_dir.mkdir(parents=True, exist_ok=True)
    graph = {
        "graph_id": "graph", "version": 1, "nodes": {}, "edges": {}, "evidence_links": [],
        "scope_context": {"seed_claim": "seed"},
    }
    (export_dir / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (export_dir / "edge_table.csv").write_text(
        "edge_id,relation_type,status,verdict,confidence,open_risks\n", encoding="utf-8"
    )
    (export_dir / "audit_memo.md").write_text("# audit memo\n", encoding="utf-8")


class FakePipeline:
    def __init__(
        self,
        *,
        surfaced: tuple[str, ...] = ("h1", "h2"),
        fail: bool = False,
        gate: threading.Event | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.confirmed: list[str] | None = None
        self.surfaced = tuple(surfaced)
        self.fail = fail
        self.gate = gate  # when given, the pipeline waits on it before finishing

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        report_progress("Retrieval", "planning sub-queries")
        export_dir = Path(kwargs["export_dir"])
        write_minimal_artifacts(export_dir)
        trace_dir = Path(kwargs["trace_dir"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "index.html").write_text("<html>trace</html>", encoding="utf-8")
        confirm = kwargs.get("confirm_fn")
        if confirm is not None:
            self.confirmed = list(confirm([scored_candidate(cid) for cid in self.surfaced]))
        if self.gate is not None:
            self.gate.wait(timeout=10)
        if self.fail:
            raise RuntimeError("boom")
        kwargs["output_fn"]("done: 2 hypotheses surfaced")
        report_progress("Writing reports", "saved", status="completed")
        return SimpleNamespace(
            version=3,
            surfaced=[
                SimpleNamespace(
                    rank=1, candidate_id="h1", rank_score=0.7,
                    experiment_plan={"design": "controlled"}, hypothesis_edge_ids=("e1",),
                )
            ],
            edge_table=[SimpleNamespace(status="supported"), SimpleNamespace(status="")],
            open_risks=("one risk",),
        )
