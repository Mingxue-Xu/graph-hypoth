"""Serve the web app over a mock pipeline, for trying the UI without credentials.

The mock plays the eight stages with short pauses, asks for confirmation through the injected hook,
and writes a realistic run directory (graph.json with statuses, evidence links, and plans; an
events database with the retrieval pool; edge table; audit memo; trace pages). No model or network
calls; the corpus is invented and every URL points at example.org. Runs land under
``runtime_artifacts/web-demo`` by default.

Usage from the repository root, with the ``web`` extra installed:
  python scripts/web_demo.py [--port 8765] [--fast] [--runs-root DIR]
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from src.cycles.hypothesis import HypothesisCandidate, ScoredCandidate
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
from src.progress import report_progress
from src.web.app import create_app
from src.web.runs import RunManager

PAUSE = 1.5


def beat(factor: float = 1.0) -> None:
    time.sleep(PAUSE * factor)


# --- the mock corpus ---------------------------------------------------------------------------
EVIDENCE = [
    {"evidence_id": "ev-1", "title": "Head pruning halves decoder latency at 40% sparsity",
     "url": "https://example.org/papers/head-pruning-latency", "source": "openalex",
     "quote": "structured removal of 40% of attention heads reduced per-token latency by 48% on a "
              "single accelerator without kernel changes", "authors": ["A. Demo", "B. Example"],
     "published_date": "2024-05-02"},
    {"evidence_id": "ev-2", "title": "Block pruning for efficient inference",
     "url": "https://example.org/papers/block-pruning", "source": "crossref",
     "quote": "block-structured pruning reduces FLOPs proportionally and the savings translate to "
              "wall-clock gains because dense kernels stay dense", "authors": ["C. Sample"],
     "published_date": "2023-11-14"},
    {"evidence_id": "ev-3", "title": "What pruning forgets: factual recall under compression",
     "url": "https://example.org/papers/pruning-forgets", "source": "arxiv",
     "quote": "factual accuracy stayed within one point of the dense model up to 30% sparsity and "
              "dropped sharply beyond it", "authors": ["D. Placeholder"], "published_date": "2024-02-20"},
    {"evidence_id": "ev-4", "title": "Are sixteen heads really better than one?",
     "url": "https://example.org/papers/sixteen-heads", "source": "openalex",
     "quote": "a large fraction of attention heads can be removed at test time with no drop in "
              "quality, which is what makes structured pruning viable", "authors": ["E. Fixture"],
     "published_date": "2019-05-25"},
    {"evidence_id": "ev-5", "title": "Calibration set size and pruning quality",
     "url": "https://example.org/papers/calibration-size", "source": "crossref",
     "quote": "increasing the calibration set from 128 to 1024 examples changed factual recall "
              "by less than the run-to-run variance", "authors": ["F. Mock"], "published_date": "2024-08-09"},
    {"evidence_id": "ev-6", "title": "Knowledge neurons in pretrained transformers",
     "url": "https://example.org/papers/knowledge-neurons", "source": "openalex",
     "quote": "factual associations concentrate in a small set of knowledge neurons in the "
              "feed-forward layers, and suppressing them erases specific facts", "authors": ["G. Stub"],
     "published_date": "2022-03-30"},
    {"evidence_id": "ev-7", "title": "Activation sparsity does not mean speed",
     "url": "https://example.org/papers/sparsity-speed", "source": "arxiv",
     "quote": "activation sparsity alone did not reduce wall-clock latency on dense kernels; only "
              "structured removal did", "authors": ["H. Dummy"], "published_date": "2023-07-01"},
    {"evidence_id": "ev-8", "title": "Importance scores for head selection",
     "url": "https://example.org/papers/importance-scores", "source": "crossref",
     "quote": "layer-wise importance scores computed on 128 calibration examples guide which heads "
              "to prune and beat magnitude pruning at every sparsity level", "authors": ["I. Test"],
     "published_date": "2024-01-17"},
]
QUERIES = [
    "structured pruning inference latency transformer",
    "pruning factual accuracy knowledge retention",
    "attention head redundancy importance scores",
    "calibration data size pruning quality",
]

CLAIM = [{"source": "claim"}]


def mined(span: str, paper_id: str) -> list[dict]:
    return [{"source": "literature_enrichment", "matched_quote_span": span, "paper_id": paper_id}]


PRUNING = build_node(label="structured pruning", type="intervention",
                     definition="removing whole heads, neurons, or layers rather than single weights",
                     provenance=CLAIM)
COST = build_node(label="inference cost", type="outcome",
                  definition="latency and FLOPs per generated token", provenance=CLAIM)
ACCURACY = build_node(label="factual accuracy", type="outcome",
                      definition="share of factual questions answered correctly on a held-out benchmark",
                      provenance=CLAIM)
REDUNDANCY = build_node(label="attention head redundancy", type="mechanism",
                        definition="the fraction of heads whose removal leaves outputs unchanged",
                        provenance=CLAIM)
CALIBRATION = build_node(label="calibration data size", type="condition",
                         definition="number of examples used to score importance before pruning",
                         provenance=CLAIM)
NEURONS = build_node(label="knowledge neurons", type="mechanism",
                     definition="feed-forward units whose activation stores a specific fact",
                     provenance=mined("knowledge neurons in the feed-forward layers", "6"))
SPARSITY = build_node(label="activation sparsity", type="mechanism",
                      definition="the share of activations that are exactly zero at inference time",
                      provenance=mined("activation sparsity alone did not reduce wall-clock latency", "7"))
IMPORTANCE = build_node(label="layer-wise importance scores", type="method",
                        definition="per-layer head scores computed on a calibration set",
                        provenance=mined("layer-wise importance scores computed on 128 calibration examples", "8"))
# Proposer-coined concepts (no provenance).
MARGIN = build_node(label="fact-retention margin", type="measure",
                    definition="the accuracy gap between the pruned and the dense model on facts the dense model answers correctly")
REASSIGN = build_node(label="head-role reassignment", type="mechanism",
                      definition="surviving heads take over the attention patterns of pruned heads after brief fine-tuning")
DISTILL = build_node(label="sparsity-aware distillation", type="intervention",
                     definition="distilling from the dense model while the sparsity mask is fixed")


def edge(source, target, relation, mechanism=""):
    return build_edge(source_node_ids=[source.node_id], target_node_ids=[target.node_id],
                      direction="causal", relation_type=relation, mechanism=mechanism)


# (edge, status, confidence, verdict, risks, evidence ids)
BASE_EDGES = [
    (edge(PRUNING, COST, "reduces", "fewer heads and blocks mean fewer FLOPs per token"),
     EdgeStatus.SUPPORTED, 0.86, "supported", [], ["ev-1", "ev-2"]),
    (edge(PRUNING, ACCURACY, "degrades", "pruned units carried facts"),
     EdgeStatus.UNVERIFIED, None, "qualified",
     ["qualified: the drop appears only above 30% sparsity (ev-3); commit withheld"], ["ev-3"]),
    (edge(REDUNDANCY, PRUNING, "enables"), EdgeStatus.SUPPORTED, 0.71, "supported", [], ["ev-4"]),
    (edge(CALIBRATION, ACCURACY, "moderates"), EdgeStatus.INSUFFICIENT, 0.34, "insufficient",
     ["insufficient: one study, effect within run-to-run variance"], ["ev-5"]),
    (edge(NEURONS, ACCURACY, "localizes", "facts live in a small set of feed-forward units"),
     EdgeStatus.SUPPORTED, 0.78, "supported", [], ["ev-6"]),
    (edge(SPARSITY, COST, "reduces"), EdgeStatus.UNVERIFIED, None, "contradicted",
     ["contradicted: dense kernels ignore activation sparsity (ev-7); commit withheld"], ["ev-7"]),
    (edge(IMPORTANCE, PRUNING, "guides"), EdgeStatus.SUPPORTED, 0.64, "supported", [], ["ev-8"]),
]

H1_EDGES = [edge(MARGIN, ACCURACY, "predicts", "the margin isolates facts the dense model knew")]
H2_EDGES = [edge(PRUNING, REASSIGN, "triggers", "brief fine-tuning re-routes attention"),
            edge(REASSIGN, ACCURACY, "preserves", "re-routed heads keep the retrieval patterns")]
H3_EDGES = [edge(PRUNING, DISTILL, "requires"), edge(DISTILL, ACCURACY, "restores")]

SCAFFOLD_H1 = {
    "problem": "Benchmarks average over facts the dense model never knew, hiding what pruning removes.",
    "lever": "Score only the facts the dense model answers correctly and track the gap.",
    "method": "Paired evaluation of dense and pruned checkpoints on the same fact set.",
    "experiment": "Sweep sparsity from 10% to 50% and record the margin at each level.",
}
SCAFFOLD_H2 = {
    "problem": "Head removal is treated as a loss; the recovery mechanism is not measured.",
    "lever": "Measure whether surviving heads reproduce the pruned heads' attention patterns.",
    "method": "Attention-pattern similarity before and after 200 fine-tuning steps.",
    "experiment": "Ablate reassignment by freezing surviving heads and compare accuracy.",
}
SCAFFOLD_H3 = {
    "problem": "Accuracy lost to pruning may be recoverable without changing the mask.",
    "lever": "Distill from the dense model with the sparsity mask fixed.",
    "method": "Distillation on a generic corpus, then factual evaluation.",
}


def candidate(cid, nodes, edges, chain, rationale, scaffold, novelty, sat, cross, hyp, rank):
    return ScoredCandidate(
        candidate=HypothesisCandidate(
            candidate_id=cid, new_nodes=tuple(nodes), new_edges=tuple(edges),
            mechanism_chain=tuple(chain),
            source_quotes=({"evidence_id": "ev-3", "quote_span": "dropped sharply beyond it",
                            "role_in_hypothesis": "motivation"},),
            assumptions=("the fact benchmark is stable across runs",),
            rationale=rationale, idea_scaffold=scaffold,
            novelty_graded=novelty, saturation=sat, cross_concept=cross,
        ),
        hyp_score=hyp, rank_score=rank,
    )


CANDIDATES = [
    candidate("h1", [MARGIN], H1_EDGES,
              [{"from": "fact-retention margin", "relation": "predicts", "to": "factual accuracy",
                "mechanism": "the margin isolates facts the dense model knew"}],
              "A fact-retention margin predicts when structured pruning preserves factual accuracy "
              "better than aggregate benchmark scores do.", SCAFFOLD_H1, 0.74, 0.22, True, 0.68, 0.71),
    candidate("h2", [REASSIGN], H2_EDGES,
              [{"from": "structured pruning", "relation": "triggers", "to": "head-role reassignment",
                "mechanism": "brief fine-tuning re-routes attention"},
               {"from": "head-role reassignment", "relation": "preserves", "to": "factual accuracy",
                "mechanism": "re-routed heads keep the retrieval patterns"}],
              "Surviving heads take over the roles of pruned heads after brief fine-tuning, which is "
              "why accuracy survives moderate sparsity.", SCAFFOLD_H2, 0.67, 0.31, True, 0.61, 0.64),
    candidate("h3", [DISTILL], H3_EDGES,
              [{"from": "sparsity-aware distillation", "relation": "restores", "to": "factual accuracy",
                "mechanism": "the dense teacher re-supplies the facts"}],
              "Distilling from the dense model with a fixed sparsity mask restores most of the lost "
              "factual accuracy.", SCAFFOLD_H3, 0.58, 0.40, False, 0.52, 0.49),
]
NEW_NODES = {"h1": [MARGIN], "h2": [REASSIGN], "h3": [DISTILL]}
NEW_EDGES = {"h1": H1_EDGES, "h2": H2_EDGES, "h3": H3_EDGES}
PLANS = {
    "h1": ExperimentPlan(
        hypothesis_under_test="A larger fact-retention margin predicts preserved factual accuracy under structured pruning.",
        operationalization="Margin = accuracy(dense) - accuracy(pruned) on facts the dense model answers correctly.",
        design=ExperimentDesign.CONTROLLED_OBSERVATIONAL,
        design_rationale="The margin is a measurement, so the sparsity sweep is observed rather than manipulated.",
        intervention_or_manipulation="Structured head pruning at 10, 20, 30, 40, and 50% sparsity.",
        comparison_baseline="The dense checkpoint on the identical fact set.",
        procedure=["Select 2,000 facts the dense model answers correctly.",
                   "Prune with layer-wise importance scores at each sparsity level.",
                   "Record accuracy on the fact set and the aggregate benchmark.",
                   "Fit the margin against sparsity and against the aggregate score."],
        expected_outcome="The margin rises sharply above 30% sparsity while the aggregate score moves little.",
        falsification="The margin tracks the aggregate score within noise at every sparsity level.",
    ),
    "h2": ExperimentPlan(
        hypothesis_under_test="Head-role reassignment during brief fine-tuning preserves factual accuracy after pruning.",
        operationalization="Attention-pattern similarity between surviving heads and pruned heads before and after fine-tuning.",
        design=ExperimentDesign.ABLATION,
        design_rationale="Freezing surviving heads removes the proposed mechanism while keeping everything else fixed.",
        intervention_or_manipulation="200 fine-tuning steps with surviving heads trainable versus frozen.",
        comparison_baseline="The pruned model without fine-tuning.",
        procedure=["Prune 30% of heads.", "Fine-tune with and without frozen surviving heads.",
                   "Measure attention-pattern similarity to the pruned heads.",
                   "Evaluate factual accuracy in both arms."],
        expected_outcome="Only the trainable arm recovers accuracy, and its similarity to pruned heads rises.",
        falsification="Accuracy recovers equally with surviving heads frozen.",
    ),
    "h3": ExperimentPlan(
        hypothesis_under_test="Sparsity-aware distillation restores factual accuracy lost to pruning.",
        design=ExperimentDesign.BENCHMARK_COMPARISON,
        procedure=["Prune 40% of heads.", "Distill from the dense model with the mask fixed.",
                   "Evaluate factual accuracy against the pruned and dense models."],
        expected_outcome="Distillation closes most of the gap to the dense model.",
        falsification="No improvement over the pruned model.",
    ),
}


def format_ranking(surfaced) -> str:
    lines = [f"surfaced hypotheses ({len(surfaced)}):"]
    for rank, scored in enumerate(surfaced, start=1):
        c = scored.candidate
        labels = " + ".join(n.label for n in c.new_nodes)
        tag = "cross-concept" if c.cross_concept else "within-concept"
        lines.append(f"  #{rank} {c.candidate_id}  f_nov {c.ranking_novelty:.2f}  sat {c.saturation:.2f}  "
                     f"RankScore {scored.rank_score:.3f}  [{tag}]")
        lines.append(f"       {labels}")
    return "\n".join(lines)


def trace_page(scored, seed: str) -> str:
    c = scored.candidate
    rows = "".join(f"<li>{html.escape(k)}: {html.escape(v)}</li>" for k, v in c.idea_scaffold.items())
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        f"<title>Proposed hypothesis {c.candidate_id}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:60rem}.k{color:#555}</style></head><body>"
        f"<h1>Proposed hypothesis {html.escape(c.candidate_id)}</h1>"
        f"<p class=k>Research question: {html.escape(seed)}</p>"
        f"<p>{html.escape(c.rationale)}</p><h2>Idea scaffold</h2><ul>{rows}</ul>"
        "<p class=k>Mock trace page written by the demo pipeline.</p></body></html>"
    )


def write_run(run_dir: Path, run_id: str, seed: str, confirmed: list[str], surfaced) -> tuple[dict, dict]:
    nodes = [PRUNING, COST, ACCURACY, REDUNDANCY, CALIBRATION, NEURONS, SPARSITY, IMPORTANCE]
    edges, links, plans, table_rows, hyp_edges = {}, [], {}, [], {}
    for index, (e, status, conf, verdict, risks, ev_ids) in enumerate(BASE_EDGES, start=1):
        committed = e.model_copy(update={"status": status, "confidence": conf, "open_risks": list(risks)})
        edges[committed.edge_id] = committed
        table_rows.append((committed.edge_id, e.relation_type, status.value, verdict, conf, risks))
        for ev_id in ev_ids:
            item = next(x for x in EVIDENCE if x["evidence_id"] == ev_id)
            links.append(EvidenceLink(
                target_id=compute_target_id(e.edge_id, EvidenceRole.SUPPORT),
                verification_task_id=f"task-{index}", evidence_id=ev_id,
                evidence_role=EvidenceRole.SUPPORT,
                signal={"matched_quote_span": item["quote"][:60], "association_score": round(0.55 + 0.05 * index, 2)},
                retrieval_event_id="batch-demo", committed_transaction_id=f"tx-{index}", trust_tier="green",
            ))
    for cid in confirmed:
        nodes.extend(NEW_NODES[cid])
        hyp_edges[cid] = []
        for e in NEW_EDGES[cid]:
            edges[e.edge_id] = e
            hyp_edges[cid].append(e.edge_id)
            table_rows.append((e.edge_id, e.relation_type, "unverified", "", None, []))
        plans[hyp_edges[cid][-1]] = PLANS[cid]
    store = CausalClaimGraphStore(
        version=len(edges) + 3, nodes={n.node_id: n for n in nodes}, edges=edges,
        evidence_links=links, scope_context={"seed_claim": seed}, experiment_plans=plans,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "graph.json").write_text(json.dumps(store.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    with (run_dir / "edge_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["edge_id", "relation_type", "status", "verdict", "confidence", "open_risks"])
        for edge_id, relation, status, verdict, conf, risks in table_rows:
            writer.writerow([edge_id, relation, status, verdict, "" if conf is None else conf, "; ".join(risks)])
    risks_md = "\n".join(f"- {r}" for _, _, _, _, _, rs in table_rows for r in rs)
    (run_dir / "audit_memo.md").write_text(
        f"# Audit memo (mock run {run_id})\n\n{len(edges)} edges committed; {len(links)} evidence links; "
        f"{len(plans)} experiment plans.\n\n## Open risks\n\n{risks_md}\n\n## Receipts\n\n"
        f"{len(edges) + len(links) + len(plans)} accepted transactions, 0 rejected.\n", encoding="utf-8")
    # The retrieval pool, in the same events database shape the pipeline writes.
    connection = sqlite3.connect(run_dir / "events.sqlite")
    connection.execute(
        "INSERT OR REPLACE INTO retrieval_cache (run_id, input_hash, retrieval_config_hash, payload_hash, payload) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, "demo-input", "demo-config", "demo-payload",
         json.dumps({"query": QUERIES[0], "evidence": [{**item, "tool_call_id": "batch-demo", "metadata": {}} for item in EVIDENCE]})),
    )
    connection.commit()
    connection.close()
    trace = run_dir / "trace"
    trace.mkdir(exist_ok=True)
    rows = "".join(
        f"<tr><td>{rank}</td><td><a href='{s.candidate.candidate_id}.html'>{s.candidate.candidate_id}</a></td>"
        f"<td>{s.candidate.ranking_novelty:.2f}</td><td>{s.candidate.saturation:.2f}</td>"
        f"<td>{s.hyp_score:.3f}</td><td>{s.rank_score:.3f}</td></tr>"
        for rank, s in enumerate(surfaced, start=1))
    (trace / "index.html").write_text(
        "<!doctype html><html lang=en><head><meta charset=utf-8><title>Research Synthesist hypotheses</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:60rem}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:.3rem .6rem;text-align:left}.k{color:#555}</style></head><body>"
        f"<h1>Research Synthesist hypotheses (mock run)</h1><p class=k>Research question: {html.escape(seed)}</p>"
        f"<p class=k>{len(surfaced)} surfaced</p><table><tr><th>rank</th><th>id</th><th>field novelty</th>"
        f"<th>saturation</th><th>HypScore</th><th>RankScore</th></tr>{rows}</table></body></html>", encoding="utf-8")
    for s in surfaced:
        (trace / f"{s.candidate.candidate_id}.html").write_text(trace_page(s, seed), encoding="utf-8")
    (trace / "plain_language.json").write_text(json.dumps({
        "claim": seed, "seed_kind": "research_question", "hypotheses": {},
        "hypothesis_edge_ids": hyp_edges,
    }, indent=2), encoding="utf-8")
    return hyp_edges, {cid: PLANS[cid].model_dump(mode="json") for cid in confirmed}


def demo_pipeline(**kwargs):
    from src.research_profile import load_research_profile

    run_id, run_dir = kwargs["run_id"], Path(kwargs["export_dir"])
    kwargs["log_store"].setup()
    profile = load_research_profile(kwargs["profile_path"])
    seed = profile.research_question or profile.claim or "the research question"
    report_progress("Preparing run", "loading configuration and checking credentials"); beat()
    report_progress("Preparing run", "loading evidence embedder"); beat()
    for i, query in enumerate(QUERIES, start=1):
        report_progress("Retrieval", f"sub-query: {query}", current=i, total=len(QUERIES)); beat(0.8)
    report_progress("Retrieval", f"fused pool: {len(EVIDENCE)} passages from 3 sources"); beat(0.6)
    report_progress("Extraction", "seed question -> 5 concepts and 4 source-asserted edges, all unverified"); beat()
    for i in range(1, 5):
        report_progress("Evidence review", f"edge {i}: grading short-listed passages", current=i, total=4); beat(0.7)
    report_progress("Priority", "weighted topics recorded on 2 matching nodes"); beat(0.7)
    report_progress("Hypothesis development", "mining concepts from 8 passages"); beat()
    report_progress("Hypothesis development", "proposing candidates over the committed graph"); beat()
    report_progress("Hypothesis development", "Critic Panel round 1: seats 1 and 2 grading"); beat()
    report_progress("Hypothesis development", "revising survivors; gates and ranking"); beat(0.8)
    surfaced = list(CANDIDATES)
    kwargs["output_fn"](format_ranking(surfaced))
    confirm = kwargs.get("confirm_fn")
    confirmed = list(confirm(surfaced)) if confirm is not None else ["h1", "h2"]
    for i, cid in enumerate(confirmed, start=1):
        report_progress("Experiment design", f"{cid}: Designer draft, Validator grading", current=i, total=len(confirmed)); beat()
        report_progress("Experiment design", f"{cid}: one revision round, plan content-complete", current=i, total=len(confirmed)); beat(0.7)
    report_progress("Writing reports", "rendering trace pages and exporting the graph")
    hyp_edges, plans = write_run(run_dir, run_id, seed, confirmed, surfaced)
    beat(0.7)
    report_progress("Writing reports", f"saved to {run_dir / 'trace'}", status="completed")
    edge_count = len(BASE_EDGES) + sum(len(v) for v in hyp_edges.values())
    kwargs["output_fn"](f"done: graph v{edge_count + 3}, {len(surfaced)} hypotheses surfaced, {edge_count} edges in graph")
    return SimpleNamespace(
        version=edge_count + 3,
        surfaced=[SimpleNamespace(rank=i, candidate_id=s.candidate.candidate_id, rank_score=s.rank_score,
                                  experiment_plan=plans.get(s.candidate.candidate_id),
                                  hypothesis_edge_ids=tuple(hyp_edges.get(s.candidate.candidate_id, ())))
                  for i, s in enumerate(surfaced, start=1)],
        edge_table=[SimpleNamespace(status="x")] * edge_count,
        open_risks=tuple(r for _, _, _, _, risks, _ in BASE_EDGES for r in risks),
    )


def main() -> int:
    global PAUSE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--runs-root", type=Path, default=REPO / "runtime_artifacts" / "web-demo")
    args = parser.parse_args()
    if args.fast:
        PAUSE = 0.2
    import uvicorn

    manager = RunManager(args.runs_root, pipeline=demo_pipeline, quiet=False)
    app = create_app(manager)
    print(f"demo web app at http://127.0.0.1:{args.port}/ (runs under {manager.runs_root})", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
