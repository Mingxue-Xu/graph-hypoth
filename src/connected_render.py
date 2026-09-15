"""Render a connected-component graph view for every surfaced hypothesis in a run.

The renderer uses one shared visual theme while deriving each page's focus node, title, counts, and
reachable component from run data.

Extraction criteria (identical to h3-connected): for one hypothesis, render every node reachable
from its proposer-coined new concept by following committed edges, treated as UNDIRECTED, any number
of hops (BFS connected component). Node colour = provenance source; edge style = committed status.

On top of the graph, each page also carries (data-derived, no LLM):
  - the hypothesis's quantified ranking signals (field novelty / saturation / HypScore / RankScore);
  - a "Sources" section tracing every literature-mined concept in the view back to the paper it was
    mined from (verbatim quote-match against the run's retrieval pool), and the same mined concepts
    are CLICKABLE in the diagram (each links to its source paper).

Inputs — a run's own artifacts only (NO LLM, NO network):
  <run-dir>/graph.json       the committed graph (nodes + edges)
  <run-dir>/trace/h*.html    the surfaced ranking: candidate id, rank, ranking signals, new concept(s)
  <events-db>.sqlite         the run's SQLiteLogStore retrieval_cache (the source pool; auto-selected
                             as the DB whose pool resolves the most of this run's mined concepts)

Output:
  <run-dir>/<candidate_id>-connected.html    one page per surfaced hypothesis

Deterministic: pure read of the artifacts; re-running overwrites. A hypothesis whose new-concept
label is not an EXACT node label in graph.json is skipped with a warning (no guessing).
"""

from __future__ import annotations

import collections
import glob
import html
import json
import re
from pathlib import Path

from src.experiment_plan_render import (
    EXPERIMENT_PLAN_FIELDS,
    collect_evidence_references,
    real_evidence_id,
)
from src.experiment_plan_render import (
    plain_design_text as _plain_design_text,
)
from src.hypothesis_scores import (
    events_db_for_run,
    load_audit_events,
    proposer_signals,
)
from src.provenance_links import (
    load_retrieval_batches,
    node_sources,
    select_best_db,
)
from src.relation_labels import normalize_relation_label
from src.report_context import SEED_LABELS, relationship_section


def sani(s: str) -> str:
    """Make a label safe inside a Mermaid ``["..."]`` node AND a ``|...|`` edge label. Beyond the
    quote, pipe, and angle-bracket swaps, escape the shape-delimiter chars
    ``()[]{}`` as HTML entities: Mermaid's edge-label parser treats them as node-shape syntax and
    breaks on them (e.g. a nested ``SO(3)`` inside a relation label like ``moderation (amplifies on
    SO(3))``), which silently kills the WHOLE diagram. ``htmlLabels:true`` decodes the entities back
    for display, so the rendered text is unchanged."""
    s = (s or "").replace('"', "'").replace("|", "/").replace("<", "‹").replace(">", "›")
    for ch, ent in (("(", "&#40;"), (")", "&#41;"), ("[", "&#91;"),
                    ("]", "&#93;"), ("{", "&#123;"), ("}", "&#125;")):
        s = s.replace(ch, ent)
    return " ".join(s.split())


def node_class(n: dict) -> str:
    """Map a node's provenance source to its palette class."""
    if n.get("user_priority"):
        return "prio"
    srcs = {p.get("source") for p in (n.get("provenance") or [])}
    if not srcs:
        return "hypo"
    if "literature_enrichment" in srcs and "claim" not in srcs:
        return "mined"
    return "claim"


# --- enumerate the surfaced hypotheses from the run's trace pages -------------------------
def parse_surfaced(trace_dir: Path) -> list[dict]:
    """Read every ``trace/h*.html`` into {rank, cid, cell, scores}. ``cell`` is the candidate's
    proposer-coined new concept(s); ``scores`` is the full ranking-signal table (field novelty,
    saturation, HypScore, RankScore, cross-concept, ...). Sorted by rank (rank-1 first)."""
    rows: list[dict] = []
    for page in sorted(glob.glob(str(trace_dir / "h*.html"))):
        text = Path(page).read_text(encoding="utf-8")
        cid_m = re.search(r"<tr><td class=k>candidate</td><td>(.*?)</td></tr>", text)
        nc_m = re.search(r"new concept\(s\)</td><td>(.*?)</td>", text)
        if not (cid_m and nc_m):
            print(f"  ! {Path(page).name}: no candidate/new-concept cell — skipped")
            continue
        scores = {
            html.unescape(k): html.unescape(v)
            for k, v in re.findall(r"<tr><td class=k>(.*?)</td><td>(.*?)</td></tr>", text)
        }
        rows.append({
            "cid": html.unescape(cid_m.group(1)).strip(),
            "rank": int(scores["rank"]) if scores.get("rank", "").isdigit() else 10**6,
            "cell": html.unescape(nc_m.group(1)).strip(),
            "scores": scores,
        })
    rows.sort(key=lambda r: (r["rank"], r["cid"]))
    return rows


def resolve_focus_ids(cell: str, by_label: dict[str, str]) -> list[str]:
    """Map a ``new concept(s)`` cell to graph node ids by EXACT label match. Try the whole cell as a
    single label first (so a label that itself contains ", " is not wrongly split); else split on
    the ", " join trace_render used. Returns [] if nothing matches (caller warns + skips)."""
    if cell in by_label:
        return [by_label[cell]]
    return [by_label[part] for part in cell.split(", ") if part in by_label]


# --- connected component (BFS over committed edges, undirected) ---------------------------
def build(graph: dict, focus_ids: list[str]) -> tuple:
    nodes, edges = graph["nodes"], graph["edges"]
    adj = collections.defaultdict(set)
    edge_list = []
    for e in edges.values():
        s = (e.get("source_node_ids") or [None])[0]
        t = (e.get("target_node_ids") or [None])[0]
        if s in nodes and t in nodes:
            adj[s].add(t)
            adj[t].add(s)
            edge_list.append((s, t, e.get("relation_type", ""), e.get("status", ""),
                              e.get("confidence")))
    dist = {fid: 0 for fid in focus_ids}
    q = collections.deque(focus_ids)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return nodes, set(dist), dist, edge_list


# --- Mermaid rendering ---------------------------------------------------------------
def mermaid(nodes, focus_ids, cid, reachable, dist, edge_list, node_links):
    focus = set(focus_ids)
    ids = sorted(reachable, key=lambda x: (dist[x], nodes[x]["label"]))
    idx = {nid: f"n{i}" for i, nid in enumerate(ids)}
    lines = ["flowchart LR"]
    for nid in ids:
        if nid in focus:
            lab = f'★ {cid.upper()} ({sani(nodes[nid]["label"])})'
        else:
            lab = sani(nodes[nid]["label"])
        lines.append(f'  {idx[nid]}["{lab}"]')
    # induced edges (both endpoints reachable)
    neutral_links, unverified_links, verified_links = [], [], []
    k = 0
    for s, t, rel, status, conf in edge_list:
        if s in reachable and t in reachable:
            label = f"{sani(rel)} · {conf:.2f}" if (status == "insufficient" and conf is not None) else sani(rel)
            arrow = "-.->" if status == "unverified" else "-->"
            lines.append(f'  {idx[s]} {arrow}|"{label}"| {idx[t]}')
            if status == "unverified":
                unverified_links.append(k)
            elif status == "supported":
                verified_links.append(k)
            else:
                neutral_links.append(k)
            k += 1
    lines += [
        # Node-provenance palette: extracted, mined, hypothesized, and priority-authored.
        "  classDef prio fill:#fff8e1,stroke:#f9a825,stroke-width:3px,color:#5f4339;",
        "  classDef hypo fill:#f7ede7,stroke:#B25A36,color:#B25A36;",
        "  classDef mined fill:#f4efe8,stroke:#8A6A4A,color:#6e5238;",
        "  classDef claim fill:#eef3f7,stroke:#2E5570,color:#2E5570;",
        "  classDef focus fill:#f1e1d8,stroke:#B25A36,stroke-width:4px,color:#8f3f22;",
    ]
    for klass in ("prio", "hypo", "mined", "claim"):
        members = [idx[nid] for nid in ids if node_class(nodes[nid]) == klass and nid not in focus]
        if members:
            lines.append(f'  class {",".join(members)} {klass};')
    lines.append(f'  class {",".join(idx[fid] for fid in focus_ids)} focus;')
    # evidence-verified (structural) edges -> green line; unverified hypothesis edges -> orange dashed
    if verified_links:
        lines.append(
            f'  linkStyle {",".join(map(str, verified_links))} '
            "stroke:#4F6A46,stroke-width:3px;"
        )
    if neutral_links:
        lines.append(
            f'  linkStyle {",".join(map(str, neutral_links))} '
            "stroke:#4F6A46,stroke-width:1.5px;"
        )
    if unverified_links:
        lines.append(
            f'  linkStyle {",".join(map(str, unverified_links))} '
            "stroke:#B25A36,stroke-width:2px,stroke-dasharray:6 4;"
        )
    # clickable literature-mined nodes -> their source paper (securityLevel:"loose" enables links)
    for nid in ids:
        link = node_links.get(nid)
        if link and link["url"].startswith(("http://", "https://")):
            tip = (sani(link["title"]).replace('"', "'")[:80]) or "source paper"
            lines.append(f'  click {idx[nid]} href "{link["url"]}" "{tip}" _blank')
    n_induced = len([1 for s, t, *_ in edge_list if s in reachable and t in reachable])
    return "\n".join(lines), n_induced


# Shared visual theme.
STYLE = """
*{box-sizing:border-box}body{font-family:'DM Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;color:#1a1a1a;background:#f5f6f8}
/* extracted style: DM Sans for node text, DM Mono for edge labels */
.mermaid .nodeLabel,.mermaid .nodeLabel *{font-family:'DM Sans',sans-serif!important;font-weight:600;font-size:13.5px}
.mermaid .edgeLabel,.mermaid .edgeLabel *{font-family:'DM Mono',ui-monospace,monospace!important;font-size:12px!important;color:#6B6258!important}
header{background:#0d1b2a;color:#fff;padding:22px 28px}header h1{margin:0 0 6px;font-size:22px}header p{margin:0;color:#b8c4d0;font-size:14px;max-width:1150px}
main{padding:20px 28px 60px;max-width:1480px;margin:0 auto}
section{background:#fff;border:1px solid #e0e4e8;border-radius:10px;padding:18px 20px;margin:20px 0;box-shadow:0 1px 3px rgba(0,0,0,.05);overflow-x:auto}
h2{font-size:18px;margin:0 0 8px}.sub{color:#5a6672;font-size:13px;margin:0 0 12px;max-width:1150px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:6px 0;font-size:13px}.legend span{display:inline-flex;align-items:center;gap:6px}
.swatch{width:14px;height:14px;border-radius:3px;border:1px solid rgba(0,0,0,.25)}.edgekey{display:inline-block;width:26px;vertical-align:middle}
.mermaid{overflow-x:auto;margin-top:8px;text-align:center}.mermaid svg{display:block;max-width:100%;height:auto;margin:0 auto}
.note{font-size:12.5px;color:#5a6672;border-left:3px solid #cfd6dd;padding:4px 0 4px 12px;margin:12px 0 0}code{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:12.5px}
"""

# Feature-specific styling (kept separate so the theme block above stays byte-verbatim to canonical).
EXTRA_STYLE = (
    "<style>"
    "table.scores{border-collapse:collapse;font-size:13px;width:100%;max-width:920px}"
    "table.scores th,table.scores td{border:1px solid #e0e4e8;padding:6px 11px;text-align:left;vertical-align:top}"
    "table.scores th{background:#f0f3f6;font-weight:600}"
    "table.scores td.v{font-family:'DM Mono',ui-monospace,monospace;white-space:nowrap;color:#2E5570}"
    "table.scores td.m{color:#5a6672}"
    "table.scores tr.grp td{background:#f7f8fa;font-weight:600;color:#5f4339}"
    "ul.srcs{list-style:none;padding:0;margin:0}ul.srcs>li{margin:.7rem 0}"
    "ul.srcs .tag{color:#8A6A4A;font-size:12px}"
    "ul.srcs ul{list-style:disc;margin:.35rem 0 .35rem 1.4rem;padding-left:1rem}"
    "ul.srcs ul li{margin:.3rem 0}"
    "ul.srcs .q{color:#555;font-style:italic;font-size:12.5px;margin:.1rem 0 0 0}"
    "</style>"
)

# Detailed scores in user-facing language. OUTCOME = persisted ranking scores (every run, from the
# trace); PROPOSER = the proposer model's own per-candidate self-assessments (only when the run
# logged its LLM calls, from audit_events). (trace key | label | what it means).
OUTCOME_SCORES = [
    ("field novelty", "Novelty (field-relative)", "How new this hypothesis is versus the retrieved literature, judged independently."),
    ("saturation", "Saturation", "How crowded / already-explored this area is (lower means more open)."),
    ("HypScore", "Composite hypothesis score", "Overall quality, combining the signals below."),
    ("RankScore", "Final rank score", "The value used to order the surfaced hypotheses."),
]
OUTCOME_FLAGS = [
    ("cross-concept", "Bridges distinct concepts", "Links concepts that were not previously connected."),
    ("common-sense", "Trivially true (common sense)", "Whether the hypothesis is obviously / trivially true."),
]
PROPOSER_SCORES = [
    ("novelty", "Novelty (proposer's estimate)", "The proposer model's own novelty rating."),
    ("duplication", "Overlap with existing work", "How much it duplicates known ideas (lower means less overlap)."),
    ("testability", "Testability", "How readily it can be tested empirically."),
    ("scope_fit", "Fit to the research scope", "How well it fits your stated research focus."),
    ("plausibility", "Plausibility", "How likely the proposed mechanism is to hold."),
    ("expected_yield", "Expected payoff", "Expected usefulness if it holds up."),
    ("centrality", "Centrality", "How central it is to the graph's structure."),
    ("mechanism_specificity", "Mechanism specificity", "How concretely the mechanism is spelled out."),
]


def _score_row(label: str, value: str, meaning: str) -> str:
    return f'<tr><td>{label}</td><td class="v">{value}</td><td class="m">{meaning}</td></tr>'


def _compact_plan_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool | int | float | str):
        return html.escape(str(value))
    if isinstance(value, list):
        parts = [_compact_plan_value(v) for v in value]
        return "; ".join(p for p in parts if p)
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            rendered = _compact_plan_value(val)
            if rendered:
                parts.append(f"<b>{html.escape(str(key).replace('_', ' '))}:</b> {rendered}")
        return "; ".join(parts)
    return html.escape(str(value))


def _plan_value_html(value) -> str:
    if value is None or value == "" or value == [] or value == {}:
        return ""
    if isinstance(value, bool | int | float | str):
        return f"<p>{html.escape(str(value))}</p>"
    if isinstance(value, list):
        items = []
        for item in value:
            rendered = _compact_plan_value(item)
            if rendered:
                items.append(f"<li>{rendered}</li>")
        return f"<ul>{''.join(items)}</ul>" if items else ""
    if isinstance(value, dict):
        rows = []
        for key, val in value.items():
            rendered = _plan_value_html(val)
            if rendered:
                rows.append(
                    f"<dt>{html.escape(str(key).replace('_', ' '))}</dt><dd>{rendered}</dd>"
                )
        return f"<dl>{''.join(rows)}</dl>" if rows else ""
    return f"<p>{html.escape(str(value))}</p>"


PLAN_FIELDS = list(EXPERIMENT_PLAN_FIELDS)


def _first_sentence(text: object, *, max_chars: int = 280) -> str:
    value = " ".join(str(text or "").split())
    if not value:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", value, maxsplit=1)[0].strip()
    sentence = sentence or value
    return sentence if len(sentence) <= max_chars else sentence[: max_chars - 1].rstrip() + "…"


def _experiment_core_summary(plan: dict) -> str:
    plain_design = _plain_design_text(plan)
    return f"<p>{html.escape(plain_design)}</p>" if plain_design else ""


def _source_lookup(pool: list[dict] | None) -> dict[str, dict]:
    """Index one retrieval batch by its batch-local evidence ID.

    Duplicate IDs inside a single malformed batch are omitted rather than resolved first-wins. The
    normal service/ledger contract makes IDs unique within a batch; cross-batch duplicates are
    handled before this helper is called.
    """
    out: dict[str, dict] = {}
    duplicate_ids: set[str] = set()
    for item in pool or []:
        if not isinstance(item, dict):
            continue
        eid = str(item.get("evidence_id") or item.get("id") or "").strip()
        if not eid or eid in duplicate_ids:
            continue
        if eid in out:
            duplicate_ids.add(eid)
            out.pop(eid, None)
        else:
            out[eid] = item
    return out


def _synthetic_retrieval_batch(pool: list[dict] | None) -> dict | None:
    """Wrap the historical flat ``pool=`` API as one explicitly trusted synthetic batch."""
    if not pool:
        return None
    return {
        "batch_id": "synthetic-flat-pool",
        "run_id": "",
        "input_hash": "",
        "retrieval_config_hash": "",
        "query": "",
        "evidence": list(pool),
        "synthetic": True,
    }


def _normalized_quote(value: object) -> str:
    # Some cached API abstracts preserve JSON-style escaped newlines as the two characters ``\\n``
    # rather than decoding them. Treat those separators like ordinary whitespace for verbatim-span
    # matching; no lexical fuzzing or semantic guess is introduced.
    text = str(value or "")
    for escaped in ("\\n", "\\r", "\\t"):
        text = text.replace(escaped, " ")
    return " ".join(text.split()).casefold()


def _batch_lookup(batch: dict) -> dict[str, dict]:
    evidence = batch.get("evidence") or []
    return _source_lookup(evidence if isinstance(evidence, list) else [])


def _explicit_plan_batch_id(plan: dict) -> str:
    """Future-compatible explicit scope; absent from legacy ExperimentPlan artifacts."""
    return str(plan.get("retrieval_batch_id") or "").strip()


def _grounding_quote_refs(plan: dict) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for item in plan.get("grounding") or []:
        if not isinstance(item, dict):
            continue
        eid = real_evidence_id(item.get("evidence_id"))
        quote = str(item.get("quote_span") or "").strip()
        if eid and _normalized_quote(quote):
            refs.append((eid, quote))
    return refs


def _resolve_plan_batch(
    plan: dict,
    batches: list[dict],
) -> tuple[dict | None, str]:
    """Resolve a plan to one cache batch without treating evidence IDs as globally unique.

    Explicit future scope wins. Legacy plans are matched by requiring every non-empty grounding
    quote span to occur in the same-ID record in one batch. With no grounding quotes, a batch must
    uniquely contain every cited ID. A historical flat ``pool=`` is one trusted synthetic batch so
    existing programmatic callers retain their behavior.
    """
    explicit = _explicit_plan_batch_id(plan)
    if explicit:
        matches = [batch for batch in batches if str(batch.get("batch_id") or "") == explicit]
        return (matches[0], "explicit") if len(matches) == 1 else (None, "unresolved")

    if len(batches) == 1 and batches[0].get("synthetic"):
        return batches[0], "synthetic"

    grounding_refs = _grounding_quote_refs(plan)
    if grounding_refs:
        matches = []
        for batch in batches:
            lookup = _batch_lookup(batch)
            if all(
                eid in lookup
                and _normalized_quote(quote) in _normalized_quote(lookup[eid].get("quote"))
                for eid, quote in grounding_refs
            ):
                matches.append(batch)
    else:
        cited_ids = {eid for eid, _role, _quote in _plan_evidence_refs(plan)}
        matches = [
            batch for batch in batches
            if cited_ids and cited_ids <= set(_batch_lookup(batch))
        ]

    if len(matches) == 1:
        return matches[0], "legacy_quote_match" if grounding_refs else "legacy_id_match"
    return None, "ambiguous" if len(matches) > 1 else "unresolved"


def _plan_evidence_refs(plan: dict) -> list[tuple[str, str, str]]:
    """(evidence_id, role, quote) triples for every citation the shared recursive collector finds
    in ``plan`` (placeholder ids already stripped). ``grounding`` / ``materials_or_data`` /
    ``metrics`` keep their role + quote presentation; any other evidence-bearing field a future
    schema adds falls back to its own field name as the role."""
    refs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(eid: str, role: str, quote: object = "") -> None:
        record = (eid, role, str(quote or "").strip())
        if record not in seen:
            seen.add(record)
            refs.append(record)

    for path_label, eid in collect_evidence_references(plan):
        match = re.match(r"^([A-Za-z_]+)\[(\d+)\]", path_label)
        field, index = (match.group(1), int(match.group(2))) if match else (path_label.split(".")[0], -1)
        if field == "grounding":
            add(eid, "grounding", plan["grounding"][index].get("quote_span"))
        elif field == "materials_or_data":
            item = plan["materials_or_data"][index].get("item")
            add(eid, f"material/data: {item or 'material/data'}")
        elif field == "metrics":
            metric = plan["metrics"][index].get("metric")
            add(eid, f"metric: {metric or 'metric'}")
        else:
            add(eid, field)
    refs.sort(key=lambda ref: (ref[0], ref[1], ref[2]))
    return refs


def _experiment_traceback_html(
    plan: dict,
    pool: list[dict] | None,
    *,
    batches: list[dict] | None = None,
) -> str:
    refs = _plan_evidence_refs(plan)
    if not refs:
        grounding_status = str(plan.get("grounding_status") or "")
        if grounding_status == "no_relevant_methods":
            detail = (
                "Methods retrieval was attempted, but the Designer found no relevant passage "
                "that could be cited honestly. This committed plan is an explicitly ungrounded draft."
            )
        elif grounding_status == "no_methods_retrieved":
            detail = (
                "No methods passage was retrieved. This committed plan is an explicitly "
                "ungrounded draft."
            )
        else:
            detail = (
                "No retrieved source is cited by this committed plan; it is therefore traceable "
                "to the Experiment Designer/Experiment Validator design step, but not to a "
                "specific retrieved methods passage."
            )
        return (
            "<h4>Traceback</h4>"
            f'<p class="sub">{html.escape(detail)}</p>'
        )

    available_batches = list(batches or [])
    if batches is None:
        synthetic = _synthetic_retrieval_batch(pool)
        available_batches = [synthetic] if synthetic else []
    selected_batch, scope_status = _resolve_plan_batch(plan, available_batches)
    sources = _batch_lookup(selected_batch) if selected_batch is not None else {}
    batch_id = str((selected_batch or {}).get("batch_id") or "")

    # One top-level item per resolved paper/composite reference, with every plan-use nested below.
    # This preserves the complete citation-use record without presenting repeated uses as sources.
    groups: dict[tuple[str, ...], dict] = {}
    for eid, role, quote_span in refs:
        source = sources.get(eid, {})
        title = str(source.get("title") or "").strip()
        url = str(source.get("url") or "").strip()
        paper_key = url or title
        key = (
            ("resolved", batch_id, paper_key or eid)
            if source
            else ("unresolved", eid)
        )
        group = groups.setdefault(
            key,
            {
                "source": source,
                "evidence_ids": [],
                "uses": [],
            },
        )
        if eid not in group["evidence_ids"]:
            group["evidence_ids"].append(eid)
        group["uses"].append((eid, role, quote_span))

    items = []
    resolved_source_keys: set[str] = set()
    for group in groups.values():
        source = group["source"]
        evidence_ids = group["evidence_ids"]
        uses = group["uses"]
        title = str(source.get("title") or "").strip()
        url = str(source.get("url") or "").strip()
        source_name = str(source.get("source") or "").strip()
        source_quote = str(source.get("quote") or "").strip()
        ids_text = ", ".join(evidence_ids)
        ids_html = html.escape(ids_text)
        title_html = (
            f'<a href="{html.escape(url)}">{html.escape(title or ids_text)}</a>'
            if url.startswith(("http://", "https://"))
            else html.escape(title or ids_text)
        )
        meta = f' <span class="tag">({html.escape(source_name)})</span>' if source_name else ""
        if source:
            resolved_source_keys.add(url or title or f"{batch_id}:{'|'.join(evidence_ids)}")
            source_label = f"{title_html}{meta}"
        else:
            reason = (
                "ambiguous legacy retrieval batch"
                if scope_status == "ambiguous"
                else "unresolved retrieval batch or evidence ID"
            )
            source_label = f'<span class="tag">({html.escape(reason)})</span>'
        source_quote_html = (
            f'<div class="q">Source excerpt: &ldquo;'
            f'{html.escape(_first_sentence(source_quote, max_chars=360))}&rdquo;</div>'
            if source_quote else ""
        )
        use_items = []
        show_use_id = len(evidence_ids) > 1
        for use_eid, role, quote_span in uses:
            use_label = (
                f"{html.escape(use_eid)} — {html.escape(role)}"
                if show_use_id
                else html.escape(role)
            )
            quote_html = (
                f'<div class="q">&ldquo;{html.escape(_first_sentence(quote_span, max_chars=360))}&rdquo;</div>'
                if quote_span else ""
            )
            use_items.append(f"<li>{use_label}{quote_html}</li>")
        items.append(
            f"<li><b>{ids_html}</b> — {source_label}{source_quote_html}"
            f"<ul>{''.join(use_items)}</ul></li>"
        )
    source_count = len(resolved_source_keys)
    source_word = "source" if source_count == 1 else "sources"
    use_word = "use" if len(refs) == 1 else "uses"
    scope_note = (
        ""
        if selected_batch is not None
        else " Retrieval scope could not be resolved uniquely, so no potentially wrong paper was attached."
    )
    return (
        "<h4>Traceback</h4>"
        f'<p class="sub">{source_count} resolved unique {source_word}; {len(refs)} citation '
        f'{use_word}. Uses are grouped by retrieval-batch-scoped evidence reference.{scope_note}</p>'
        f"<ul>{''.join(items)}</ul>"
    )


def experiment_plan_section(
    graph: dict, focus_ids: list[str], pool: list[dict] | None = None,
    edge_ids: tuple[str, ...] | None = None,
    batches: list[dict] | None = None,
    attempt: dict | None = None,
) -> str:
    """Render committed Experiment Designer/Experiment Validator experiment plans bound to this hypothesis.

    ``graph["experiment_plans"]`` is keyed by edge id. ``edge_ids`` (the trace sidecar's
    per-hypothesis ``hypothesis_edge_ids``) is the confirmed candidate's own
    committed new-edge ids; when present, only the plan(s) keyed by those exact ids render. Old
    artifacts carry no ``edge_ids``, so the section falls back to focus-node adjacency (any plan
    edge touching the focused node) — weaker, since it can mis-bind when two hypothesis edges
    share a focus node. Empty/absent experiment plans remain additive and render nothing.
    """
    plans = graph.get("experiment_plans") or {}
    if edge_ids:
        matched_edge_ids = [edge_id for edge_id in edge_ids if edge_id in plans]
    else:
        focus = set(focus_ids)
        matched_edge_ids = []
        for edge_id in sorted(plans):
            edge = graph.get("edges", {}).get(edge_id, {})
            edge_nodes = set(edge.get("source_node_ids") or []) | set(edge.get("target_node_ids") or [])
            if focus.intersection(edge_nodes):
                matched_edge_ids.append(edge_id)
    blocks = []
    for edge_id in matched_edge_ids:
        edge = graph.get("edges", {}).get(edge_id, {})
        plan = plans.get(edge_id) or {}
        relation = edge.get("relation_type") or edge_id[:8]
        rows = []
        grounding_status = str(plan.get("grounding_status") or "")
        if grounding_status in {"no_relevant_methods", "no_methods_retrieved"}:
            reason = (
                "retrieval returned no relevant, citable methods passage"
                if grounding_status == "no_relevant_methods"
                else "no methods passage was retrieved"
            )
            rows.append(
                '<p class="sub"><b>Ungrounded draft.</b> '
                f'{html.escape(reason.capitalize())}; the design is preserved, but its '
                "materials, baselines, and metrics are not literature-backed by this run.</p>"
            )
        summary = _experiment_core_summary(plan)
        if summary:
            rows.append(f"<h4>Core idea</h4>{summary}")
        for key, label in PLAN_FIELDS:
            rendered = _plan_value_html(plan.get(key))
            if rendered:
                rows.append(f"<h4>{html.escape(label)}</h4>{rendered}")
        rows.append(_experiment_traceback_html(plan, pool, batches=batches))
        if rows:
            blocks.append(
                f'<div class="exp-plan"><p class="sub">Committed plan for edge '
                f'<code>{html.escape(edge_id)}</code> ({html.escape(str(relation))}).</p>'
                f"{''.join(rows)}</div>"
            )
    if not blocks:
        if attempt and attempt.get("status") != "committed":
            reason = str(attempt.get("reason") or "the experiment attempt did not commit a plan")
            edge_id = str(attempt.get("primary_edge_id") or "")
            edge_note = (
                f' for edge <code>{html.escape(edge_id)}</code>' if edge_id else ""
            )
            return (
                "<section><h2>Experiment plan</h2>"
                f'<p class="sub"><b>No committed plan{edge_note}.</b> '
                f'{html.escape(reason)}.</p></section>'
            )
        return ""
    title = "Experiment plan" if len(blocks) == 1 else "Experiment plans"
    return (
        f"<section><h2>{title}</h2>"
        '<p class="sub">Detailed committed Experiment Designer/Experiment Validator plan attached to this hypothesis edge.</p>'
        f"{''.join(blocks)}</section>"
    )


def _why_surfaced_panel(why: dict | None) -> str:
    """Collapsible "Why surfaced" panel: deterministic rank, tournament rank, Elo, the
    two-sided pairwise record + debate rationale. Empty when the tournament did not run (additive)."""
    if not why:
        return ""
    items = []
    if why.get("deterministic_rank") is not None and why.get("tournament_rank") is not None:
        items.append(f"deterministic rank #{why['deterministic_rank']} &rarr; "
                     f"tournament rank #{why['tournament_rank']}")
    if why.get("elo") is not None:
        items.append(f"final Elo {float(why['elo']):.0f}")
    if why.get("record"):
        items.append(f"pairwise record (W-L-T) {html.escape(str(why['record']))}")
    lis = "".join(f"<li>{i}</li>" for i in items)
    rationale = (f'<p class="sub">{html.escape(str(why.get("rationale", "")))}</p>'
                 if why.get("rationale") else "")
    return f"<details><summary>Why surfaced</summary><ul>{lis}</ul>{rationale}</details>"


def _lineage_panel(lineage: dict | None) -> str:
    """Collapsible "Lineage" panel: the derivation DAG's parents,
    strategy, round, grounding status. Empty for proposer candidates / when evolution did not run."""
    if not lineage:
        return ""
    parents = lineage.get("wasDerivedFrom") or lineage.get("derived_from") or []
    parts = [f"derived from {html.escape(' & '.join(str(p) for p in parents))}" if parents
             else "proposer candidate (no parent)"]
    if lineage.get("strategy"):
        parts.append(f"strategy: {html.escape(str(lineage['strategy']))}")
    if lineage.get("round") is not None:
        parts.append(f"round {lineage['round']}")
    if lineage.get("grounding"):
        parts.append(f"grounding: {html.escape(str(lineage['grounding']))}")
    if lineage.get("agent"):
        parts.append(f"by: {html.escape(str(lineage['agent']))}")
    lis = "".join(f"<li>{p}</li>" for p in parts)
    return f"<details><summary>Lineage</summary><ul>{lis}</ul></details>"


def scores_section(scores: dict, signals: dict, *, why_surfaced: dict | None = None,
                   lineage: dict | None = None) -> str:
    """Detailed per-hypothesis scores in plain language: the ranking-outcome scores (persisted, every
    run) plus the proposer's 0–1 self-assessments (only when the run logged them). ``why_surfaced`` /
    ``lineage`` add the collapsible tournament and derivation-DAG panels when present."""
    if not scores and not signals and not why_surfaced and not lineage:
        return ""
    rows = ['<tr class="grp"><td colspan="3">Ranking outcome (what the system used to rank it)</td></tr>']
    for key, label, meaning in OUTCOME_SCORES:
        if key in scores:
            rows.append(_score_row(label, html.escape(scores[key]), meaning))
    for key, label, meaning in OUTCOME_FLAGS:
        if key in scores:
            val = {"True": "yes", "False": "no"}.get(scores[key], html.escape(scores[key]))
            rows.append(_score_row(label, val, meaning))
    if signals:
        rows.append('<tr class="grp"><td colspan="3">Proposer\'s self-assessment (the model\'s own 0–1 ratings)</td></tr>')
        for key, label, meaning in PROPOSER_SCORES:
            if key in signals:
                try:
                    val = f"{float(signals[key]):.2f}"
                except (TypeError, ValueError):
                    val = html.escape(str(signals[key]))
                rows.append(_score_row(label, val, meaning))
    fallback = (
        ""
        if signals
        else '<p class="sub">Finer proposer self-assessments (overlap, testability, fit, …) were not '
        "logged by this run, so only the ranking-outcome scores are shown.</p>"
    )
    return (
        '<section><h2>Scores for this hypothesis</h2>'
        '<p class="sub">Plain-language scores for this hypothesis. The ranking-outcome scores are what '
        "the system used to surface and rank it; the proposer self-assessments are the proposer model's "
        "own 0–1 ratings (shown when the run logged them). Edge-level confidence appears on the edges "
        "in the diagram for verified edges; unverified hypothesis edges carry none by design.</p>"
        f"{fallback}"
        '<table class="scores"><tr><th>Score</th><th>Value</th><th>What it means</th></tr>'
        f'{"".join(rows)}</table>'
        f"{_why_surfaced_panel(why_surfaced)}{_lineage_panel(lineage)}</section>"
    )


def sources_section(items: list[tuple[str, list[dict]]]) -> str:
    """The 'Sources' section, grouped BY PAPER: each source paper listed once, with the
    literature-mined concepts that trace to it bulleted underneath (label + verbatim quote).
    ``items`` is [(concept_label, [source dicts])]; empty -> no section."""
    if not items:
        return ""
    # group concepts under their source paper (key = url-or-title); a concept under >1 paper appears
    # under each. Papers ordered by title, concepts under each ordered by label (deterministic).
    groups: dict[str, dict] = {}
    for label, srcs in items:
        for s in srcs:
            key = s.get("url") or s.get("title")
            grp = groups.setdefault(key, {"paper": s, "concepts": []})
            grp["concepts"].append((label, s.get("quote", "")))
    blocks = []
    for key in sorted(groups, key=lambda k: groups[k]["paper"].get("title", "").lower()):
        s = groups[key]["paper"]
        url = s.get("url", "")
        name = (
            f'<a href="{html.escape(url)}">{html.escape(s["title"])}</a>'
            if url.startswith(("http://", "https://"))
            else html.escape(s["title"])
        )
        tag = f' <span class="tag">({html.escape(s["source"])})</span>' if s.get("source") else ""
        bullets = "".join(
            f"<li><b>{html.escape(lbl)}</b>"
            + (f'<div class="q">&ldquo;{html.escape(q)}&rdquo;</div>' if q else "")
            + "</li>"
            for lbl, q in sorted(groups[key]["concepts"])
        )
        blocks.append(f"<li>{name}{tag}<ul>{bullets}</ul></li>")
    return (
        '<section><h2>Sources &mdash; where the literature-mined concepts came from</h2>'
        '<p class="sub">Each source paper the concepts in this view were mined from (by verbatim '
        "quote-match against the run's retrieval pool), with the concepts that trace to it listed "
        "underneath. These concepts are also clickable in the diagram above.</p>"
        f'<ul class="srcs">{"".join(blocks)}</ul></section>'
    )


def _claim_from_profile(path) -> str:
    """Read the seed claim from a research-profile YAML (the authoritative source). Empty on any
    failure so the renderer still produces pages."""
    if not path:
        return ""
    try:
        from src.research_profile import load_research_profile
        return load_research_profile(path).claim
    except Exception:
        return ""


def claim_banner(claim: str | None, label: str = "Claim") -> str:
    """Show the run's claim or claimless research lens atop each page."""
    return (
        f'<section class="claim"><h2>{html.escape(label)}</h2><p>{html.escape(claim)}</p></section>'
        if claim else ""
    )


_SEED_KIND_LABELS = SEED_LABELS


def seed_context_card(claim: str | None, seed_kind: str | None) -> str:
    """The user's original seed, shown as its own context card next to a
    promoted Hypothesis card — the connected page otherwise drops the seed once a concise
    headline exists. Labeled by ``seed_kind`` ('claim' -> "Claim to investigate", 'research_goal' -> "Research
    goal", any other kind, e.g. a raw-message entry point, -> "Starting input"). Empty when
    ``seed_kind`` is absent (old sidecars keep hiding the seed — additive, default-off)."""
    if not seed_kind:
        return ""
    return claim_banner(claim, _SEED_KIND_LABELS.get(seed_kind, "Starting input"))


def concise_hypothesis_text(
    graph: dict, focus_ids: list[str], plain: dict | None = None
) -> str:
    """Reader-facing one-sentence hypothesis for the connected page.

    Only the reader card supplies prose. Experiment specifications deliberately contain
    graph notation and must stay in the experiment section. Keep the graph/focus arguments
    for callers of the existing renderer API.
    """
    if plain:
        headline = str(plain.get("headline", "") or plain.get("plain_sentence", "")).strip()
        if headline:
            return headline
    return "Plain-language summary unavailable for this hypothesis."


def _key_terms_html(key_terms: list | None) -> str:
    items = []
    for term in key_terms or []:
        if not isinstance(term, dict):
            continue
        label = html.escape(str(term.get("term", "")))
        # : a verified deeper-dive ``link`` (resolved driver-side, persisted in the sidecar)
        # makes the term clickable, ``link_kind`` as the tooltip; a ``familiar_gloss`` (the
        # reader-translation seam's phrasing) appends after the plain definition. Entries
        # without these fields render byte-identically (mirrors trace_render._key_terms_html).
        link = str(term.get("link", "") or "")
        if link:
            link_kind = html.escape(str(term.get("link_kind", "") or ""))
            head = f'<b><a href="{html.escape(link)}" title="{link_kind}">{label}</a></b>'
        else:
            head = f"<b>{label}</b>"
        definition = html.escape(str(
            term.get("plain_meaning", "")
            or term.get("universal_definition", "")
            or term.get("definition", "")
        ))
        gloss = f" — {definition}" if definition else ""
        familiar = str(term.get("familiar_gloss", "") or "")
        fam = (f'<span class="sub"> — in your terms: {html.escape(familiar)}</span>'
               if familiar else "")
        source = html.escape(str(term.get("source", "") or ""))
        cite = f' <span class="sub">({source})</span>' if source else ""
        items.append(f"<li>{head}{gloss}{fam}{cite}</li>")
    return f"<ul>{''.join(items)}</ul>" if items else ""


def _translations_html(translations: list | None, home_field: str = "") -> str:
    """: local twin of ``trace_render._translations_html`` (this module stays
    self-contained — no cross-import). The audited ``{original -> familiar}`` record,
    marked [confirmed] / [≈ approximate] with the translator's rationale; a rejected pair
    renders as "kept original" (the failed analogy phrasing never ships). Empty/absent -> ""."""
    items = []
    for record in translations or []:
        if not isinstance(record, dict):
            continue
        original = str(record.get("original", "") or "")
        if not original:
            continue
        if str(record.get("faithfulness", "") or "") == "rejected":
            items.append(
                f"<li><b>{html.escape(original)}</b> — kept original — proposed analogy "
                "failed the faithfulness check</li>"
            )
            continue
        mark = "[confirmed]" if record.get("faithfulness") == "confirmed" else "[≈ approximate]"
        rationale = str(record.get("rationale", "") or "")
        note = f' <span class="sub">{html.escape(rationale)}</span>' if rationale else ""
        items.append(
            f"<li><b>{html.escape(original)}</b> &rarr; "
            f"{html.escape(str(record.get('familiar', '') or ''))} "
            f'<span class="sub">{mark}</span>{note}</li>'
        )
    if not items:
        return ""
    reader = f"a {html.escape(home_field)} reader" if home_field else "this reader"
    return (
        f"<details><summary>Translated for {reader}</summary>"
        f"<ul>{''.join(items)}</ul></details>"
    )


def _next_steps_html(next_steps: list | None) -> str:
    """Mirror the trace renderer's ordered next-step list (flat
    plain strings, in doing order; defensively truncated at 6 items). Empty/absent -> ""
    (legacy cards render byte-identically)."""
    items = [
        f"<li>{html.escape(step)}</li>"
        for step in next_steps or []
        if isinstance(step, str) and step.strip()
    ][:6]
    if not items:
        return ""
    return f"<p><b>What to do next:</b></p><ol>{''.join(items)}</ol>"


def _mechanism_steps_section(steps: list | None) -> str:
    """Render the audited concept-to-relation-to-concept chain from sidecar data,
    chain, mirroring trace_render's section (warn badges ONLY for residual vague/false steps).
    Empty/absent -> "" (pages without audits byte-identical)."""
    items = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        arrow = f"{step.get('from', '')} --{normalize_relation_label(step.get('relation', ''))}--> {step.get('to', '')}"
        mech = str(step.get("mechanism", "") or "")
        mech_html = f' <span class="sub">{html.escape(mech)}</span>' if mech else ""
        verdict = str(step.get("verdict", "") or "")
        badge = ""
        if verdict in ("vague", "false"):
            note = str(step.get("note", "") or "")
            badge = f" <b>&#9888; [{html.escape(verdict)}]</b>"
            if note:
                badge += f' <span class="sub">{html.escape(note)}</span>'
        items.append(f"<li>{html.escape(arrow)}{mech_html}{badge}</li>")
    if not items:
        return ""
    return f'<section><h2>How the pieces connect</h2><ol>{"".join(items)}</ol></section>'


def _term_warnings_section(term_audit: list | None) -> str:
    """Render residual misused or stretched terms with the original
    source quote (mirrors trace_render's block; no_source renders as an info line). Empty -> ""."""
    items = []
    for verdict in term_audit or []:
        if not isinstance(verdict, dict):
            continue
        term = str(verdict.get("term", "") or "")
        grade = str(verdict.get("verdict", "") or "")
        if not term:
            continue
        if grade in ("misused", "stretched"):
            quote = str(verdict.get("source_quote", "") or "")
            title = str(verdict.get("source_title", "") or "")
            usage = str(verdict.get("usage", "") or "")
            note = str(verdict.get("note", "") or "")
            parts = [f"<b>&#9888; '{html.escape(term)}'</b> "
                     f'<span class="sub">[{html.escape(grade)}]</span>']
            if quote:
                cite = f' <span class="sub">({html.escape(title)})</span>' if title else ""
                parts.append(
                    f" &mdash; the source uses this as: &ldquo;{html.escape(quote)}&rdquo;{cite}"
                )
            if usage:
                parts.append(f"; this hypothesis: &ldquo;{html.escape(usage)}&rdquo;")
            if note:
                parts.append(f' <span class="sub">{html.escape(note)}</span>')
            items.append(f"<li>{''.join(parts)}</li>")
        elif grade == "no_source":
            items.append(
                f'<li><span class="sub">\'{html.escape(term)}\' &mdash; no original source '
                "occurrence found; its meaning here is unverified.</span></li>"
            )
    if not items:
        return ""
    return f'<section><h2>Term check</h2><ul>{"".join(items)}</ul></section>'


def hypothesis_prose(plain: dict | None, audits: dict | None = None,
                     reader: dict | None = None, include_headline: bool = True) -> str:
    """Render the hypothesis's plain-language card.

    Falls back to ``plain_sentence`` and ``elaboration`` when structured card fields are absent.
    The sidecar's per-candidate ``audits`` entry appends deterministic mechanism-chain and
    terminology-warning sections. Its top-level ``reader`` block names the home field in the
    translation summary. Connected pages that
    already promote the headline into a top Hypothesis card can pass ``include_headline=False``
    so the explanation starts with the mechanism instead of repeating the hypothesis."""
    card = ""
    headline = ""
    if plain:
        headline = str(plain.get("headline", "") or plain.get("plain_sentence", "")).strip()
    if plain and headline:
        sections = [
            ("What might be happening", plain.get("what_might_be_happening")),
            ("Why it matters", plain.get("why_it_matters")),
            ("How to check", plain.get("how_to_check")),
            ("What would change our mind", plain.get("what_would_change_our_mind")),
            ("Status", plain.get("status")),
            ("What problem this solves", plain.get("problem_solved")),
        ]
        rows = [
            f"<p><b>{html.escape(label)}:</b> {html.escape(str(value))}</p>"
            for label, value in sections if value
        ]
        if not rows and plain.get("elaboration"):
            rows.append(f"<p>{html.escape(str(plain.get('elaboration', '')))}</p>")
        home_field = str((reader or {}).get("home_field", "") or "")
        lead = f"<p><b>{html.escape(headline)}</b></p>" if include_headline else ""
        card = (
            '<section class="plain">'
            "<h2>Possible explanation</h2>"
            f"{lead}"
            f"{''.join(rows)}"
            f"{_next_steps_html(plain.get('next_steps'))}"
            f"{_key_terms_html(plain.get('key_terms'))}"
            f"{_translations_html(plain.get('translations'), home_field)}"
            "</section>"
        )
    if not audits:
        return card
    return (
        card
        + _mechanism_steps_section(audits.get("mechanism_steps"))
        + _term_warnings_section(audits.get("term_audit"))
    )


def render_page(graph, run_name, row, focus_ids, pool, signals, claim=None, plain=None,
                audits=None, reader=None, banner_label="Claim", edge_ids=None, seed_kind=None,
                retrieval_batches=None, experiment_attempt=None):
    nodes, reachable, dist, edge_list = build(graph, focus_ids)
    # resolve the literature-mined concepts in this component back to their source paper(s)
    mined_items: list[tuple[str, list[dict]]] = []
    node_links: dict[str, dict] = {}
    for nid in reachable:
        n = nodes[nid]
        if node_class(n) != "mined" or not pool:
            continue
        srcs = node_sources(n, pool)
        if srcs:
            mined_items.append((n["label"], srcs))
            if srcs[0]["url"].startswith(("http://", "https://")):
                node_links[nid] = {"url": srcs[0]["url"], "title": srcs[0]["title"]}
    mined_items.sort(key=lambda x: x[0])

    diagram, n_edges = mermaid(nodes, focus_ids, row["cid"], reachable, dist, edge_list, node_links)
    total_n, total_e = len(graph["nodes"]), len(graph["edges"])
    excluded = total_n - len(reachable)
    focus_label = ", ".join(graph["nodes"][fid]["label"] for fid in focus_ids)
    cid = row["cid"]
    hypothesis_text = concise_hypothesis_text(graph, focus_ids, plain)
    top_label = "Proposed hypothesis" if hypothesis_text else banner_label
    top_text = hypothesis_text or claim
    # Once the hypothesis headline is promoted to the top card, the seed would otherwise be
    # hidden entirely — show it in its own context card, labeled by seed_kind (additive/default-off).
    seed_card = seed_context_card(claim, seed_kind) if hypothesis_text else ""
    relationship = relationship_section(plain, seed_kind) if claim else ""
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Hypothesis {html.escape(cid)} — connected component</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/mermaid@11.11.0/dist/mermaid.min.js"></script><style>{STYLE}</style>{EXTRA_STYLE}</head><body>
<header><h1>Hypothesis {html.escape(cid)} — connected component (everything reachable from it)</h1>
<p>Every node in <code>{html.escape(run_name)}/graph.json</code> (v{graph['version']}) connected to hypothesis
<b>{html.escape(cid)}</b> (<i>{html.escape(focus_label)}</i>, rank {row['rank']}) directly or indirectly, following
committed edges. <b>{len(reachable)} of {total_n} nodes</b>, {n_edges} of {total_e} edges.
The remaining {excluded} nodes are not connected to it (isolated literature-mined concepts no hypothesis
linked, plus the separately-connected hypotheses).</p></header>
<main>
  {seed_card}{claim_banner(top_text, top_label)}{relationship}
  {hypothesis_prose(plain, audits, reader, include_headline=not bool(hypothesis_text))}
  {experiment_plan_section(graph, focus_ids, pool, edge_ids=edge_ids, batches=retrieval_batches,
                           attempt=experiment_attempt)}
  <section>
    <h2>Hypothesis {html.escape(cid)}'s reachable neighbourhood</h2>
    <p class="sub">Node colour = provenance source (bold violet = this hypothesis). Edge style = committed status:
    solid green = insufficient (evidence-verified; label = confidence), dashed orange = unverified hypothesis edge
    (unverified edges have not been through verification, so they carry no confidence value).</p>
    <div class="legend">
      <span><span class="swatch" style="background:#f1e1d8;border:2px solid #B25A36"></span> ★ this hypothesis</span>
      <span><span class="swatch" style="background:#f7ede7;border:1px solid #B25A36"></span> other hypothesis concept</span>
      <span><span class="swatch" style="background:#f4efe8;border:1px solid #8A6A4A"></span> literature-mined concepts</span>
      <span><span class="swatch" style="background:#eef3f7;border:1px solid #2E5570"></span> claim concept</span>
      <span><span class="swatch" style="background:#fff8e1;border:2px solid #f9a825"></span> ★ user-priority</span>
      <span><span class="edgekey" style="border-top:2px solid #4F6A46"></span> insufficient (evidence-verified)</span>
      <span><span class="edgekey" style="border-top:2px dashed #B25A36"></span> unverified</span>
    </div>
    <div class="mermaid">
{diagram}
    </div>
  </section>
  {scores_section(row.get("scores"), signals)}
  {sources_section(mined_items)}
</main>
<script>mermaid.initialize({{startOnLoad:true,theme:"default",themeVariables:{{fontFamily:"'DM Sans', sans-serif"}},flowchart:{{useMaxWidth:true,htmlLabels:true}},securityLevel:"loose"}});</script>
</body></html>"""
    return page, len(reachable)


def pick_run_dir() -> Path:
    """Auto-pick the newest out/*/ directory that holds a graph.json (the 'new round')."""
    candidates = [Path(p).parent for p in glob.glob("out/*/graph.json")]
    if not candidates:
        raise SystemExit("no out/*/graph.json found — pass --run-dir explicitly")
    return max(candidates, key=lambda d: (d / "graph.json").stat().st_mtime)


def render_connected_hypotheses(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
    events_db: Path | None = None,
    run_id: str | None = None,
    claim: str | None = None,
    lens: str | None = None,
    profile: Path | None = None,
    only: str | None = None,
) -> list[Path]:
    """Render one ``<candidate_id>-connected.html`` page per surfaced hypothesis of ``run_dir``.

    Reads ``run_dir/graph.json``, ``run_dir/trace/h*.html``, and the optional
    ``run_dir/trace/plain_language.json`` sidecar; resolves provenance links + proposer
    self-assessments from ``events_db`` (or the best-matching ``*.sqlite`` under the cwd/run
    dir) when present. ``run_id`` identifies the exact logical run in that event DB; it defaults
    to ``run_dir.name`` for backward compatibility with the one-directory-per-run layout. Offline
    and deterministic: no LLM calls, no network calls, no graph-state mutation — only reads run
    artifacts and writes the connected pages. Returns the list of pages written, in surfaced
    order. ``only`` restricts rendering to a single candidate id (the CLI's ``--only``).
    """
    out_dir = out_dir or run_dir
    graph_path = run_dir / "graph.json"
    if not graph_path.exists():
        raise SystemExit(f"no graph.json under {run_dir}")
    graph = json.loads(graph_path.read_text())
    by_label = {n["label"]: nid for nid, n in graph["nodes"].items()}
    surfaced = parse_surfaced(run_dir / "trace")
    if not surfaced:
        raise SystemExit(f"no trace/h*.html under {run_dir} — nothing to render")

    # Plain-language layer (claim + per-hypothesis LLM paraphrase), persisted by write_trace.
    plain_path = run_dir / "trace" / "plain_language.json"
    plain_data = json.loads(plain_path.read_text(encoding="utf-8")) if plain_path.exists() else {}
    plain_by_cid = plain_data.get("hypotheses", {})
    audits_by_cid = plain_data.get("audits", {})  # Optional finalization audits.
    reader_ctx = plain_data.get("reader", {})  # Optional reader context for the panel summary.
    # Each hypothesis's committed edge IDs; old artifacts fall back to focus-node adjacency.
    edge_ids_by_cid = plain_data.get("hypothesis_edge_ids", {})
    experiment_attempts_by_cid = plain_data.get("experiment_attempts", {})
    # The run's seed kind ('claim' | 'research_goal' | 'raw_message'), set by the profile at
    # load time (additive key; absent for old sidecars -> no seed context card).
    seed_kind = plain_data.get("seed_kind")
    # Banner resolution: explicit claim, profile claim, sidecar claim, then the claimless lens.
    claim = claim or _claim_from_profile(profile) or plain_data.get("claim", "") or ""
    banner_label = "Claim"
    if not claim and lens:
        claim, banner_label = lens, "Research lens"
    print(f"  plain-language: claim={'set' if claim else '(none)'}, "
          f"paraphrases for {sum(1 for v in plain_by_cid.values() if v)} candidate(s)"
          f" (from {plain_path if plain_path.exists() else 'no sidecar'})")

    # candidate event DBs: explicit events_db, else every *.sqlite in cwd + the run dir.
    cands = ([str(events_db)] if events_db is not None
             else sorted(set(glob.glob("*.sqlite")) | set(glob.glob(str(run_dir / "*.sqlite")))))
    # source-link pool: the DB whose retrieval pool resolves the most of THIS run's mined concepts.
    mined_nodes = [n for n in graph["nodes"].values() if node_class(n) == "mined"]
    pool_db = select_best_db(cands, mined_nodes) if cands else None
    retrieval_batches = (
        load_retrieval_batches(pool_db, run_id=run_id) if pool_db else []
    )
    pool = [item for batch in retrieval_batches for item in batch["evidence"]]
    if mined_nodes:
        resolved = sum(1 for n in mined_nodes if pool and node_sources(n, pool))
        print(f"  provenance: db={pool_db or '(none)'} pool={len(pool)} — resolved {resolved}/{len(mined_nodes)} mined concepts")
    # proposer self-assessed scores: ONLY from the DB that logged THIS run, never another run's
    # log (candidate ids like h1 recur across runs). Older callers omit the logical id and retain
    # the historical one-directory-per-run convention.
    logical_run_id = run_id or run_dir.name
    audit_db = events_db_for_run(cands, logical_run_id)
    signals_by_cid = (
        proposer_signals(load_audit_events(audit_db, run_id=logical_run_id)) if audit_db else {}
    )
    print(f"  detailed scores: proposer self-assessments for {len(signals_by_cid)} candidate(s) "
          f"({'from ' + audit_db if audit_db else 'no log for run_id=' + logical_run_id})")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    unconfirmed: list[str] = []
    for row in surfaced:
        if only and row["cid"] != only:
            continue
        # A current sidecar's exact committed-edge bindings are authoritative. A focus label can
        # still exist because another candidate committed the same node, or because an older
        # malformed node-only hypothesis transaction was accepted; neither makes this candidate
        # a committed hypothesis. Legacy sidecars omit this mapping and retain adjacency fallback.
        if edge_ids_by_cid and row["cid"] not in edge_ids_by_cid:
            unconfirmed.append(row["cid"])
            print(f"  – {row['cid']} (rank {row['rank']}): not confirmed, so "
                  f"{row['cell']!r} was never committed — no connected page")
            continue
        focus_ids = resolve_focus_ids(row["cell"], by_label)
        if not focus_ids:
            # A surfaced candidate the confirm policy did not confirm was never
            # committed, so its new concept is not a node and no connected component
            # exists to draw. That is the ordinary outcome of a `top_k` or
            # `threshold` policy, not a rendering fault, and saying so keeps a
            # normal run from reading like a partial failure. The distinction needs
            # the committed-edge sidecar; without it, report the label mismatch.
            print(f"  ! {row['cid']}: new concept {row['cell']!r} is not an exact node label — SKIPPED (no guess)")
            continue
        page = out_dir / f"{row['cid']}-connected.html"
        html_text, n_reach = render_page(graph, run_dir.name, row, focus_ids, pool,
                                         signals_by_cid.get(row["cid"], {}),
                                         claim=claim, plain=plain_by_cid.get(row["cid"]),
                                         audits=audits_by_cid.get(row["cid"]),
                                         reader=reader_ctx, banner_label=banner_label,
                                         edge_ids=edge_ids_by_cid.get(row["cid"]),
                                         seed_kind=seed_kind,
                                         retrieval_batches=retrieval_batches,
                                         experiment_attempt=experiment_attempts_by_cid.get(row["cid"]))
        page.write_text(html_text, encoding="utf-8")
        print(f"  ✓ {row['cid']} (rank {row['rank']}): {page}  [{n_reach}/{len(graph['nodes'])} nodes]")
        written.append(page)
    if only and only not in {r["cid"] for r in surfaced}:
        print(f"  ! --only {only!r} matched no surfaced hypothesis; available: "
              + ", ".join(r["cid"] for r in surfaced))
    tail = (
        f" ({len(unconfirmed)} surfaced candidate(s) not confirmed: "
        + ", ".join(unconfirmed) + ")"
        if unconfirmed
        else ""
    )
    print(f"done: wrote {len(written)} hypothesis page(s) into {out_dir}{tail}")
    return written


def render_connected_after_trace(
    run_root: Path,
    *,
    enabled: bool = True,
    run_id: str | None = None,
    events_db: Path | None = None,
) -> list[Path]:
    """Render connected pages automatically once a run's artifacts are on disk.

    ``synthesist_run.py`` calls this immediately after ``write_trace()`` finishes.

    ``enabled=False`` is the ``--no-connected-render`` escape hatch: skip silently (an intentional
    opt-out, not a defect). Otherwise, render only when ``run_root`` already has both
    ``graph.json`` and a ``trace/`` directory; an incomplete layout emits ONE explicit warning
    line and skips rather than guessing at missing artifacts. Offline and deterministic
    (delegates to ``render_connected_hypotheses``): no LLM calls, no network calls, no
    graph-state mutation. ``run_id`` and ``events_db`` are forwarded when the caller knows the
    exact audit-log identity; omitted values preserve the renderer's artifact-discovery behavior.
    Never raises — a rendering failure must not fail the run.
    """
    if not enabled:
        return []
    if not (run_root / "graph.json").is_file() or not (run_root / "trace").is_dir():
        print(
            f"connected-render: skipped — {run_root} does not yet have both graph.json and a "
            "trace/ directory"
        )
        return []
    try:
        return render_connected_hypotheses(
            run_root,
            run_id=run_id,
            events_db=events_db,
        )
    except SystemExit as exc:
        print(f"connected-render: skipped — {exc}")
        return []
