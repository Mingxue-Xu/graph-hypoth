#!/usr/bin/env python3
"""Render runtime_logs/<run>/trace.jsonl into static HTML reports.

Usage:
    python scripts/render_runtime_logs.py [--root runtime_logs]

Produces:
    runtime_logs/index.html              # listing of all runs
    runtime_logs/<run>/report.html       # per-run timeline
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

EVENT_TYPE_COLORS = {
    "artifact": "#90caf9",
    "agent_tool_call": "#a5d6a7",
    "agent_with_api": "#ffcc80",
}

EVENT_TYPE_BG = {
    "artifact": "#e7f0fa",
    "agent_tool_call": "#e8f5e9",
    "agent_with_api": "#fff3e0",
}

STATUS_BG = {
    "success": "#c8e6c9",
    "failed": "#ffcdd2",
    "skipped": "#eeeeee",
}


@dataclass
class Run:
    run_dir: Path
    manifest: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.manifest.get("label") or self.run_dir.name

    @property
    def pid(self) -> Any:
        return self.manifest.get("pid", "")

    @property
    def created_at(self) -> str:
        return self.manifest.get("created_at") or ""


def discover_runs(root: Path) -> list[Path]:
    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir()
        and (child / "manifest.json").is_file()
        and (child / "trace.jsonl").is_file()
    )


def load_run(run_dir: Path) -> Run:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events: list[dict[str, Any]] = []
    with (run_dir / "trace.jsonl").open() as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(
                    f"warning: skipping malformed JSON in {run_dir.name} line {lineno}: {exc}",
                    file=sys.stderr,
                )
    return Run(run_dir=run_dir, manifest=manifest, events=events)


def parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def event_type_counts(events: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ev in events:
        et = ev.get("event_type") or "unknown"
        counts[et] = counts.get(et, 0) + 1
    return counts


def aggregate_source_statuses(events: list[dict]) -> list[dict]:
    """Aggregate per-source status counts from agent_tool_call response events."""
    bucket: dict[str, dict] = {}
    for ev in events:
        if ev.get("event_type") != "agent_tool_call" or ev.get("direction") != "response":
            continue
        for ss in (ev.get("payload") or {}).get("source_statuses") or []:
            src = ss.get("source")
            if not src:
                continue
            slot = bucket.setdefault(
                src, {"source": src, "calls": 0, "ok": 0, "fail": 0, "skip": 0, "errors": []}
            )
            slot["calls"] += 1
            status = ss.get("status")
            if status == "success":
                slot["ok"] += 1
            elif status == "failed":
                slot["fail"] += 1
                for e in ss.get("errors") or []:
                    if e and e not in slot["errors"]:
                        slot["errors"].append(e)
            elif status == "skipped":
                slot["skip"] += 1
    return list(bucket.values())


def format_delta(seconds: float) -> str:
    if seconds < 1:
        return f"+{int(seconds * 1000)}ms"
    if seconds < 60:
        return f"+{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    return f"+{int(minutes)}m{int(secs):02d}s"


def trunc(s: str, n: int = 80) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; background: #fafbfc; color: #24292e; }
header.run { position: sticky; top: 0; z-index: 10; background: #fff;
             border-bottom: 1px solid #e1e4e8; padding: 16px 24px; }
header.run h1 { margin: 0 0 4px; font-size: 18px; }
header.run .meta { font-size: 12px; color: #586069; }
header.run .meta a { color: #0366d6; text-decoration: none; }
header.run .chips { margin-top: 8px; }
.chip { display: inline-block; padding: 2px 8px; margin: 2px 4px 2px 0;
        border-radius: 12px; font-size: 11px; font-weight: 500; color: #1b1b1b; }
.timeline { padding: 16px 24px 64px; }
details.event { margin: 6px 0; border: 1px solid #e1e4e8; border-radius: 4px;
                background: #fff; }
details.event > summary { padding: 8px 12px; cursor: pointer; list-style: none;
                          font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                          font-size: 12px; }
details.event > summary::-webkit-details-marker { display: none; }
details.event[open] > summary { border-bottom: 1px solid #e1e4e8; }
.event .body { padding: 12px; }
.event .body pre { background: #f6f8fa; padding: 12px; border-radius: 4px;
                   overflow: auto; font-size: 12px; line-height: 1.4;
                   max-height: 480px; white-space: pre-wrap; word-wrap: break-word; }
.event .body details.raw { margin-top: 12px; }
.event .body details.raw > summary { cursor: pointer; font-size: 11px; color: #6a737d; }
.event .body details.raw pre { max-height: 240px; }
.event .body .field { margin: 4px 0; font-size: 12px; }
.event .body .field .key { color: #6a737d; display: inline-block; min-width: 110px; }
.event .body ul.items { margin: 6px 0; padding-left: 20px; }
.event .body ul.items li { margin: 4px 0; font-size: 12px; }
.event .body a { color: #0366d6; }
.delta { color: #6a737d; min-width: 64px; display: inline-block; }
.actor { color: #0366d6; font-weight: 500; }
.target { color: #6f42c1; }
.eventtype { font-weight: 600; }
.dim { color: #6a737d; }
table.runs { border-collapse: collapse; width: 100%; background: #fff;
             border: 1px solid #e1e4e8; border-radius: 4px; }
table.runs th, table.runs td { padding: 8px 12px; text-align: left;
                                border-bottom: 1px solid #e1e4e8; font-size: 13px;
                                vertical-align: top; }
table.runs th { background: #f6f8fa; font-weight: 600; }
table.runs td a { color: #0366d6; text-decoration: none;
                  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                  font-size: 12px; }
table.runs td a:hover { text-decoration: underline; }
.status-ok { color: #2e7d32; font-weight: 600; }
.status-fail { color: #c62828; font-weight: 600; }
.status-skip { color: #757575; }
"""


def chip(label: str, color: str) -> str:
    return f'<span class="chip" style="background:{color}">{html.escape(label)}</span>'


def event_type_chip(et: str, count: int) -> str:
    return chip(f"{et}: {count}", EVENT_TYPE_BG.get(et, "#eeeeee"))


def source_chip(s: dict) -> str:
    label = f'{s["source"]}: {s["ok"]}✓ {s["fail"]}✗ {s["skip"]}⊘'
    color = STATUS_BG["success"] if s["fail"] == 0 and s["ok"] > 0 else (
        STATUS_BG["failed"] if s["fail"] > 0 and s["ok"] == 0 else "#fff9c4"
    )
    if s["ok"] == 0 and s["fail"] == 0 and s["skip"] > 0:
        color = STATUS_BG["skipped"]
    return chip(label, color)


def json_block(payload: Any, max_chars: int | None = None) -> str:
    try:
        text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(payload)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n… [{len(text) - max_chars} more chars truncated]"
    return f"<pre>{html.escape(text)}</pre>"


def field_row(key: str, value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return (
        f'<div class="field"><span class="key">{html.escape(key)}</span>'
        f"<span>{html.escape(str(value))}</span></div>"
    )


def text_block(label: str, content: str) -> str:
    if not content:
        return ""
    return (
        f'<div class="field"><span class="key">{html.escape(label)}</span></div>'
        f"<pre>{html.escape(content)}</pre>"
    )


def keyfact(et: str, direction: str, payload: dict) -> str:
    if et == "agent_tool_call":
        if direction == "request":
            srcs = payload.get("sources")
            srcs_str = f" sources={srcs}" if srcs else ""
            return f'query="{trunc(payload.get("query") or "", 60)}"{srcs_str}'
        if direction == "response":
            return (
                f'{len(payload.get("evidence") or [])} evidence, '
                f'{len(payload.get("errors") or [])} errors, '
                f'elapsed={payload.get("elapsed_ms")}ms'
            )
    if et == "agent_with_api":
        if direction == "request":
            q = payload.get("query") or payload.get("model") or ""
            return f'query="{trunc(q, 60)}"' if q else ""
        if direction == "response":
            return f'{payload.get("status") or ""}, {len(payload.get("results") or [])} results'
        if direction == "raw_response":
            n = len(payload.get("results") or payload.get("documents") or [])
            return f"{n} raw results"
    if et == "artifact":
        if direction == "setup":
            return f'path={payload.get("path", "")}'
        if direction == "write":
            if "event_type" in payload and "sender_role" in payload:
                tu = (payload.get("token_usage") or {}).get("total_tokens")
                tu_str = f" tokens={tu}" if tu else ""
                return (
                    f'sender={payload.get("sender_role")}  '
                    f'type={payload.get("event_type")}{tu_str}'
                )
            if "input_hash" in payload:
                return f'input_hash={(payload.get("input_hash") or "")[:12]}…'
        if direction == "read_hit":
            return f'input_hash={(payload.get("input_hash") or "")[:12]}…'
    return ""


def summary_html(ev: dict, t0: datetime | None) -> str:
    et = ev.get("event_type") or "unknown"
    direction = ev.get("direction") or ""
    actor = ev.get("actor") or "—"
    target = ev.get("target") or "—"
    payload = ev.get("payload") or {}

    ts = parse_ts(ev.get("timestamp") or "")
    delta = format_delta((ts - t0).total_seconds()) if t0 and ts else ""

    fact = keyfact(et, direction, payload)
    fact_html = f' · <span class="dim">{html.escape(fact)}</span>' if fact else ""

    return (
        f'<span class="delta">{html.escape(delta)}</span> '
        f'<span class="eventtype">{html.escape(et)}</span> · '
        f'<span class="dim">{html.escape(direction)}</span> · '
        f'<span class="actor">{html.escape(str(actor))}</span> → '
        f'<span class="target">{html.escape(str(target))}</span>'
        f"{fact_html}"
    )


def render_audit_event_body(payload: dict) -> str:
    parts = [
        field_row("sender_role", payload.get("sender_role")),
        field_row("event_type", payload.get("event_type")),
        field_row("checkpoint_id", payload.get("checkpoint_id")),
        field_row("round_index", payload.get("round_index")),
        field_row("model", payload.get("model")),
        field_row("latency_ms", payload.get("latency_ms")),
    ]
    tu = payload.get("token_usage") or {}
    if tu:
        parts.append(
            field_row(
                "tokens",
                f'prompt={tu.get("prompt_tokens")} '
                f'completion={tu.get("completion_tokens")} '
                f'total={tu.get("total_tokens")}',
            )
        )
    parts.append(text_block("content", payload.get("content") or ""))
    sp = payload.get("structured_payload") or {}
    if sp:
        parts.append('<div class="field"><span class="key">structured_payload</span></div>')
        parts.append(json_block(sp, max_chars=4000))
    tcs = payload.get("tool_calls") or []
    if tcs:
        parts.append('<div class="field"><span class="key">tool_calls</span></div>')
        parts.append(json_block(tcs, max_chars=4000))
    return "".join(parts)


def render_tool_call_response(payload: dict) -> str:
    parts = [
        field_row("query", payload.get("query")),
        field_row("sources", payload.get("sources")),
        field_row("elapsed_ms", payload.get("elapsed_ms")),
    ]
    statuses = payload.get("source_statuses") or []
    if statuses:
        parts.append('<div class="field"><span class="key">source_statuses</span></div>')
        parts.append('<ul class="items">')
        for s in statuses:
            status = str(s.get("status", ""))
            cls = (
                "status-ok"
                if status == "success"
                else "status-fail"
                if status == "failed"
                else "status-skip"
            )
            errs = s.get("errors") or []
            err_str = (
                f' — <span class="dim">{html.escape("; ".join(str(e) for e in errs))}</span>'
                if errs
                else ""
            )
            parts.append(
                f'<li><span class="{cls}">[{html.escape(status)}]</span> '
                f'{html.escape(str(s.get("source", "")))}: '
                f'{s.get("result_count", 0)} results{err_str}</li>'
            )
        parts.append("</ul>")
    evidences = payload.get("evidence") or []
    if evidences:
        parts.append(
            f'<div class="field"><span class="key">evidence</span>'
            f"<span>{len(evidences)} items</span></div>"
        )
        parts.append('<ul class="items">')
        for e in evidences:
            title = html.escape(str(e.get("title", "(no title)")))
            src = html.escape(str(e.get("source", "")))
            url = e.get("url") or ""
            link = (
                f' <a href="{html.escape(url)}" target="_blank" rel="noopener">[link]</a>'
                if url
                else ""
            )
            parts.append(
                f'<li><strong>{title}</strong> '
                f'<span class="dim">({src})</span>{link}</li>'
            )
        parts.append("</ul>")
    return "".join(parts)


def render_api_event(payload: dict) -> str:
    parts = [
        field_row("query", payload.get("query")),
        field_row("model", payload.get("model")),
        field_row("status", payload.get("status")),
    ]
    results = payload.get("results")
    if isinstance(results, list) and results:
        parts.append(
            f'<div class="field"><span class="key">results</span>'
            f"<span>{len(results)} items</span></div>"
        )
        parts.append('<ul class="items">')
        for r in results[:10]:
            if not isinstance(r, dict):
                continue
            title = html.escape(str(r.get("title", "(no title)")))
            url = r.get("url") or ""
            link = (
                f' <a href="{html.escape(url)}" target="_blank" rel="noopener">[link]</a>'
                if url
                else ""
            )
            parts.append(f"<li>{title}{link}</li>")
        if len(results) > 10:
            parts.append(f'<li class="dim">… and {len(results) - 10} more</li>')
        parts.append("</ul>")
    errors = payload.get("errors") or []
    if errors:
        parts.append('<div class="field"><span class="key">errors</span></div>')
        parts.append('<ul class="items">')
        for e in errors:
            parts.append(f'<li class="status-fail">{html.escape(str(e))}</li>')
        parts.append("</ul>")
    return "".join(parts)


def render_event_body(ev: dict) -> str:
    et = ev.get("event_type")
    direction = ev.get("direction")
    payload = ev.get("payload") or {}

    rich = ""
    if et == "artifact" and direction == "write" and "event_type" in payload and "sender_role" in payload:
        rich = render_audit_event_body(payload)
    elif et == "agent_tool_call" and direction == "response":
        rich = render_tool_call_response(payload)
    elif et == "agent_with_api":
        rich = render_api_event(payload)

    raw = (
        '<details class="raw"><summary>Raw event JSON</summary>'
        f"{json_block(ev, max_chars=20000)}"
        "</details>"
    )
    return f'<div class="body">{rich}{raw}</div>'


def render_event(ev: dict, t0: datetime | None) -> str:
    et = ev.get("event_type") or "unknown"
    border = EVENT_TYPE_COLORS.get(et, "#cccccc")
    return (
        f'<details class="event" style="border-left:4px solid {border}">'
        f"<summary>{summary_html(ev, t0)}</summary>"
        f"{render_event_body(ev)}"
        "</details>"
    )


def render_run_html(run: Run) -> str:
    events = run.events
    counts = event_type_counts(events)
    sources = aggregate_source_statuses(events)

    timestamps = [t for t in (parse_ts(e.get("timestamp") or "") for e in events) if t]
    t0 = timestamps[0] if timestamps else None
    t_end = timestamps[-1] if timestamps else None
    elapsed = (t_end - t0).total_seconds() if t0 and t_end else 0.0

    type_chips = "".join(event_type_chip(et, n) for et, n in sorted(counts.items()))
    src_chips = "".join(source_chip(s) for s in sources) or '<span class="dim">no retrieval calls</span>'
    timeline = "\n".join(render_event(ev, t0) for ev in events)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Run: {html.escape(run.label)}</title>
<style>{CSS}</style>
</head><body>
<header class="run">
  <h1>{html.escape(run.label)}</h1>
  <div class="meta">
    pid={html.escape(str(run.pid))} ·
    created_at={html.escape(run.created_at)} ·
    {len(events)} events ·
    elapsed={elapsed:.2f}s ·
    <a href="../index.html">← all runs</a>
  </div>
  <div class="chips">{type_chips}</div>
  <div class="chips">{src_chips}</div>
</header>
<div class="timeline">
{timeline}
</div>
</body></html>
"""


def render_index_html(runs: list[Run], aggregates: list[Path]) -> str:
    sorted_runs = sorted(runs, key=lambda r: r.created_at, reverse=True)

    rows = []
    for run in sorted_runs:
        counts = event_type_counts(run.events)
        sources = aggregate_source_statuses(run.events)
        type_chips = "".join(event_type_chip(et, n) for et, n in sorted(counts.items()))
        src_chips = "".join(source_chip(s) for s in sources) or '<span class="dim">—</span>'
        link = f'<a href="{html.escape(run.run_dir.name)}/report.html">{html.escape(run.run_dir.name)}</a>'
        rows.append(
            "<tr>"
            f"<td>{html.escape(run.created_at)}</td>"
            f"<td>{html.escape(run.label)}</td>"
            f"<td>{html.escape(str(run.pid))}</td>"
            f"<td>{len(run.events)}</td>"
            f"<td>{type_chips}</td>"
            f"<td>{src_chips}</td>"
            f"<td>{link}</td>"
            "</tr>"
        )

    aggregate_links = ""
    if aggregates:
        links = ", ".join(
            f'<a href="{html.escape(p.name)}">{html.escape(p.name)}</a>' for p in aggregates
        )
        aggregate_links = (
            f'<p class="meta" style="margin:12px 0">External aggregate index: {links}</p>'
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>runtime_logs index</title>
<style>{CSS}</style>
</head><body>
<header class="run">
  <h1>Runtime trace index</h1>
  <div class="meta">{len(runs)} runs · sorted newest first</div>
</header>
<div class="timeline">
{aggregate_links}
<table class="runs">
<thead><tr>
  <th>Created (UTC)</th><th>Label</th><th>PID</th><th>Events</th>
  <th>Event types</th><th>Sources</th><th>Run dir</th>
</tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</div>
</body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runtime_logs"),
        help="Root directory containing run subdirs (default: runtime_logs)",
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    run_dirs = discover_runs(root)
    if not run_dirs:
        print(f"warning: no runs found under {root}", file=sys.stderr)

    runs: list[Run] = []
    for run_dir in run_dirs:
        try:
            run = load_run(run_dir)
        except Exception as exc:
            print(f"error: failed to load {run_dir}: {exc}", file=sys.stderr)
            continue
        runs.append(run)
        report_path = run_dir / "report.html"
        report_path.write_text(render_run_html(run))
        print(f"wrote {report_path}")

    aggregates = sorted(p for p in root.glob("*-index*.json"))
    index_path = root / "index.html"
    index_path.write_text(render_index_html(runs, aggregates))
    print(f"wrote {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
