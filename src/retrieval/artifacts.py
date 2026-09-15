from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.runtime_trace import (
    record_runtime_event,
    redact_runtime_payload,
    runtime_artifact_dir,
)


@dataclass(frozen=True)
class RetrievalArtifactPaths:
    json_path: Path | None
    html_path: Path | None

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, str | None]:
        del args, kwargs
        payload = {
            "json_path": str(self.json_path) if self.json_path is not None else None,
            "html_path": str(self.html_path) if self.html_path is not None else None,
        }
        return redact_runtime_payload(payload)


def write_retrieval_artifacts(
    *,
    run_id: str,
    tool_call_id: str,
    source: str,
    payload: dict[str, Any],
    write_json: bool,
    write_html: bool,
) -> RetrievalArtifactPaths | None:
    if not write_json and not write_html:
        return None
    artifact_root = runtime_artifact_dir()
    if artifact_root is None:
        return None

    artifact_dir = artifact_root / "retrieval"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe_name(tool_call_id)}-{_safe_name(source)}"
    redacted = redact_runtime_payload(
        {
            "run_id": run_id,
            "tool_call_id": tool_call_id,
            "source": source,
            "payload": payload,
        }
    )

    json_path = artifact_dir / f"{stem}.json" if write_json else None
    html_path = artifact_dir / f"{stem}.html" if write_html else None
    if json_path is not None:
        json_path.write_text(
            json.dumps(redacted, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        record_runtime_event(
            "artifact",
            run_id=run_id,
            target="retrieval_json",
            direction="write",
            artifact_path=json_path,
            payload={"tool_call_id": tool_call_id, "source": source},
        )
    if html_path is not None:
        html_path.write_text(_render_html(redacted), encoding="utf-8")
        record_runtime_event(
            "artifact",
            run_id=run_id,
            target="retrieval_html",
            direction="write",
            artifact_path=html_path,
            payload={"tool_call_id": tool_call_id, "source": source},
        )
    return RetrievalArtifactPaths(json_path=json_path, html_path=html_path)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-") or "retrieval"


def _render_html(payload: dict[str, Any]) -> str:
    body = payload.get("payload", {})
    evidence = body.get("evidence", []) if isinstance(body, dict) else []
    rows = []
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata", {})
            selection = (
                metadata.get("quote_selection", {})
                if isinstance(metadata, dict)
                else {}
            )
            anchor = metadata.get("exa_anchor_query") if isinstance(metadata, dict) else None
            rows.append(
                "<tr>"
                f"<td>{_cell(item.get('evidence_id'))}</td>"
                f"<td>{_cell(item.get('title'))}</td>"
                f"<td><a href=\"{_attr(item.get('url'))}\">{_cell(item.get('url'))}</a></td>"
                f"<td>{_cell(anchor or selection.get('query'))}</td>"
                f"<td>{_cell(selection.get('candidate_excerpt'))}</td>"
                f"<td>{_cell(selection.get('verified_quote') or item.get('quote'))}</td>"
                f"<td>{_cell(selection.get('verification_status'))}</td>"
                f"<td>{_cell(selection.get('match_type'))}</td>"
                "</tr>"
            )
    return (
        "<!doctype html>\n"
        "<html><head><meta charset=\"utf-8\">"
        "<title>GraphHypoth Retrieval Evidence</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "margin:24px;color:#17202a;background:#f8fafc;}"
        "table{border-collapse:collapse;width:100%;background:white;}"
        "th,td{border:1px solid #d8dee9;padding:8px;text-align:left;"
        "vertical-align:top;font-size:13px;}"
        "th{background:#eef2f7;}"
        "td:nth-child(5),td:nth-child(6){max-width:420px;}"
        "</style></head><body>"
        "<h1>GraphHypoth Retrieval Evidence</h1>"
        f"<p>Run: {_cell(payload.get('run_id'))} | "
        f"Tool call: {_cell(payload.get('tool_call_id'))} | "
        f"Source: {_cell(payload.get('source'))}</p>"
        "<table><thead><tr><th>ID</th><th>Title</th><th>URL</th>"
        "<th>Anchor</th><th>Candidate Excerpt</th><th>Verified Quote</th>"
        "<th>Verification</th><th>Match</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</body></html>\n"
    )


def _cell(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _attr(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)
