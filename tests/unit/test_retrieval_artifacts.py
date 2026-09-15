from __future__ import annotations

import json

from src import runtime_trace
from src.retrieval.artifacts import write_retrieval_artifacts


def test_retrieval_artifacts_write_redacted_json_and_html(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(runtime_trace, "_TRACE_FILE", None)
    monkeypatch.setenv("GRAPH_HYPOTH_RUNTIME_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("GRAPH_HYPOTH_RUNTIME_RUN_LABEL", "retrieval-artifacts")
    monkeypatch.setenv("EXA_API_KEY", "exa-secret-value")

    result = write_retrieval_artifacts(
        run_id="run-artifacts",
        tool_call_id="tool_000001",
        source="exa",
        payload={
            "evidence": [
                {
                    "evidence_id": "ev_000001",
                    "title": "Known Paper",
                    "url": "https://arxiv.org/html/2309.10668v2",
                    "quote": "Verified quote.",
                    "metadata": {
                        "exa_anchor_query": "anchor claim",
                        "quote_selection": {
                            "verification_status": "accepted",
                            "match_type": "exact",
                            "verified_quote": "Verified quote.",
                            "candidate_excerpt": "exa-secret-value",
                        }
                    },
                }
            ]
        },
        write_json=True,
        write_html=True,
    )

    assert result is not None
    assert result.json_path is not None
    assert result.html_path is not None
    assert result.json_path.exists()
    assert result.html_path.exists()
    serialized = (
        result.json_path.read_text(encoding="utf-8")
        + result.html_path.read_text(encoding="utf-8")
    )
    assert "exa-secret-value" not in serialized
    assert "[REDACTED_EXA_API_KEY]" in serialized
    assert "ev_000001" in serialized
    assert "anchor claim" in serialized
    assert "Verified quote." in serialized
    assert "accepted" in serialized

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["tool_call_id"] == "tool_000001"
    assert payload["source"] == "exa"


def test_retrieval_artifacts_are_noop_without_runtime_log_dir(monkeypatch) -> None:
    monkeypatch.setattr(runtime_trace, "_TRACE_FILE", None)
    monkeypatch.delenv("GRAPH_HYPOTH_RUNTIME_LOG_DIR", raising=False)

    assert (
        write_retrieval_artifacts(
            run_id="run-artifacts",
            tool_call_id="tool_000001",
            source="exa",
            payload={},
            write_json=True,
            write_html=True,
        )
        is None
    )
