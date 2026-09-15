from __future__ import annotations

import json

from src import runtime_trace


def test_runtime_trace_writes_redacted_jsonl_event(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(runtime_trace, "_TRACE_FILE", None)
    monkeypatch.setenv("GRAPH_HYPOTH_RUNTIME_RUN_LABEL", "unit-trace")
    monkeypatch.setenv("EXA_API_KEY", "exa-secret-value")
    home = tmp_path / "home" / "sensitive-user"
    runtime_root = home / "runtime_logs"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GRAPH_HYPOTH_RUNTIME_LOG_DIR", str(runtime_root))

    path = runtime_trace.record_runtime_event(
        "agent_with_api",
        run_id="run-trace",
        actor="builder",
        target="model_api",
        direction="request",
        payload={
            "messages": [{"role": "user", "content": "hello sk-test-secret-value"}],
            "headers": {"Authorization": "Bearer exa-secret-value"},
            "cache_path": str(home / "api-keys" / "provider-api-key.txt"),
            "artifact": home / "runtime_logs" / "payload-path.json",
        },
        artifact_path=home / "runtime_logs" / "artifact.json",
    )

    assert path is not None
    assert path.parent.parent == runtime_root
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["event_type"] == "agent_with_api"
    assert row["run_id"] == "run-trace"
    assert row["actor"] == "builder"
    assert row["target"] == "model_api"
    assert row["direction"] == "request"
    serialized = json.dumps(row)
    # "sk-..." is redacted by the shape heuristic, not by exact-value env scrubbing.
    assert "sk-test-secret-value" not in serialized
    assert "exa-secret-value" not in serialized
    assert "[REDACTED_EXA_API_KEY]" in serialized
    assert str(home) not in serialized
    assert "$HOME/api-keys/provider-api-key.txt" in serialized
    assert "$HOME/runtime_logs/artifact.json" in serialized
    assert "$HOME/runtime_logs/payload-path.json" in serialized

    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    assert str(home) not in json.dumps(manifest)
    assert "$HOME/runtime_logs" in manifest["trace_file"]


def test_runtime_trace_is_noop_without_log_dir(monkeypatch) -> None:
    monkeypatch.setattr(runtime_trace, "_TRACE_FILE", None)
    monkeypatch.delenv("GRAPH_HYPOTH_RUNTIME_LOG_DIR", raising=False)

    assert runtime_trace.record_runtime_event("artifact", payload={"ok": True}) is None


def test_runtime_trace_can_reset_between_in_process_runs(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GRAPH_HYPOTH_RUNTIME_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("GRAPH_HYPOTH_RUNTIME_RUN_LABEL", "first")
    first = runtime_trace.record_runtime_event("artifact", payload={"run": 1})
    assert first is not None

    runtime_trace.reset_runtime_trace()
    monkeypatch.setenv("GRAPH_HYPOTH_RUNTIME_RUN_LABEL", "second")
    second = runtime_trace.record_runtime_event("artifact", payload={"run": 2})

    assert second is not None
    assert second.parent != first.parent


def test_current_runtime_artifact_dir_does_not_create_trace(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runtime_trace, "_TRACE_FILE", None)
    monkeypatch.setenv("GRAPH_HYPOTH_RUNTIME_LOG_DIR", str(tmp_path))

    assert runtime_trace.current_runtime_artifact_dir() is None
    assert list(tmp_path.glob("*")) == []


def test_runtime_artifact_dir_returns_current_trace_directory(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(runtime_trace, "_TRACE_FILE", None)
    monkeypatch.setenv("GRAPH_HYPOTH_RUNTIME_RUN_LABEL", "artifact-dir")
    monkeypatch.setenv("GRAPH_HYPOTH_RUNTIME_LOG_DIR", str(tmp_path))

    artifact_dir = runtime_trace.runtime_artifact_dir()

    assert artifact_dir is not None
    assert artifact_dir.parent == tmp_path
    assert (artifact_dir / "manifest.json").exists()
