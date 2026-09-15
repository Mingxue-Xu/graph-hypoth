"""Progress must stay live during blocking calls without changing their results."""

from __future__ import annotations

import io
import json
import threading
from types import SimpleNamespace

import pytest

from src import cli, synthesist_run
from src.camel_adapter import _install_logging_backend_hook
from src.progress import ProgressReporter, progress_operation, report_progress


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class _HeartbeatStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.heartbeat_seen = threading.Event()
        self.flushes = 0

    def write(self, text):
        result = super().write(text)
        if "Still waiting:" in text:
            self.heartbeat_seen.set()
        return result

    def flush(self):
        self.flushes += 1
        super().flush()


@pytest.mark.parametrize("provider", ["claude-cli", "codex-cli"])
def test_blocked_backend_emits_flushed_heartbeat_and_preserves_io(tmp_path, provider):
    stream = _HeartbeatStream()
    path = tmp_path / "progress.jsonl"
    response = object()
    messages = [{"content": "PRIVATE PROMPT"}]
    calls = []

    class Backend:
        def run(self, actual_messages):
            assert actual_messages is messages
            # The model call is blocked on this thread until the reporter speaks.
            assert stream.heartbeat_seen.wait(2), "heartbeat stopped during model call"
            return response

    backend = _install_logging_backend_hook(
        Backend(),
        lambda *args: calls.append(args),
        role="evidence_reviewer",
        provider=provider,
    )
    with ProgressReporter(
        "r1", stream=stream, jsonl_path=path, heartbeat_seconds=0.02
    ) as reporter:
        report_progress("Checking evidence", "reviewing link", current=4, total=12)
        assert backend.run(messages) is response
        # JSONL is flushed while the run is still live, so tailers can read it.
        rows = _events(path)
        heartbeat = next(row for row in rows if row["event"] == "heartbeat")
        assert heartbeat["provider"] == provider
        assert heartbeat["stage"] == "Checking evidence"
        assert (heartbeat["current"], heartbeat["total"]) == (4, 12)
        assert heartbeat["operation_elapsed_seconds"] >= 0.02
        assert heartbeat["run_id"] == "r1"
    assert reporter._thread is not None and not reporter._thread.is_alive()
    assert stream.flushes > 0
    assert calls == [("evidence_reviewer", messages, response)]
    assert "PRIVATE PROMPT" not in stream.getvalue() + path.read_text()
    assert _events(path)[-1]["status"] == "completed"


def test_interactive_confirmation_pauses_heartbeat(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("src.progress.time.monotonic", lambda: clock[0])
    stream = io.StringIO()
    path = tmp_path / "progress.jsonl"
    with ProgressReporter("r", stream=stream, jsonl_path=path) as reporter:

        def confirm_input(_prompt):
            clock[0] += 31
            reporter.heartbeat()
            assert "Still waiting:" not in stream.getvalue()
            assert _events(path)[-1]["status"] == "waiting"
            return "none"

        confirm = synthesist_run.interactive_confirm_fn(
            input_fn=confirm_input,
            output_fn=lambda _: None,
        )
        monkeypatch.setattr(synthesist_run, "_format_surfaced", lambda _: "ranking")
        assert confirm([SimpleNamespace()]) == []
        clock[0] += 31
        reporter.heartbeat()
        assert "Still waiting:" in stream.getvalue()


@pytest.mark.parametrize("error", [RuntimeError("broken"), KeyboardInterrupt()])
def test_failure_and_cancellation_stop_reporter_and_propagate(tmp_path, error):
    path = tmp_path / "progress.jsonl"
    with (
        pytest.raises(type(error)),
        ProgressReporter("r", quiet=True, jsonl_path=path) as reporter,
    ):
        report_progress("Mapping claim")
        with progress_operation("builder", provider="codex-cli"):
            raise error
    assert reporter._thread is not None and not reporter._thread.is_alive()
    expected = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    rows = _events(path)
    assert rows[-1]["event"] == "run_finished"
    assert rows[-1]["status"] == expected
    assert rows[-2]["status"] == expected
    # A later library call cannot accidentally reuse the closed reporter.
    report_progress("should not appear")
    assert _events(path) == rows


def test_nested_operations_restore_parent_and_keep_run_logs_separate(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    with ProgressReporter("a", quiet=True, jsonl_path=a):
        report_progress("Designing experiments", "hypothesis", current=1, total=2)
        with progress_operation("retrieving methods"):
            with progress_operation("source attempt") as outcome:
                outcome.status = "skipped"
            with ProgressReporter("b", quiet=True, jsonl_path=b):
                report_progress("Mapping claim")
        with progress_operation("drafting plan"):
            pass
    assert {row["run_id"] for row in _events(a)} == {"a"}
    assert {row["run_id"] for row in _events(b)} == {"b"}
    draft = next(row for row in _events(a) if "drafting plan" in row["detail"])
    assert draft["detail"] == "hypothesis · drafting plan"
    assert (draft["current"], draft["total"]) == (1, 2)


def test_unavailable_output_does_not_break_the_run(tmp_path):
    class BrokenStream(io.StringIO):
        def write(self, _text):
            raise BrokenPipeError("closed pipe")

    path = tmp_path / "progress.jsonl"
    with ProgressReporter("r", stream=BrokenStream(), jsonl_path=path):
        report_progress("Mapping claim")
    assert _events(path)[-1]["status"] == "completed"


@pytest.mark.parametrize("entrypoint", ["orchestrate", "synthesist"])
@pytest.mark.parametrize("quiet", [False, True])
def test_cli_progress_goes_to_stderr_and_quiet_keeps_jsonl(
    tmp_path,
    monkeypatch,
    capsys,
    entrypoint,
    quiet,
):
    path = tmp_path / "progress.jsonl"
    result = SimpleNamespace(edge_table=[], version=1, surfaced=[])

    def fake_run(*_args, **_kwargs):
        report_progress("Mapping claim", "extracting concepts")
        return result

    args = ["--run-id", "test-run", "--progress-jsonl", str(path)]
    if quiet:
        args.append("--quiet")
    if entrypoint == "synthesist":
        monkeypatch.setattr(synthesist_run, "run_synthesist", fake_run)
        synthesist_run.main(["--profile", "unused.yaml", *args])
    else:
        monkeypatch.setattr(cli, "run_graph_state_orchestration", fake_run)
        monkeypatch.setattr(cli, "_print_graph_state_result", lambda _: print("done"))
        cli.main(["--claim", "claim", *args])
    output = capsys.readouterr()
    assert "done" in output.out
    assert "Mapping claim" not in output.out
    assert ("Mapping claim" in output.err) is not quiet
    if quiet:
        assert output.err == ""
    rows = _events(path)
    assert [row["event"] for row in rows] == ["run_started", "progress", "run_finished"]
    assert all(row["run_id"] == "test-run" for row in rows)


def test_progress_file_appends_runs_and_invalid_destination_fails_early(tmp_path):
    path = tmp_path / "progress.jsonl"
    for run_id in ("a", "b"):
        with ProgressReporter(run_id, quiet=True, jsonl_path=path):
            pass
    assert [row["run_id"] for row in _events(path)] == ["a", "a", "b", "b"]
    with (
        pytest.raises(OSError),
        ProgressReporter("bad", quiet=True, jsonl_path=tmp_path),
    ):
        pytest.fail("run must not start with an invalid progress destination")


def test_example_runner_forwards_progress_options(tmp_path, monkeypatch):
    from scripts.run_example import main

    config = tmp_path / "config.yaml"
    config.touch()
    calls = []
    monkeypatch.setattr(synthesist_run, "main", lambda args: calls.append(args) or 0)
    path = tmp_path / "progress.jsonl"
    main(["--config", str(config), "--quiet", "--progress-jsonl", str(path)])
    assert "--quiet" in calls[0]
    assert calls[0][-2:] == ["--progress-jsonl", str(path)]


def test_library_calls_without_reporter_stay_silent(capsys):
    report_progress("Mapping claim")
    with progress_operation("builder"):
        pass
    output = capsys.readouterr()
    assert output.out == output.err == ""
