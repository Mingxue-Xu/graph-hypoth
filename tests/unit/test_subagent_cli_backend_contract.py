"""The process-transport contract shared by every subagent CLI backend.

The Codex and Claude backends each drive a local CLI as a child process. Their
*command construction* and *response parsing* are backend-specific and stay in
``test_codex_cli_backend.py`` / ``test_claude_cli_backend.py``. What is identical
between them -- and what this file owns, once, parametrised over both -- is the
transport hardening: the timeout escalation, the interrupt path, temp-directory
isolation under concurrency, and executable resolution.

Each invariant still runs once per backend; the parametrisation is what stops the
two copies drifting apart the way they already had.
"""

from __future__ import annotations

import json
import signal
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import src.claude_cli_backend as claude_backend
import src.codex_cli_backend as codex_backend
from src.claude_cli_backend import ClaudeSubagentBackend, resolve_claude_executable
from src.codex_cli_backend import CodexSubagentBackend, resolve_codex_executable


@dataclass(frozen=True)
class _BackendCase:
    """Everything that differs between two backends running the same transport."""

    name: str
    module: Any                       # module whose ``subprocess.Popen`` is patched
    signal_module: Any                # module whose ``os.getpgid``/``killpg`` the transport calls
    backend_class: type
    model: str
    executable_env_var: str
    executable_candidates_name: str
    resolve_executable: Callable[[], str]
    # (command, popen_kwargs, reply_text) -> (stdout, stderr); may write a completion file
    reply: Callable[[list[str], dict[str, Any], str], tuple[str, str]]


def _codex_reply(
    command: list[str], _kwargs: dict[str, Any], text: str
) -> tuple[str, str]:
    # Codex returns its completion through --output-last-message, not stdout.
    Path(command[command.index("--output-last-message") + 1]).write_text(
        text, encoding="utf-8"
    )
    return "", ""


def _claude_reply(
    _command: list[str], _kwargs: dict[str, Any], text: str
) -> tuple[str, str]:
    # Claude returns a single JSON result record on stdout.
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "num_turns": 1,
            "session_id": "session-123",
            "stop_reason": "end_turn",
            "total_cost_usd": 0.0125,
            "usage": {"input_tokens": 10, "output_tokens": 2},
            "permission_denials": [],
            "result": text,
        }
    ), ""


CASES = [
    _BackendCase(
        name="codex",
        module=codex_backend,
        signal_module=codex_backend,
        backend_class=CodexSubagentBackend,
        model="gpt-test",
        executable_env_var="GRAPH_HYPOTH_CODEX_BIN",
        executable_candidates_name="_codex_executable_candidates",
        resolve_executable=resolve_codex_executable,
        reply=_codex_reply,
    ),
    _BackendCase(
        name="claude",
        module=claude_backend,
        # ``_stop_process_group`` is shared with the Codex transport, so the signal
        # calls it makes live in that module's namespace for both backends.
        signal_module=codex_backend,
        backend_class=ClaudeSubagentBackend,
        model="claude-test",
        executable_env_var="GRAPH_HYPOTH_CLAUDE_BIN",
        executable_candidates_name="_claude_executable_candidates",
        resolve_executable=resolve_claude_executable,
        reply=_claude_reply,
    ),
]

backends = pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])


def _executable(tmp_path: Path, case: _BackendCase) -> Path:
    path = tmp_path / case.name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _backend(tmp_path: Path, case: _BackendCase, **overrides: Any) -> Any:
    return case.backend_class(
        role_name="critic_panel",
        model=case.model,
        executable=_executable(tmp_path, case),
        **overrides,
    )


@backends
def test_backend_timeout_terminates_and_reaps_process_group(
    case: _BackendCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {"communicate_calls": 0, "signals": []}

    class TimeoutProcess:
        pid = 8123
        returncode = None

        def communicate(self, **kwargs: Any) -> tuple[str, str]:
            capture["communicate_calls"] += 1
            if capture["communicate_calls"] <= 2:
                raise subprocess.TimeoutExpired(case.name, kwargs.get("timeout", 0))
            return "", ""

        def terminate(self) -> None:
            capture["terminated"] = True

        def kill(self) -> None:
            capture["killed"] = True

    def fake_popen(command: list[str], **kwargs: Any) -> TimeoutProcess:
        capture["command"] = command
        capture["popen_kwargs"] = kwargs
        return TimeoutProcess()

    monkeypatch.setattr(case.module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(case.signal_module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        case.signal_module.os,
        "killpg",
        lambda pid, sig: capture["signals"].append((pid, sig)),
    )
    backend = _backend(tmp_path, case, timeout_seconds=0.01)

    with pytest.raises(RuntimeError, match="timed out"):
        backend.run([{"role": "user", "content": "x"}])

    # SIGTERM first, SIGKILL only after the process failed to exit -- and the child
    # is reaped, so a timed-out run leaves no orphan and no temp directory behind.
    assert capture["signals"] == [(8123, signal.SIGTERM), (8123, signal.SIGKILL)]
    assert capture["communicate_calls"] == 3
    assert capture["popen_kwargs"]["start_new_session"] is True
    assert not Path(capture["popen_kwargs"]["cwd"]).exists()


@backends
def test_backend_interrupt_stops_process_group(
    case: _BackendCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptProcess:
        pid = 8125
        returncode = None

        def communicate(self, **kwargs: Any) -> tuple[str, str]:
            del kwargs
            raise KeyboardInterrupt

    process = InterruptProcess()
    monkeypatch.setattr(
        case.module.subprocess, "Popen", lambda command, **kwargs: process
    )
    stopped: list[Any] = []
    monkeypatch.setattr(case.module, "_stop_process_group", stopped.append)
    backend = _backend(tmp_path, case)

    # Ctrl-C must reach the child's process group, not just unwind the parent.
    with pytest.raises(KeyboardInterrupt):
        backend.run([{"role": "user", "content": "x"}])

    assert stopped == [process]


@backends
def test_concurrent_calls_use_distinct_temporary_directories(
    case: _BackendCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directories: list[Path] = []

    class ConcurrentProcess:
        returncode = 0
        pid = 99

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            self.command = command
            self.kwargs = kwargs

        def communicate(
            self, input: str | None = None, timeout: float | None = None
        ) -> tuple[str, str]:
            del timeout
            _prompt, request_json = (input or "").split(
                "\n\nChat-completion request JSON follows:\n", maxsplit=1
            )
            content = json.loads(request_json)["messages"][0]["content"]
            directories.append(Path(self.kwargs["cwd"]))
            return case.reply(self.command, self.kwargs, f"reply:{content}")

    monkeypatch.setattr(
        case.module.subprocess,
        "Popen",
        lambda command, **kwargs: ConcurrentProcess(command, **kwargs),
    )
    backend = _backend(tmp_path, case)

    def call(value: str) -> str:
        response = backend.run([{"role": "user", "content": value}])
        return response["choices"][0]["message"]["content"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(call, ("one", "two")))

    # Two in-flight calls must not share a scratch directory, or one would read
    # the other's completion file; both are removed when the calls finish.
    assert results == ["reply:one", "reply:two"]
    assert len(set(directories)) == 2
    assert all(not directory.exists() for directory in directories)


@backends
def test_resolve_executable_honors_env_override_and_fails_cleanly(
    case: _BackendCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, case)
    monkeypatch.setenv(case.executable_env_var, str(executable))
    assert case.resolve_executable() == str(executable)

    monkeypatch.delenv(case.executable_env_var)
    monkeypatch.setattr(case.module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(case.module, case.executable_candidates_name, lambda: ())
    with pytest.raises(RuntimeError, match=case.executable_env_var):
        case.resolve_executable()


@backends
def test_resolve_executable_detects_a_known_install_outside_path(
    case: _BackendCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path, case)
    monkeypatch.delenv(case.executable_env_var, raising=False)
    monkeypatch.setattr(case.module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        case.module,
        case.executable_candidates_name,
        lambda: (tmp_path / "missing", executable),
    )

    assert case.resolve_executable() == str(executable.resolve())
