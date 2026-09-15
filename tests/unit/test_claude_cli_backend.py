from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import src.claude_cli_backend as claude_backend
from src.claude_cli_backend import (
    CLAUDE_REASONING_EFFORTS,
    ClaudeSubagentBackend,
    claude_cli_preflight,
)


def _executable(tmp_path: Path) -> Path:
    path = tmp_path / "claude"
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _result_json(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 1,
        "session_id": "session-123",
        "stop_reason": "end_turn",
        "total_cost_usd": 0.0125,
        "usage": {
            "input_tokens": 10,
            "cache_read_input_tokens": 4,
            "output_tokens": 2,
            # Nested/non-integer members must not reach the token record.
            "service_tier": "standard",
            "output_tokens_details": {"thinking_tokens": 1},
        },
        "modelUsage": {"claude-model-id": {"inputTokens": 10}},
        "permission_denials": [],
        "result": "completion",
    }
    payload.update(overrides)
    return json.dumps(payload)


class _FakeProcess:
    def __init__(
        self,
        command: list[str],
        *,
        capture: dict[str, Any],
        planned_stdout: str = "",
        planned_stderr: str = "",
        planned_returncode: int = 0,
        **popen_kwargs: Any,
    ) -> None:
        self.command = command
        self.capture = capture
        self.stdout = planned_stdout
        self.stderr = planned_stderr
        self.returncode = planned_returncode
        self.pid = 43210
        capture.update(command=command, popen_kwargs=popen_kwargs, process=self)

    def communicate(
        self,
        input: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        self.capture["input"] = input
        self.capture["timeout"] = timeout
        return self.stdout, self.stderr

    def terminate(self) -> None:
        self.capture["terminated"] = True

    def kill(self) -> None:
        self.capture["killed"] = True


def _install_fake_process(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> dict[str, Any]:
    capture: dict[str, Any] = {}

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        return _FakeProcess(
            command,
            capture=capture,
            planned_stdout=stdout,
            planned_stderr=stderr,
            planned_returncode=returncode,
            **kwargs,
        )

    monkeypatch.setattr(claude_backend.subprocess, "Popen", fake_popen)
    return capture


def test_claude_backend_builds_isolated_command_and_normalizes_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _install_fake_process(monkeypatch, stdout=_result_json())
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-home"))
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "must-not-reach-child-either")
    monkeypatch.setenv("SOME_TOKEN", "must-not-reach-child-either")

    backend = ClaudeSubagentBackend(
        role_name="research_synthesist",
        model="claude-test",
        reasoning_effort="xhigh",
        timeout_seconds=42,
        executable=_executable(tmp_path),
    )
    injected = "literal $(touch nope) `uname`\nmore"
    response = backend.run([{"role": "user", "content": injected}], tools=[])

    command = capture["command"]
    assert command[:2] == [str(tmp_path / "claude"), "--print"]
    assert ["--model", "claude-test"] == command[2:4]
    assert ["--effort", "xhigh"] == command[4:6]
    assert command[command.index("--output-format") + 1] == "json"
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--setting-sources") + 1] == ""
    assert "completion backend" in command[command.index("--system-prompt") + 1]
    for flag in (
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--no-chrome",
        "--safe-mode",
        "--no-session-persistence",
    ):
        assert flag in command
    assert "--dangerously-skip-permissions" not in command
    assert "--allow-dangerously-skip-permissions" not in command
    assert "--max-budget-usd" not in command
    assert injected not in command
    assert capture["popen_kwargs"]["shell"] is False
    assert capture["popen_kwargs"]["start_new_session"] is True
    assert capture["timeout"] == 42

    prompt, request_json = capture["input"].split(
        "\n\nChat-completion request JSON follows:\n", maxsplit=1
    )
    assert "completion backend" in prompt
    request = json.loads(request_json)
    assert request == {"messages": [{"role": "user", "content": injected}]}
    child_env = capture["popen_kwargs"]["env"]
    assert child_env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "claude-home")
    assert child_env["NO_COLOR"] == "1"
    assert "CLAUDE_CODE_SESSION_ID" not in child_env
    assert "SOME_TOKEN" not in child_env

    message = response["choices"][0]["message"]
    assert message["content"] == "completion"
    assert message["role"] == "assistant"
    info = response["info"]
    assert info["provider"] == "claude-cli"
    assert info["model"] == "claude-test"
    assert info["resolved_model"] == "claude-model-id"
    assert info["claude_session_id"] == "session-123"
    assert info["id"] == "session-123"
    assert info["num_turns"] == 1
    assert info["permission_denials"] == 0
    assert info["usage"] == {
        "input_tokens": 10,
        "cache_read_input_tokens": 4,
        "output_tokens": 2,
        "cost_usd": 0.0125,
    }
    assert not capture["popen_kwargs"]["cwd"].exists()


def test_claude_backend_passes_an_optional_budget_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _install_fake_process(monkeypatch, stdout=_result_json())
    backend = ClaudeSubagentBackend(
        role_name="builder",
        model="claude-test",
        executable=_executable(tmp_path),
        max_budget_usd=2.5,
    )

    backend.run([{"role": "user", "content": "x"}])

    command = capture["command"]
    assert command[command.index("--max-budget-usd") + 1] == "2.5"


def test_claude_backend_runs_a_hermetic_fake_executable(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "fake-claude"
    executable.write_text(
        f"""#!{sys.executable}
import json
import sys

args = sys.argv[1:]
if args[args.index("--tools") + 1] != "":
    raise SystemExit(7)
prompt, request_json = sys.stdin.read().split(
    "\\n\\nChat-completion request JSON follows:\\n", maxsplit=1
)
if "completion backend" not in prompt:
    raise SystemExit(8)
request = json.loads(request_json)
content = request["messages"][0]["content"]
print(json.dumps({{
    "type": "result",
    "is_error": False,
    "session_id": "fake-session",
    "result": "fake:" + content,
    "total_cost_usd": 0.5,
    "usage": {{"input_tokens": 3, "output_tokens": 1}},
    "modelUsage": {{"claude-model-id": {{}}}},
    "num_turns": 1,
    "permission_denials": [],
}}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    backend = ClaudeSubagentBackend(
        role_name="builder",
        model="claude-test",
        executable=executable,
    )

    response = backend.run([{"role": "user", "content": "hello"}])

    assert response["choices"][0]["message"]["content"] == "fake:hello"
    assert response["info"]["claude_session_id"] == "fake-session"
    assert response["info"]["usage"] == {
        "input_tokens": 3,
        "output_tokens": 1,
        "cost_usd": 0.5,
    }


def test_claude_backend_fails_closed_on_empty_error_or_nonzero_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path)
    _install_fake_process(monkeypatch, stdout=_result_json(result=""))
    backend = ClaudeSubagentBackend(
        role_name="builder", model="claude-test", executable=executable
    )
    with pytest.raises(RuntimeError, match="returned no completion"):
        backend.run([{"role": "user", "content": "x"}])

    # The CLI reports auth/API failures through ``is_error`` on a zero exit code.
    _install_fake_process(
        monkeypatch,
        stdout=_result_json(
            is_error=True,
            terminal_reason="api_error",
            result="Not logged in",
        ),
    )
    backend = ClaudeSubagentBackend(
        role_name="builder", model="claude-test", executable=executable
    )
    with pytest.raises(RuntimeError, match="reported an error") as error_info:
        backend.run([{"role": "user", "content": "x"}])
    assert "api_error" in str(error_info.value)
    assert "Not logged in" in str(error_info.value)

    proxy_secret = "https://name:secret@example.test:443"
    monkeypatch.setenv("HTTPS_PROXY", proxy_secret)
    _install_fake_process(
        monkeypatch,
        stdout="",
        stderr=f"connection through {proxy_secret} failed",
        returncode=7,
    )
    backend = ClaudeSubagentBackend(
        role_name="builder", model="claude-test", executable=executable
    )
    with pytest.raises(RuntimeError, match="exit code 7") as exc_info:
        backend.run([{"role": "user", "content": "x"}])
    assert proxy_secret not in str(exc_info.value)
    assert "[redacted]" in str(exc_info.value)


def test_claude_backend_ignores_unparsable_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_process(monkeypatch, stdout="not json at all")
    backend = ClaudeSubagentBackend(
        role_name="builder", model="claude-test", executable=_executable(tmp_path)
    )

    with pytest.raises(RuntimeError, match="returned no completion"):
        backend.run([{"role": "user", "content": "x"}])


def test_claude_backend_reads_the_last_result_of_a_streamed_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init"}),
            _result_json(result="streamed completion"),
        ]
    )
    _install_fake_process(monkeypatch, stdout=stdout)
    backend = ClaudeSubagentBackend(
        role_name="builder", model="claude-test", executable=_executable(tmp_path)
    )

    response = backend.run([{"role": "user", "content": "x"}])

    assert response["choices"][0]["message"]["content"] == "streamed completion"






def test_claude_environment_is_an_explicit_allowlist() -> None:
    source = {
        "HOME": "/home/test",
        "PATH": "/usr/bin",
        "CLAUDE_CONFIG_DIR": "/claude",
        "HTTPS_PROXY": "https://proxy.test",
        "SSL_CERT_FILE": "/cert.pem",
        "NODE_EXTRA_CA_CERTS": "/node-cert.pem",
        "CLAUDE_CODE_ENTRYPOINT": "outer-session",
        "CLAUDECODE": "1",
        "CUSTOM_API_KEY": "secret-3",
        "ACCESS_TOKEN": "secret-4",
        "UNRELATED": "discard-me",
    }

    assert claude_backend._claude_environment(source) == {
        "HOME": "/home/test",
        "PATH": "/usr/bin",
        "CLAUDE_CONFIG_DIR": "/claude",
        "HTTPS_PROXY": "https://proxy.test",
        "SSL_CERT_FILE": "/cert.pem",
        "NODE_EXTRA_CA_CERTS": "/node-cert.pem",
        "NO_COLOR": "1",
    }


def test_claude_backend_rejects_nonempty_tool_exchange(tmp_path: Path) -> None:
    backend = ClaudeSubagentBackend(
        role_name="builder",
        model="claude-test",
        executable=_executable(tmp_path),
    )

    with pytest.raises(NotImplementedError, match="tool-call exchange"):
        backend.run(
            [{"role": "user", "content": "x"}],
            tools=[{"type": "function"}],
        )


def test_claude_backend_rejects_unsupported_settings(tmp_path: Path) -> None:
    executable = _executable(tmp_path)
    with pytest.raises(ValueError, match="reasoning_effort"):
        ClaudeSubagentBackend(
            role_name="builder",
            model="claude-test",
            reasoning_effort="minimal",
            executable=executable,
        )
    with pytest.raises(ValueError, match="timeout_seconds"):
        ClaudeSubagentBackend(
            role_name="builder",
            model="claude-test",
            timeout_seconds=0,
            executable=executable,
        )
    with pytest.raises(ValueError, match="max_budget_usd"):
        ClaudeSubagentBackend(
            role_name="builder",
            model="claude-test",
            max_budget_usd=0,
            executable=executable,
        )
    assert CLAUDE_REASONING_EFFORTS == {"low", "medium", "high", "xhigh", "max"}






def test_preflight_reports_a_missing_executable_without_probing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GRAPH_HYPOTH_CLAUDE_BIN", raising=False)
    monkeypatch.setattr(claude_backend.shutil, "which", lambda _name: None)
    monkeypatch.setattr(claude_backend, "_claude_executable_candidates", lambda: ())

    report = claude_cli_preflight()

    assert report["available"] is False
    assert report["logged_in"] is False
    assert report["executable"] is None
    assert any("GRAPH_HYPOTH_CLAUDE_BIN" in message for message in report["errors"])


def test_preflight_reads_version_and_saved_login_state(tmp_path: Path) -> None:
    executable = tmp_path / "fake-claude"
    executable.write_text(
        f"""#!{sys.executable}
import sys

args = sys.argv[1:]
if args == ["--version"]:
    print("9.9.9 (Claude Code)")
elif args == ["auth", "status", "--json"]:
    print('{{"loggedIn": true, "authMethod": "oauth"}}')
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    report = claude_cli_preflight(executable)

    assert report["available"] is True
    assert report["version"] == "9.9.9 (Claude Code)"
    assert report["logged_in"] is True
    assert report["auth_method"] == "oauth"
    assert report["errors"] == []


def test_preflight_flags_a_signed_out_cli(tmp_path: Path) -> None:
    """A signed-out CLI prints its JSON and exits non-zero; the JSON wins."""
    executable = tmp_path / "fake-claude"
    executable.write_text(
        f"""#!{sys.executable}
import sys

args = sys.argv[1:]
if args == ["--version"]:
    print("9.9.9 (Claude Code)")
else:
    print('{{"loggedIn": false, "authMethod": "none"}}')
    raise SystemExit(1)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    report = claude_cli_preflight(executable)

    assert report["available"] is True
    assert report["logged_in"] is False
    assert report["auth_method"] == "none"
    assert any("not logged in" in message for message in report["errors"])
    assert not any("parseable JSON" in message for message in report["errors"])


def test_preflight_reports_an_unreadable_auth_status(tmp_path: Path) -> None:
    executable = tmp_path / "fake-claude"
    executable.write_text(
        f"""#!{sys.executable}
import sys

args = sys.argv[1:]
if args == ["--version"]:
    print("9.9.9 (Claude Code)")
else:
    print("something went very wrong", file=sys.stderr)
    raise SystemExit(3)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    report = claude_cli_preflight(executable)

    assert report["logged_in"] is False
    assert any("parseable JSON" in message for message in report["errors"])
    assert any("something went very wrong" in message for message in report["errors"])


def test_resolved_model_is_the_generating_model_not_the_first_reported() -> None:
    """``modelUsage`` can hold an auxiliary model alongside the requested one.

    Claude Code bills small internal turns to a fast model, and that entry can
    be serialised first.  Reporting whichever key came first recorded a false
    model in every attempt's provenance, so the requested model wins when it is
    present and the highest output-token entry wins otherwise.
    """

    payload = {
        "type": "result",
        "is_error": False,
        "session_id": "s-1",
        "result": "ok",
        "modelUsage": {
            # Auxiliary work, serialised first, tiny output.
            "claude-model-id-aux": {"inputTokens": 1665, "outputTokens": 17},
            # The model that actually produced the completion.
            "claude-model-id": {"inputTokens": 2, "outputTokens": 241},
        },
    }
    parsed = claude_backend._claude_metadata(json.dumps(payload), "claude-model-id")
    assert parsed.resolved_model == "claude-model-id"
    # The auxiliary model must stay visible rather than being discarded.
    assert parsed.model_usage is not None
    assert set(parsed.model_usage) == {
        "claude-model-id-aux",
        "claude-model-id",
    }


def test_resolved_model_falls_back_to_the_highest_output_producer() -> None:
    payload = {
        "type": "result",
        "is_error": False,
        "session_id": "s-2",
        "result": "ok",
        "modelUsage": {
            "claude-model-id-aux": {"inputTokens": 900, "outputTokens": 12},
            "claude-model-id-alt": {"inputTokens": 4, "outputTokens": 300},
        },
    }
    # The requested model is absent from the report, so the generating model is
    # inferred rather than silently taken from insertion order.
    parsed = claude_backend._claude_metadata(json.dumps(payload), "claude-model-id")
    assert parsed.resolved_model == "claude-model-id-alt"


def test_resolved_model_is_none_without_model_usage() -> None:
    payload = {"type": "result", "is_error": False, "session_id": "s-3", "result": "ok"}
    parsed = claude_backend._claude_metadata(json.dumps(payload), "claude-model-id")
    assert parsed.resolved_model is None
    assert parsed.model_usage is None
