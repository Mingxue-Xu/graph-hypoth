from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import src.codex_cli_backend as codex_backend
from src.camel_adapter import (
    _create_camel_model_backend,
    _create_direct_model_backend,
)
from src.codex_cli_backend import CodexSubagentBackend
from src.config import AgentConfig, ModelConfig, load_config


def _executable(tmp_path: Path) -> Path:
    path = tmp_path / "codex"
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


class _FakeProcess:
    def __init__(
        self,
        command: list[str],
        *,
        capture: dict[str, Any],
        completion: str | None = "completion",
        planned_stdout: str = "",
        planned_stderr: str = "",
        planned_returncode: int = 0,
        **popen_kwargs: Any,
    ) -> None:
        self.command = command
        self.capture = capture
        self.completion = completion
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
        output_path = Path(
            self.command[self.command.index("--output-last-message") + 1]
        )
        self.capture["output_path"] = output_path
        if self.completion is not None:
            output_path.write_text(self.completion, encoding="utf-8")
        return self.stdout, self.stderr

    def terminate(self) -> None:
        self.capture["terminated"] = True

    def kill(self) -> None:
        self.capture["killed"] = True


def _install_fake_process(
    monkeypatch: pytest.MonkeyPatch,
    *,
    completion: str | None = "completion",
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> dict[str, Any]:
    capture: dict[str, Any] = {}

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        return _FakeProcess(
            command,
            capture=capture,
            completion=completion,
            planned_stdout=stdout,
            planned_stderr=stderr,
            planned_returncode=returncode,
            **kwargs,
        )

    monkeypatch.setattr(codex_backend.subprocess, "Popen", fake_popen)
    return capture


def test_codex_backend_builds_isolated_command_and_normalizes_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-123"}),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 10,
                        "cached_input_tokens": 4,
                        "output_tokens": 2,
                    },
                }
            ),
        ]
    )
    capture = _install_fake_process(
        monkeypatch,
        completion='{"answer":"from output file"}',
        stdout=stdout,
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("SOME_TOKEN", "must-not-reach-child-either")

    backend = CodexSubagentBackend(
        role_name="research_synthesist",
        model="gpt-test",
        reasoning_effort="xhigh",
        timeout_seconds=42,
        executable=_executable(tmp_path),
    )
    injected = "literal $(touch nope) `uname`\nmore"
    response = backend.run([{"role": "user", "content": injected}], tools=[])

    command = capture["command"]
    assert command[:3] == [str(tmp_path / "codex"), "exec", "--strict-config"]
    assert ["--model", "gpt-test"] == command[3:5]
    assert 'model_reasoning_effort="xhigh"' in command
    assert 'shell_environment_policy.inherit="none"' in command
    assert 'approval_policy="never"' in command
    assert "agents.enabled=false" in command
    assert "features.shell_tool=false" in command
    assert "tools.web_search=false" in command
    assert "features.view_image=false" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    for flag in (
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--ephemeral",
        "--json",
    ):
        assert flag in command
    assert injected not in command
    assert capture["popen_kwargs"]["shell"] is False
    assert capture["popen_kwargs"]["start_new_session"] is True
    assert capture["timeout"] == 42

    assert command[-1] == "-"
    prompt, request_json = capture["input"].split(
        "\n\nChat-completion request JSON follows:\n", maxsplit=1
    )
    assert "completion backend" in prompt
    request = json.loads(request_json)
    assert request == {"messages": [{"role": "user", "content": injected}]}
    child_env = capture["popen_kwargs"]["env"]
    assert child_env["CODEX_HOME"] == str(tmp_path / "codex-home")
    assert "OPENROUTER_API_KEY" not in child_env
    assert "SOME_TOKEN" not in child_env

    message = response["choices"][0]["message"]
    assert message["content"] == '{"answer":"from output file"}'
    assert message["role"] == "assistant"
    assert response["info"]["provider"] == "codex-cli"
    assert response["info"]["codex_thread_id"] == "thread-123"
    assert response["info"]["usage"] == {
        "input_tokens": 10,
        "cached_input_tokens": 4,
        "output_tokens": 2,
    }
    assert not capture["output_path"].exists()
    assert not capture["popen_kwargs"]["cwd"].exists()


def test_codex_backend_falls_back_to_jsonl_agent_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "plain prose is valid"},
        }
    )
    _install_fake_process(monkeypatch, completion=None, stdout=stdout)
    backend = CodexSubagentBackend(
        role_name="elaboration_writer",
        model="gpt-test",
        executable=_executable(tmp_path),
    )

    response = backend.run([{"role": "user", "content": "write prose"}])

    assert response["choices"][0]["message"]["content"] == "plain prose is valid"


def test_codex_backend_runs_a_hermetic_fake_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "fake-codex"
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import sys

if os.environ.get("OPENROUTER_API_KEY"):
    raise SystemExit(9)
args = sys.argv[1:]
output = pathlib.Path(args[args.index("--output-last-message") + 1])
prompt, request_json = sys.stdin.read().split(
    "\\n\\nChat-completion request JSON follows:\\n", maxsplit=1
)
if "completion backend" not in prompt:
    raise SystemExit(8)
request = json.loads(request_json)
content = request["messages"][0]["content"]
output.write_text("fake:" + content, encoding="utf-8")
print(json.dumps({{"type": "thread.started", "thread_id": "fake-thread"}}))
print(json.dumps({{
    "type": "turn.completed",
    "usage": {{"input_tokens": 3, "output_tokens": 1}},
}}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-for-the-child")
    backend = CodexSubagentBackend(
        role_name="builder",
        model="gpt-test",
        executable=executable,
    )

    response = backend.run([{"role": "user", "content": "hello"}])

    assert response["choices"][0]["message"]["content"] == "fake:hello"
    assert response["info"]["codex_thread_id"] == "fake-thread"
    assert response["info"]["usage"] == {
        "input_tokens": 3,
        "output_tokens": 1,
    }


def test_codex_backend_fails_closed_on_empty_or_nonzero_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _executable(tmp_path)
    _install_fake_process(monkeypatch, completion=None)
    backend = CodexSubagentBackend(
        role_name="builder", model="gpt-test", executable=executable
    )
    with pytest.raises(RuntimeError, match="returned no completion"):
        backend.run([{"role": "user", "content": "x"}])

    proxy_secret = "https://name:secret@example.test:443"
    monkeypatch.setenv("HTTPS_PROXY", proxy_secret)
    _install_fake_process(
        monkeypatch,
        completion=None,
        stderr=f"connection through {proxy_secret} failed",
        returncode=7,
    )
    backend = CodexSubagentBackend(
        role_name="builder", model="gpt-test", executable=executable
    )
    with pytest.raises(RuntimeError, match="exit code 7") as exc_info:
        backend.run([{"role": "user", "content": "x"}])
    assert proxy_secret not in str(exc_info.value)
    assert "[redacted]" in str(exc_info.value)




def test_codex_backend_timeout_uses_process_fallbacks_after_io_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {"communicate_calls": 0}

    class TimeoutProcess:
        pid = 8124
        returncode = None

        def communicate(self, **kwargs: Any) -> tuple[str, str]:
            capture["communicate_calls"] += 1
            if capture["communicate_calls"] == 1:
                raise subprocess.TimeoutExpired("codex", kwargs.get("timeout", 0))
            if capture["communicate_calls"] == 2:
                raise OSError("pipe already closed")
            return "", ""

        def terminate(self) -> None:
            capture["terminated"] = True

        def kill(self) -> None:
            capture["killed"] = True

    monkeypatch.setattr(
        codex_backend.subprocess,
        "Popen",
        lambda command, **kwargs: TimeoutProcess(),
    )
    monkeypatch.setattr(codex_backend.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        codex_backend.os,
        "killpg",
        lambda pid, sig: (_ for _ in ()).throw(PermissionError("denied")),
    )
    backend = CodexSubagentBackend(
        role_name="critic_panel",
        model="gpt-test",
        timeout_seconds=0.01,
        executable=_executable(tmp_path),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        backend.run([{"role": "user", "content": "x"}])

    assert capture == {
        "communicate_calls": 3,
        "terminated": True,
        "killed": True,
    }




def test_codex_environment_is_an_explicit_allowlist() -> None:
    source = {
        "HOME": "/home/test",
        "PATH": "/usr/bin",
        "CODEX_HOME": "/codex",
        "HTTPS_PROXY": "https://proxy.test",
        "SSL_CERT_FILE": "/cert.pem",
        "CUSTOM_API_KEY": "secret-2",
        "ACCESS_TOKEN": "secret-3",
        "UNRELATED": "discard-me",
    }

    assert codex_backend._codex_environment(source) == {
        "HOME": "/home/test",
        "PATH": "/usr/bin",
        "CODEX_HOME": "/codex",
        "HTTPS_PROXY": "https://proxy.test",
        "SSL_CERT_FILE": "/cert.pem",
        "NO_COLOR": "1",
    }


def test_codex_backend_rejects_nonempty_tool_exchange(
    tmp_path: Path,
) -> None:
    backend = CodexSubagentBackend(
        role_name="builder",
        model="gpt-test",
        executable=_executable(tmp_path),
    )

    with pytest.raises(NotImplementedError, match="tool-call exchange"):
        backend.run(
            [{"role": "user", "content": "x"}],
            tools=[{"type": "function"}],
        )






def _agent_model(**model_overrides: Any) -> AgentConfig:
    model_values: dict[str, Any] = {
        "provider": "codex-cli",
        "model_id": "gpt-role",
        "api_key_env": None,
        "base_url": None,
        "timeout_seconds": 321,
        "reasoning_effort": "xhigh",
    }
    model_values.update(model_overrides)
    return AgentConfig(
        temperature=0.0,
        model=ModelConfig(**model_values),
    )


def test_direct_model_resolver_dispatches_codex_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeCodexBackend:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(codex_backend, "CodexSubagentBackend", FakeCodexBackend)

    backend = _create_direct_model_backend(
        _agent_model(), role_name="research_synthesist"
    )

    assert isinstance(backend, FakeCodexBackend)
    assert captured == {
        "role_name": "research_synthesist",
        "model": "gpt-role",
        "reasoning_effort": "xhigh",
        "timeout_seconds": 321.0,
    }


def test_direct_model_resolver_leaves_camel_providers_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    import src.camel_adapter as camel_adapter

    monkeypatch.setattr(
        camel_adapter,
        "_create_camel_model_backend",
        lambda agent_config, *, role_name: sentinel,
    )
    config = AgentConfig(
        temperature=0.0,
        model=ModelConfig(provider="openrouter", model_id="x/test-model"),
    )

    assert _create_direct_model_backend(config, role_name="builder") is sentinel


def test_codex_yaml_composes_all_direct_graph_role_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.graph_state_runtime as graph_runtime
    from src.graph_state_runtime import SynthesistBuildSpec, build_graph_state_deps
    from src.log_store import SQLiteLogStore

    config_data = load_config("config/evidence-evaluation.yaml").model_dump(
        mode="json"
    )
    for role, model_id in (
        ("builder", "codex-builder"),
        ("skeptical_verifier", "codex-verifier"),
    ):
        config_data["agents"][role]["model"] = {
            "provider": "codex-cli",
            "model_id": model_id,
            "api_key_env": None,
            "base_url": None,
            "reasoning_effort": "high",
            "timeout_seconds": 30,
        }
    config_path = tmp_path / "codex.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    config = load_config(config_path)

    calls: list[tuple[str, str]] = []

    class FakeCodexBackend:
        def __init__(self, *, role_name: str, model: str, **kwargs: Any) -> None:
            del kwargs
            calls.append((role_name, model))

    monkeypatch.setattr(codex_backend, "CodexSubagentBackend", FakeCodexBackend)
    monkeypatch.setattr(graph_runtime, "default_merge_embedder", lambda: None)
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()

    deps = build_graph_state_deps(
        config,
        claim="claim",
        log_store=store,
        run_id="run",
        thread_id="thread",
        synthesist=SynthesistBuildSpec(),
        evidence=[],
    )
    assert deps.research_synthesist is not None
    assert deps.experiment_designer is not None
    deps.research_synthesist()
    deps.experiment_designer()

    assert calls == [
        ("builder", "codex-builder"),
        ("evidence_reviewer", "codex-verifier"),
        ("builder", "codex-builder"),
        ("critic_panel", "codex-verifier"),
        ("skeptical_verifier", "codex-verifier"),
        ("experiment_validator", "codex-verifier"),
        ("research_synthesist", "codex-builder"),
        ("experiment_designer", "codex-builder"),
    ]


@pytest.mark.parametrize(
    ("agent_config", "message"),
    [
        (_agent_model(api_key_env="TEST_MODEL_API_KEY"), "api_key_env to null"),
        (_agent_model(base_url="https://example.test"), "base_url to null"),
        (_agent_model(max_tokens=100), "must omit max_tokens"),
    ],
)
def test_direct_model_resolver_rejects_unsupported_codex_settings(
    agent_config: AgentConfig,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _create_direct_model_backend(agent_config, role_name="builder")


def test_camel_factory_rejects_codex_provider() -> None:
    with pytest.raises(RuntimeError, match="CAMEL.*ModelFactory"):
        _create_camel_model_backend(_agent_model(), role_name="builder")


def test_model_config_validates_codex_reasoning_effort() -> None:
    assert ModelConfig(
        provider="codex-cli",
        model_id="gpt-test",
        reasoning_effort="high",
    ).reasoning_effort == "high"
    with pytest.raises(ValidationError):
        ModelConfig(
            provider="codex-cli",
            model_id="gpt-test",
            reasoning_effort="extreme",  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        ModelConfig(
            provider="codex-cli",
            model_id="gpt-test",
            reasoning_effortt="xhigh",  # type: ignore[call-arg]
        )
    with pytest.raises(ValidationError):
        ModelConfig(
            provider="codex-cli",
            model_id="gpt-test",
            reasoning_effort="ultra",  # type: ignore[arg-type]
        )
