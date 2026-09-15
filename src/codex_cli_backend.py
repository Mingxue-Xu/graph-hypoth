"""Codex CLI as a process-per-call backend for GraphHypoth's direct LLM seams.

The graph-state pipeline consumes a small CAMEL-compatible surface:
``backend.run(messages)`` returns an OpenAI ChatCompletion-shaped object.  This
module implements that surface with a fresh, constrained ``codex exec`` process for
each call.  It deliberately does not implement GraphHypoth's model tool-call
exchange; retrieval and other tools remain in the audited host pipeline.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_CODEX_TIMEOUT_SECONDS = 600.0
DEFAULT_CODEX_REASONING_EFFORT = "high"

_BACKEND_PROMPT = (
    "Act only as the language-model completion backend for an automated research "
    "pipeline. The piped JSON contains a chat-completion message list. Follow its "
    "embedded system and user messages exactly, including any strict output format. "
    "Use only the information in those messages: do not invent papers, citations, "
    "metrics, sources, or experimental results. Do not inspect files, run commands, "
    "browse, call tools, or modify anything. Return only the raw assistant completion, "
    "with no preamble, commentary, or Markdown fence."
)

_ENV_ALLOWLIST = frozenset(
    {
        "HOME",
        "USER",
        "LOGNAME",
        "PATH",
        "SHELL",
        "CODEX_HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
    }
)

# The CLI does not currently expose a single "web-search and nothing else"
# allowlist switch. Keep this explicit denylist shared by completion and
# retrieval subprocesses; strict config makes an unknown/removed key fail closed.
_CODEX_DISABLED_FEATURE_OVERRIDES = (
    "agents.enabled=false",
    "features.shell_tool=false",
    "features.unified_exec=false",
    "features.shell_snapshot=false",
    "features.skill_search=false",
    "features.skill_mcp_dependency_install=false",
    "features.apps=false",
    "features.plugins=false",
    "features.plugin_sharing=false",
    "features.remote_plugin=false",
    "features.browser_use=false",
    "features.browser_use_external=false",
    "features.browser_use_full_cdp_access=false",
    "features.in_app_browser=false",
    "features.computer_use=false",
    "features.workspace_dependencies=false",
    "features.image_generation=false",
    "features.view_image=false",
    "features.multi_agent=false",
    "features.goals=false",
    "features.hooks=false",
    "features.tool_suggest=false",
    "features.auth_elicitation=false",
    "features.tool_call_mcp_elicitation=false",
    "features.code_mode=false",
)


def _config_override_args(overrides: Sequence[str]) -> list[str]:
    return [item for override in overrides for item in ("-c", override)]


def _codex_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the minimal inherited environment needed by the Codex executable.

    Provider credentials such as ``*_API_KEY`` and ``*_TOKEN`` are intentionally
    absent.  Codex must therefore use saved CLI authentication (normally created by
    ``codex login``).  The separate ``shell_environment_policy.inherit=none`` CLI
    override also prevents any model-invoked shell from inheriting this process
    environment.
    """

    inherited = os.environ if source is None else source
    env = {key: value for key, value in inherited.items() if key in _ENV_ALLOWLIST}
    env["NO_COLOR"] = "1"
    return env


def _codex_executable_candidates() -> tuple[Path, ...]:
    """Return common Codex CLI locations not necessarily inherited by IDEs."""

    user_home = Path.home()
    configured_home = os.environ.get("CODEX_HOME")
    codex_home = (
        Path(configured_home).expanduser() if configured_home else user_home / ".codex"
    )
    return (
        user_home / ".local/bin/codex",
        codex_home / "packages/standalone/current/bin/codex",
        Path("/opt/homebrew/bin/codex"),
        Path("/usr/local/bin/codex"),
        user_home / "Applications/ChatGPT.app/Contents/Resources/codex",
        Path("/Applications/ChatGPT.app/Contents/Resources/codex"),
    )


def _resolved_executable(path: Path) -> str | None:
    """Return the canonical path when *path* names an executable regular file."""

    candidate = path.expanduser().resolve()
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def resolve_codex_executable(executable: str | Path | None = None) -> str:
    """Resolve an override, environment setting, PATH entry, or known install."""

    requested = str(executable or os.environ.get("GRAPH_HYPOTH_CODEX_BIN") or "codex")
    expanded = Path(requested).expanduser()
    has_path_component = expanded.is_absolute() or expanded.parent != Path(".")

    if has_path_component:
        resolved = _resolved_executable(expanded)
        if resolved:
            return resolved
    else:
        resolved = shutil.which(requested)
        if resolved:
            return resolved
        if requested == "codex":
            for candidate in _codex_executable_candidates():
                resolved = _resolved_executable(candidate)
                if resolved:
                    return resolved

    raise RuntimeError(
        f"Codex CLI executable {requested!r} was not found or is not executable. "
        "Install Codex or set GRAPH_HYPOTH_CODEX_BIN to its executable path."
    )


def _json_default(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    return str(value)


def _request_payload(messages: Sequence[Mapping[str, Any]]) -> str:
    request_json = json.dumps(
        {"messages": list(messages)},
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )
    return (
        f"{_BACKEND_PROMPT}\n\n"
        "Chat-completion request JSON follows:\n"
        f"{request_json}"
    )


def _jsonl_metadata(
    stdout: str,
) -> tuple[str | None, dict[str, int], str | None, str | None]:
    """Extract thread id, usage, last agent message, and error from Codex JSONL."""

    thread_id: str | None = None
    usage: dict[str, int] = {}
    last_message: str | None = None
    error_message: str | None = None

    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "thread.started" and isinstance(event.get("thread_id"), str):
            thread_id = event["thread_id"]
        elif event_type == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = {
                str(key): value
                for key, value in event["usage"].items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
        elif event_type == "item.completed" and isinstance(event.get("item"), dict):
            item = event["item"]
            if item.get("type") == "agent_message" and isinstance(
                item.get("text"), str
            ):
                last_message = item["text"]
        elif event_type == "error":
            raw_error = event.get("message") or event.get("error")
            if isinstance(raw_error, str):
                error_message = raw_error
            elif isinstance(raw_error, dict) and isinstance(
                raw_error.get("message"), str
            ):
                error_message = raw_error["message"]
    return thread_id, usage, last_message, error_message


def _compact_diagnostic(
    value: str | None,
    *,
    secret_values: Sequence[str] = (),
    limit: int = 1000,
) -> str:
    if not value:
        return ""
    redacted = value
    for secret in secret_values:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    return " ".join(redacted.split())[:limit]


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate a timed-out Codex process and its children, then reap it."""

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (AttributeError, OSError, ProcessLookupError, PermissionError):
        try:
            process.terminate()
        except OSError:
            pass

    try:
        process.communicate(timeout=5)
        return
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (AttributeError, OSError, ProcessLookupError, PermissionError):
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.communicate(timeout=5)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        try:
            process.wait(timeout=5)
        except (AttributeError, OSError, ValueError, subprocess.TimeoutExpired):
            pass


class CodexSubagentBackend:
    """CAMEL-shaped backend backed by one isolated ``codex exec`` per call.

    The class intentionally remains mutable because GraphHypoth's logging hook
    replaces the instance's ``run`` attribute at runtime.
    """

    def __init__(
        self,
        *,
        role_name: str,
        model: str,
        reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
        timeout_seconds: float = DEFAULT_CODEX_TIMEOUT_SECONDS,
        executable: str | Path | None = None,
    ) -> None:
        if not role_name.strip():
            raise ValueError("role_name must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if not reasoning_effort.strip():
            raise ValueError("reasoning_effort must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.role_name = role_name
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = float(timeout_seconds)
        self.executable = resolve_codex_executable(executable)

    def _command(self, *, work_dir: Path, output_path: Path) -> list[str]:
        return [
            self.executable,
            "exec",
            "--strict-config",
            "--model",
            self.model,
            "-c",
            f"model_reasoning_effort={json.dumps(self.reasoning_effort)}",
            "-c",
            'shell_environment_policy.inherit="none"',
            "-c",
            'approval_policy="never"',
            *_config_override_args(_CODEX_DISABLED_FEATURE_OVERRIDES),
            "-c",
            "tools.web_search=false",
            "--sandbox",
            "read-only",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--ephemeral",
            "--color",
            "never",
            "--json",
            "--cd",
            str(work_dir),
            "--output-last-message",
            str(output_path),
            "-",
        ]

    def run(
        self,
        messages: Sequence[Mapping[str, Any]],
        *args: Any,
        tools: object | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if args:
            raise TypeError(
                "codex-cli backend accepts only messages and keyword arguments"
            )
        if tools:
            raise NotImplementedError(
                "codex-cli direct backend does not implement GraphHypoth "
                "tool-call exchange"
            )
        unsupported = {key: value for key, value in kwargs.items() if value is not None}
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise NotImplementedError(
                f"codex-cli backend does not support keyword(s): {names}"
            )

        payload = _request_payload(messages)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="graph-hypoth-codex-") as temp_name:
            work_dir = Path(temp_name)
            output_path = work_dir / "last-message.txt"
            command = self._command(work_dir=work_dir, output_path=output_path)
            codex_env = _codex_environment()
            diagnostic_secrets = tuple(
                value
                for key, value in codex_env.items()
                if "proxy" in key.lower()
            )
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=work_dir,
                    env=codex_env,
                    shell=False,
                    start_new_session=True,
                )
            except OSError as exc:
                raise RuntimeError(
                    f"failed to start Codex CLI for role {self.role_name!r}: "
                    f"{_compact_diagnostic(str(exc), secret_values=diagnostic_secrets)}"
                ) from exc

            try:
                stdout, stderr = process.communicate(
                    input=payload,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                _stop_process_group(process)
                raise RuntimeError(
                    f"Codex CLI timed out for role {self.role_name!r} after "
                    f"{self.timeout_seconds:g}s"
                ) from exc
            except BaseException:
                # The child owns a new process group, so cancellation must reap
                # it explicitly instead of relying on parent signal delivery.
                _stop_process_group(process)
                raise

            thread_id, usage, streamed_message, event_error = _jsonl_metadata(
                stdout or ""
            )
            completion = (
                output_path.read_text(encoding="utf-8").strip()
                if output_path.is_file()
                else ""
            )
            completion = completion or (streamed_message or "").strip()
            elapsed_ms = max(0, round((time.monotonic() - started) * 1000))

            if process.returncode != 0:
                detail = _compact_diagnostic(
                    event_error or stderr,
                    secret_values=diagnostic_secrets,
                )
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    f"Codex CLI failed for role {self.role_name!r} with exit code "
                    f"{process.returncode}{suffix}"
                )
            if not completion:
                detail = _compact_diagnostic(
                    event_error or stderr,
                    secret_values=diagnostic_secrets,
                )
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    "Codex CLI returned no completion for role "
                    f"{self.role_name!r}{suffix}"
                )

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": completion,
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                }
            ],
            "info": {
                "provider": "codex-cli",
                "model": self.model,
                "role": self.role_name,
                "latency_ms": elapsed_ms,
                "termination_reason": "stop",
                "usage": usage,
                "codex_thread_id": thread_id,
            },
        }


# A concise alias for callers that care about the transport rather than the
# process-per-call orchestration semantics.
CodexCliBackend = CodexSubagentBackend


__all__ = [
    "DEFAULT_CODEX_REASONING_EFFORT",
    "DEFAULT_CODEX_TIMEOUT_SECONDS",
    "CodexCliBackend",
    "CodexSubagentBackend",
    "resolve_codex_executable",
]
