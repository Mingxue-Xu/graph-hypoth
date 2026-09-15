"""Claude Code CLI as a process-per-call backend for GraphHypoth's direct LLM seams.

The graph-state pipeline consumes a small CAMEL-compatible surface:
``backend.run(messages)`` returns an OpenAI ChatCompletion-shaped object.  This
module implements that surface with a fresh, constrained ``claude --print``
process for each call.  It deliberately does not implement GraphHypoth's model
tool-call exchange; retrieval and other tools remain in the audited host pipeline.

This is the Claude analogue of :mod:`src.codex_cli_backend`.  The backend prompt,
request payload framing, diagnostic redaction, and process-group cancellation are
imported from that module rather than re-implemented, so the two transports send a
byte-identical instruction and cancel identically -- a fairness precondition for
the backend-sensitivity comparison.  A freeze or provenance record therefore has
to hash both source files, not just this one.

Four boundaries could not be mirrored exactly, and an audit has to disclose them:

1. No kernel-level sandbox.  Codex passes ``--sandbox read-only``; the Claude CLI
   exposes no sandbox flag.  Isolation rests on removing the entire tool surface
   and running in an empty temporary working directory -- a broader restriction
   in surface, but not an OS-enforced one.
2. No ``--strict-config`` analogue.  Every constraint here is a command-line flag
   and no configuration file is read, so the fail-closed mode is ``unknown
   option`` on a renamed or removed flag.  The drift watchdog for it is
   ``tests/unit/test_claude_cli_strict_config.py``.
3. Failure signalling differs.  The CLI can report ``is_error`` with the failure
   text in ``result`` while ``subtype`` still reads ``"success"``, so ``is_error``
   is treated as authoritative and an error message is never promoted to a
   completion.
4. Reasoning-effort vocabularies differ: Codex has ``minimal``, Claude has
   ``max``.  This one has teeth: the CLI treats an unrecognised ``--effort`` as a
   *warning* and silently falls back to its default effort, which would make a
   controlled comparison run at an unknown setting.  The value is therefore
   validated here against :data:`CLAUDE_REASONING_EFFORTS` before a process is
   spawned, rather than against ``src.config.ModelConfig``, whose ``Literal``
   still carries Codex's vocabulary.

``provider: claude-cli`` is selectable from configuration and resolved by
``src.camel_adapter._create_direct_model_backend``, which requires
``api_key_env: null``, ``base_url: null``, no ``max_tokens`` and ``tools: []`` so
the subprocess runs on saved ``claude`` CLI credentials and receives no API key.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.codex_cli_backend import (
    _BACKEND_PROMPT,
    _compact_diagnostic,
    _request_payload,
    _stop_process_group,
)

DEFAULT_CLAUDE_TIMEOUT_SECONDS = 600.0
DEFAULT_CLAUDE_REASONING_EFFORT = "high"

# ``claude --effort`` accepts exactly these levels (verified against CLI 2.1.247).
# Codex's ``minimal`` has no Claude equivalent and Claude's ``max`` has no Codex
# equivalent.  An unrecognised value only warns and falls back to the default
# effort, so it must be rejected here rather than inherited unchecked from
# ``ModelConfig.reasoning_effort``.
CLAUDE_REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})

_ENV_ALLOWLIST = frozenset(
    {
        "HOME",
        "USER",
        "LOGNAME",
        "PATH",
        "SHELL",
        "CLAUDE_CONFIG_DIR",
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
        "NODE_EXTRA_CA_CERTS",
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

# The CLI has no single "plain completion and nothing else" switch.  Keep this
# explicit list of constraining flags together: an unknown or removed flag makes
# the CLI exit non-zero with ``unknown option``, so the boundary fails closed.
_CLAUDE_CONSTRAINT_FLAGS = (
    # Disable every built-in tool: no shell, file, browser, or image tools.
    ("--tools", ""),
    # No skills, no MCP servers, no Chrome integration.
    ("--disable-slash-commands",),
    ("--strict-mcp-config",),
    ("--no-chrome",),
    # Ignore user/project/local settings, project memory files, plugins, hooks, and agents.
    ("--safe-mode",),
    ("--setting-sources", ""),
    # Leave no resumable session behind.
    ("--no-session-persistence",),
)


def _constraint_flag_args() -> list[str]:
    return [item for flag in _CLAUDE_CONSTRAINT_FLAGS for item in flag]


def _claude_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the minimal inherited environment needed by the Claude executable.

    Provider credentials — every ``*_API_KEY`` and ``*_TOKEN`` — are
    intentionally absent.  Claude must therefore use saved CLI authentication
    (normally created by ``claude auth login`` or ``claude setup-token``).  The
    host session's own ``CLAUDE_CODE_*`` variables are also withheld so the
    child never inherits an outer Claude Code session's identity, transport, or
    entrypoint.
    """

    inherited = os.environ if source is None else source
    env = {key: value for key, value in inherited.items() if key in _ENV_ALLOWLIST}
    env["NO_COLOR"] = "1"
    return env


def _claude_executable_candidates() -> tuple[Path, ...]:
    """Return common Claude CLI locations not necessarily inherited by IDEs."""

    user_home = Path.home()
    return (
        user_home / ".local/bin/claude",
        user_home / ".claude/local/claude",
        Path("/opt/homebrew/bin/claude"),
        Path("/usr/local/bin/claude"),
    )


def _resolved_executable(path: Path) -> str | None:
    """Return the canonical path when *path* names an executable regular file."""

    candidate = path.expanduser().resolve()
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def resolve_claude_executable(executable: str | Path | None = None) -> str:
    """Resolve an override, environment setting, PATH entry, or known install."""

    requested = str(
        executable or os.environ.get("GRAPH_HYPOTH_CLAUDE_BIN") or "claude"
    )
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
        if requested == "claude":
            for candidate in _claude_executable_candidates():
                resolved = _resolved_executable(candidate)
                if resolved:
                    return resolved

    raise RuntimeError(
        f"Claude CLI executable {requested!r} was not found or is not executable. "
        "Install the Claude Code CLI or set GRAPH_HYPOTH_CLAUDE_BIN to its "
        "executable path."
    )



def _generating_model(
    model_usage: object, requested_model: str | None = None
) -> str | None:
    """Return the model that actually produced the completion.

    ``modelUsage`` is keyed by model and can hold more than one entry: Claude
    Code bills small auxiliary turns to a fast model alongside the requested
    one.  Taking the first key reported whichever model happened to be inserted
    first -- typically the auxiliary one -- which would record a false model in
    every attempt's provenance and trip the release verifier.  The generating
    model is the one that emitted the output tokens.
    """

    if not isinstance(model_usage, Mapping) or not model_usage:
        return None
    names = [name for name in model_usage if isinstance(name, str)]
    if not names:
        return None
    # An exact match on the requested model is authoritative when present.
    if requested_model in names:
        return requested_model

    def output_tokens(name: str) -> int:
        stats = model_usage.get(name)
        value = stats.get("outputTokens") if isinstance(stats, Mapping) else None
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    return max(names, key=output_tokens)

def _int_fields(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, int) and not isinstance(item, bool)
    }


@dataclass(frozen=True)
class _ClaudeResult:
    """The fields this backend reads out of one ``--output-format json`` object."""

    session_id: str | None = None
    text: str = ""
    is_error: bool = False
    usage: dict[str, int] | None = None
    cost_usd: float | None = None
    resolved_model: str | None = None
    model_usage: dict[str, Any] | None = None
    num_turns: int | None = None
    permission_denials: int | None = None
    stop_reason: str | None = None
    error_message: str | None = None


def _result_object(stdout: str) -> dict[str, Any] | None:
    """Return the CLI's final ``type: "result"`` object, if stdout carries one."""

    text = (stdout or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    # Defensive: tolerate a stream-shaped stdout if the CLI ever emits one here.
    latest: dict[str, Any] | None = None
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            latest = event
    return latest


def _claude_metadata(
    stdout: str, requested_model: str | None = None
) -> _ClaudeResult:
    """Extract session id, completion, usage, cost, and error state from stdout."""

    payload = _result_object(stdout)
    if payload is None:
        return _ClaudeResult()

    is_error = bool(payload.get("is_error"))
    raw_text = payload.get("result")
    text = raw_text if isinstance(raw_text, str) else ""

    usage = _int_fields(payload.get("usage"))
    cost_raw = payload.get("total_cost_usd")
    cost_usd = (
        float(cost_raw)
        if isinstance(cost_raw, (int, float)) and not isinstance(cost_raw, bool)
        else None
    )

    model_usage = payload.get("modelUsage")
    resolved_model = _generating_model(model_usage, requested_model)

    denials = payload.get("permission_denials")
    permission_denials = len(denials) if isinstance(denials, list) else None

    error_message: str | None = None
    if is_error:
        # ``result`` carries the failure text when ``is_error`` is set, so it must
        # never be promoted to a completion.
        details = [
            payload.get("terminal_reason"),
            payload.get("api_error_status"),
            text,
        ]
        error_message = "; ".join(str(item) for item in details if item) or None
        text = ""

    return _ClaudeResult(
        session_id=(
            payload["session_id"]
            if isinstance(payload.get("session_id"), str)
            else None
        ),
        text=text,
        is_error=is_error,
        usage=usage,
        cost_usd=cost_usd,
        resolved_model=resolved_model if isinstance(resolved_model, str) else None,
        model_usage=dict(model_usage) if isinstance(model_usage, Mapping) else None,
        num_turns=(
            payload["num_turns"]
            if isinstance(payload.get("num_turns"), int)
            and not isinstance(payload.get("num_turns"), bool)
            else None
        ),
        permission_denials=permission_denials,
        stop_reason=(
            payload["stop_reason"]
            if isinstance(payload.get("stop_reason"), str)
            else None
        ),
        error_message=error_message,
    )


class ClaudeSubagentBackend:
    """CAMEL-shaped backend backed by one isolated ``claude --print`` per call.

    The class intentionally remains mutable because GraphHypoth's logging hook
    replaces the instance's ``run`` attribute at runtime.
    """

    def __init__(
        self,
        *,
        role_name: str,
        model: str,
        reasoning_effort: str = DEFAULT_CLAUDE_REASONING_EFFORT,
        timeout_seconds: float = DEFAULT_CLAUDE_TIMEOUT_SECONDS,
        executable: str | Path | None = None,
        max_budget_usd: float | None = None,
    ) -> None:
        if not role_name.strip():
            raise ValueError("role_name must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if not reasoning_effort.strip():
            raise ValueError("reasoning_effort must not be empty")
        if reasoning_effort not in CLAUDE_REASONING_EFFORTS:
            allowed = ", ".join(sorted(CLAUDE_REASONING_EFFORTS))
            raise ValueError(
                f"reasoning_effort {reasoning_effort!r} is not supported by the "
                f"Claude CLI; use one of: {allowed}"
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_budget_usd is not None and max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive when set")

        self.role_name = role_name
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = float(timeout_seconds)
        self.max_budget_usd = (
            float(max_budget_usd) if max_budget_usd is not None else None
        )
        self.executable = resolve_claude_executable(executable)

    def _command(self) -> list[str]:
        budget = (
            ["--max-budget-usd", f"{self.max_budget_usd:g}"]
            if self.max_budget_usd is not None
            else []
        )
        return [
            self.executable,
            "--print",
            "--model",
            self.model,
            "--effort",
            self.reasoning_effort,
            "--output-format",
            "json",
            "--system-prompt",
            _BACKEND_PROMPT,
            *_constraint_flag_args(),
            *budget,
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
                "claude-cli backend accepts only messages and keyword arguments"
            )
        if tools:
            raise NotImplementedError(
                "claude-cli direct backend does not implement GraphHypoth "
                "tool-call exchange"
            )
        unsupported = {key: value for key, value in kwargs.items() if value is not None}
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise NotImplementedError(
                f"claude-cli backend does not support keyword(s): {names}"
            )

        payload = _request_payload(messages)
        command = self._command()
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="graph-hypoth-claude-") as temp_name:
            work_dir = Path(temp_name)
            claude_env = _claude_environment()
            diagnostic_secrets = tuple(
                value for key, value in claude_env.items() if "proxy" in key.lower()
            )
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=work_dir,
                    env=claude_env,
                    shell=False,
                    start_new_session=True,
                )
            except OSError as exc:
                raise RuntimeError(
                    f"failed to start Claude CLI for role {self.role_name!r}: "
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
                    f"Claude CLI timed out for role {self.role_name!r} after "
                    f"{self.timeout_seconds:g}s"
                ) from exc
            except BaseException:
                # The child owns a new process group, so cancellation must reap
                # it explicitly instead of relying on parent signal delivery.
                _stop_process_group(process)
                raise

            parsed = _claude_metadata(stdout or "", self.model)
            completion = parsed.text.strip()
            elapsed_ms = max(0, round((time.monotonic() - started) * 1000))

            def _detail() -> str:
                text = _compact_diagnostic(
                    parsed.error_message or stderr,
                    secret_values=diagnostic_secrets,
                )
                return f": {text}" if text else ""

            if process.returncode != 0:
                raise RuntimeError(
                    f"Claude CLI failed for role {self.role_name!r} with exit code "
                    f"{process.returncode}{_detail()}"
                )
            if parsed.is_error:
                raise RuntimeError(
                    "Claude CLI reported an error for role "
                    f"{self.role_name!r}{_detail()}"
                )
            if not completion:
                raise RuntimeError(
                    "Claude CLI returned no completion for role "
                    f"{self.role_name!r}{_detail()}"
                )

        usage: dict[str, Any] = dict(parsed.usage or {})
        if parsed.cost_usd is not None:
            # Preserve the CLI-reported figure for direct response consumers.
            # Integer-only token normalization deliberately ignores it.
            usage["cost_usd"] = parsed.cost_usd

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
                "provider": "claude-cli",
                "model": self.model,
                "role": self.role_name,
                "latency_ms": elapsed_ms,
                "termination_reason": "stop",
                "usage": usage,
                # ``id`` is the generic provider-issued call identifier; the
                # aliased key makes the issuing transport explicit.
                "id": parsed.session_id,
                "claude_session_id": parsed.session_id,
                "resolved_model": parsed.resolved_model,
                "model_usage": parsed.model_usage,
                "reasoning_effort": self.reasoning_effort,
                "num_turns": parsed.num_turns,
                "permission_denials": parsed.permission_denials,
                "stop_reason": parsed.stop_reason,
            },
        }


def claude_cli_preflight(
    executable: str | Path | None = None,
    *,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Report CLI presence, version, and saved-login state without a model call.

    This is the Claude analogue of the ``codex --version`` / ``codex login status``
    pair recorded by the preflight seam in ``src/cli.py`` and
    ``src/retrieval/preflight.py``.  It runs under the same constrained
    environment as a real call, so a login that only works with an inherited
    provider API key is reported as absent.
    """

    errors: list[str] = []
    try:
        resolved = resolve_claude_executable(executable)
    except RuntimeError as exc:
        return {
            "executable": None,
            "version": None,
            "available": False,
            "logged_in": False,
            "auth_method": None,
            "errors": [str(exc)],
        }

    env = _claude_environment()

    def _probe(arguments: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                [resolved, *arguments],
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"claude {' '.join(arguments)} failed: {exc}")
            return None

    version_run = _probe(["--version"])
    version = (
        version_run.stdout.strip()
        if version_run is not None and version_run.returncode == 0
        else None
    )
    if version_run is not None and version_run.returncode != 0:
        errors.append(
            "claude --version exited with "
            f"{version_run.returncode}: {_compact_diagnostic(version_run.stderr)}"
        )

    logged_in = False
    auth_method: str | None = None
    status_run = _probe(["auth", "status", "--json"])
    if status_run is not None:
        # ``auth status`` exits non-zero when signed out but still prints its
        # JSON, so the payload is authoritative and the exit code is not.
        status: object = None
        if status_run.stdout.strip():
            try:
                status = json.loads(status_run.stdout)
            except (TypeError, ValueError):
                status = None
        if isinstance(status, Mapping):
            logged_in = bool(status.get("loggedIn"))
            raw_method = status.get("authMethod")
            auth_method = raw_method if isinstance(raw_method, str) else None
        else:
            errors.append(
                "claude auth status exited with "
                f"{status_run.returncode} without parseable JSON: "
                f"{_compact_diagnostic(status_run.stderr or status_run.stdout)}"
            )
        if not logged_in:
            errors.append(
                "claude CLI is not logged in; run `claude auth login` or "
                "`claude setup-token` for the account this study bills to"
            )

    return {
        "executable": resolved,
        "version": version,
        "available": version is not None,
        "logged_in": logged_in,
        "auth_method": auth_method,
        "errors": errors,
    }


# A concise alias for callers that care about the transport rather than the
# process-per-call orchestration semantics.
ClaudeCliBackend = ClaudeSubagentBackend


__all__ = [
    "CLAUDE_REASONING_EFFORTS",
    "DEFAULT_CLAUDE_REASONING_EFFORT",
    "DEFAULT_CLAUDE_TIMEOUT_SECONDS",
    "ClaudeCliBackend",
    "ClaudeSubagentBackend",
    "claude_cli_preflight",
    "resolve_claude_executable",
]
