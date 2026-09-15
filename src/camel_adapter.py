from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable
from typing import Any

from src import _warning_filters as _warning_filters  # noqa: F401
from src.config import AgentConfig, OrchestrationConfig
from src.progress import progress_operation


def _response_info(response: object) -> dict[str, Any]:
    info = (
        response.get("info")
        if isinstance(response, dict)
        else getattr(response, "info", None)
    )
    return info if isinstance(info, dict) else {}


def _model_dump_payload(response: object) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        payload = model_dump()
        return payload if isinstance(payload, dict) else {}
    return {}


def _response_usage(response: object) -> dict[str, Any]:
    info = _response_info(response)
    usage = info.get("usage") or info.get("token_usage")
    if isinstance(usage, dict):
        return usage
    payload = _model_dump_payload(response)
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else {}


def _response_token_usage(response: object) -> dict[str, int]:
    usage = _response_usage(response)

    normalized: dict[str, int] = {}
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            normalized[str(key)] = value
    if "total_tokens" not in normalized:
        # Cached/reasoning token fields are components of the input/output totals in
        # Codex JSONL and several provider responses, not additional tokens. Prefer
        # canonical input+output pairs before falling back to the legacy sum-all rule.
        total = 0
        for input_key, output_key in (
            ("input_tokens", "output_tokens"),
            ("prompt_tokens", "completion_tokens"),
            ("input", "output"),
        ):
            if input_key in normalized or output_key in normalized:
                total = normalized.get(input_key, 0) + normalized.get(output_key, 0)
                break
        if not total:
            total = sum(normalized.values())
        if total:
            normalized["total_tokens"] = total
    return normalized


def _install_logging_backend_hook(
    backend: object,
    sink: Callable[[str, Any, Any], None],
    *,
    role: str,
    provider: str | None = None,
) -> object:
    """Wrap ``backend.run`` to log every ``(role, request messages, raw response)`` via ``sink``,
    then delegate untouched.

    Used to capture each LLM seam's raw I/O in the standalone run (the intermediate artifacts the
    subagent setting surfaces). Logging must NEVER break the LLM turn, so sink errors are swallowed.
    """
    if not hasattr(backend, "run") or getattr(
        backend,
        "_graph_hypoth_logging_hooked",
        False,
    ):
        return backend
    original_run = backend.run  # type: ignore[attr-defined]

    def run(*args: Any, **kwargs: Any) -> Any:
        with progress_operation(role.replace("_", " "), provider=provider):
            response = original_run(*args, **kwargs)
        try:
            messages = args[0] if args else kwargs.get("messages")
            sink(role, messages, response)
        except Exception:
            pass  # observability must never break the LLM turn
        return response

    try:
        setattr(backend, "run", run)
        setattr(backend, "_graph_hypoth_logging_hooked", True)
    except Exception:
        return backend
    return backend


def _is_placeholder_model_id(model_id: str) -> bool:
    """True for an unedited config placeholder such as ``<MODEL_ID>``.

    The shipped ``config/evidence-evaluation.yaml`` carries placeholders rather
    than a recommended model, and no layer substitutes one — an unedited value
    is handed straight to the provider. Angle-bracketed values are reserved for
    that role in the shipped config and its documented variants, so treating one
    as a real model name costs nothing and catching it saves a whole run.
    """

    stripped = model_id.strip()
    return stripped.startswith("<") and stripped.endswith(">")


def _preflight_model_credentials(
    config: OrchestrationConfig,
    roles: Iterable[str] = ("builder", "evidence_reviewer"),
) -> None:
    missing_credentials: list[str] = []
    placeholder_models: list[str] = []
    for role_name in dict.fromkeys(roles):
        agent_config = config.agents.for_role(role_name)
        model = _required_model_config(agent_config, role_name=role_name)
        if _is_placeholder_model_id(model.model_id):
            placeholder_models.append(
                f"{role_name} model provider={model.provider!r} "
                f"model_id={model.model_id!r}"
            )
        if model.api_key_env and not os.environ.get(model.api_key_env):
            missing_credentials.append(
                f"{role_name} model provider={model.provider!r} "
                f"model_id={model.model_id!r} requires {model.api_key_env}"
            )
    if placeholder_models:
        # Reported here rather than at config load because the shipped config is
        # loaded, unedited, by much of the deterministic suite. The first model
        # call happens after claim-level retrieval, so leaving this to the
        # provider spends the whole retrieval stage before returning a 404 on a
        # model named ``<MODEL_ID>``.
        raise RuntimeError(
            "model_id is still a config placeholder:\n"
            + "\n".join(f"- {item}" for item in placeholder_models)
            + "\nCopy config/evidence-evaluation.yaml to an untracked file, "
            "replace each placeholder with a real model, and pass the copy with "
            "--config. For provider: claude-cli or codex-cli use a model your "
            "saved CLI login can use (what you would pass to `claude --model`); "
            "for provider: openrouter use a catalog slug from "
            "https://openrouter.ai/models."
        )
    if missing_credentials:
        raise RuntimeError(
            "missing model provider API keys:\n"
            + "\n".join(f"- {item}" for item in missing_credentials)
            + "\nExport the missing keys or pass an external key file with "
            "--api-keys-file before running. Set OPENROUTER_API_KEY (base_url "
            "https://openrouter.ai/api/v1; pick a model slug at "
            "https://openrouter.ai/models), or switch the role to a key-free "
            "local CLI provider (provider: claude-cli or codex-cli with "
            "api_key_env: null)."
        )


def _extract_json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match is None:
            raise ValueError("model response did not contain JSON") from None
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON must be an object")
    return parsed


def _first_response_message(response: object) -> object:
    choices = (
        response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    )
    if not choices:
        raise RuntimeError("model response did not include choices")
    first_choice = choices[0]
    message = (
        first_choice.get("message")
        if isinstance(first_choice, dict)
        else getattr(first_choice, "message", None)
    )
    if message is None:
        raise RuntimeError("model response did not include a message")
    return message


def _message_content(message: object) -> object:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def backend_json(
    backend: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    catch_backend_errors: bool = False,
    return_response: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], object | None]:
    """The shared LLM-seam scaffold: build ``[system, user]`` messages, run the backend, and
    parse STRICT JSON from its response — degrading to ``{}`` on empty/malformed output rather
    than crashing. With ``catch_backend_errors`` also swallows a ``RuntimeError`` from
    ``backend.run``/``_first_response_message`` (else it propagates, matching the extraction/
    hypothesis call sites); without it, only the JSON-parse ``ValueError`` is caught. With
    ``return_response`` also returns the raw backend response (e.g. for token-usage capture)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response: object | None = None
    if catch_backend_errors:
        try:
            response = backend.run(messages)
            content = _message_content(_first_response_message(response))
            data = _extract_json_object(str(content)) if content else {}
        except (ValueError, RuntimeError):
            data = {}
    else:
        response = backend.run(messages)
        content = _message_content(_first_response_message(response))
        if not content:
            data = {}
        else:
            try:
                data = _extract_json_object(str(content))
            except ValueError:
                data = {}
    return (data, response) if return_response else data


def _camel_model_config_dict(agent_config: AgentConfig) -> dict[str, object]:
    model_config_dict: dict[str, object] = {
        "temperature": agent_config.temperature,
    }
    if agent_config.model and agent_config.model.max_tokens is not None:
        model_config_dict["max_tokens"] = agent_config.model.max_tokens
    return model_config_dict


# Providers answered by a constrained local CLI subprocess rather than through
# CAMEL. They share the direct ``.run(messages)`` contract and its guards, and
# neither implements CAMEL's model/tool exchange.
_DIRECT_CLI_PROVIDERS = frozenset({"claude-cli", "codex-cli"})


def _required_model_config(agent_config: AgentConfig, *, role_name: str):
    if agent_config.model is None:
        raise RuntimeError(f"{role_name} model config is required")
    return agent_config.model


def _configured_model_api_key(
    agent_config: AgentConfig,
    *,
    role_name: str,
) -> str | None:
    model = _required_model_config(agent_config, role_name=role_name)
    if model.api_key_env is None:
        return None
    api_key = os.environ.get(model.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{role_name} model requires {model.api_key_env}, but it is not set"
        )
    return api_key


def _redact_sensitive_values(message: str, values: list[str | None]) -> str:
    redacted = message
    for value in values:
        if value:
            redacted = redacted.replace(value, "[redacted]")
    return redacted


def _create_camel_model_backend(
    agent_config: AgentConfig,
    *,
    role_name: str,
) -> object:
    model_config = _required_model_config(agent_config, role_name=role_name)

    if model_config.provider in _DIRECT_CLI_PROVIDERS:
        raise RuntimeError(
            f"provider={model_config.provider!r} is supported by the direct "
            "graph-state seams; it cannot be constructed through CAMEL's "
            "ModelFactory"
        )

    try:
        from camel.models import ModelFactory
    except ImportError as exc:
        raise RuntimeError(
            "CAMEL is required to build configured model backends."
        ) from exc

    api_key = _configured_model_api_key(agent_config, role_name=role_name)

    try:
        backend = ModelFactory.create(
            model_platform=model_config.provider,
            model_type=model_config.model_id,
            model_config_dict=_camel_model_config_dict(agent_config),
            api_key=api_key,
            url=model_config.base_url,
            timeout=model_config.timeout_seconds,
        )
        return backend
    except Exception as exc:
        safe_error = _redact_sensitive_values(str(exc), [api_key])
        raise RuntimeError(
            "installed camel-ai cannot support "
            f"provider={model_config.provider!r}, "
            f"model_id={model_config.model_id!r}: {safe_error}"
        ) from exc


def _create_direct_model_backend(
    agent_config: AgentConfig,
    *,
    role_name: str,
) -> object:
    """Build a backend for GraphHypoth's direct ``.run(messages)`` seams.

    Normal configured providers continue through CAMEL unchanged. ``codex-cli``
    and ``claude-cli`` instead launch a fresh CLI process for each call. These
    direct CLI backends intentionally implement no provider-side tool exchange.
    """

    model_config = _required_model_config(agent_config, role_name=role_name)
    provider = model_config.provider
    if provider not in _DIRECT_CLI_PROVIDERS:
        return _create_camel_model_backend(
            agent_config,
            role_name=role_name,
        )

    # Both CLI backends authenticate from a saved CLI login, take no per-call
    # token cap, and implement no tool-call exchange, so the guards are shared.
    login_hint = (
        "saved `codex login` authentication"
        if provider == "codex-cli"
        else "saved `claude auth login` authentication"
    )
    if model_config.api_key_env is not None:
        raise RuntimeError(
            f"{role_name} {provider} model must set api_key_env to null; "
            f"the backend uses {login_hint}"
        )
    if model_config.base_url is not None:
        raise RuntimeError(
            f"{role_name} {provider} model must set base_url to null"
        )
    if model_config.max_tokens is not None:
        raise RuntimeError(
            f"{role_name} {provider} model must omit max_tokens; the CLI does not "
            "expose an equivalent per-call limit"
        )
    if provider == "codex-cli":
        from src.codex_cli_backend import (
            DEFAULT_CODEX_REASONING_EFFORT,
            DEFAULT_CODEX_TIMEOUT_SECONDS,
            CodexSubagentBackend,
        )

        backend: object = CodexSubagentBackend(
            role_name=role_name,
            model=model_config.model_id,
            reasoning_effort=(
                model_config.reasoning_effort or DEFAULT_CODEX_REASONING_EFFORT
            ),
            timeout_seconds=(
                model_config.timeout_seconds or DEFAULT_CODEX_TIMEOUT_SECONDS
            ),
        )
    else:
        from src.claude_cli_backend import (
            DEFAULT_CLAUDE_REASONING_EFFORT,
            DEFAULT_CLAUDE_TIMEOUT_SECONDS,
            ClaudeSubagentBackend,
        )

        backend = ClaudeSubagentBackend(
            role_name=role_name,
            model=model_config.model_id,
            reasoning_effort=(
                model_config.reasoning_effort or DEFAULT_CLAUDE_REASONING_EFFORT
            ),
            timeout_seconds=(
                model_config.timeout_seconds or DEFAULT_CLAUDE_TIMEOUT_SECONDS
            ),
        )
    return backend
