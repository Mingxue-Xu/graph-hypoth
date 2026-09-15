from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_TRACE_FILE: Path | None = None
_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"(?i)(authorization|api[-_ ]?key|bearer)(['\":=\s]+)[^\[,'\"\s)}]+"
    ),
]


def record_runtime_event(
    event_type: str,
    *,
    run_id: str | None = None,
    thread_id: str | None = None,
    actor: str | None = None,
    target: str | None = None,
    direction: str | None = None,
    payload: Any = None,
    artifact_path: str | Path | None = None,
) -> Path | None:
    trace_file = _trace_file()
    if trace_file is None:
        return None

    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        "run_id": run_id,
        "thread_id": thread_id,
        "actor": actor,
        "target": target,
        "direction": direction,
        "artifact_path": (
            _redact_text(str(artifact_path)) if artifact_path is not None else None
        ),
        "payload": _redact(payload),
    }
    with trace_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return trace_file


def runtime_artifact_dir() -> Path | None:
    trace_file = _trace_file()
    if trace_file is None:
        return None
    return trace_file.parent


def reset_runtime_trace() -> None:
    global _TRACE_FILE
    _TRACE_FILE = None


def current_runtime_artifact_dir() -> Path | None:
    if _TRACE_FILE is None:
        return None
    return _TRACE_FILE.parent


def redact_runtime_payload(value: Any) -> Any:
    return _redact(value)


def _trace_file() -> Path | None:
    global _TRACE_FILE

    root = os.environ.get("GRAPH_HYPOTH_RUNTIME_LOG_DIR")
    if not root:
        return None
    if _TRACE_FILE is not None:
        return _TRACE_FILE

    label = _safe_label(os.environ.get("GRAPH_HYPOTH_RUNTIME_RUN_LABEL", "run"))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = Path(root) / f"{timestamp}-{label}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _TRACE_FILE = run_dir / "trace.jsonl"
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "label": label,
        "trace_file": _redact_text(str(_TRACE_FILE)),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _TRACE_FILE


def _safe_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip())
    return cleaned.strip("-") or "run"


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_redact(item) for item in value]
    if isinstance(value, os.PathLike):
        return _redact_text(os.fspath(value))
    if hasattr(value, "model_dump"):
        return _redact(value.model_dump(mode="json"))
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact_text(value: str) -> str:
    redacted = value
    for env_name in ("OPENROUTER_API_KEY", "EXA_API_KEY"):
        secret = os.environ.get(env_name)
        if secret:
            redacted = redacted.replace(secret, f"[REDACTED_{env_name}]")
    for home_path in _home_paths():
        redacted = redacted.replace(home_path, "$HOME")
    redacted = re.sub(r"/Users/[^/,\s\"')}\]]+", "$HOME", redacted)
    redacted = re.sub(r"/home/[^/,\s\"')}\]]+", "$HOME", redacted)
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(_redact_match, redacted)
    return redacted


def _home_paths() -> list[str]:
    candidates = []
    if os.environ.get("HOME"):
        candidates.append(os.environ["HOME"])
    try:
        candidates.append(str(Path.home()))
    except RuntimeError:
        pass
    return sorted(
        {path.rstrip("/") for path in candidates if path},
        key=len,
        reverse=True,
    )


def _redact_match(match: re.Match[str]) -> str:
    if match.re.pattern.startswith("\\bsk-"):
        return "[REDACTED_KEY]"
    return f"{match.group(1)}{match.group(2)}[REDACTED]"
