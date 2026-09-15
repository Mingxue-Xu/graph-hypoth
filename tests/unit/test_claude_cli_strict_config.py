"""Opt-in local-CLI drift probe for the exact isolated subagent command."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.claude_cli_backend import (
    ClaudeSubagentBackend,
    _claude_environment,
    resolve_claude_executable,
)


def _claude_is_absent() -> bool:
    try:
        resolve_claude_executable()
    except RuntimeError:
        return True
    return False


@pytest.mark.live
@pytest.mark.drift_watchdog
@pytest.mark.skipif(_claude_is_absent(), reason="Claude CLI is absent")
def test_exact_subagent_command_parses_and_stops_before_a_model_turn(
    tmp_path: Path,
) -> None:
    """Empty stdin makes the CLI validate flags and stop before any model turn."""
    executable = resolve_claude_executable()
    completion = ClaudeSubagentBackend(
        role_name="strict_flag_probe",
        model="probe-no-model-call",
        executable=executable,
    )

    completed = subprocess.run(
        completion._command(),
        input="",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=_claude_environment(),
        timeout=60,
        check=False,
    )

    diagnostic = f"{completed.stderr}\n{completed.stdout}".lower()
    assert completed.returncode != 0, diagnostic
    assert "input must be provided" in diagnostic, diagnostic
    assert "unknown option" not in diagnostic, diagnostic
    assert "unknown argument" not in diagnostic, diagnostic
    assert "invalid" not in diagnostic, diagnostic
