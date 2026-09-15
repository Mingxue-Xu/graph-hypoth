"""Opt-in local-CLI drift probe for the exact isolated subagent commands."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.codex_cli_backend import CodexSubagentBackend, _codex_environment
from src.config import CodexWebSourceConfig
from src.retrieval.codex_web import (
    CodexCliWebRunner,
    codex_web_output_schema,
)


@pytest.mark.live
@pytest.mark.drift_watchdog
@pytest.mark.skipif(shutil.which("codex") is None, reason="Codex CLI is absent")
def test_exact_subagent_commands_pass_strict_config_parsing(
    tmp_path: Path,
) -> None:
    """Empty stdin makes Codex validate config and stop before any model turn."""
    executable = str(shutil.which("codex"))
    completion = CodexSubagentBackend(
        role_name="strict_config_probe",
        model="probe-no-model-call",
        executable=executable,
    )
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(codex_web_output_schema(max_records=1)),
        encoding="utf-8",
    )
    web = CodexCliWebRunner(
        CodexWebSourceConfig(model_id="probe-no-model-call"),
        executable=executable,
    )
    commands = {
        "completion": completion._command(
            work_dir=tmp_path,
            output_path=tmp_path / "completion.json",
        ),
        "retrieval": web._command(
            executable=executable,
            work_dir=tmp_path,
            output_path=tmp_path / "retrieval.json",
            schema_path=schema_path,
            allowed_domains=["example.org"],
        ),
    }

    for name, command in commands.items():
        completed = subprocess.run(
            command,
            input="",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_codex_environment(),
            timeout=15,
            check=False,
        )
        diagnostic = f"{completed.stderr}\n{completed.stdout}".lower()
        assert completed.returncode != 0, name
        assert "no prompt provided via stdin" in diagnostic, diagnostic
        assert "unknown configuration" not in diagnostic, diagnostic
        assert "failed to load configuration" not in diagnostic, diagnostic
