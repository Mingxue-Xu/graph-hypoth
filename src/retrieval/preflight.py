"""Fail-fast checks for retrieval sources required by a CLI run."""

from __future__ import annotations

import os
from typing import Any

from src.codex_cli_backend import resolve_codex_executable


def preflight_required_retrieval(config: Any) -> None:
    retrieval = config.retrieval
    if not retrieval.enabled or retrieval.fake_mode:
        return
    required_sources = set(retrieval.sources)
    if "exa" in required_sources:
        key_env = retrieval.source_limits.exa.require_api_key_env
        if not os.environ.get(key_env):
            raise RuntimeError(
                f"required retrieval source exa requires {key_env}, but it is not set"
            )
    if "codex_web" in required_sources:
        try:
            resolve_codex_executable()
        except RuntimeError as exc:
            raise RuntimeError(
                "required retrieval source codex_web requires an executable "
                "Codex CLI; install Codex or set GRAPH_HYPOTH_CODEX_BIN"
            ) from exc


__all__ = ["preflight_required_retrieval"]
