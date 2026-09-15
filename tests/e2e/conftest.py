"""Shared e2e gating helpers.

Live e2e tests must gate on the key the *configured* provider actually reads.
Hardcoding a provider's env var drifts the moment the shipped default config
changes provider -- which is exactly how six gates in this package came to skip
on a hardcoded legacy provider key while ``config/evidence-evaluation.yaml``
pins every agent to OpenRouter, so on a machine holding only that stale key they
ran and died in ``_preflight_model_credentials``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pytest

from src.config import load_config


DEFAULT_CONFIG_PATH = Path("config/evidence-evaluation.yaml")


@lru_cache(maxsize=None)
def live_model_key_env(config_path: str = str(DEFAULT_CONFIG_PATH)) -> str:
    """The env var the shipped default config's builder model authenticates with."""
    return load_config(Path(config_path)).agents.builder.model.api_key_env


def require_live_model_key(reason: str = "live CAMEL/LLM e2e tests") -> str:
    """Skip unless the configured provider's API key is present. Returns its name."""
    env_var = live_model_key_env()
    if not os.environ.get(env_var):
        pytest.skip(f"{env_var} is required for {reason}")
    return env_var
