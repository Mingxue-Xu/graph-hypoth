import os
from pathlib import Path

import pytest

from src.camel_adapter import _preflight_model_credentials
from src.config import load_config
from src.credentials import load_api_keys_file, resolve_api_key
from tests.support import openrouter_provider_config


def test_load_api_keys_file_sets_missing_environment_values(monkeypatch, tmp_path):
    api_keys_file = tmp_path / "api_keys.env"
    api_keys_file.write_text(
        """
# Blank lines and comments are ignored.
GRAPH_HYPOTH_TEST_PLAIN_KEY=from-file
export GRAPH_HYPOTH_TEST_EXPORTED_KEY="from-file-quoted"
GRAPH_HYPOTH_TEST_SINGLE_QUOTED_KEY='from-file-single-quoted'
OPENROUTER_API_KEY=
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("GRAPH_HYPOTH_TEST_PLAIN_KEY", raising=False)
    monkeypatch.delenv("GRAPH_HYPOTH_TEST_EXPORTED_KEY", raising=False)
    monkeypatch.delenv("GRAPH_HYPOTH_TEST_SINGLE_QUOTED_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    loaded = load_api_keys_file(api_keys_file)

    assert loaded == api_keys_file
    assert Path(api_keys_file).exists()
    assert os.environ["GRAPH_HYPOTH_TEST_PLAIN_KEY"] == "from-file"
    assert os.environ["GRAPH_HYPOTH_TEST_EXPORTED_KEY"] == "from-file-quoted"
    assert (
        os.environ["GRAPH_HYPOTH_TEST_SINGLE_QUOTED_KEY"] == "from-file-single-quoted"
    )
    # An empty value assigns nothing, so the variable stays unset.
    assert "OPENROUTER_API_KEY" not in os.environ


def test_load_api_keys_file_preserves_existing_environment(monkeypatch, tmp_path):
    api_keys_file = tmp_path / "api_keys.env"
    api_keys_file.write_text(
        "GRAPH_HYPOTH_TEST_PLAIN_KEY=from-file\n", encoding="utf-8"
    )
    monkeypatch.setenv("GRAPH_HYPOTH_TEST_PLAIN_KEY", "from-shell")

    load_api_keys_file(api_keys_file)

    assert os.environ["GRAPH_HYPOTH_TEST_PLAIN_KEY"] == "from-shell"


def test_load_api_keys_file_tolerates_bare_value_lines(monkeypatch, tmp_path):  # env
    # Bare-value key files (e.g. ~/api-keys/exa-api-key.txt hold just the token, no
    # NAME=) must NOT crash --api-keys-file; skip the unmappable line with a warning.
    api_keys_file = tmp_path / "exa-api-key.txt"
    api_keys_file.write_text("sk-bare-token-no-name\n", encoding="utf-8")

    loaded = load_api_keys_file(api_keys_file)  # must not raise

    assert loaded == api_keys_file


def test_preflight_model_credentials_reports_all_missing_provider_keys(
    monkeypatch,
):
    config = openrouter_provider_config()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        _preflight_model_credentials(config)

    message = str(exc_info.value)
    assert "builder" in message
    assert "openrouter" in message
    assert "OPENROUTER_API_KEY" in message
    assert "evidence_reviewer" in message
    assert "--api-keys-file" in message
    assert "provider: claude-cli or codex-cli" in message


def test_preflight_rejects_the_shipped_config_model_id_placeholder():
    # The shipped config ships `<MODEL_ID>` rather than a recommended model, and
    # nothing substitutes one. The first model call is on the far side of
    # claim-level retrieval, so the placeholder has to be caught at preflight or
    # a whole retrieval stage is spent before the provider returns a 404.
    config = load_config(Path("config/evidence-evaluation.yaml"))

    with pytest.raises(RuntimeError) as exc_info:
        _preflight_model_credentials(config)

    message = str(exc_info.value)
    assert "placeholder" in message
    assert "<MODEL_ID>" in message
    assert "--config" in message
    assert "builder" in message


def test_preflight_accepts_a_real_claude_cli_model_id():
    config = load_config(Path("config/evidence-evaluation.yaml")).model_copy(deep=True)
    for role in ("builder", "skeptical_verifier"):
        getattr(config.agents, role).model.model_id = "claude-sonnet-5"

    _preflight_model_credentials(config)  # must not raise: claude-cli takes no key


def test_resolve_api_key_env_mode_reads_environment(monkeypatch):
    monkeypatch.setenv("GRAPH_HYPOTH_TEST_KEY", "from-env")
    assert resolve_api_key("GRAPH_HYPOTH_TEST_KEY") == "from-env"
    monkeypatch.delenv("GRAPH_HYPOTH_TEST_KEY", raising=False)
    assert resolve_api_key("GRAPH_HYPOTH_TEST_KEY") is None
    assert resolve_api_key(None) is None


def test_resolve_api_key_proxy_mode_holds_no_client_key(monkeypatch):
    # Even when an env key exists, proxy mode returns None (proxy injects auth).
    monkeypatch.setenv("GRAPH_HYPOTH_TEST_KEY", "from-env")
    assert resolve_api_key("GRAPH_HYPOTH_TEST_KEY", key_source="proxy") is None
