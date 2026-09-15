"""Regression coverage for the surviving ``--adapter`` switch.

The legacy CAMEL debate adapter is gone, but ``--adapter fake`` remains the only
CLI switch for the deterministic, no-credential retrieval path: it forces
``retrieval.fake_mode``, which is otherwise settable only from a config file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import cli
from src.graph_state_runtime import GraphStateDeps
from src.run_path import GraphRunResult
from tests.support import runnable_default_config

pytestmark = pytest.mark.smoke


def _empty_graph_result() -> GraphRunResult:
    return GraphRunResult(
        graph_id="g",
        version=0,
        statuses={},
        receipts=(),
        open_risks=(),
        graph_json={},
        edge_table=(),
        audit_memo="memo",
    )


class _UnusedExtractor:
    def extract(self, claim):  # pragma: no cover - the workflow is stubbed out
        raise AssertionError("the workflow is stubbed in this module")


class _UnusedReviewer:
    def review(self, task, evidences):  # pragma: no cover - workflow is stubbed
        raise AssertionError("the workflow is stubbed in this module")


def _inert_deps() -> GraphStateDeps:
    return GraphStateDeps(
        extractor=_UnusedExtractor(),
        embedder=None,
        evidence_reviewer=_UnusedReviewer(),
        retrieve_for_target=lambda task: [],
    )


def _run_cli(monkeypatch, tmp_path: Path, extra_argv: list[str]):
    """Run ``cli.main`` with a stubbed workflow and return the config it used."""

    config = runnable_default_config()
    for agent in (config.agents.builder, config.agents.skeptical_verifier):
        assert agent.model is not None
        agent.model.api_key_env = None
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cli, "build_graph_state_deps", lambda *_args, **_kwargs: _inert_deps()
    )
    monkeypatch.setattr(
        cli,
        "run_graph_state_workflow",
        lambda *_args, **_kwargs: _empty_graph_result(),
    )

    cli.main(
        [
            "--claim",
            "smoking increases lung cancer",
            "--events-db",
            str(tmp_path / "events.sqlite"),
            *extra_argv,
        ]
    )
    return config


def test_adapter_fake_forces_retrieval_fake_mode(monkeypatch, tmp_path) -> None:
    config = _run_cli(monkeypatch, tmp_path, ["--adapter", "fake"])

    assert config.retrieval.fake_mode is True
    assert config.retrieval.selected_source_names() == ["fake"]


def test_default_adapter_leaves_retrieval_fake_mode_off(monkeypatch, tmp_path) -> None:
    config = _run_cli(monkeypatch, tmp_path, [])

    assert config.retrieval.fake_mode is False
    assert "fake" not in config.retrieval.selected_source_names()


def test_explicit_real_adapter_leaves_retrieval_fake_mode_off(
    monkeypatch, tmp_path
) -> None:
    config = _run_cli(monkeypatch, tmp_path, ["--adapter", "real"])

    assert config.retrieval.fake_mode is False


def test_orchestration_args_default_adapter_is_real() -> None:
    assert cli.OrchestrationArgs(claim="claim").adapter == "real"


def test_removed_camel_adapter_value_is_rejected(capsys) -> None:
    """``camel`` named the deleted debate adapter, so it is no longer accepted."""

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--claim", "claim", "--adapter", "camel"])

    assert exc_info.value.code == 2
    assert "invalid choice: 'camel'" in capsys.readouterr().err
