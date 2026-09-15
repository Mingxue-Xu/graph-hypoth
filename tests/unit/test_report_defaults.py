"""Reader prose is enabled consistently through both command-line entry points."""

from types import SimpleNamespace

import pytest

from src import synthesist_run


@pytest.mark.parametrize("flags,enabled", [([], True), (["--elaborate"], True), (["--no-elaborate"], False)])
def test_example_runner_forwards_prose_choice(tmp_path, monkeypatch, flags, enabled):
    from scripts.run_example import main

    config = tmp_path / "config.yaml"
    config.touch()
    calls = []
    monkeypatch.setattr(synthesist_run, "main", lambda args: calls.append(args) or 0)
    assert main(["--config", str(config), *flags]) == 0
    assert ("--elaborate" in calls[0]) is enabled
    assert ("--no-elaborate" in calls[0]) is (not enabled)


@pytest.mark.parametrize("flags,enabled", [([], True), (["--elaborate"], True), (["--no-elaborate"], False)])
def test_synthesist_cli_prose_choice(tmp_path, monkeypatch, flags, enabled):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(edge_table=[], version=1, surfaced=[])

    monkeypatch.setattr(synthesist_run, "run_synthesist", run)
    assert synthesist_run.main([
        "--profile", "profile.yaml", "--events-db", str(tmp_path / "events.sqlite"),
        "--trace-dir", str(tmp_path / "trace"), *flags,
    ]) == 0
    assert calls[0]["elaborate"] is enabled
