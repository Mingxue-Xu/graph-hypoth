"""Claim-based CLI coverage for the graph-state workflow."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src import cli
from src.cli import GraphStateRunResult, OrchestrationArgs, run_graph_state_orchestration
from src.config import DEFAULT_CONFIG_PATH
from src.cycles import verification as v
from src.cycles.extraction import ClaimExtraction
from src.delta import build_edge, build_node
from src.graph_state_runtime import GraphStateDeps
from src.graph_store import EdgeStatus
from src.run_path import GraphRunResult
from src.state import RetrievedEvidence
from tests.support import openrouter_provider_config, runnable_default_config

pytestmark = pytest.mark.smoke


def _result() -> GraphStateRunResult:
    return GraphStateRunResult(
        run_id="r",
        thread_id="t",
        result=GraphRunResult(
            graph_id="g",
            version=0,
            statuses={},
            receipts=(),
            open_risks=(),
            graph_json={},
            edge_table=(),
            audit_memo="memo",
        ),
        export_dir=None,
    )


def test_orchestration_args_default_config_path() -> None:
    assert OrchestrationArgs(claim="claim").config == DEFAULT_CONFIG_PATH


def test_main_runs_only_the_graph_state_path(monkeypatch, capsys) -> None:
    calls: list[OrchestrationArgs] = []
    notices_before_orchestration: list[str] = []

    def fake_run(args, *, parser=None):
        del parser
        output = capsys.readouterr()
        assert output.out == ""
        notices_before_orchestration.append(output.err)
        calls.append(args)
        return _result()

    monkeypatch.setattr(cli, "run_graph_state_orchestration", fake_run)
    cli.main(["--claim", "smoking increases lung cancer"])

    assert len(calls) == 1
    assert calls[0].claim == "smoking increases lung cancer"
    assert calls[0].config == Path("config/evidence-evaluation.yaml")
    assert "Starting run" in notices_before_orchestration[0]


@pytest.mark.parametrize(
    "removed_flag",
    [
        "--legacy",
        "--checkpoint-db",
        "--review-evidence",
        "--allow-redact-all",
        "--cost-report",
        "--no-cost-report",
        "--reconcile-openrouter",
    ],
)
def test_removed_legacy_flags_are_rejected(removed_flag, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--claim", "claim", removed_flag])

    assert exc_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


_CAUSE = build_node(label="smoking", type="exposure/intervention")
_EFFECT = build_node(label="lung cancer", type="outcome")
_EDGE = build_edge(
    source_node_ids=[_CAUSE.node_id],
    target_node_ids=[_EFFECT.node_id],
    direction="causal",
    relation_type="increases",
)


class _FakeExtractor:
    def extract(self, claim):
        return ClaimExtraction(nodes=(_CAUSE, _EFFECT), edges=(_EDGE,))


class _FakeReviewer:
    def review(self, task, evidences):
        coherence = v.CoherenceSignal(
            entailment={"supports": 1.0},
            contradiction_risk={},
            context_fit=1.0,
            construct_match=1.0,
        )
        methods = v.MethodsSignal(1.0, 1.0, 1.0, 1.0, 0.0)
        return v.EdgeReview(
            coherence={ev.evidence_id: coherence for ev in evidences},
            methods={ev.evidence_id: methods for ev in evidences},
            appraisal=v.VerifierAppraisal(open_risks=(), qualifiers=()),
        )


def _fake_deps() -> GraphStateDeps:
    evidence = [
        RetrievedEvidence(
            evidence_id="ev-1",
            source="openalex",
            source_id="s1",
            title="A cohort study of smoking and lung cancer",
            quote="smoking increases lung cancer risk",
            relevance="high",
            retrieved_by="retriever",
            tool_call_id="tool-ev-1",
            rank=1,
            trust_tier="green",
        )
    ]
    return GraphStateDeps(
        extractor=_FakeExtractor(),
        embedder=lambda texts: [[1.0, 0.0] for _ in texts],
        evidence_reviewer=_FakeReviewer(),
        retrieve_for_target=lambda task: evidence,
    )


def test_run_graph_state_orchestration_runs_end_to_end(tmp_path) -> None:
    args = OrchestrationArgs(
        claim="smoking increases lung cancer",
        run_id="r1",
        events_db=tmp_path / "events.db",
    )
    result = run_graph_state_orchestration(args, deps=_fake_deps())

    assert result.run_id == "r1"
    assert result.result.statuses[_EDGE.edge_id] == EdgeStatus.SUPPORTED.value
    assert result.export_dir is not None
    assert (result.export_dir / "graph.json").is_file()
    assert (result.export_dir / "edge_table.csv").is_file()
    assert (result.export_dir / "audit_memo.md").is_file()


def test_default_ids_share_one_generated_identity(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "uuid4", lambda: "generated-run-id")

    result = run_graph_state_orchestration(
        OrchestrationArgs(
            claim="smoking increases lung cancer",
            events_db=tmp_path / "events.sqlite",
        ),
        deps=_fake_deps(),
    )

    assert result.run_id == "generated-run-id"
    assert result.thread_id == "generated-run-id"


def test_default_events_db_routes_to_runtime_artifact_or_working_directory(
    tmp_path, monkeypatch
) -> None:
    artifact_dir = tmp_path / "runtime-trace"
    monkeypatch.setattr(cli, "runtime_artifact_dir", lambda: artifact_dir)
    assert cli._default_events_db_path() == artifact_dir / "events.sqlite"

    monkeypatch.setattr(cli, "runtime_artifact_dir", lambda: None)
    monkeypatch.chdir(tmp_path)
    assert cli._default_events_db_path() == Path(".graph-hypoth-events.sqlite")


def test_dependency_build_enables_llm_audit_logging(tmp_path, monkeypatch) -> None:
    config = runnable_default_config()
    config.retrieval.enabled = False
    for agent in (config.agents.builder, config.agents.skeptical_verifier):
        assert agent.model is not None
        agent.model.api_key_env = None

    captured = {}

    def fake_build(config, **kwargs):
        del config
        captured.update(kwargs)
        return _fake_deps()

    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(cli, "build_graph_state_deps", fake_build)

    run_graph_state_orchestration(
        OrchestrationArgs(
            claim="smoking increases lung cancer",
            run_id="logged",
            events_db=tmp_path / "events.db",
        )
    )

    assert captured["log_llm_calls"] is True


def test_api_keys_file_is_loaded_before_model_preflight(
    tmp_path, monkeypatch
) -> None:
    config = runnable_default_config()
    config.retrieval.enabled = False
    for agent in (config.agents.builder, config.agents.skeptical_verifier):
        assert agent.model is not None
        agent.model.api_key_env = "GRAPH_HYPOTH_TEST_MODEL_KEY"
    monkeypatch.delenv("GRAPH_HYPOTH_TEST_MODEL_KEY", raising=False)
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(cli, "build_graph_state_deps", lambda *_args, **_kwargs: _fake_deps())
    key_file = tmp_path / "keys.env"
    key_file.write_text(
        "GRAPH_HYPOTH_TEST_MODEL_KEY=from-file\n", encoding="utf-8"
    )

    run_graph_state_orchestration(
        OrchestrationArgs(
            claim="smoking increases lung cancer",
            api_keys_file=key_file,
            events_db=tmp_path / "events.sqlite",
        )
    )

    assert os.environ["GRAPH_HYPOTH_TEST_MODEL_KEY"] == "from-file"


def test_required_retrieval_preflight_stops_before_dependency_build(
    tmp_path, monkeypatch
) -> None:
    config = runnable_default_config()
    for agent in (config.agents.builder, config.agents.skeptical_verifier):
        assert agent.model is not None
        agent.model.api_key_env = None
    config.retrieval.sources = ["exa"]
    config.retrieval.optional_sources = []
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_config", lambda _path: config)

    def fail_build(*_args, **_kwargs):
        raise AssertionError("dependency construction should not start")

    monkeypatch.setattr(cli, "build_graph_state_deps", fail_build)

    with pytest.raises(RuntimeError, match="required retrieval source exa"):
        run_graph_state_orchestration(
            OrchestrationArgs(
                claim="claim",
                events_db=tmp_path / "events.sqlite",
            )
        )


def test_optional_exa_without_a_key_allows_dependency_build(
    tmp_path, monkeypatch
) -> None:
    config = runnable_default_config()
    for agent in (config.agents.builder, config.agents.skeptical_verifier):
        assert agent.model is not None
        agent.model.api_key_env = None
    config.retrieval.sources = ["arxiv"]
    config.retrieval.optional_sources = ["exa"]
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    built: list[bool] = []

    def fake_build(*_args, **_kwargs):
        built.append(True)
        return _fake_deps()

    monkeypatch.setattr(cli, "build_graph_state_deps", fake_build)

    run_graph_state_orchestration(
        OrchestrationArgs(
            claim="claim",
            events_db=tmp_path / "events.sqlite",
        )
    )

    assert built == [True]


def test_required_codex_web_preflight_stops_before_dependency_build(
    tmp_path, monkeypatch
) -> None:
    config = runnable_default_config()
    for agent in (config.agents.builder, config.agents.skeptical_verifier):
        assert agent.model is not None
        agent.model.api_key_env = None
    config.retrieval.sources = ["codex_web"]
    config.retrieval.optional_sources = []
    config.retrieval.source_limits.codex_web.model_id = "gpt-test"
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(
        "src.retrieval.preflight.resolve_codex_executable",
        lambda: (_ for _ in ()).throw(RuntimeError("not found")),
    )
    monkeypatch.setattr(
        cli,
        "build_graph_state_deps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dependency construction should not start")
        ),
    )

    with pytest.raises(RuntimeError, match="requires an executable Codex CLI"):
        run_graph_state_orchestration(
            OrchestrationArgs(
                claim="claim",
                events_db=tmp_path / "events.sqlite",
            )
        )


def test_missing_model_key_exits_cleanly_before_dependency_build(
    tmp_path, monkeypatch, capsys
) -> None:
    # The shipped config runs key-free claude-cli subagents, so this needs a config
    # whose roles actually demand a provider key.
    config = openrouter_provider_config()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cli,
        "build_graph_state_deps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dependency construction should not start")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--claim",
                "claim",
                "--events-db",
                str(tmp_path / "events.sqlite"),
            ]
        )

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert "missing model provider API keys" in error
    assert "Traceback" not in error


def test_cli_reports_removed_config_fields_without_a_traceback(
    tmp_path, capsys
) -> None:
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        DEFAULT_CONFIG_PATH.read_text(encoding="utf-8") + "\ngraph: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--claim", "claim", "--config", str(config_path)])

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert "graph" in error
    assert "Extra inputs are not permitted" in error
    assert "Traceback" not in error


def test_cli_reports_dependency_construction_errors_cleanly(
    tmp_path, monkeypatch, capsys
) -> None:
    config = runnable_default_config()
    config.retrieval.enabled = False
    for agent in (config.agents.builder, config.agents.skeptical_verifier):
        assert agent.model is not None
        agent.model.api_key_env = None
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cli,
        "build_graph_state_deps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("backend unavailable")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--claim",
                "claim",
                "--events-db",
                str(tmp_path / "events.sqlite"),
            ]
        )

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert "backend unavailable" in error
    assert "Traceback" not in error


def test_cli_reports_workflow_execution_errors_cleanly(
    tmp_path, monkeypatch, capsys
) -> None:
    config = runnable_default_config()
    config.retrieval.enabled = False
    for agent in (config.agents.builder, config.agents.skeptical_verifier):
        assert agent.model is not None
        agent.model.api_key_env = None
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cli, "build_graph_state_deps", lambda *_args, **_kwargs: _fake_deps()
    )
    monkeypatch.setattr(
        cli,
        "run_graph_state_workflow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("extractor backend failed")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--claim",
                "claim",
                "--events-db",
                str(tmp_path / "events.sqlite"),
            ]
        )

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert "extractor backend failed" in error
    assert "Traceback" not in error
