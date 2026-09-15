"""The standalone Research Synthesist driver's confirmation hooks.

Interactive mode prints the surfaced ranking and reads the owner's selection; the auto modes apply
the profile's ConfirmPolicy. All hermetic — stdin/stdout are injected, the ScoredCandidates are real.
The live run assembly (``run_synthesist``) is exercised by the gated live end-to-end test.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _scored(cid, *, field_novelty=0.7, saturation=0.25, cross_concept=True, common_sense=False):
    from src.cycles.hypothesis import HypothesisCandidate, ScoredCandidate
    from src.delta import build_node

    candidate = HypothesisCandidate(
        candidate_id=cid, new_nodes=(build_node(label=f"node-{cid}"),),
        novelty_graded=field_novelty, saturation=saturation,
        cross_concept=cross_concept, common_sense=common_sense,
    )
    hyp_score = 0.5
    return ScoredCandidate(candidate=candidate, hyp_score=hyp_score, rank_score=hyp_score * (1 - saturation))


def test_interactive_confirm_fn_parses_a_comma_separated_id_list():  #  interactive
    from src.synthesist_run import interactive_confirm_fn

    fn = interactive_confirm_fn(input_fn=lambda _prompt: "h5, h6", output_fn=lambda _s: None)
    assert fn([_scored("h5"), _scored("h6"), _scored("h3")]) == ["h5", "h6"]


def test_interactive_confirm_fn_handles_all_and_none():
    from src.synthesist_run import interactive_confirm_fn

    surfaced = [_scored("h5"), _scored("h6")]
    assert interactive_confirm_fn(input_fn=lambda _p: "all", output_fn=lambda _s: None)(surfaced) == ["h5", "h6"]
    assert interactive_confirm_fn(input_fn=lambda _p: "none", output_fn=lambda _s: None)(surfaced) == []
    assert interactive_confirm_fn(input_fn=lambda _p: "", output_fn=lambda _s: None)(surfaced) == []


def test_interactive_confirm_fn_shows_the_ranking_to_the_owner():
    from src.synthesist_run import interactive_confirm_fn

    shown: list[str] = []
    interactive_confirm_fn(input_fn=lambda _p: "none", output_fn=shown.append)([_scored("h5")])
    assert any("h5" in line for line in shown)  # the ranking was printed before the prompt


def test_confirm_fn_for_threshold_policy_filters_by_scores():  #  auto-policy
    from src.synthesist_run import confirm_fn_for
    from src.research_profile import ConfirmPolicy

    surfaced = [_scored("h5", field_novelty=0.70, saturation=0.25),
                _scored("h3", field_novelty=0.55, saturation=0.42)]
    fn = confirm_fn_for(ConfirmPolicy(mode="threshold", min_field_novelty=0.6, max_saturation=0.35))
    assert fn(surfaced) == ["h5"]  # h3 fails (novelty 0.55 < 0.6, saturation 0.42 > 0.35)


def test_confirm_fn_for_all_policy_confirms_everything():
    from src.synthesist_run import confirm_fn_for
    from src.research_profile import ConfirmPolicy

    fn = confirm_fn_for(ConfirmPolicy(mode="all"))
    assert fn([_scored("h5"), _scored("h6")]) == ["h5", "h6"]


def test_evidence_to_records_adapts_retrieved_evidence():  # corpus-curation pool adaptation
    from src.synthesist_run import evidence_to_records
    from src.state import RetrievedEvidence

    ev = RetrievedEvidence(
        evidence_id="ev-1", source="openalex", source_id="s1", title="A study",
        quote="the body snippet", relevance="high", retrieved_by="r", tool_call_id="t", rank=1,
        trust_tier="green",
    )
    records = evidence_to_records([ev])
    assert records[0]["evidence_id"] == "ev-1"
    assert records[0]["title"] == "A study"
    assert records[0]["body"] == "the body snippet"


def test_passages_from_evidence_dedupes_quotes_for_claimless_mining():  # claimless miner passages
    from src.synthesist_run import passages_from_evidence
    from src.state import RetrievedEvidence

    def ev(eid, quote):
        return RetrievedEvidence(
            evidence_id=eid, source="exa", source_id=eid, title="t", quote=quote,
            relevance="high", retrieved_by="r", tool_call_id=eid, rank=1, trust_tier="green",
        )

    # claimless mode schedules no verification work, so the miner's passages come straight from the
    # retrieved pool: deduped, empties dropped, retrieval order preserved.
    passages = passages_from_evidence(
        [ev("e1", "alpha"), ev("e2", "alpha"), ev("e3", "beta"), ev("e4", "")]
    )
    assert passages == ["alpha", "beta"]


def test_passages_from_evidence_does_not_truncate_long_quotes():  # byte-identical regression:
    # Unlike ``run_path._passages_from_work``, claimless mining is deliberately uncapped;
    # this preserves the known passage-length asymmetry.
    from src.synthesist_run import passages_from_evidence
    from src.state import RetrievedEvidence

    long_quote = "q" * 5000
    ev = RetrievedEvidence(
        evidence_id="e1", source="exa", source_id="e1", title="t", quote=long_quote,
        relevance="high", retrieved_by="r", tool_call_id="e1", rank=1, trust_tier="green",
    )
    passages = passages_from_evidence([ev])
    assert passages == [long_quote]
    assert len(passages[0]) == 5000


def test_node_definitions_maps_label_to_universal_definition():  # B: feed defs to the elaborator
    from src.synthesist_run import node_definitions

    graph_json = {"nodes": {
        "n1": {"label": "LLC threshold", "definition": "a loss-landscape degeneracy measure"},
        "n2": {"label": "", "definition": "dropped (no label)"},
    }}
    assert node_definitions(graph_json) == {"LLC threshold": "a loss-landscape degeneracy measure"}


def test_run_synthesist_builds_the_planner_with_the_direct_backend_resolver(
    tmp_path, monkeypatch
):
    import src.camel_adapter as camel_adapter
    import src.synthesist_run as synthesist_run
    from src.config import load_config
    from src.log_store import SQLiteLogStore
    from src.research_profile import ResearchProfile

    sentinel_backend = object()
    captured = {}

    def fake_direct_backend(agent_config, *, role_name, **kwargs):
        del agent_config, kwargs
        captured["role_name"] = role_name
        return sentinel_backend

    class FakePlanner:
        def __init__(self, backend):
            captured["backend"] = backend

    class StopAfterPlanner(Exception):
        pass

    monkeypatch.setattr(
        camel_adapter, "_create_direct_model_backend", fake_direct_backend
    )
    monkeypatch.setattr(
        synthesist_run,
        "load_config",
        lambda _path: load_config(Path("config/evidence-evaluation.yaml")),
    )
    monkeypatch.setattr(
        synthesist_run, "_preflight_model_credentials", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        synthesist_run,
        "load_research_profile",
        lambda _path: ResearchProfile(claim="a claim"),
    )
    monkeypatch.setattr(synthesist_run, "default_merge_embedder", lambda: object())
    monkeypatch.setattr(synthesist_run, "LLMRetrievalPlanner", FakePlanner)
    def stop_after_planner(*args, **kwargs):
        del args, kwargs
        raise StopAfterPlanner

    monkeypatch.setattr(
        synthesist_run, "retrieve_claim_evidence_planned", stop_after_planner
    )

    store = SQLiteLogStore(tmp_path / "events.sqlite")
    with pytest.raises(StopAfterPlanner):
        synthesist_run.run_synthesist(
            profile_path="profile.yaml",
            config_path="config.yaml",
            log_store=store,
            run_id="run",
            thread_id="thread",
        )

    assert captured == {
        "role_name": "retrieval_planner",
        "backend": sentinel_backend,
    }


def test_run_synthesist_preflights_required_codex_before_planner(
    tmp_path,
    monkeypatch,
):
    import src.synthesist_run as synthesist_run
    from src.config import load_config
    from src.log_store import SQLiteLogStore

    config = load_config(Path("config/evidence-evaluation.yaml")).model_copy(
        deep=True
    )
    config.retrieval.sources = ["codex_web"]
    config.retrieval.optional_sources = []
    config.retrieval.source_limits.codex_web.model_id = "gpt-test"
    monkeypatch.setattr(synthesist_run, "load_config", lambda _path: config)
    monkeypatch.setattr(
        "src.retrieval.preflight.resolve_codex_executable",
        lambda: (_ for _ in ()).throw(RuntimeError("not found")),
    )
    monkeypatch.setattr(
        synthesist_run,
        "load_research_profile",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("profile/planner work should not start")
        ),
    )

    with pytest.raises(RuntimeError, match="requires an executable Codex CLI"):
        synthesist_run.run_synthesist(
            profile_path="profile.yaml",
            config_path="config.yaml",
            log_store=SQLiteLogStore(tmp_path / "events.sqlite"),
            run_id="run",
            thread_id="thread",
        )


@pytest.mark.parametrize(
    ("k_judges", "trace_enabled", "elaborate", "reader_lexicon", "extra_roles"),
    [
        (2, False, True, object(), set()),
        (
            3,
            True,
            True,
            object(),
            {
                "skeptical_verifier",
                "elaboration_writer",
                "reader_translator",
                "translation_verifier",
            },
        ),
    ],
)
def test_run_synthesist_preflights_only_backends_it_will_construct(
    tmp_path,
    monkeypatch,
    k_judges,
    trace_enabled,
    elaborate,
    reader_lexicon,
    extra_roles,
):
    import src.synthesist_run as synthesist_run
    from src.log_store import SQLiteLogStore

    class StopAfterPreflight(Exception):
        pass

    captured: list[str] = []
    profile = SimpleNamespace(
        k_judges=k_judges,
        reader_lexicon=reader_lexicon,
        # Read by the authored-priority preflight that runs just before this one.
        concepts=[],
        priority_author="",
    )
    monkeypatch.setattr(synthesist_run, "load_config", lambda _path: object())
    monkeypatch.setattr(
        synthesist_run, "preflight_required_retrieval", lambda _config: None
    )
    monkeypatch.setattr(
        synthesist_run, "load_research_profile", lambda _path: profile
    )

    def capture_roles(_config, roles):
        captured.extend(roles)
        raise StopAfterPreflight

    monkeypatch.setattr(
        synthesist_run, "_preflight_model_credentials", capture_roles
    )

    with pytest.raises(StopAfterPreflight):
        synthesist_run.run_synthesist(
            profile_path="profile.yaml",
            config_path="config.yaml",
            log_store=SQLiteLogStore(tmp_path / "events.sqlite"),
            run_id="run",
            thread_id="thread",
            trace_dir=(tmp_path / "trace") if trace_enabled else None,
            elaborate=elaborate,
        )

    required_roles = {
        "builder",
        "evidence_reviewer",
        "research_synthesist",
        "critic_panel",
        "experiment_designer",
        "experiment_validator",
    }
    assert set(captured) == required_roles | extra_roles


@pytest.mark.parametrize("prose_options", [{}, {"elaborate": False}])
def test_run_synthesist_threads_exact_audit_identity_to_connected_renderer(
    tmp_path, monkeypatch, prose_options
):
    import src.graph_state_runtime as graph_state_runtime
    import src.synthesist_run as synthesist_run
    from src.log_store import SQLiteLogStore
    from src.research_profile import ResearchProfile

    profile = ResearchProfile(claim="a claim", confirm={"mode": "all"})
    result = SimpleNamespace(surfaced=[SimpleNamespace(candidate_id="h1")], graph_json={})
    deps = SimpleNamespace(
        extractor=object(),
        embedder=object(),
        retrieve_for_target=object(),
        evidence_reviewer=object(),
        research_synthesist=object(),
        critic_panel=object(),
        experiment_designer=object(),
        experiment_validator=object(),
        experiment_refine_rounds=0,
        retrieve_experiment_methods=object(),
        retrieval_open_risks=(),
        retrieval_open_risks_provider=lambda: (),
    )
    captured = {}
    saved_cards = []
    cards = {"h1": {"headline": "Calibration may preserve factual accuracy."}}
    prose_calls = []

    monkeypatch.setattr(synthesist_run, "load_config", lambda _path: object())
    monkeypatch.setattr(synthesist_run, "preflight_required_retrieval", lambda _config: None)
    monkeypatch.setattr(
        synthesist_run, "_preflight_model_credentials", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(synthesist_run, "load_research_profile", lambda _path: profile)
    monkeypatch.setattr(synthesist_run, "default_merge_embedder", lambda: object())
    monkeypatch.setattr(graph_state_runtime, "logged_backend", lambda *args, **kwargs: object())
    monkeypatch.setattr(synthesist_run, "LLMRetrievalPlanner", lambda _backend: object())
    monkeypatch.setattr(synthesist_run, "retrieve_claim_evidence_planned", lambda *args, **kwargs: [])
    monkeypatch.setattr(synthesist_run, "corpus_paper_list", lambda _records: [])
    monkeypatch.setattr(synthesist_run, "build_graph_state_deps", lambda *args, **kwargs: deps)
    monkeypatch.setattr(synthesist_run, "emphasis_reranker", lambda *args, **kwargs: object())
    monkeypatch.setattr(synthesist_run, "run_graph_state_workflow", lambda *args, **kwargs: result)
    def fake_elaborate(writer, surfaced, **kwargs):
        prose_calls.append(surfaced)
        return cards

    monkeypatch.setattr(synthesist_run, "elaborate_all", fake_elaborate)
    monkeypatch.setattr(
        synthesist_run, "write_trace",
        lambda *args, **kwargs: saved_cards.append(kwargs.get("elaborations")) or [],
    )

    def fake_render(run_root, **kwargs):
        assert saved_cards == [cards if prose_options.get("elaborate", True) else None]
        captured["run_root"] = run_root
        captured.update(kwargs)
        return []

    monkeypatch.setattr(synthesist_run, "render_connected_after_trace", fake_render)

    events_db = tmp_path / "events.sqlite"
    export_dir = tmp_path / "artifacts"
    returned = synthesist_run.run_synthesist(
        profile_path="profile.yaml",
        config_path="config.yaml",
        log_store=SQLiteLogStore(events_db),
        run_id="logical-run-id",
        thread_id="thread",
        export_dir=export_dir,
        trace_dir=export_dir / "trace",
        **prose_options,
    )

    assert returned is result
    assert len(prose_calls) == (0 if prose_options else 1)
    assert captured == {
        "run_root": export_dir,
        "enabled": True,
        "run_id": "logical-run-id",
        "events_db": events_db,
    }


def test_synthesist_cli_generates_a_fresh_default_run_id(
    tmp_path,
    monkeypatch,
) -> None:
    import src.synthesist_run as synthesist_run

    captured = {}

    def fake_run_synthesist(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(edge_table=[], version=1, surfaced=[])

    monkeypatch.setattr(synthesist_run, "run_synthesist", fake_run_synthesist)
    monkeypatch.setattr(synthesist_run, "uuid4", lambda: "fresh-run-id")

    result = synthesist_run.main(
        [
            "--profile",
            str(tmp_path / "profile.yaml"),
            "--config",
            str(tmp_path / "config.yaml"),
            "--events-db",
            str(tmp_path / "events.sqlite"),
        ]
    )

    assert result == 0
    assert captured["run_id"] == "fresh-run-id"
    assert captured["thread_id"] == "fresh-run-id"


def test_synthesist_cli_reports_preflight_errors_without_traceback(
    tmp_path, monkeypatch, capsys
) -> None:
    import src.synthesist_run as synthesist_run

    monkeypatch.setattr(
        synthesist_run,
        "run_synthesist",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("missing model provider API keys")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        synthesist_run.main(
            [
                "--profile",
                str(tmp_path / "profile.yaml"),
                "--events-db",
                str(tmp_path / "events.sqlite"),
            ]
        )

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert "missing model provider API keys" in error
    assert "Traceback" not in error


def test_authored_priority_preflight_stops_before_retrieval(tmp_path, monkeypatch):
    # Weighted `concepts` with a blank `priority_author` cannot build a valid
    # UserPriorityAnnotation, so the run aborts at the priority stage -- after
    # planned retrieval and evidence review. Catch it while the profile is still
    # being read, before any source is called.
    import src.synthesist_run as synthesist_run
    from src.config import load_config
    from src.log_store import SQLiteLogStore
    from src.research_profile import ResearchProfile

    config = load_config(Path("config/evidence-evaluation.yaml")).model_copy(deep=True)
    for role in ("builder", "skeptical_verifier"):
        getattr(config.agents, role).model.model_id = "test-model"
    monkeypatch.setattr(synthesist_run, "load_config", lambda _path: config)
    monkeypatch.setattr(
        synthesist_run,
        "load_research_profile",
        lambda _path: ResearchProfile(
            claim="a claim", concepts=[{"term": "structured pruning", "weight": 0.9}]
        ),
    )

    def fail(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("retrieval must not start")

    monkeypatch.setattr(synthesist_run, "retrieve_claim_evidence_planned", fail)

    store = SQLiteLogStore(tmp_path / "events.sqlite")
    with pytest.raises(ValueError) as exc_info:
        synthesist_run.run_synthesist(
            profile_path="profile.yaml",
            config_path="config.yaml",
            log_store=store,
            run_id="run",
            thread_id="thread",
        )

    message = str(exc_info.value)
    assert "priority_author" in message
    assert "structured pruning" in message
