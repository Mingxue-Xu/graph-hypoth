"""``run_synthesist`` accepts an injected confirm hook, which the web app uses to wait for a browser
selection, and otherwise still resolves the profile's own policy. Hermetic: every live seam is
monkeypatched away, the same way the driver's other unit tests do it."""

from __future__ import annotations

from types import SimpleNamespace

from tests.unit.test_synthesist_run import _scored


def test_run_synthesist_prefers_an_injected_confirm_fn_over_the_profile_policy(
    tmp_path, monkeypatch
):
    from src import graph_state_runtime, synthesist_run
    from src.log_store import SQLiteLogStore
    from src.research_profile import ResearchProfile

    profile = ResearchProfile(claim="a claim", confirm={"mode": "all"})
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
    monkeypatch.setattr(synthesist_run, "load_config", lambda _path: object())
    monkeypatch.setattr(synthesist_run, "preflight_required_retrieval", lambda _config: None)
    monkeypatch.setattr(
        synthesist_run, "_preflight_model_credentials", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(synthesist_run, "load_research_profile", lambda _path: profile)
    monkeypatch.setattr(synthesist_run, "default_merge_embedder", lambda: object())
    monkeypatch.setattr(graph_state_runtime, "logged_backend", lambda *args, **kwargs: object())
    monkeypatch.setattr(synthesist_run, "LLMRetrievalPlanner", lambda _backend: object())
    monkeypatch.setattr(
        synthesist_run, "retrieve_claim_evidence_planned", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(synthesist_run, "corpus_paper_list", lambda _records: [])
    monkeypatch.setattr(synthesist_run, "build_graph_state_deps", lambda *args, **kwargs: deps)
    monkeypatch.setattr(synthesist_run, "emphasis_reranker", lambda *args, **kwargs: object())

    def fake_workflow(*args, **kwargs):
        captured["confirm"] = kwargs["expansion_confirm_fn"]
        return SimpleNamespace(surfaced=[], graph_json={})

    monkeypatch.setattr(synthesist_run, "run_graph_state_workflow", fake_workflow)

    def browser_confirm(surfaced):
        return []

    common = {
        "profile_path": "profile.yaml",
        "config_path": "config.yaml",
        "log_store": SQLiteLogStore(tmp_path / "events.sqlite"),
        "run_id": "run",
        "thread_id": "thread",
    }
    synthesist_run.run_synthesist(**common, confirm_fn=browser_confirm)
    assert captured["confirm"] is browser_confirm  # the injected hook passes through untouched

    synthesist_run.run_synthesist(**common)
    assert captured["confirm"] is not browser_confirm
    assert captured["confirm"]([_scored("h5"), _scored("h6")]) == ["h5", "h6"]  # `all` policy
