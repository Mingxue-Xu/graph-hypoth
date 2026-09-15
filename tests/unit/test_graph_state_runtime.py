"""Standalone Research Synthesist wiring into ``GraphStateDeps``.

The real backend creation is live-only; here we inject a fake ``backend_factory`` so the wiring
(which role each seam is built on, how many judges, the steering threaded through) is deterministic
and hermetic — no network, no credentials.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.events import EventType
from src.log_store import SQLiteLogStore
from src.research_profile import ResearchProfile
from src.retrieval.planner import RetrievalPlan
from tests.support import openrouter_provider_config


def _config():
    return openrouter_provider_config()


def _embed(texts):
    return [[1.0] for _ in texts]


def test_retrieve_claim_evidence_planned_fans_out_dedupes_and_fuses(tmp_path) -> None:
    # The planned path runs every sub-query into ONE ledger (identity dedup across
    # plans), fuses ranks via RRF, and scores claim-relevance. Hermetic via fake_mode.
    from src.graph_state_runtime import retrieve_claim_evidence_planned

    config = openrouter_provider_config()
    config.retrieval.fake_mode = True
    config.retrieval.final_top_k = 5
    config.retrieval.per_source_top_k = 5
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()

    class _Planner:
        def plan(self, profile):
            return [
                RetrievalPlan(query=profile.claim),
                RetrievalPlan(query=f"{profile.claim} extra"),
            ]

    profile = ResearchProfile(claim="evidence grounded debate")
    pool = retrieve_claim_evidence_planned(
        config, profile, log_store=store, run_id="r", thread_id="t",
        planner=_Planner(), embedder=_embed,
    )

    assert pool
    titles = [item.title for item in pool]
    assert len(titles) == len(set(titles))  # deduped across the two plans
    assert any("rrf_score" in (item.metadata or {}) for item in pool)
    assert all("claim_relevance" in (item.metadata or {}) for item in pool)
    tool_events = [
        event for event in store.list_events("r") if event.event_type == EventType.TOOL_EVENT
    ]
    assert [event.structured_payload["query"] for event in tool_events] == [
        profile.claim,
        f"{profile.claim} extra",
    ]
    assert all(event.sender_role == "retrieve_evidence" for event in tool_events)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {
                "source_statuses": [
                    {"source": "fake", "status": "failed", "errors": ["forced failure"]}
                ],
                "evidence": [],
            },
            ("retrieval failed for all selected sources",),
        ),
        (
            {
                "source_statuses": [{"source": "fake", "status": "success"}],
                "evidence": [],
            },
            ("retrieval returned no evidence for selected sources",),
        ),
        (
            {
                "source_statuses": [
                    {
                        "source": "fake",
                        "status": "partial_failure",
                        "warnings": ["truncated response"],
                    }
                ],
                "evidence": [{"evidence_id": "ev-1", "metadata": {}}],
            },
            ("retrieval partial failure source fake: truncated response",),
        ),
        (
            {
                "source_statuses": [
                    {"source": "fake", "status": "success"},
                    {
                        "source": "exa",
                        "status": "skipped",
                        "warnings": ["missing EXA_API_KEY"],
                    },
                ],
                "evidence": [{"evidence_id": "ev-1", "metadata": {}}],
            },
            ("retrieval skipped source exa: missing EXA_API_KEY",),
        ),
        (
            {
                "source_statuses": [{"source": "exa", "status": "success"}],
                "evidence": [
                    {
                        "evidence_id": "ev-1",
                        "metadata": {
                            "quote_selection": {
                                "candidate_verification_status": "rejected",
                                "verification_status": "summary_fallback",
                            }
                        },
                    }
                ],
            },
            (
                "retrieval candidate excerpt rejected for ev-1",
                "retrieval quote summary_fallback for ev-1",
            ),
        ),
    ],
)
def test_retrieval_diagnostics_become_current_run_risks(payload, expected) -> None:
    from src.graph_state_runtime import _retrieval_open_risks

    assert _retrieval_open_risks([payload]) == expected


def test_build_synthesist_seams_constructs_the_seven_seam_seams():  # standalone wiring
    from src.cycles.experiment import ExperimentDesigner, ExperimentValidator
    from src.cycles.panel import LLMCriticPanelJudge
    from src.graph_state_runtime import SynthesistBuildSpec, build_synthesist_seams

    calls = []

    def fake_factory(agent_config, *, role_name):
        calls.append(role_name)
        return object()  # sentinel backend; never .run in this test

    spec = SynthesistBuildSpec(
        judge_reference_field="LLM compression", judge_corpus="P1: a paper", k_judges=3,
    )
    seams = build_synthesist_seams(
        _config(), claim="c", embedder=_embed, spec=spec, backend_factory=fake_factory
    )
    # The Research Synthesist and Experiment Designer are per-use factories;
    # Factories are lazy; the three-judge Critic Panel and Experiment Validator are
    # eager (reused across the run). No legacy seams remain.
    assert set(seams) == {"research_synthesist", "critic_panel", "experiment_designer", "experiment_validator"}
    assert callable(seams["research_synthesist"]) and callable(seams["experiment_designer"])
    assert len(seams["critic_panel"]) == 3
    assert all(isinstance(j, LLMCriticPanelJudge) for j in seams["critic_panel"])
    assert isinstance(seams["experiment_validator"], ExperimentValidator)
    assert isinstance(seams["experiment_designer"](), ExperimentDesigner)  # invoke the Experiment Designer factory
    # eager seams call their role backends at build time (three judges, then Experiment Validator); the two
    # factories call their role lazily (experiment_designer's call above appends last).
    assert calls == ["builder", "critic_panel", "skeptical_verifier", "experiment_validator", "experiment_designer"]


def test_build_synthesist_seams_honors_k_judges_panel_size():  # DESIGN §4 hyperparameter
    from src.graph_state_runtime import SynthesistBuildSpec, build_synthesist_seams

    calls = []

    def fake_factory(agent_config, *, role_name):
        calls.append(role_name)
        return object()

    spec = SynthesistBuildSpec(judge_reference_field="LLM compression", k_judges=2)
    seams = build_synthesist_seams(
        _config(), claim="c", embedder=_embed, spec=spec, backend_factory=fake_factory
    )
    assert len(seams["critic_panel"]) == 2          # k_judges=2 -> 2 base judges, no tie-breaker
    # two panel backends plus the eager Experiment Validator (other factories build lazily)
    assert calls == ["builder", "critic_panel", "experiment_validator"]


def test_build_synthesist_seams_rejects_k_judges_below_two():  # no silent one-judge downgrade
    from src.graph_state_runtime import SynthesistBuildSpec, build_synthesist_seams

    def fake_factory(agent_config, *, role_name):
        return object()

    spec = SynthesistBuildSpec(judge_reference_field="LLM compression", k_judges=1)
    with pytest.raises(ValueError, match="k_judges"):
        build_synthesist_seams(
            _config(), claim="c", embedder=_embed, spec=spec, backend_factory=fake_factory
        )


def test_build_synthesist_seams_threads_profile_steering_into_synthesist():  # DESIGN §4 inherited hooks
    from src.graph_state_runtime import SynthesistBuildSpec, build_synthesist_seams

    def fake_factory(agent_config, *, role_name):
        return object()

    spec = SynthesistBuildSpec(
        miner_focus="mine tensor-train compression effects",
        judgeability="must be testable", venue_preference="ICML",
    )
    seams = build_synthesist_seams(
        _config(), claim="c", embedder=_embed, spec=spec, backend_factory=fake_factory
    )
    synth = seams["research_synthesist"]()  # invoke the FACTORY to build the thread
    assert synth._miner_focus == "mine tensor-train compression effects"  # miner_focus ->
    assert "must be testable" in synth._propose_steering and "ICML" in synth._propose_steering  # ->


def test_proposer_steering_appends_venue_as_soft_preference():  # soft venue preference
    from src.graph_state_runtime import _proposer_steering

    combined = _proposer_steering("must be testable", "CVPR, ACL")
    assert "must be testable" in combined
    assert "CVPR, ACL" in combined
    assert "not a hard constraint" in combined.lower()
    # venue empty -> just the judgeability; both empty -> None (no steering injected).
    assert _proposer_steering("must be testable", "") == "must be testable"
    assert _proposer_steering("", "") is None


def test_build_synthesist_seams_anchors_the_synthesist_on_the_lens():  # claimless scope anchor
    from src.graph_state_runtime import SynthesistBuildSpec, build_synthesist_seams

    def fake_factory(agent_config, *, role_name):
        return object()

    seams = build_synthesist_seams(
        _config(), claim="", anchor="tensor factorization for LLM compression",
        embedder=_embed, spec=SynthesistBuildSpec(judge_reference_field="LLM compression"),
        backend_factory=fake_factory,
    )
    # The Research Synthesist's scope anchor is the lens, not the empty claim.
    assert seams["research_synthesist"]()._claim == "tensor factorization for LLM compression"


def test_build_synthesist_seams_anchor_defaults_to_claim():  # claim-present fallback
    from src.graph_state_runtime import SynthesistBuildSpec, build_synthesist_seams

    def fake_factory(agent_config, *, role_name):
        return object()

    seams = build_synthesist_seams(
        _config(), claim="the seed claim", embedder=_embed,
        spec=SynthesistBuildSpec(), backend_factory=fake_factory,
    )
    assert seams["research_synthesist"]()._claim == "the seed claim"  # anchor defaults to the claim


def test_make_experiment_methods_retriever_uses_called_by_experiment_design(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import src.retrieval.service as service_mod
    from src import graph_config_defaults as gcd
    from src.graph_state_runtime import make_experiment_methods_retriever
    from src.state import RetrievedEvidence

    captured = {}

    class _FakeService:
        def __init__(self, **kwargs):
            captured["rerank_embedder"] = kwargs.get("rerank_embedder")

        def search_papers(self, *, run_id, query, sources, limit, filters, called_by, tool_call_id):
            captured.update(
                run_id=run_id, query=query, limit=limit, called_by=called_by, tool_call_id=tool_call_id
            )
            return SimpleNamespace(evidence=[{
                "evidence_id": "ev-m1", "source": "exa", "source_id": None,
                "title": "Ablation Benchmark", "authors": [], "published_date": None, "url": None,
                "quote": "we ablate the rank on TruthfulQA",
                "relevance": "Retrieved for query: ablation baselines", "retrieval_method": "search_papers",
                "retrieved_by": called_by, "tool_call_id": tool_call_id, "score": None, "rank": 1,
                "trust_tier": "web_research_paper", "redacted": False,
            }])

        def result_to_dict(self, result):
            return {
                "tool_call_id": captured["tool_call_id"],
                "query": captured["query"],
                "filters": {},
                "sources": ["exa"],
                "evidence": result.evidence,
                "source_statuses": [
                    {
                        "source": "exa",
                        "status": "success",
                        "source_query": captured["query"],
                        "result_count": len(result.evidence),
                        "warnings": [],
                        "errors": [],
                        "metadata": {},
                    }
                ],
                "warnings": [],
                "errors": [],
                "elapsed_ms": 1,
                "input_hash": "input-hash",
                "retrieval_config_hash": "config-hash",
            }

    monkeypatch.setattr(service_mod, "RetrievalService", _FakeService)
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    risks = []
    retrieve = make_experiment_methods_retriever(
        _config(), store, "run-x", "thread-x", embedder=_embed,
        retrieval_risks=risks,
    )
    items = retrieve("ablation baselines for compression")

    assert captured["called_by"] == "experiment_design"           # its own audit attribution (not term_audit)
    assert captured["query"] == "ablation baselines for compression" and captured["run_id"] == "run-x"
    assert captured["limit"] == gcd.EXPERIMENT_METHODS_QUOTES      # knob-derived, not a hard-coded literal
    assert captured["rerank_embedder"] is _embed                 # full-pool relevance reranker
    assert items and isinstance(items[0], RetrievedEvidence)       # normalized to the pool shape
    assert items[0].quote == "we ablate the rank on TruthfulQA"
    events = store.list_events("run-x")
    assert len(events) == 1
    assert events[0].event_type == EventType.TOOL_EVENT
    assert events[0].checkpoint_id == "experiment_methods"
    assert events[0].node_name == "experiment_design"
    assert events[0].sender_role == "experiment_design"
    assert events[0].structured_payload["query"] == captured["query"]
    assert risks == []


def test_graph_state_deps_experiment_fields_default_empty():  # backward-compat: evidence-only deps
    from src.graph_state_runtime import GraphStateDeps

    deps = GraphStateDeps(
        extractor=object(), embedder=None, evidence_reviewer=object(),
        retrieve_for_target=lambda _t: [],
    )
    assert deps.experiment_designer is None
    assert deps.experiment_validator is None
    assert deps.experiment_refine_rounds == 0
    assert deps.retrieve_experiment_methods is None
    assert deps.retrieval_open_risks == ()
    assert deps.retrieval_open_risks_provider is None


def test_build_graph_state_deps_wires_the_experiment_stage(tmp_path, monkeypatch):
    import src.graph_state_runtime as gsr
    from src.graph_state_runtime import SynthesistBuildSpec, build_graph_state_deps

    captured = {}

    monkeypatch.setattr(gsr, "default_merge_embedder", lambda: _embed)

    def fake_methods_retriever(
        config,
        log_store,
        run_id,
        thread_id,
        *,
        embedder=None,
        retrieval_risks=None,
    ):
        del config, log_store, run_id, thread_id
        captured["embedder"] = embedder
        captured["retrieval_risks"] = retrieval_risks
        return lambda _query: []

    monkeypatch.setattr(gsr, "make_experiment_methods_retriever", fake_methods_retriever)

    def fake_factory(agent_config, *, role_name):
        return object()

    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    deps = build_graph_state_deps(
        _config(), claim="c", log_store=store, run_id="r", thread_id="t",
        synthesist=SynthesistBuildSpec(experiment_refine_rounds=2),
        backend_factory=fake_factory, evidence=[],
    )
    assert deps.experiment_refine_rounds == 2         # threaded from the profile spec
    assert callable(deps.experiment_designer)         # Experiment Designer factory injected
    assert deps.experiment_validator is not None      # Experiment Validator injected
    assert callable(deps.retrieve_experiment_methods)  # methods retriever injected
    assert captured["embedder"] is _embed             # the already-loaded embedder is reused
    assert captured["retrieval_risks"] == []
    assert deps.retrieval_open_risks_provider() == ()


def test_experiment_retrieval_cache_audit_and_risk_reach_graph_result(
    tmp_path, monkeypatch
) -> None:
    from src import graph_state_runtime as gsr
    from src.retrieval import service as service_mod
    from src.graph_state_runtime import SynthesistBuildSpec, build_graph_state_deps
    from src.retrieval.service import RetrievalService
    from src.run_path import run_graph_state_workflow

    config = _config().model_copy(deep=True)
    config.retrieval.sources = ["exa"]
    config.retrieval.optional_sources = []
    config.retrieval.fake_mode = False
    config.retrieval.artifacts.enabled = False
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    service = RetrievalService(config.retrieval, store)
    source_calls = []

    class _FailingSource:
        def search(self, *, query, limit, filters):
            del limit, filters
            source_calls.append(query)
            raise RuntimeError("forced source failure")

    service.sources = {"exa": _FailingSource()}
    monkeypatch.setattr(service_mod, "RetrievalService", lambda **_kwargs: service)
    monkeypatch.setattr(gsr, "default_merge_embedder", lambda: _embed)

    deps = build_graph_state_deps(
        config,
        claim="claim",
        log_store=store,
        run_id="run-integrated",
        thread_id="thread-integrated",
        synthesist=SynthesistBuildSpec(),
        backend_factory=lambda _agent_config, *, role_name: object(),
        evidence=[],
    )
    query = "methods for a falsifiable hypothesis"
    assert deps.retrieve_experiment_methods is not None
    assert deps.retrieve_experiment_methods(query) == []
    assert deps.retrieve_experiment_methods(query) == []

    # The source runs once; the second identical request is a deterministic SQLite replay.
    assert source_calls == [query]
    tool_events = [
        event
        for event in store.list_events("run-integrated")
        if event.event_type == EventType.TOOL_EVENT
    ]
    assert len(tool_events) == 2
    assert tool_events[0].structured_payload["source_statuses"][0]["status"] == "failed"
    assert (
        tool_events[1].structured_payload["source_statuses"][0]["metadata"]["cache"]
        == "hit"
    )

    result = run_graph_state_workflow(
        None,
        anchor="claim",
        extractor=deps.extractor,
        embedder=deps.embedder,
        evidence_reviewer=deps.evidence_reviewer,
        enable_expansion=False,
        initial_open_risks=deps.retrieval_open_risks,
        open_risks_provider=deps.retrieval_open_risks_provider,
    )
    risk = "retrieval failed for all selected sources"
    assert risk in result.open_risks
    assert f"- {risk}" in result.audit_memo


def test_build_graph_state_deps_uses_the_direct_backend_resolver_by_default(
    tmp_path, monkeypatch
):
    import src.camel_adapter as camel_adapter
    import src.graph_state_runtime as gsr
    from src.graph_state_runtime import build_graph_state_deps

    calls = []

    def fake_direct_backend(agent_config, *, role_name, **kwargs):
        del agent_config, kwargs
        calls.append(role_name)
        return object()

    monkeypatch.setattr(
        camel_adapter, "_create_direct_model_backend", fake_direct_backend
    )
    monkeypatch.setattr(gsr, "default_merge_embedder", lambda: None)
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()

    build_graph_state_deps(
        _config(),
        claim="claim",
        log_store=store,
        run_id="run",
        thread_id="thread",
        evidence=[],
    )

    assert calls == ["builder", "evidence_reviewer"]


def test_graph_state_deps_synthesist_fields_default_empty():
    from src.graph_state_runtime import GraphStateDeps

    deps = GraphStateDeps(
        extractor=object(), embedder=None, evidence_reviewer=object(),
        retrieve_for_target=lambda _t: [],
    )
    assert deps.research_synthesist is None
    assert deps.critic_panel == ()


def test_logged_backend_builds_the_role_backend_and_logs_its_own_call(tmp_path, monkeypatch):
    """Chokepoint for the outside-build_graph_state_deps sites (elaboration_writer,
    reader_translator, translation_verifier): one call = one fresh logger, same as today's
    inline ``_install_logging_backend_hook(..., _llm_call_logger(...))`` construction."""
    import src.camel_adapter as camel_adapter
    from src.graph_state_runtime import logged_backend

    class _FakeBackend:
        def __init__(self, content):
            self._content = content

        def run(self, messages, **kwargs):
            return {"choices": [{"message": {"content": self._content}}]}

    def fake_create(agent_config, *, role_name, **kwargs):
        return _FakeBackend(f"resp-for-{role_name}")

    monkeypatch.setattr(camel_adapter, "_create_camel_model_backend", fake_create)
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()

    backend = logged_backend(_config(), "elaboration_writer", store, "run-x", "thread-x")
    backend.run([{"role": "user", "content": "write it"}])

    events = store.list_events("run-x")
    assert len(events) == 1
    assert events[0].sender_role == "elaboration_writer"
    assert events[0].structured_payload["raw_response"] == "resp-for-elaboration_writer"
    assert events[0].structured_payload["request_messages"] == [
        {"role": "user", "content": "write it"}
    ]


def test_logged_backend_gives_each_call_its_own_logger(tmp_path, monkeypatch):
    """Two separate ``logged_backend`` calls, as used by ``synthesist_run.py`` translation
    sites) must each keep their own seq-counter, not share one across seams."""
    import src.camel_adapter as camel_adapter
    from src.graph_state_runtime import logged_backend

    class _FakeBackend:
        def run(self, messages, **kwargs):
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(
        camel_adapter, "_create_camel_model_backend",
        lambda agent_config, *, role_name, **kwargs: _FakeBackend(),
    )
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()

    translator = logged_backend(_config(), "reader_translator", store, "run-y", "thread-y")
    verifier = logged_backend(_config(), "translation_verifier", store, "run-y", "thread-y")
    translator.run([{"role": "user", "content": "1"}])
    verifier.run([{"role": "user", "content": "2"}])

    events = store.list_events("run-y")
    assert len(events) == 2  # both recorded independently, no idempotency-key collision


def test_logged_backend_uses_the_direct_backend_resolver(tmp_path, monkeypatch):
    import src.camel_adapter as camel_adapter
    from src.graph_state_runtime import logged_backend

    calls = []

    class _FakeBackend:
        def run(self, messages, **kwargs):
            del messages, kwargs
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_direct_backend(agent_config, *, role_name, **kwargs):
        del agent_config, kwargs
        calls.append(role_name)
        return _FakeBackend()

    monkeypatch.setattr(
        camel_adapter, "_create_direct_model_backend", fake_direct_backend
    )
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()

    backend = logged_backend(
        _config(), "reader_translator", store, "run", "thread"
    )
    backend.run([{"role": "user", "content": "translate"}])

    assert calls == ["reader_translator"]
    assert len(store.list_events("run")) == 1
