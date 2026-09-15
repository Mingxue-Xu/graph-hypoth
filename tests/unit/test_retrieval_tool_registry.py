from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src import runtime_trace
from src.config import RetrievalConfig
from src.events import stable_hash_payload
from src.log_store import SQLiteLogStore
from src.retrieval.ledger import EvidenceLedger
from src.retrieval.models import SearchPaperFilters, SourceResult
from src.retrieval.service import RetrievalService
from src.retrieval.tool_registry import (
    RetrievalToolRegistry,
    build_retrieval_stack,
)
from src.state import RetrievedEvidence


@dataclass
class ToolBudgetConfig:
    max_calls_per_run: int | None = 12
    max_calls_per_agent_round: int | None = None


@dataclass
class RegistryConfig:
    tool_budget: ToolBudgetConfig
    sources: list[str] = field(default_factory=lambda: ["arxiv"])
    optional_sources: list[str] = field(default_factory=lambda: ["exa"])

    def selected_source_names(self) -> list[str]:
        return [*self.sources, *self.optional_sources]


class FakeService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.sources = {"arxiv": object(), "exa": object(), "fake": object()}

    def search_papers(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "tool_call_id": kwargs["tool_call_id"],
            "query": kwargs["query"],
            "filters": kwargs["filters"] or {},
            "sources": kwargs["sources"] or ["arxiv"],
            "evidence": [
                {
                    "evidence_id": "ev_000001",
                    "source": "arxiv",
                    "source_id": "useful-paper",
                    "title": "Useful paper",
                    "quote": "Evidence text",
                    "relevance": "Retrieved for testing",
                    "retrieved_by": kwargs["called_by"],
                    "tool_call_id": kwargs["tool_call_id"],
                    "score": 0.5,
                    "rank": 1,
                    "trust_tier": "authoritative_preprint",
                    "redacted": False,
                    "metadata": {"source_rank": 1, "raw_score": 0.5},
                }
            ],
            "source_statuses": [
                {
                    "source": "arxiv",
                    "status": "success",
                    "source_query": kwargs["query"],
                    "result_count": 1,
                    "warnings": [],
                    "errors": [],
                    "unused_filters": [],
                    "metadata": {"costDollars": 0.01},
                }
            ],
            "warnings": ["normalized warning"],
            "errors": [],
            "elapsed_ms": 7,
            "input_hash": "input-hash",
            "retrieval_config_hash": "config-hash",
        }

    def result_to_dict(self, result: dict[str, Any]) -> dict[str, Any]:
        return result


def _ledger() -> EvidenceLedger:
    return EvidenceLedger(
        trust_policy={"arxiv": "authoritative_preprint"},
        trust_tier_priority={"authoritative_preprint": 100},
        source_order=["arxiv"],
    )


def _registry(
    service: FakeService | None = None,
    config: RegistryConfig | None = None,
    *,
    max_calls_per_run: int | None = 12,
    max_calls_per_agent_round: int | None = None,
    ledger: EvidenceLedger | None = None,
) -> RetrievalToolRegistry:
    return RetrievalToolRegistry(
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        round_index=1,
        config=config
        or RegistryConfig(
            ToolBudgetConfig(
                max_calls_per_run=max_calls_per_run,
                max_calls_per_agent_round=max_calls_per_agent_round,
            )
        ),
        service=service or FakeService(),  # type: ignore[arg-type]
        ledger=ledger,
    )


def test_search_papers_returns_result_and_pending_event_shape() -> None:
    service = FakeService()
    registry = _registry(service, RegistryConfig(ToolBudgetConfig()))

    result = registry.search_papers(
        query="retrieval augmented debate",
        sources=["arxiv"],
        limit=2,
        filters={"text": True},
        called_by="builder",
    )
    payloads = registry.drain_pending_event_payloads()

    assert result["tool_call_id"] == "tool_000001"
    assert result["input_hash"] == "input-hash"
    assert result["retrieval_config_hash"] == "config-hash"
    assert result["sources"] == ["arxiv"]
    assert result["source_statuses"][0]["status"] == "success"
    assert result["warnings"] == ["normalized warning"]
    assert result["errors"] == []
    assert result["elapsed_ms"] == 7
    assert len(service.calls) == 1
    assert payloads == [
        {
            "tool_name": "search_papers",
            "tool_call_id": "tool_000001",
            "called_by": "builder",
            "query": "retrieval augmented debate",
            "filters": {"text": True},
            "sources": ["arxiv"],
            "source_queries": {"arxiv": "retrieval augmented debate"},
            "input_hash": "input-hash",
            "retrieval_config_hash": "config-hash",
            "result_count": 1,
            "evidence_ids": ["ev_000001"],
            "evidence_hashes": [stable_hash_payload(result["evidence"][0])],
            "source_statuses": result["source_statuses"],
            "warnings": ["normalized warning"],
            "errors": [],
            "elapsed_ms": 7,
            "cost_dollars": [0.01],
            "verification_status_counts": {},
            "evidence": result["evidence"],
        }
    ]


def test_search_papers_writes_runtime_retrieval_artifacts(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(runtime_trace, "_TRACE_FILE", None)
    monkeypatch.setenv("GRAPH_HYPOTH_RUNTIME_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("GRAPH_HYPOTH_RUNTIME_RUN_LABEL", "registry-artifacts")
    registry = _registry(FakeService(), RegistryConfig(ToolBudgetConfig()))

    result = registry.search_papers(
        query="retrieval augmented debate",
        sources=["arxiv"],
        limit=2,
        filters={"text": True},
        called_by="builder",
    )
    payload = registry.drain_pending_event_payloads()[0]

    artifacts = result["retrieval_artifacts"]
    assert artifacts["json_path"].endswith("tool_000001-search_papers.json")
    assert artifacts["html_path"].endswith("tool_000001-search_papers.html")
    assert payload["retrieval_artifacts"] == artifacts
    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert any((path / "retrieval").exists() for path in run_dirs)


def test_search_papers_returns_structured_error_when_service_rejects_call() -> None:
    class RejectingService(FakeService):
        def search_papers(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            raise ValueError("unsupported retrieval source: openreview.net")

    service = RejectingService()
    registry = _registry(service)

    result = registry.search_papers(
        query="paper query",
        sources=["semantic_scholar"],
        limit=1,
        filters={"text": True},
        called_by="builder",
    )

    assert result["evidence"] == []
    assert result["errors"] == [
        "search_papers failed: unsupported retrieval source: openreview.net"
    ]
    assert registry.drain_pending_event_payloads()[0]["errors"] == result["errors"]


def test_budget_limit_per_run_returns_warning_without_service_call() -> None:
    service = FakeService()
    registry = _registry(
        service,
        RegistryConfig(ToolBudgetConfig(max_calls_per_run=0, max_calls_per_agent_round=2)),
    )

    result = registry.search_papers(
        query="retrieval augmented debate",
        sources=["arxiv"],
        limit=2,
        filters={},
        called_by="builder",
    )

    assert service.calls == []
    # The truncation warning names the tripped cap so callers see why retrieval
    # stopped (not a silent empty).
    assert result["warnings"] == [
        "retrieval tool budget exceeded (max_calls_per_run=0); query skipped"
    ]
    assert result["evidence"] == []
    assert result["tool_call_id"] == "tool_000001"
    assert registry.drain_pending_event_payloads()[0]["called_by"] == "builder"


def test_budget_limit_per_agent_round_returns_warning_without_second_service_call() -> None:
    service = FakeService()
    registry = _registry(
        service,
        RegistryConfig(ToolBudgetConfig(max_calls_per_run=12, max_calls_per_agent_round=1)),
    )

    registry.search_papers(
        query="first query",
        sources=["arxiv"],
        limit=2,
        filters={},
        called_by="builder",
    )
    result = registry.search_papers(
        query="second query",
        sources=["arxiv"],
        limit=2,
        filters={},
        called_by="builder",
    )

    assert len(service.calls) == 1
    assert result["warnings"] == [
        "retrieval tool budget exceeded (max_calls_per_agent_round=1); query skipped"
    ]
    assert result["query"] == "second query"


def test_none_per_agent_round_budget_allows_repeated_same_round_calls() -> None:
    service = FakeService()
    registry = _registry(
        service=service,
        max_calls_per_run=12,
        max_calls_per_agent_round=None,
    )

    for index in range(3):
        result = registry.search_papers(
            query=f"query {index}",
            sources=["arxiv"],
            limit=2,
            filters={},
            called_by="builder",
        )
        assert result["warnings"] == ["normalized warning"]

    assert len(service.calls) == 3


def test_none_run_budget_allows_calls_past_old_run_cap() -> None:
    service = FakeService()
    registry = _registry(
        service=service,
        max_calls_per_run=None,
        max_calls_per_agent_round=None,
    )

    for index in range(13):
        result = registry.search_papers(
            query=f"query {index}",
            sources=["arxiv"],
            limit=2,
            filters={},
            called_by="builder",
        )
        assert result["warnings"] == ["normalized warning"]

    assert len(service.calls) == 13


def test_disabling_per_agent_round_budget_keeps_run_budget() -> None:
    service = FakeService()
    registry = _registry(
        service=service,
        max_calls_per_run=2,
        max_calls_per_agent_round=None,
    )

    registry.search_papers(query="first", called_by="builder")
    registry.search_papers(query="second", called_by="builder")
    result = registry.search_papers(query="third", called_by="builder")

    assert len(service.calls) == 2
    assert result["warnings"] == [
        "retrieval tool budget exceeded (max_calls_per_run=2); query skipped"
    ]


def test_registry_admits_service_evidence_into_run_ledger() -> None:
    service = FakeService()
    ledger = _ledger()
    registry = _registry(service=service, ledger=ledger)
    ledger.add_many(
        results=[],
        query="empty",
        retrieved_by="test",
        tool_call_id="tool_000000",
    )

    result = registry.search_papers(query="registry admission", called_by="builder")
    payload = registry.drain_pending_event_payloads()[0]

    assert result["evidence"][0]["evidence_id"] == "ev_000001"
    assert payload["evidence_ids"] == ["ev_000001"]
    assert payload["evidence"] == result["evidence"]
    assert ledger.all_evidence()[0].retrieved_by == "builder"


def test_registry_admission_appends_after_existing_run_ledger_ids() -> None:
    service = FakeService()
    ledger = _ledger()
    ledger.add_many(
        results=[
            SourceResult(
                source="arxiv",
                source_id="existing",
                title="Existing",
                text="existing quote",
            )
        ],
        query="initial",
        retrieved_by="retrieve_evidence",
        tool_call_id="tool_000000",
    )
    registry = _registry(service=service, ledger=ledger)

    result = registry.search_papers(query="registry admission", called_by="builder")

    assert [item.evidence_id for item in ledger.all_evidence()] == [
        "ev_000001",
        "ev_000002",
    ]
    assert result["evidence"][0]["evidence_id"] == "ev_000002"


def test_per_round_budget_resets_each_round() -> None:
    registry = _registry(max_calls_per_agent_round=1)

    first = registry.search_papers(query="round zero", called_by="builder")
    registry.round_index = 2
    second = registry.search_papers(query="round one", called_by="builder")

    assert first["warnings"] == ["normalized warning"]
    assert second["warnings"] == ["normalized warning"]


def test_per_round_budget_is_separate_per_agent() -> None:
    registry = _registry(max_calls_per_agent_round=1)

    builder = registry.search_papers(query="builder", called_by="builder")
    verifier = registry.search_papers(query="verifier", called_by="verifier")

    assert builder["warnings"] == ["normalized warning"]
    assert verifier["warnings"] == ["normalized warning"]


def test_max_calls_per_agent_round_zero_blocks_all_agent_round_calls() -> None:
    registry = _registry(max_calls_per_agent_round=0)

    result = registry.search_papers(query="blocked", called_by="builder")

    assert result["warnings"] == [
        "retrieval tool budget exceeded (max_calls_per_agent_round=0); query skipped"
    ]


def _fake_retrieval_config() -> RetrievalConfig:
    config = RetrievalConfig()
    config.fake_mode = True
    config.sources = ["fake"]
    config.optional_sources = []
    return config


def test_build_retrieval_stack_wires_service_ledger_and_registry_together(
    tmp_path,
) -> None:
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    config = _fake_retrieval_config()

    service, ledger, registry = build_retrieval_stack(
        config,
        store,
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="retrieve_evidence",
    )

    assert isinstance(service, RetrievalService)
    assert isinstance(ledger, EvidenceLedger)
    assert isinstance(registry, RetrievalToolRegistry)
    assert registry.service is service
    assert registry.ledger is ledger
    assert registry.checkpoint_id == "retrieve_evidence"
    assert registry.round_index == 0

    registry.search_papers(
        query="evidence grounded debate",
        sources=None,
        limit=3,
        filters=SearchPaperFilters(),
        called_by="retrieve_evidence",
    )

    assert ledger.all_evidence()  # the fake source admitted evidence into the shared ledger


def test_build_retrieval_stack_reuses_a_passed_in_service(tmp_path) -> None:
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    config = _fake_retrieval_config()
    existing_service = RetrievalService(config=config, log_store=store)

    service, _ledger, registry = build_retrieval_stack(
        config,
        store,
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="retrieve_evidence",
        service=existing_service,
    )

    assert service is existing_service
    assert registry.service is existing_service


def test_build_retrieval_stack_seeds_ledger_with_existing_evidence(tmp_path) -> None:
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    config = _fake_retrieval_config()
    seed = RetrievedEvidence(
        evidence_id="ev_000042",
        source="fake",
        source_id="seed-paper",
        title="Seed paper",
        quote="Seed quote",
        relevance="seeded before graph-state retrieval",
        retrieved_by="retrieve_evidence",
        tool_call_id="tool_000000",
        score=0.9,
        rank=1,
        trust_tier="deterministic_test",
    )

    _service, ledger, _registry = build_retrieval_stack(
        config,
        store,
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="retrieve_evidence",
        existing=[seed],
    )

    assert [item.evidence_id for item in ledger.all_evidence()] == ["ev_000042"]


def test_build_retrieval_stack_initial_tool_index_defaults_to_one(tmp_path) -> None:
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    config = _fake_retrieval_config()

    _service, _ledger, default_registry = build_retrieval_stack(
        config,
        store,
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="retrieve_evidence",
    )
    _service, _ledger, custom_registry = build_retrieval_stack(
        config,
        store,
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="retrieve_evidence",
        initial_tool_index=7,
    )

    assert default_registry._assign_tool_call_id() == "tool_000001"
    assert custom_registry._assign_tool_call_id() == "tool_000007"
