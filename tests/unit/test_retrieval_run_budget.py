from __future__ import annotations

from types import SimpleNamespace

from src.retrieval.models import SearchPaperFilters
from src.retrieval.sources import ExaPaperSource


def test_exa_observed_cost_accumulates_and_budget_guard_short_circuits(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeExa:
        def __init__(self, api_key: str) -> None:
            del api_key

        def search(self, **kwargs):
            calls.append(kwargs)
            return {
                "results": [{"id": "1", "title": "Paper", "url": "https://e.test"}],
                "cost": {"costDollars": 0.06},
            }

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    monkeypatch.setattr(
        "src.retrieval.sources.Exa",
        FakeExa,
    )
    source = ExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_cost_dollars_per_run=0.10,
            max_qps=1000,
        )
    )

    first = source.search("first", limit=1, filters=SearchPaperFilters())
    second = source.search("second", limit=1, filters=SearchPaperFilters())
    third = source.search("third", limit=1, filters=SearchPaperFilters())

    assert first.status.status == "success"
    assert first.status.metadata["costDollars"] == 0.06
    assert second.status.status == "success"
    assert second.status.metadata["costDollars"] == 0.06
    assert third.status.status == "skipped"
    assert third.status.metadata["costDollars"] == 0.12
    assert len(calls) == 2


def test_exa_cost_extraction_handles_current_response_shape(monkeypatch) -> None:
    class FakeExa:
        def __init__(self, api_key: str) -> None:
            del api_key

        def search(self, **kwargs):
            return {
                "results": [{"id": "1", "title": "Paper", "url": "https://e.test"}],
                "cost_dollars": {
                    "contents": None,
                    "search": {"neural": 0.007},
                    "total": 0.007,
                },
            }

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    monkeypatch.setattr(
        "src.retrieval.sources.Exa",
        FakeExa,
    )
    source = ExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_cost_dollars_per_run=0.10,
            max_qps=1000,
        )
    )

    result = source.search("query", limit=1, filters=SearchPaperFilters())

    assert result.status.status == "success"
    assert result.status.metadata["costDollars"] == 0.007


def test_exa_projected_run_cost_skips_before_sdk_call(monkeypatch) -> None:
    called = False

    class ProjectedCostExaPaperSource(ExaPaperSource):
        def _estimate_call_cost_dollars(self, *, limit: int, text: bool) -> float:
            return 0.03

        def _call_exa_sdk(self, **kwargs):
            nonlocal called
            called = True
            return {"results": []}

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    source = ProjectedCostExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_cost_dollars_per_run=0.05,
            max_cost_dollars_per_call=0.05,
            max_qps=1000,
        )
    )
    source._observed_cost_dollars = 0.04

    result = source.search("query", limit=1, filters=SearchPaperFilters())

    assert called is False
    assert result.status.status == "skipped"
    assert result.status.warnings == [
        "exa run cost cap would be exceeded; skipping exa"
    ]
    assert result.status.metadata["estimatedCostDollars"] == 0.03
    assert result.status.metadata["observedCostDollars"] == 0.04
    assert result.status.metadata["maxCostDollarsPerRun"] == 0.05
