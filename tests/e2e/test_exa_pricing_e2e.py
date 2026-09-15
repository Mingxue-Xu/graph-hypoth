from __future__ import annotations

import pytest

from src.retrieval.models import SearchPaperFilters

from tests.e2e.test_exa_drift_e2e import (
    _assert_non_empty_results,
    exa_adapter_search,
    finite_cost_total,
)


pytestmark = [
    pytest.mark.live,
    pytest.mark.live_exa,
    pytest.mark.drift_watchdog,
]


def test_search_vs_search_and_contents_cost() -> None:
    query = "evidence-grounded multi-agent debate research papers"

    search_only_response = exa_adapter_search(
        query,
        limit=5,
        filters=SearchPaperFilters(text=False),
    )
    _assert_non_empty_results(search_only_response, "search-only pricing probe")

    search_with_text_response = exa_adapter_search(
        query,
        limit=5,
        filters=SearchPaperFilters(text=True),
    )
    _assert_non_empty_results(search_with_text_response, "text pricing probe")

    search_only_total = finite_cost_total(search_only_response)
    search_with_text_total = finite_cost_total(search_with_text_response)

    allowed_delta = max(
        0.001,
        min(search_only_total, search_with_text_total) * 0.20,
    )
    assert abs(search_only_total - search_with_text_total) <= allowed_delta


def test_text_disabled_search_observed_cost_stays_under_per_call_cap() -> None:
    cap = 0.05
    response = exa_adapter_search(
        "evidence-grounded multi-agent debate research papers",
        limit=5,
        filters=SearchPaperFilters(text=False),
        max_cost_dollars_per_call=cap,
    )
    _assert_non_empty_results(response, "text-disabled pricing watchdog")

    assert finite_cost_total(response) <= cap
