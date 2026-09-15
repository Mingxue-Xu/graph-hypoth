from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from src.retrieval.models import SearchPaperFilters
from src.retrieval.sources import ExaPaperSource


pytestmark = [
    pytest.mark.live,
    pytest.mark.live_exa,
    pytest.mark.drift_watchdog,
]


def _exa_adapter_source(**overrides: Any) -> ExaPaperSource:
    if not os.environ.get("EXA_API_KEY"):
        pytest.skip("EXA_API_KEY is required for live Exa drift e2e tests")
    pytest.importorskip("exa_py", reason="exa-py is required for live Exa e2e tests")
    config = SimpleNamespace(
        require_api_key_env="EXA_API_KEY",
        max_qps=1000,
        **overrides,
    )
    return ExaPaperSource(config)


def exa_adapter_search(
    query: str,
    *,
    limit: int,
    filters: SearchPaperFilters | None = None,
    **overrides: Any,
) -> Any:
    source = _exa_adapter_source(**overrides)
    return source.search(query, limit=limit, filters=filters or SearchPaperFilters())


def _as_mapping(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return response
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(response, "dict"):
        dumped = response.dict()
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(response, "__dict__"):
        return vars(response)
    return {}


def _results_from(response: Any) -> list[Any]:
    if hasattr(response, "results"):
        return list(response.results)
    if isinstance(response, list):
        return response

    mapping = _as_mapping(response)
    results = mapping.get("results")
    if isinstance(results, list):
        return results
    if results is not None:
        return list(results)
    return []


def _cost_metadata_from(response: Any) -> Any:
    if hasattr(response, "status"):
        return response.status.metadata.get("costDollars") or response.status.metadata.get(
            "cost"
        )
    mapping = _as_mapping(response)
    return mapping.get("costDollars") or mapping.get("cost")


def _redact(text: str) -> str:
    secret = os.environ.get("EXA_API_KEY")
    if secret:
        text = text.replace(secret, "[REDACTED_EXA_API_KEY]")
    text = re.sub(
        r"(?i)(api[-_ ]?key|authorization|bearer)(['\":=\s]+)[^,'\"\s)]+",
        r"\1\2[REDACTED]",
        text,
    )
    return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED_KEY]", text)


def _fail_on_provider_error(response: Any, label: str) -> None:
    if hasattr(response, "status") and response.status.errors:
        pytest.fail(f"{label} provider error: {_redact(str(response.status.errors))}")
    mapping = _as_mapping(response)
    if "error" not in mapping:
        return
    pytest.fail(f"{label} provider error: {_redact(str(mapping['error']))}")


def _assert_non_empty_results(response: Any, label: str) -> None:
    _fail_on_provider_error(response, label)
    results = _results_from(response)
    assert results, f"{label} should return non-empty Exa results"


def test_exa_adapter_round_trip() -> None:
    response = exa_adapter_search(
        "evidence-grounded multi-agent debate",
        limit=3,
    )

    _assert_non_empty_results(response, "adapter search")
    assert _cost_metadata_from(response), (
        "adapter search should surface Exa costDollars/cost metadata"
    )


def test_text_filters_round_trip() -> None:
    include_response = exa_adapter_search(
        "transformer architecture research papers",
        limit=3,
        filters=SearchPaperFilters(include_text=["transformer"]),
    )
    _assert_non_empty_results(include_response, "include_text search")

    exclude_response = exa_adapter_search(
        "machine learning evaluation research papers",
        limit=3,
        filters=SearchPaperFilters(exclude_text=["LLM"]),
    )
    _assert_non_empty_results(exclude_response, "exclude_text search")


def test_known_url_contents_round_trip() -> None:
    response = exa_adapter_search(
        "Language Modeling Is Compression impressive compression rates",
        limit=1,
        filters=SearchPaperFilters(
            target_urls=["https://arxiv.org/html/2309.10668v2"],
            anchor_queries=["impressive compression rates"],
            text=True,
            highlights=True,
            text_max_characters=3000,
            max_age_hours=0,
            target_scope="known_url",
        ),
        max_cost_dollars_per_call=0.01,
    )

    _assert_non_empty_results(response, "known-url contents probe")
    assert response.status.metadata["exa_request_kind"] == "contents"
    assert response.status.metadata["exa_anchor_queries"] == [
        "impressive compression rates"
    ]
    assert "text" in response.status.metadata["exa_content_modes"]
    assert "highlights" in response.status.metadata["exa_content_modes"]
    assert response.results[0].metadata["exa_request_kind"] == "contents"
    assert response.results[0].text


def finite_cost_total(response: Any) -> float:
    cost = _cost_metadata_from(response)
    if isinstance(cost, Mapping):
        total = cost["total"] if "total" in cost else cost.get("search")
    else:
        total = cost
    assert total is not None, "Exa response should include a cost total"
    value = float(total)
    assert math.isfinite(value), f"Exa cost total should be finite, got {value!r}"
    return value
