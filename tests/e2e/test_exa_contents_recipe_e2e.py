from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from src.retrieval.models import SearchPaperFilters
from src.retrieval.sources import ExaPaperSource


pytestmark = [
    pytest.mark.live,
    pytest.mark.drift_watchdog,
]


def _require_exa() -> None:
    if not os.environ.get("EXA_API_KEY"):
        pytest.skip("EXA_API_KEY is required for live Exa contents tests")
    pytest.importorskip("exa_py", reason="exa-py is required for live Exa contents tests")


def _exa_source(**overrides: Any) -> ExaPaperSource:
    return ExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_qps=1000,
            max_cost_dollars_per_run=0.10,
            max_cost_dollars_per_call=0.02,
            text=False,
            highlights=True,
            highlight_max_characters=1200,
            text_max_characters=20000,
            max_age_hours=None,
            livecrawl_timeout=None,
            include_html_tags=False,
            **overrides,
        )
    )


@pytest.mark.live_exa
def test_search_discovery_can_find_arxiv_candidate_url() -> None:
    _require_exa()

    result = _exa_source().search(
        "Language Modeling Is Compression arxiv 2309.10668",
        limit=3,
        filters=SearchPaperFilters(
            include_domains=["arxiv.org"],
            highlights=True,
        ),
    )

    assert result.status.status == "success"
    assert result.results
    assert any("arxiv.org" in (item.url or "") for item in result.results)

