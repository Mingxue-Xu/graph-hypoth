from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from tests.e2e.test_arxiv_loader_e2e import _arxiv_loader_class


pytestmark = [
    pytest.mark.live,
    pytest.mark.live_arxiv,
    pytest.mark.drift_watchdog,
]


def test_long_query_succeeds(
    paced_arxiv_call: Callable[[Callable[[], list[Any]]], list[Any]],
) -> None:
    query = (
        "graph neural network reasoning for scientific claim verification with "
        "retrieval augmented evidence ranking debate calibration uncertainty "
        "estimation benchmark construction reproducibility analysis citation "
        "grounding multi agent critique synthesis paper selection methodology "
        "experimental controls ablation transfer evaluation data human expert review"
    )
    assert len(query) == 350
    ArxivLoader = _arxiv_loader_class()
    loader = ArxivLoader(query=query, load_max_docs=2, load_all_available_meta=True)

    docs = paced_arxiv_call(loader.get_summaries_as_docs)

    assert docs, "arXiv should accept a 350-character natural-language query"
