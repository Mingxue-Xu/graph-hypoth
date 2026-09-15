from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


pytestmark = [
    pytest.mark.live,
    pytest.mark.live_arxiv,
    pytest.mark.drift_watchdog,
]


def _arxiv_loader_class() -> type[Any]:
    pytest.importorskip("arxiv", reason="arxiv is required for live arXiv e2e tests")
    pytest.importorskip(
        "pymupdf",
        reason="PyMuPDF is required for live arXiv full-PDF e2e tests",
    )
    loaders = pytest.importorskip(
        "langchain_community.document_loaders",
        reason="langchain-community is required for live arXiv e2e tests",
    )
    return loaders.ArxivLoader


def test_attention_is_all_you_need_full_pdf_extraction(
    paced_arxiv_call: Callable[[Callable[[], list[Any]]], list[Any]],
) -> None:
    ArxivLoader = _arxiv_loader_class()
    loader = ArxivLoader(query="1706.03762", load_max_docs=1)

    docs = paced_arxiv_call(loader.load)

    assert docs, "ArxivLoader should return a document for 1706.03762"
    document = docs[0]
    assert len(document.page_content) >= 5000
    assert document.metadata.get("Title")
    assert document.metadata.get("Authors")
