from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from src.retrieval.models import SearchPaperFilters, SourceStatus
from src.retrieval.sources import ArxivPaperSource


pytestmark = [
    pytest.mark.live,
    pytest.mark.live_arxiv,
]


def _import_arxiv_metadata_dependencies() -> None:
    pytest.importorskip("arxiv", reason="arxiv is required for live arXiv e2e tests")
    loaders = pytest.importorskip(
        "langchain_community.document_loaders",
        reason="langchain-community is required for live arXiv e2e tests",
    )
    if not hasattr(loaders, "ArxivLoader"):
        pytest.skip("langchain-community ArxivLoader is required")


def _import_arxiv_full_pdf_dependencies() -> None:
    _import_arxiv_metadata_dependencies()
    pytest.importorskip(
        "pymupdf",
        reason="PyMuPDF is required for live arXiv full-PDF e2e tests",
    )
    fitz = pytest.importorskip(
        "fitz",
        reason="PyMuPDF fitz alias is required for live arXiv full-PDF e2e tests",
    )
    if not callable(getattr(fitz, "open", None)) or getattr(fitz, "Document", None) is None:
        pytest.skip("PyMuPDF fitz alias is unavailable or malformed")


def _arxiv_source(*, mode: str, doc_content_chars_max: int) -> ArxivPaperSource:
    return ArxivPaperSource(
        SimpleNamespace(
            query_max_chars=300,
            mode=mode,
            min_delay_seconds=0.0,
            load_max_docs=1,
            load_all_available_meta=True,
            doc_content_chars_max=doc_content_chars_max,
            continue_on_failure=False,
        )
    )


def _assert_successful_status(
    status: SourceStatus,
    *,
    result_count: int,
    source_query: str,
) -> None:
    assert status.status == "success"
    assert status.errors == []
    assert status.source == "arxiv"
    assert status.source_query == source_query
    assert status.result_count == result_count
    assert result_count >= 1


def _assert_attention_result_identity(result: Any) -> None:
    assert result.source == "arxiv"
    assert (
        "attention" in result.title.lower()
        or "transformer" in result.title.lower()
    )
    assert result.authors
    assert "1706.03762" in (result.source_id or result.url or "")


def test_arxiv_source_metadata_only_search_returns_normalized_result(
    paced_arxiv_call: Callable[[Callable[[], Any]], Any],
) -> None:
    _import_arxiv_metadata_dependencies()
    source = _arxiv_source(mode="metadata_only", doc_content_chars_max=4000)

    search_result = paced_arxiv_call(
        lambda: source.search(
            "1706.03762",
            limit=1,
            filters=SearchPaperFilters(include_text=["attention"], text=True),
        )
    )

    results = search_result.results
    status = search_result.status
    _assert_successful_status(
        status,
        result_count=len(results),
        source_query="1706.03762",
    )
    assert search_result.errors == []
    assert status.unused_filters == ["include_text", "text"]
    first = results[0]
    _assert_attention_result_identity(first)
    assert first.text


def test_arxiv_source_full_pdf_search_returns_normalized_pdf_text(
    paced_arxiv_call: Callable[[Callable[[], Any]], Any],
) -> None:
    _import_arxiv_full_pdf_dependencies()
    source = _arxiv_source(mode="full_pdf", doc_content_chars_max=8000)

    search_result = paced_arxiv_call(
        lambda: source.search(
            "1706.03762",
            limit=1,
            filters=SearchPaperFilters(),
        )
    )

    results = search_result.results
    status = search_result.status
    _assert_successful_status(
        status,
        result_count=len(results),
        source_query="1706.03762",
    )
    assert search_result.errors == []
    first = results[0]
    _assert_attention_result_identity(first)
    assert len(first.text or "") >= 5000
