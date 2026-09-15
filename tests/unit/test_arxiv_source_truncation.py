from __future__ import annotations

from types import SimpleNamespace

from src.retrieval.models import SearchPaperFilters
from src.retrieval.sources import ArxivPaperSource


class _Document:
    page_content = "paper body"
    metadata = {
        "Title": "Ledger Paper",
        "Authors": ["A. Researcher"],
        "entry_id": "https://arxiv.org/abs/1234.56789",
        "Published": "2026-05-01",
        "Summary": "summary",
    }


def test_arxiv_truncates_long_query_and_reports_unused_filters(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeArxivLoader:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def get_summaries_as_docs(self):
            return [_Document()]

    def fake_import_module(name: str):
        if name == "arxiv":
            return object()
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(
        "src.retrieval.sources.importlib.import_module",
        fake_import_module,
    )
    monkeypatch.setattr(
        "src.retrieval.sources.ArxivLoader",
        FakeArxivLoader,
    )

    source = ArxivPaperSource(
        SimpleNamespace(
            query_max_chars=300,
            mode="metadata_only",
            min_delay_seconds=0,
            doc_content_chars_max=4000,
        )
    )
    query = "q" * 350

    result = source.search(
        query,
        limit=1,
        filters=SearchPaperFilters(include_text=["ledger"], text=True),
    )

    assert result.status.status == "success"
    assert result.status.source_query == "q" * 300
    assert result.status.warnings == ["arxiv query truncated to 300 characters"]
    assert result.status.unused_filters == ["include_text", "text"]
    assert captured["query"] == "q" * 300
    assert result.results[0].title == "Ledger Paper"
