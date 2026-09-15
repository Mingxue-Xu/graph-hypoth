from __future__ import annotations

from types import SimpleNamespace

from src.retrieval.models import SearchPaperFilters
from src.retrieval.sources import ArxivPaperSource


def test_full_pdf_mode_rejects_missing_fitz_alias(monkeypatch) -> None:
    def fake_import_module(name: str):
        if name in {"arxiv", "pymupdf"}:
            return object()
        if name == "fitz":
            raise ModuleNotFoundError("No module named fitz")
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(
        "src.retrieval.sources.importlib.import_module",
        fake_import_module,
    )
    monkeypatch.setattr(
        "src.retrieval.sources.ArxivLoader",
        object,
    )

    result = ArxivPaperSource(
        SimpleNamespace(mode="full_pdf", query_max_chars=300)
    ).search("query", limit=1, filters=SearchPaperFilters())

    assert result.status.status == "failed"
    assert result.status.result_count == 0
    assert any("fitz" in error.lower() for error in result.status.errors)


def test_full_pdf_mode_rejects_malformed_fitz_squatter(monkeypatch) -> None:
    malformed_fitz = SimpleNamespace(__doc__="not PyMuPDF", open=None)

    def fake_import_module(name: str):
        if name in {"arxiv", "pymupdf"}:
            return object()
        if name == "fitz":
            return malformed_fitz
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(
        "src.retrieval.sources.importlib.import_module",
        fake_import_module,
    )
    monkeypatch.setattr(
        "src.retrieval.sources.ArxivLoader",
        object,
    )

    result = ArxivPaperSource(
        SimpleNamespace(mode="full_pdf", query_max_chars=300)
    ).search("query", limit=1, filters=SearchPaperFilters())

    assert result.status.status == "failed"
    assert result.status.result_count == 0
    assert any("pymupdf" in error.lower() for error in result.status.errors)
