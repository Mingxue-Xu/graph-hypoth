from __future__ import annotations

import json
import logging
import sys
import urllib.parse
from types import SimpleNamespace

import pytest

from src import runtime_trace
from src.config import (
    ApifySourceConfig,
    CrossrefSourceConfig,
    EuropePmcSourceConfig,
    OpenAlexSourceConfig,
)
from src.retrieval import sources as sources_module
from datetime import datetime, timezone
from src.retrieval.models import SearchPaperFilters, SourceResult
from src.retrieval.sources import (
    ArxivPaperSource,
    CrossrefPaperSource,
    EuropePmcPaperSource,
    ExaPaperSource,
    FakePaperSource,
    OpenAlexPaperSource,
    _filter_by_recency,
    _normalize_doi,
)


# sentinel: an Exa test case that supplies no direct-PDF title expects no such lookup
_UNSET = object()


def test_filter_by_recency_drops_stale_keeps_fresh_and_undated() -> None:
    # Post-fetch recency cutoff for arxiv (max_age_hours is otherwise an Exa-only
    # no-op). Undated records are kept — we never drop what we cannot prove is stale.
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    results = [
        SourceResult(source="arxiv", title="fresh", published_date="2026-05-15"),
        SourceResult(source="arxiv", title="stale", published_date="2023-01-01"),
        SourceResult(source="arxiv", title="undated", published_date=None),
    ]
    kept, dropped = _filter_by_recency(results, max_age_hours=24 * 90, now=now)
    titles = [r.title for r in kept]
    assert "fresh" in titles
    assert "undated" in titles
    assert "stale" not in titles
    assert dropped == 1


def test_filter_by_recency_is_a_no_op_when_disabled() -> None:
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    results = [SourceResult(source="arxiv", title="old", published_date="2000-01-01")]
    for disabled in (None, 0):
        kept, dropped = _filter_by_recency(results, max_age_hours=disabled, now=now)
        assert [r.title for r in kept] == ["old"]
        assert dropped == 0


def test_fake_paper_source_returns_deterministic_success_results() -> None:
    source = FakePaperSource()
    filters = SearchPaperFilters()

    first = source.search("  retrieval   ledger  ", limit=2, filters=filters)
    second = source.search("  retrieval   ledger  ", limit=2, filters=filters)

    assert first.status.status == "success"
    assert first.status.result_count == 2
    assert [result.model_dump() for result in first.results] == [
        result.model_dump() for result in second.results
    ]
    assert first.results[0].source == "fake"
    assert first.results[0].title == (
        "Deterministic Retrieval Result 1 for retrieval ledger"
    )


def test_arxiv_pacing_uses_cross_process_lock_timestamp(
    monkeypatch,
    tmp_path,
) -> None:
    lock_path = tmp_path / "arxiv-rate-limit.lock"
    lock_path.write_text("100.0", encoding="utf-8")
    sleeps: list[float] = []
    now = {"value": 103.0}
    source = ArxivPaperSource(SimpleNamespace(rate_limit_lock_path=str(lock_path)))
    ArxivPaperSource._last_call_at = 0.0

    monkeypatch.setattr(sources_module.time, "monotonic", lambda: now["value"])

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now["value"] += seconds

    monkeypatch.setattr(sources_module.time, "sleep", fake_sleep)

    with source._paced_call(min_delay_seconds=5.0):
        pass

    assert sleeps == [2.0]
    assert lock_path.read_text(encoding="utf-8") == "105.0"


def test_arxiv_full_pdf_load_prepares_pymupdf_and_restores_diagnostics(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeTools:
        def __init__(self) -> None:
            self.errors = True
            self.warnings = True

        def mupdf_display_errors(self, value=None):
            if value is None:
                return self.errors
            self.errors = value
            return value

        def mupdf_display_warnings(self, value=None):
            if value is None:
                return self.warnings
            self.warnings = value
            return value

    tools = FakeTools()
    fake_fitz = SimpleNamespace(TOOLS=tools, FileDataError=RuntimeError)
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    class FakeArxivLoader:
        def __init__(self, **kwargs):
            del kwargs

        def load(self):
            assert fake_fitz.fitz is fake_fitz
            assert tools.errors is False
            assert tools.warnings is False
            return [SimpleNamespace(metadata={}, page_content="loaded PDF text")]

    monkeypatch.setattr(sources_module, "ArxivLoader", FakeArxivLoader)
    source = ArxivPaperSource(
        SimpleNamespace(
            mode="full_pdf",
            rate_limit_lock_path=str(tmp_path / "arxiv-rate-limit.lock"),
            min_delay_seconds=0,
        )
    )

    documents = source._load_documents("projection compression", limit=1)

    assert documents[0].page_content == "loaded PDF text"
    assert not hasattr(fake_fitz, "fitz")
    assert tools.errors is True
    assert tools.warnings is True


def test_arxiv_full_pdf_load_records_trace_after_restoring_working_directory(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPH_HYPOTH_RUNTIME_LOG_DIR", "runtime_logs")
    monkeypatch.setenv("GRAPH_HYPOTH_RUNTIME_RUN_LABEL", "unit")
    runtime_trace.reset_runtime_trace()
    trace_file = runtime_trace.record_runtime_event("artifact", payload={"ok": True})
    assert trace_file is not None

    class FakeArxivLoader:
        def __init__(self, **kwargs):
            del kwargs

        def load(self):
            return [SimpleNamespace(metadata={"Title": "Paper"}, page_content="text")]

    monkeypatch.setattr(sources_module, "ArxivLoader", FakeArxivLoader)
    source = ArxivPaperSource(
        SimpleNamespace(
            mode="full_pdf",
            rate_limit_lock_path=str(tmp_path / "arxiv-rate-limit.lock"),
            min_delay_seconds=0,
        )
    )

    try:
        documents = source._load_documents("projection compression", limit=1)
    finally:
        runtime_trace.reset_runtime_trace()

    assert documents[0].page_content == "text"
    assert '"direction": "raw_response"' in trace_file.read_text(encoding="utf-8")


def test_arxiv_search_captures_recoverable_loader_errors_without_logging(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    def fake_import_module(name: str):
        if name == "arxiv":
            return SimpleNamespace()
        return original_import_module(name)

    original_import_module = sources_module.importlib.import_module
    monkeypatch.setattr(sources_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        ArxivPaperSource,
        "_ensure_pymupdf_available",
        lambda self: None,
    )

    class FakeArxivLoader:
        def __init__(self, **kwargs):
            del kwargs

        def load(self):
            logging.getLogger("langchain_community.utilities.arxiv").error(
                "<urlopen error retrieval incomplete: got only 1 out of 2 bytes>"
            )
            return []

    monkeypatch.setattr(sources_module, "ArxivLoader", FakeArxivLoader)
    source = ArxivPaperSource(
        SimpleNamespace(
            mode="full_pdf",
            rate_limit_lock_path=str(tmp_path / "arxiv-rate-limit.lock"),
            min_delay_seconds=0,
        )
    )
    caplog.set_level(logging.ERROR)

    result = source.search("projection compression", limit=1, filters=SearchPaperFilters())

    assert result.status.status == "partial_failure"
    assert "urlopen error retrieval incomplete" not in caplog.text
    assert any(
        "urlopen error retrieval incomplete" in warning
        for warning in result.status.warnings
    )


def test_exa_missing_api_key_skips_without_leaking_secrets(monkeypatch) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    source = ExaPaperSource(SimpleNamespace(require_api_key_env="EXA_API_KEY"))

    result = source.search("test query", limit=3, filters=SearchPaperFilters())

    assert result.status.status == "skipped"
    assert result.status.result_count == 0
    joined = " ".join(result.status.warnings + result.warnings)
    assert "EXA_API_KEY" in joined
    assert "secret" not in joined.lower()


def test_exa_text_filter_validation_failure_is_structured(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    source = ExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            include_text=["one", "two"],
        )
    )

    result = source.search("test query", limit=3, filters=SearchPaperFilters())

    assert result.status.status == "failed"
    assert result.status.result_count == 0
    assert result.status.errors == ["exa include_text accepts at most one string"]
    assert "test-secret-not-for-output" not in " ".join(result.status.errors)


def test_exa_sdk_error_dict_returns_failed_status_with_raw_error(
    monkeypatch,
) -> None:
    class FakeExa:
        def __init__(self, api_key: str) -> None:
            del api_key

        def search(self, **kwargs):
            return {"error": f"bad request for {kwargs['query']}"}

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    monkeypatch.setattr(
        "src.retrieval.sources.Exa",
        FakeExa,
    )
    source = ExaPaperSource(
        SimpleNamespace(require_api_key_env="EXA_API_KEY", max_qps=1000)
    )

    result = source.search("test query", limit=3, filters=SearchPaperFilters())

    assert result.status.status == "failed"
    assert result.status.errors == ["bad request for test query"]
    assert result.status.metadata == {"raw": {"error": "bad request for test query"}}


def test_exa_source_requests_highlights_by_default_and_records_metadata(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeExa:
        def __init__(self, api_key: str) -> None:
            del api_key

        def search(self, **kwargs):
            calls.append(kwargs)
            return {
                "results": [
                    {
                        "id": "paper-1",
                        "title": "Paper 1",
                        "url": "https://example.test/paper-1",
                        "highlights": ["A useful claim-relevant sentence."],
                        "highlightScores": [0.91],
                    }
                ],
                "statuses": [
                    {"id": "paper-1", "status": "success"},
                ],
                "costDollars": 0.007,
            }

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    monkeypatch.setattr(
        "src.retrieval.sources.Exa",
        FakeExa,
    )
    source = ExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_qps=1000,
            max_cost_dollars_per_run=1.0,
            max_cost_dollars_per_call=1.0,
        )
    )

    result = source.search("claim target", limit=1, filters=SearchPaperFilters())

    assert calls[0]["contents"] == {
        "highlights": {"query": "claim target", "max_characters": 600}
    }
    assert result.status.metadata["exa_content_modes"] == ["highlights"]
    assert result.status.metadata["exa_statuses"] == [
        {"id": "paper-1", "status": "success"}
    ]
    metadata = result.results[0].metadata
    assert metadata["exa_highlights"] == ["A useful claim-relevant sentence."]
    assert metadata["exa_highlight_scores"] == [0.91]
    assert metadata["exa_highlight_query"] == "claim target"
    assert metadata["exa_content_modes"] == ["highlights"]
    assert metadata["source_text_chars_available"] == 0


# --- Exa title recovery ---------------------------------------------------------------------
# Eight cases over one provider double. Exa often returns an empty or placeholder
# title for a PDF, so the source recovers one from the contents text and, failing
# that, by fetching the PDF directly. Each case below varies only the provider's
# search result, what get_contents returns, and the direct-PDF stub.
_EXA_PDF_TEXT = (
    "Understanding LLM Behaviors via Compression: Data\n"
    "Generation, Knowledge Acquisition and Scaling Laws\n"
    "Abstract\n"
    "Large Language Models have demonstrated..."
)
_EXA_RECOVERED_TITLE = (
    "Understanding LLM Behaviors via Compression: Data Generation, "
    "Knowledge Acquisition and Scaling Laws"
)


def _install_fake_exa(
    monkeypatch,
    *,
    search_results: list[dict],
    contents_results: list[dict] | None = None,
    direct_title: object = _UNSET,
    **config_extras: object,
):
    """Install a canned Exa provider and return ``(source, calls, direct_calls)``.

    ``contents_results=None`` makes ``get_contents`` fail the test if the source
    calls it, which is how the "provider gave a usable title" cases prove no
    second billable request is made. ``direct_title`` left unset likewise fails
    the test if the direct-PDF fallback fires.
    """
    calls: list[dict[str, object]] = []
    direct_calls: list[str] = []

    class FakeExa:
        def __init__(self, api_key: str) -> None:
            del api_key

        def search(self, **kwargs):
            calls.append({"method": "search", **kwargs})
            return {"results": search_results, "costDollars": 0.007}

        def get_contents(self, urls, **kwargs):
            calls.append({"method": "get_contents", "urls": urls, **kwargs})
            if contents_results is None:
                raise AssertionError("get_contents should not be called for this case")
            return {"results": contents_results, "costDollars": {"total": 0.001}}

    def fake_pdf_title(url: str):
        direct_calls.append(url)
        if direct_title is _UNSET:
            raise AssertionError(f"direct PDF fallback should not fetch {url}")
        return direct_title

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    monkeypatch.setattr("src.retrieval.sources.Exa", FakeExa)
    monkeypatch.setattr("src.retrieval.sources.resolve_pdf_title_from_url", fake_pdf_title)
    source = ExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_qps=1000,
            max_cost_dollars_per_run=1.0,
            max_cost_dollars_per_call=1.0,
            **config_extras,
        )
    )
    return source, calls, direct_calls


def _exa_hit(url: str, title: str, *, result_id: str | None = None) -> dict:
    return {
        "id": result_id or url,
        "title": title,
        "url": url,
        "highlights": ["Prediction and compression are related."],
    }


def _search_one(source):
    return source.search("claim target", limit=1, filters=SearchPaperFilters())


def test_exa_source_does_not_invent_generic_title_when_provider_omits_title(
    monkeypatch,
) -> None:
    source, _calls, _direct = _install_fake_exa(
        monkeypatch,
        search_results=[
            {
                "id": "paper-untitled",
                "title": "",
                "url": "https://example.test/paper-untitled",
                "highlights": ["A useful claim-relevant sentence."],
            }
        ],
        contents_results=[],
        direct_title=None,
    )

    result = _search_one(source)

    assert result.results[0].title == ""
    assert "Exa result" not in result.results[0].model_dump_json()


def test_exa_source_strips_arxiv_id_prefix_from_result_title(monkeypatch) -> None:
    source, _calls, _direct = _install_fake_exa(
        monkeypatch,
        search_results=[
            _exa_hit(
                "https://arxiv.org/abs/2407.06645",
                "[2407.06645] Entropy Law: The Story Behind Data "
                "Compression and LLM Performance",
            )
        ],
    )

    result = _search_one(source)

    assert result.results[0].title == (
        "Entropy Law: The Story Behind Data Compression and LLM Performance"
    )


@pytest.mark.parametrize("provider_title", ["", "(untitled)"])
def test_exa_source_looks_up_source_text_to_recover_pdf_title(
    monkeypatch, provider_title: str
) -> None:
    url = "https://example.test/openreview.pdf"
    source, calls, _direct = _install_fake_exa(
        monkeypatch,
        search_results=[_exa_hit(url, provider_title)],
        contents_results=[{"id": url, "url": url, "text": _EXA_PDF_TEXT}],
        text=False,
        highlights=True,
    )

    result = _search_one(source)

    # exactly one extra (billable) contents request, bounded to 5000 characters
    assert [call["method"] for call in calls] == ["search", "get_contents"]
    assert calls[1]["urls"] == [url]
    assert calls[1]["text"] == {"max_characters": 5000, "include_html_tags": False}
    assert result.results[0].title == _EXA_RECOVERED_TITLE
    assert result.results[0].metadata["source_text_chars_available"] > 0


def test_exa_source_matches_title_lookup_text_by_canonical_url(monkeypatch) -> None:
    # the search hit, the request url, and the contents row all spell the same
    # document differently (fragment, query string, trailing slash)
    source, _calls, _direct = _install_fake_exa(
        monkeypatch,
        search_results=[
            _exa_hit(
                "https://example.test/paper.pdf?download=1",
                "",
                result_id="https://example.test/paper.pdf#page=1",
            )
        ],
        contents_results=[
            {
                "id": "https://example.test/paper.pdf",
                "url": "https://example.test/paper.pdf/",
                "text": _EXA_PDF_TEXT,
            }
        ],
        text=False,
        highlights=True,
    )

    result = _search_one(source)

    assert result.results[0].title == _EXA_RECOVERED_TITLE


def test_exa_source_direct_pdf_title_fallback_after_empty_contents_lookup(
    monkeypatch,
) -> None:
    url = "https://example.test/paper.pdf"
    source, _calls, direct_calls = _install_fake_exa(
        monkeypatch,
        search_results=[_exa_hit(url, "")],
        contents_results=[],
        direct_title="Recovered PDF Title",
        text=False,
        highlights=True,
    )

    result = _search_one(source)

    assert direct_calls == [url]
    assert result.results[0].title == "Recovered PDF Title"
    assert result.results[0].metadata["pdf_title_lookup"] == "direct_pdf"


def test_exa_source_preserves_direct_pdf_title_lookup_warning(monkeypatch) -> None:
    source, _calls, _direct = _install_fake_exa(
        monkeypatch,
        search_results=[_exa_hit("https://example.test/paper.pdf", "")],
        contents_results=[],
        direct_title=None,
        text=False,
        highlights=True,
    )

    result = _search_one(source)

    assert result.status.status == "success"
    assert result.warnings == ["exa direct PDF title lookup did not recover title"]
    assert result.status.warnings == result.warnings


def test_exa_source_skips_direct_pdf_title_fallback_for_non_pdf_url(
    monkeypatch,
) -> None:
    # direct_title left unset: the stub raises if the non-PDF url reaches it
    source, _calls, _direct = _install_fake_exa(
        monkeypatch,
        search_results=[_exa_hit("https://example.test/page", "")],
        contents_results=[],
        text=False,
        highlights=True,
    )

    result = _search_one(source)

    assert result.results[0].title == ""


def test_exa_source_honors_text_and_highlight_filter_options(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeExa:
        def __init__(self, api_key: str) -> None:
            del api_key

        def search(self, **kwargs):
            calls.append(kwargs)
            return {
                "results": [
                    {
                        "id": "paper-1",
                        "title": "Paper 1",
                        "url": "https://example.test/paper-1",
                        "text": "Full source text.",
                        "highlights": ["Filtered highlight."],
                        "highlightScores": [0.8],
                    }
                ],
            }

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    monkeypatch.setattr(
        "src.retrieval.sources.Exa",
        FakeExa,
    )
    source = ExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_qps=1000,
            max_cost_dollars_per_run=1.0,
            max_cost_dollars_per_call=1.0,
            highlights=True,
            highlight_max_characters=600,
            text_max_characters=5000,
        )
    )

    result = source.search(
        "discovery query",
        limit=1,
        filters=SearchPaperFilters(
            text=True,
            highlights=True,
            highlight_query="specific claim",
            highlight_max_characters=300,
            text_max_characters=2000,
        ),
    )

    assert calls[0]["contents"] == {
        "highlights": {"query": "specific claim", "max_characters": 300},
        "text": {"max_characters": 2000, "include_html_tags": False},
    }
    assert result.results[0].metadata["exa_highlight_query"] == "specific claim"
    assert result.results[0].metadata["source_text_chars_available"] == len(
        "Full source text."
    )


def test_exa_source_uses_contents_endpoint_for_known_urls(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeExa:
        def __init__(self, api_key: str) -> None:
            del api_key

        def search(self, **kwargs):
            raise AssertionError(f"search should not be called: {kwargs}")

        def get_contents(self, urls, **kwargs):
            calls.append({"urls": urls, **kwargs})
            return {
                "results": [
                    {
                        "id": "https://example.test/paper",
                        "title": "Known Paper",
                        "url": "https://example.test/paper",
                        "text": "The verified sentence supports the claim.",
                        "highlights": ["The verified sentence supports the claim."],
                    }
                ],
                "statuses": [
                    {
                        "id": "https://example.test/paper",
                        "status": "success",
                        "source": "cached",
                    }
                ],
                "costDollars": {"total": 0.002},
            }

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    monkeypatch.setattr(
        "src.retrieval.sources.Exa",
        FakeExa,
    )
    source = ExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_qps=1000,
            max_cost_dollars_per_run=1.0,
            max_cost_dollars_per_call=1.0,
            highlights=True,
            text=False,
        )
    )

    result = source.search(
        "claim target",
        limit=1,
        filters=SearchPaperFilters(
            urls=["https://example.test/paper"],
            highlights=True,
            text=True,
            max_age_hours=0,
            include_html_tags=True,
        ),
    )

    assert calls == [
        {
            "urls": ["https://example.test/paper"],
            "highlights": {"query": "claim target", "max_characters": 600},
            "text": {"max_characters": 5000, "include_html_tags": True},
            "max_age_hours": 0,
        }
    ]
    assert result.status.metadata["exa_request_kind"] == "contents"
    assert result.status.metadata["exa_statuses"][0]["status"] == "success"
    assert result.results[0].metadata["exa_request_kind"] == "contents"
    assert result.results[0].metadata["exa_content_modes"] == ["highlights", "text"]


def test_exa_source_uses_one_contents_call_per_anchor_query(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeExa:
        def __init__(self, api_key: str) -> None:
            del api_key

        def get_contents(self, urls, **kwargs):
            calls.append({"urls": urls, **kwargs})
            anchor = kwargs["highlights"]["query"]
            return {
                "results": [
                    {
                        "id": "https://example.test/paper",
                        "title": "Known Paper",
                        "url": "https://example.test/paper",
                        "text": (
                            "Anchor one supports the claim. "
                            "Anchor two supports the claim."
                        )
                        if "text" in kwargs
                        else None,
                        "highlights": [f"{anchor} supports the claim."],
                    }
                ],
                "costDollars": {"total": 0.002 if "text" in kwargs else 0.001},
            }

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    monkeypatch.setattr(
        "src.retrieval.sources.Exa",
        FakeExa,
    )
    source = ExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_qps=1000,
            max_cost_dollars_per_run=1.0,
            max_cost_dollars_per_call=1.0,
            highlights=True,
            text=True,
        )
    )

    result = source.search(
        "known paper",
        limit=2,
        filters=SearchPaperFilters(
            target_urls=["https://example.test/paper"],
            anchor_queries=["Anchor one", "Anchor two"],
            text=True,
            highlights=True,
            max_age_hours=0,
            livecrawl_timeout=3000,
        ),
    )

    assert [call["highlights"]["query"] for call in calls] == [
        "Anchor one",
        "Anchor two",
    ]
    assert "text" in calls[0]
    assert "text" not in calls[1]
    assert calls[0]["livecrawl_timeout"] == 3000
    assert result.status.metadata["exa_request_kind"] == "contents"
    assert result.status.metadata["exa_anchor_queries"] == ["Anchor one", "Anchor two"]
    assert result.status.metadata["costDollars"] == 0.003
    assert [item.metadata["exa_anchor_query"] for item in result.results] == [
        "Anchor one",
        "Anchor two",
    ]


def test_exa_source_passes_domain_filters_to_search(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeExa:
        def __init__(self, api_key: str) -> None:
            del api_key

        def search(self, **kwargs):
            calls.append(kwargs)
            return {"results": [], "costDollars": 0.007}

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    monkeypatch.setattr(
        "src.retrieval.sources.Exa",
        FakeExa,
    )
    source = ExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_qps=1000,
            max_cost_dollars_per_run=1.0,
            max_cost_dollars_per_call=1.0,
        )
    )

    source.search(
        "discovery query",
        limit=2,
        filters=SearchPaperFilters(
            include_domains=["arxiv.org"],
            exclude_domains=["example.com"],
        ),
    )

    assert calls[0]["include_domains"] == ["arxiv.org"]
    assert calls[0]["exclude_domains"] == ["example.com"]


def test_exa_per_call_cap_skips_before_sdk_call(monkeypatch) -> None:
    called = False

    class CostlyExaPaperSource(ExaPaperSource):
        def _estimate_call_cost_dollars(self, *, limit: int, text: bool) -> float:
            return 0.25

        def _call_exa_sdk(self, **kwargs):
            nonlocal called
            called = True
            return {"results": []}

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    source = CostlyExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_cost_dollars_per_run=1.0,
            max_cost_dollars_per_call=0.05,
            max_qps=1000,
            text=False,
        )
    )

    result = source.search("test query", limit=3, filters=SearchPaperFilters())

    assert called is False
    assert result.status.status == "skipped"
    assert result.status.warnings == ["exa per-call cost cap exceeded; skipping exa"]
    assert result.status.metadata["estimatedCostDollars"] == 0.25


def test_exa_source_skips_call_when_projected_cost_exceeds_per_call_cap(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXA_API_KEY", "sk-test")
    monkeypatch.setattr(
        "src.retrieval.sources.Exa",
        lambda **_kwargs: pytest.fail("Exa SDK must not be constructed"),
    )
    source = ExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_qps=1000,
            max_cost_dollars_per_run=10.0,
            max_cost_dollars_per_call=0.0001,
        )
    )

    result = source.search("test query", limit=5, filters=SearchPaperFilters())

    assert result.status.status == "skipped"
    joined = " ".join(result.status.warnings + result.warnings)
    assert "exa per-call cost cap exceeded; skipping exa" in joined
    assert result.results == []


def test_exa_per_call_cap_allows_call_at_or_below_cap(monkeypatch) -> None:
    # The source builds the real `exa_py` Exa client before reaching the
    # (overridden) _call_exa_sdk, so this cap test needs the optional `exa-py`
    # package; skip cleanly when it is not installed in the default env.
    pytest.importorskip("exa_py")
    called = False

    class AffordableExaPaperSource(ExaPaperSource):
        def _estimate_call_cost_dollars(self, *, limit: int, text: bool) -> float:
            return 0.05

        def _call_exa_sdk(self, **kwargs):
            nonlocal called
            called = True
            return {"results": [], "costDollars": 0.01}

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    source = AffordableExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_cost_dollars_per_run=1.0,
            max_cost_dollars_per_call=0.05,
            max_qps=1000,
            text=False,
        )
    )

    result = source.search("test query", limit=3, filters=SearchPaperFilters())

    assert called is True
    assert result.status.status == "success"


class _FakeHttpResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def read(self, amount: int | None = None) -> bytes:
        return self._payload


def _crossref_opener(payload: bytes, captured: dict | None = None):
    def opener(request, timeout=None):
        if captured is not None:
            captured["request"] = request
            captured["timeout"] = timeout
        return _FakeHttpResponse(payload)

    return opener


def _crossref_config(**overrides):
    base = {
        "base_url": "https://api.crossref.org/works",
        "mailto": "test@graph-hypoth.ai",
        "query_max_chars": 300,
        "timeout_seconds": 10.0,
        "select": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


_CROSSREF_PAYLOAD = json.dumps(
    {
        "message": {
            "items": [
                {
                    "title": ["Graph Neural Networks for Materials Discovery"],
                    "DOI": "10.1234/ABC.def",
                    "URL": "https://doi.org/10.1234/abc.def",
                    "author": [
                        {"given": "Ada", "family": "Lovelace"},
                        {"name": "R. Feynman"},
                    ],
                    "issued": {"date-parts": [[2025, 3, 14]]},
                    "abstract": (
                        "<jats:p>We present <jats:italic>novel</jats:italic> "
                        "methods &amp; results.</jats:p>"
                    ),
                    "is-referenced-by-count": 42,
                    "reference": [
                        {"DOI": "10.1/AAA"},
                        {"key": "ref-without-doi"},
                        {"DOI": "10.2/bbb"},
                    ],
                    "container-title": ["Nature Materials"],
                    "type": "journal-article",
                    "subject": ["Materials Science"],
                },
                {"DOI": "10.9/no-title", "author": []},
            ]
        }
    }
).encode("utf-8")


def test_crossref_source_normalizes_items_with_doi_and_abstract() -> None:
    captured: dict = {}
    source = CrossrefPaperSource(
        _crossref_config(),
        opener=_crossref_opener(_CROSSREF_PAYLOAD, captured),
    )

    result = source.search("materials discovery", limit=5, filters=SearchPaperFilters())

    assert result.status.status == "success"
    assert result.status.result_count == 1  # item without a title is skipped
    item = result.results[0]
    assert item.source == "crossref"
    assert item.source_id == "10.1234/abc.def"  # DOI lowercased for cross-source dedup
    assert item.url == "https://doi.org/10.1234/abc.def"
    assert item.text is None  # metadata-only source
    assert item.summary == "We present novel methods & results."  # JATS stripped
    assert item.authors == ["Ada Lovelace", "R. Feynman"]
    assert item.published_date == "2025-03-14"
    assert item.metadata["doi"] == "10.1234/abc.def"
    assert item.metadata["references"] == ["10.1/aaa", "10.2/bbb"]
    assert item.metadata["cited_by_count"] == 42
    assert item.metadata["container_title"] == "Nature Materials"
    # polite-pool: mailto threaded into the request URL
    assert "mailto=test%40graph-hypoth.ai" in captured["request"].full_url
    assert "rows=5" in captured["request"].full_url


def test_crossref_source_reports_zero_works_warning() -> None:
    empty = json.dumps({"message": {"items": []}}).encode("utf-8")
    source = CrossrefPaperSource(
        _crossref_config(), opener=_crossref_opener(empty)
    )

    result = source.search("nothing here", limit=3, filters=SearchPaperFilters())

    assert result.status.status == "success"
    assert result.status.result_count == 0
    assert any("zero works" in warning for warning in result.status.warnings)


def test_crossref_source_reports_failed_status_on_transport_error() -> None:
    def failing_opener(request, timeout=None):
        raise OSError("connection refused")

    source = CrossrefPaperSource(_crossref_config(), opener=failing_opener)

    result = source.search("anything", limit=3, filters=SearchPaperFilters())

    assert result.status.status == "failed"
    assert result.results == []
    assert any("crossref search failed" in error for error in result.status.errors)


def test_crossref_source_truncates_long_query() -> None:
    captured: dict = {}
    source = CrossrefPaperSource(
        _crossref_config(query_max_chars=5),
        opener=_crossref_opener(
            json.dumps({"message": {"items": []}}).encode("utf-8"), captured
        ),
    )

    result = source.search("abcdefgh", limit=1, filters=SearchPaperFilters())

    assert any("truncated" in warning for warning in result.status.warnings)
    assert "query=abcde" in captured["request"].full_url


def test_crossref_source_falls_back_to_config_defaults_when_fields_absent() -> None:
    """Representative regression for the `_get(...)` literal-fallback sweep: a
    config lacking query_max_chars/timeout_seconds must still resolve
    CrossrefSourceConfig's pydantic defaults (300 / 10.0) byte-for-byte, whether
    that resolution comes from a restated literal or the config class itself."""
    captured: dict = {}
    bare_config = SimpleNamespace(
        base_url="https://api.crossref.org/works", mailto=None, select=None
    )
    source = CrossrefPaperSource(
        bare_config,
        opener=_crossref_opener(
            json.dumps({"message": {"items": []}}).encode("utf-8"), captured
        ),
    )

    result = source.search("x" * 400, limit=1, filters=SearchPaperFilters())

    assert captured["timeout"] == 10.0
    assert any("truncated" in warning for warning in result.status.warnings)
    assert "query=" + "x" * 300 in captured["request"].full_url


def _openalex_config(**overrides):
    base = {
        "base_url": "https://api.openalex.org/works",
        "mailto": "test@graph-hypoth.ai",
        "require_api_key_env": "OPENALEX_API_KEY",
        "require_api_key": False,
        "query_max_chars": 300,
        "timeout_seconds": 10.0,
        "select": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


_OPENALEX_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "id": "https://openalex.org/W123",
                "doi": "https://doi.org/10.1234/ABC.def",
                "display_name": "Cross-Domain Claim Verification",
                "publication_date": "2025-06-01",
                "publication_year": 2025,
                "authorships": [
                    {"author": {"display_name": "Grace Hopper"}},
                    {"author": {"display_name": "Alan Turing"}},
                ],
                "abstract_inverted_index": {
                    "Grounded": [0],
                    "retrieval": [1],
                    "matters": [2],
                },
                "referenced_works": [
                    "https://openalex.org/W1",
                    "https://openalex.org/W2",
                ],
                "cited_by_count": 17,
                "cited_by_api_url": "https://api.openalex.org/works?filter=cites:W123",
                "topics": [{"display_name": "Information Retrieval"}],
                "open_access": {
                    "is_oa": True,
                    "oa_status": "green",
                    "oa_url": "https://example.org/paper.pdf",
                },
                "best_oa_location": {"pdf_url": "https://example.org/paper.pdf"},
                "primary_location": {"landing_page_url": "https://example.org/paper"},
            }
        ]
    }
).encode("utf-8")


def test_openalex_source_inverts_abstract_and_captures_graph_metadata() -> None:
    captured: dict = {}
    source = OpenAlexPaperSource(
        _openalex_config(),
        opener=_crossref_opener(_OPENALEX_PAYLOAD, captured),
    )

    result = source.search("claim verification", limit=5, filters=SearchPaperFilters())

    assert result.status.status == "success"
    assert result.status.result_count == 1
    item = result.results[0]
    assert item.source == "openalex"
    assert item.source_id == "10.1234/abc.def"  # bare lowercased DOI for dedup
    assert item.summary == "Grounded retrieval matters"  # inverted abstract
    assert item.authors == ["Grace Hopper", "Alan Turing"]
    assert item.published_date == "2025-06-01"
    assert item.text is None
    assert item.metadata["references"] == [
        "https://openalex.org/W1",
        "https://openalex.org/W2",
    ]
    assert item.metadata["cited_by_count"] == 17
    assert item.metadata["topics"] == ["Information Retrieval"]
    assert item.metadata["is_oa"] is True
    assert item.metadata["oa_url"] == "https://example.org/paper.pdf"
    assert "per_page=5" in captured["request"].full_url
    assert "mailto=test%40graph-hypoth.ai" in captured["request"].full_url


def test_openalex_source_works_keyless_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    captured: dict = {}
    source = OpenAlexPaperSource(
        _openalex_config(),  # require_api_key defaults to False
        opener=_crossref_opener(_OPENALEX_PAYLOAD, captured),
    )

    result = source.search("claim", limit=3, filters=SearchPaperFilters())

    assert result.status.status == "success"  # not skipped
    assert "api_key=" not in captured["request"].full_url


def test_openalex_source_skips_when_key_required_and_absent(monkeypatch) -> None:
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    source = OpenAlexPaperSource(_openalex_config(require_api_key=True))

    result = source.search("claim", limit=3, filters=SearchPaperFilters())

    assert result.status.status == "skipped"
    assert result.status.result_count == 0
    assert any("OPENALEX_API_KEY" in w for w in result.status.warnings)


def test_openalex_source_includes_api_key_when_present(monkeypatch) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "secret-token-not-logged")
    captured: dict = {}
    source = OpenAlexPaperSource(
        _openalex_config(),
        opener=_crossref_opener(_OPENALEX_PAYLOAD, captured),
    )

    source.search("claim", limit=3, filters=SearchPaperFilters())

    assert "api_key=secret-token-not-logged" in captured["request"].full_url


def _europepmc_config(**overrides):
    base = {
        "base_url": "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        "fulltext_base_url": "https://www.ebi.ac.uk/europepmc/webservices/rest",
        "result_type": "core",
        "query_max_chars": 300,
        "timeout_seconds": 10.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


_EUROPEPMC_PAYLOAD = json.dumps(
    {
        "resultList": {
            "result": [
                {
                    "id": "PMC123",
                    "source": "PMC",
                    "pmid": "999",
                    "pmcid": "PMC123",
                    "doi": "10.5555/OA.paper",
                    "title": "Open Access Biomedical Finding",
                    "authorString": "Smith J, Doe A.",
                    "abstractText": "A grounded biomedical abstract.",
                    "firstPublicationDate": "2025-02-10",
                    "pubYear": "2025",
                    "isOpenAccess": "Y",
                    "citedByCount": 8,
                },
                {
                    "id": "MED456",
                    "source": "MED",
                    "doi": "10.5555/closed.paper",
                    "title": "Closed Access Finding",
                    "authorString": "Roe B.",
                    "abstractText": "Abstract only.",
                    "isOpenAccess": "N",
                },
            ]
        }
    }
).encode("utf-8")


def test_europepmc_source_normalizes_and_flags_oa_fulltext_url() -> None:
    source = EuropePmcPaperSource(
        _europepmc_config(), opener=_crossref_opener(_EUROPEPMC_PAYLOAD)
    )

    result = source.search("biomedical", limit=5, filters=SearchPaperFilters())

    assert result.status.status == "success"
    assert result.status.result_count == 2
    oa, closed = result.results
    assert oa.source == "europepmc"
    assert oa.source_id == "10.5555/oa.paper"
    assert oa.summary == "A grounded biomedical abstract."
    assert oa.authors == ["Smith J", "Doe A"]
    assert oa.published_date == "2025-02-10"
    assert oa.metadata["is_oa"] is True
    assert oa.metadata["jats_fulltext_url"] == (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC/PMC123/fullTextXML"
    )
    # Closed access: no JATS full-text URL exposed.
    assert closed.metadata["is_oa"] is False
    assert "jats_fulltext_url" not in closed.metadata


def test_europepmc_source_reports_failed_status_on_transport_error() -> None:
    def failing_opener(request, timeout=None):
        raise OSError("ebi unreachable")

    source = EuropePmcPaperSource(_europepmc_config(), opener=failing_opener)

    result = source.search("x", limit=3, filters=SearchPaperFilters())

    assert result.status.status == "failed"
    assert any("europepmc search failed" in e for e in result.status.errors)


def _query_params(url: str) -> dict[str, list[str]]:
    return urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)


def test_scholarly_sources_treat_doi_like_free_text_query_not_doi_lookup() -> None:
    doi = "10.1234/Guide.Case"

    crossref_capture: dict = {}
    crossref = CrossrefPaperSource(
        _crossref_config(),
        opener=_crossref_opener(
            json.dumps({"message": {"items": []}}).encode("utf-8"),
            crossref_capture,
        ),
    )
    crossref.search(doi, limit=1, filters=SearchPaperFilters())

    crossref_url = crossref_capture["request"].full_url
    assert crossref_url.startswith("https://api.crossref.org/works?")
    assert not crossref_url.startswith(f"https://api.crossref.org/works/{doi}")
    assert _query_params(crossref_url)["query"] == [doi]

    openalex_capture: dict = {}
    openalex = OpenAlexPaperSource(
        _openalex_config(),
        opener=_crossref_opener(
            json.dumps({"results": []}).encode("utf-8"),
            openalex_capture,
        ),
    )
    openalex.search(doi, limit=1, filters=SearchPaperFilters())

    openalex_params = _query_params(openalex_capture["request"].full_url)
    assert openalex_params["search"] == [doi]
    assert "filter" not in openalex_params
    assert "search.semantic" not in openalex_params

    europepmc_capture: dict = {}
    europepmc = EuropePmcPaperSource(
        _europepmc_config(),
        opener=_crossref_opener(
            json.dumps({"resultList": {"result": []}}).encode("utf-8"),
            europepmc_capture,
        ),
    )
    europepmc.search(doi, limit=1, filters=SearchPaperFilters())

    europepmc_url = europepmc_capture["request"].full_url
    assert europepmc_url.startswith("https://www.ebi.ac.uk/europepmc/")
    assert _query_params(europepmc_url)["query"] == [doi]


def test_openalex_source_uses_keyword_search_not_semantic_search() -> None:
    query = "how does retrieval augmented generation improve factual accuracy"
    captured: dict = {}
    source = OpenAlexPaperSource(
        _openalex_config(),
        opener=_crossref_opener(
            json.dumps({"results": []}).encode("utf-8"),
            captured,
        ),
    )

    source.search(query, limit=2, filters=SearchPaperFilters(highlights=True))

    params = _query_params(captured["request"].full_url)
    assert params["search"] == [query]
    assert params["per_page"] == ["2"]
    assert "search.semantic" not in params
    assert "semantic_search" not in params


_GUIDE_MATRIX_CASES = [
    pytest.param("T", True, False, False, False, "search", True, id="01-T"),
    pytest.param("D", False, True, False, False, "contents", False, id="02-D"),
    pytest.param("P", False, False, True, False, "search", True, id="03-P"),
    pytest.param("S", False, False, False, True, "search", True, id="04-S"),
    pytest.param("TD", True, True, False, False, "contents", False, id="05-TD"),
    pytest.param("TP", True, False, True, False, "search", True, id="06-TP"),
    pytest.param("TS", True, False, False, True, "search", True, id="07-TS"),
    pytest.param("DP", False, True, True, False, "contents", False, id="08-DP"),
    pytest.param("DS", False, True, False, True, "contents", False, id="09-DS"),
    pytest.param("PS", False, False, True, True, "search", True, id="10-PS"),
    pytest.param("TDP", True, True, True, False, "contents", False, id="11-TDP"),
    pytest.param("TDS", True, True, False, True, "contents", False, id="12-TDS"),
    pytest.param("TPS", True, False, True, True, "search", True, id="13-TPS"),
    pytest.param("DPS", False, True, True, True, "contents", False, id="14-DPS"),
    pytest.param("TDPS", True, True, True, True, "contents", False, id="15-TDPS"),
]


def _guide_matrix_query(*, title: bool, publisher: bool, semantic: bool) -> str:
    if title:
        return "Language Modeling Is Compression"
    if semantic:
        return "how does retrieval augmented generation improve factual accuracy"
    if publisher:
        return "publisher scoped discovery"
    return ""


def test_exa_result_count_follows_caller_limit(monkeypatch) -> None:
    captured = {}

    class FakeExa:
        def __init__(self, api_key: str) -> None:
            del api_key

        def search(self, **kwargs):
            captured.update(kwargs)
            return {"results": [], "costDollars": 0.0}

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    monkeypatch.setattr("src.retrieval.sources.Exa", FakeExa)
    source = ExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_qps=1000,
            max_cost_dollars_per_run=1.0,
            max_cost_dollars_per_call=1.0,
            search_type="auto",
            category="research paper",
            highlights=False,
            text=False,
        )
    )

    # The requested count is the caller's limit and nothing else. The Exa block carries no
    # result-count knob: `RetrievalService` already derives the per-source limit from
    # `per_source_top_k`, and `_exa_fallback_backfill` deliberately passes the larger
    # `final_top_k` to pull extra pages. A second cap here silently shrank both.
    source.search("bounded results", limit=10, filters=SearchPaperFilters())

    assert captured["num_results"] == 10


@pytest.mark.parametrize(
    (
        "combination",
        "title",
        "doi",
        "publisher",
        "semantic",
        "expected_route",
        "documented_precise_support",
    ),
    _GUIDE_MATRIX_CASES,
)
def test_paper_retrieval_guide_combination_matrix_routes_as_documented(
    monkeypatch,
    combination: str,
    title: bool,
    doi: bool,
    publisher: bool,
    semantic: bool,
    expected_route: str,
    documented_precise_support: bool,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    doi_url = "https://doi.org/10.1234/guide.case"

    class FakeExa:
        def __init__(self, api_key: str) -> None:
            del api_key

        def search(self, **kwargs):
            calls.append(("search", kwargs))
            return {
                "results": [
                    {
                        "id": "search-result",
                        "title": f"Search result for {combination}",
                        "url": "https://example.test/search-result",
                    }
                ],
                "costDollars": 0.001,
            }

        def get_contents(self, urls, **kwargs):
            calls.append(("contents", {"urls": urls, **kwargs}))
            return {
                "results": [
                    {
                        "id": urls[0],
                        "title": f"DOI contents for {combination}",
                        "url": urls[0],
                        "text": "Resolved DOI page text.",
                    }
                ],
                "costDollars": {"total": 0.001},
            }

    monkeypatch.setenv("EXA_API_KEY", "test-secret-not-for-output")
    monkeypatch.setattr(
        "src.retrieval.sources.Exa",
        FakeExa,
    )
    source = ExaPaperSource(
        SimpleNamespace(
            require_api_key_env="EXA_API_KEY",
            max_qps=1000,
            max_cost_dollars_per_run=1.0,
            max_cost_dollars_per_call=1.0,
            search_type="auto",
            category="research paper",
            highlights=False,
            text=False,
        )
    )
    filters = SearchPaperFilters(
        urls=[doi_url] if doi else None,
        include_domains=["nature.com"] if publisher else None,
        text=True if doi else None,
        highlights=True if semantic else None,
    )
    query = _guide_matrix_query(title=title, publisher=publisher, semantic=semantic)

    result = source.search(query, limit=1, filters=filters)

    assert len(calls) == 1
    call_kind, kwargs = calls[0]
    assert call_kind == expected_route
    assert result.status.status == "success"
    assert result.status.metadata["exa_request_kind"] == expected_route

    if expected_route == "search":
        assert documented_precise_support is True
        assert kwargs["query"] == query
        assert kwargs["type"] == "auto"
        assert kwargs["category"] == "research paper"
        assert kwargs["num_results"] == 1
        assert "urls" not in kwargs
        if publisher:
            assert kwargs["include_domains"] == ["nature.com"]
        else:
            assert "include_domains" not in kwargs
        if semantic:
            assert kwargs["contents"] == {
                "highlights": {"query": query, "max_characters": 600}
            }
        else:
            assert kwargs["contents"] is False
        return

    assert documented_precise_support is False
    assert kwargs["urls"] == [doi_url]
    assert "query" not in kwargs
    assert "type" not in kwargs
    assert "category" not in kwargs
    assert "num_results" not in kwargs
    assert "include_domains" not in kwargs
    assert "exclude_domains" not in kwargs
    assert kwargs["text"] == {"max_characters": 5000, "include_html_tags": False}
    if semantic:
        assert kwargs["highlights"] == {"query": query, "max_characters": 600}
    else:
        assert "highlights" not in kwargs


def test_openalex_proxy_key_source_sends_no_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "should-not-be-sent")
    captured: dict = {}
    source = OpenAlexPaperSource(
        _openalex_config(
            key_source="proxy", base_url="https://proxy.internal/openalex"
        ),
        opener=_crossref_opener(_OPENALEX_PAYLOAD, captured),
    )

    result = source.search("claim", limit=2, filters=SearchPaperFilters())

    assert result.status.status == "success"
    assert "api_key=" not in captured["request"].full_url
    assert captured["request"].full_url.startswith("https://proxy.internal/openalex")


def test_crossref_sets_doi_external_id() -> None:
    source = CrossrefPaperSource(
        _crossref_config(), opener=_crossref_opener(_CROSSREF_PAYLOAD)
    )
    result = source.search("x", limit=5, filters=SearchPaperFilters())
    assert result.results[0].external_ids == {"doi": "10.1234/abc.def"}


def test_europepmc_sets_pmid_pmcid_external_ids() -> None:
    source = EuropePmcPaperSource(
        _europepmc_config(), opener=_crossref_opener(_EUROPEPMC_PAYLOAD)
    )
    result = source.search("x", limit=5, filters=SearchPaperFilters())
    oa, closed = result.results
    assert oa.external_ids == {
        "doi": "10.5555/oa.paper",
        "pmid": "999",
        "pmcid": "PMC123",
    }
    assert closed.external_ids == {"doi": "10.5555/closed.paper"}


def test_openalex_captures_ids_block_into_external_ids() -> None:
    payload = json.dumps(
        {
            "results": [
                {
                    "id": "https://openalex.org/W55",
                    "display_name": "Bio Paper",
                    "doi": "https://doi.org/10.7/BIO",
                    "ids": {
                        "doi": "https://doi.org/10.7/bio",
                        "pmid": "https://pubmed.ncbi.nlm.nih.gov/30000001",
                        "pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7000001",
                        "mag": "2099",
                    },
                    "authorships": [],
                    "abstract_inverted_index": {"x": [0]},
                }
            ]
        }
    ).encode("utf-8")
    source = OpenAlexPaperSource(_openalex_config(), opener=_crossref_opener(payload))

    result = source.search("bio", limit=3, filters=SearchPaperFilters())

    # mag is deliberately NOT captured: it is OpenAlex-only and never bridges sources.
    assert result.results[0].external_ids == {
        "doi": "10.7/bio",
        "pmid": "30000001",
        "pmcid": "PMC7000001",
    }


def test_arxiv_external_ids_capture_arxiv_id_and_doi() -> None:
    source = ArxivPaperSource(SimpleNamespace())
    document = SimpleNamespace(
        metadata={
            "Title": "Graph Nets",
            "entry_id": "http://arxiv.org/abs/2401.00001v2",
            "doi": "10.9/ARX",
        },
        page_content="body text",
    )
    result = source._normalize_document(document)
    assert result.external_ids == {"arxiv": "2401.00001", "doi": "10.9/arx"}


def test_normalize_doi_preserves_suffix_after_prefix_and_case_normalization() -> None:
    assert _normalize_doi("https://doi.org/10.1/ABC") == "10.1/abc"
    assert _normalize_doi("10.1021/acs.jcim.2c01099.s001") == (
        "10.1021/acs.jcim.2c01099.s001"
    )
    assert _normalize_doi("10.1234/example.s12") == "10.1234/example.s12"
    assert _normalize_doi(None) is None


def test_crossref_published_date_handles_partial_and_null_date_parts() -> None:
    source = CrossrefPaperSource(_crossref_config())
    assert source._published_date({"issued": {"date-parts": [[2020, None]]}}) == "2020"
    assert source._published_date({"issued": {"date-parts": [[None]]}}) is None
    assert source._published_date({"issued": {"date-parts": [[2019, 3, None]]}}) == "2019-03"


def test_crossref_search_keeps_valid_works_when_one_has_null_date_parts() -> None:
    payload = json.dumps(
        {
            "message": {
                "items": [
                    {
                        "title": ["Bad Date Work"],
                        "DOI": "10.1/bad",
                        "issued": {"date-parts": [[2020, None]]},
                    },
                    {
                        "title": ["Good Work"],
                        "DOI": "10.1/good",
                        "issued": {"date-parts": [[2021, 6, 1]]},
                    },
                ]
            }
        }
    ).encode("utf-8")
    source = CrossrefPaperSource(_crossref_config(), opener=_crossref_opener(payload))

    result = source.search("x", limit=5, filters=SearchPaperFilters())

    assert result.status.status == "success"
    assert result.status.result_count == 2
    titles = {r.title for r in result.results}
    assert titles == {"Bad Date Work", "Good Work"}


def test_openalex_normalize_work_survives_non_dict_open_access() -> None:
    source = OpenAlexPaperSource(_openalex_config())
    result = source._normalize_work(
        {"display_name": "T", "open_access": ["x"], "ids": {}}
    )
    assert result is not None
    assert result.title == "T"
    assert result.metadata["oa_url"] is None
    assert result.metadata["is_oa"] is None


def test_openalex_url_uses_normalized_doi() -> None:
    source = OpenAlexPaperSource(_openalex_config())
    result = source._normalize_work(
        {
            "display_name": "T",
            "doi": "https://doi.org/10.7/ABC",
            "primary_location": {"landing_page_url": "https://example.org/landing"},
        }
    )
    assert result is not None
    assert result.url == "https://doi.org/10.7/abc"


def test_openalex_url_falls_back_to_landing_when_no_doi() -> None:
    source = OpenAlexPaperSource(_openalex_config())
    result = source._normalize_work(
        {
            "display_name": "T",
            "primary_location": {"landing_page_url": "https://example.org/landing"},
        }
    )
    assert result is not None
    assert result.url == "https://example.org/landing"


def test_openalex_search_returns_empty_without_network_when_limit_zero() -> None:
    def exploding_opener(request, timeout=None):
        raise AssertionError("network should not be called when limit=0")

    source = OpenAlexPaperSource(_openalex_config(), opener=exploding_opener)

    result = source.search("x", limit=0, filters=SearchPaperFilters())

    assert result.status.status == "success"
    assert result.status.result_count == 0
    assert result.results == []


def test_europepmc_percent_encodes_record_id_in_urls() -> None:
    payload = json.dumps(
        {
            "resultList": {
                "result": [
                    {
                        "id": "../etc/ pwd",
                        "source": "P MC",
                        "title": "Sneaky Id",
                        "isOpenAccess": "Y",
                    }
                ]
            }
        }
    ).encode("utf-8")
    source = EuropePmcPaperSource(_europepmc_config(), opener=_crossref_opener(payload))

    result = source.search("x", limit=5, filters=SearchPaperFilters())

    item = result.results[0]
    assert "../" not in item.metadata["jats_fulltext_url"]
    assert "%2F" in item.metadata["jats_fulltext_url"]
    assert "../" not in item.url
    assert " " not in item.url


# --- Apify actor source -----------------------------------------------------

def _apify_config(**overrides):
    base = {
        "base_url": "https://api.apify.com",
        "require_api_key_env": "APIFY_API_TOKEN",
        "actor_id": "owner/scholar-scraper",
        "query_field": "query",
        "extra_input": {},
        "query_max_chars": 300,
        "max_items": 15,
        # config.py ApifySourceConfig.timeout_seconds pins 400.0 (a live actor run
        # measured ~292s; the old 60.0 timed out every call -> circuit breaker).
        "timeout_seconds": 400.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _apify_opener(payload: bytes, captured: dict | None = None):
    def opener(request, timeout=None):
        if captured is not None:
            captured["request"] = request
            captured["timeout"] = timeout
        return _FakeHttpResponse(payload)

    return opener


def test_apify_source_skips_when_token_unset(monkeypatch) -> None:
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    source = sources_module.ApifyPaperSource(_apify_config())

    result = source.search("low-rank compression", limit=5, filters=SearchPaperFilters())

    assert result.status.status == "skipped"
    assert result.results == []
    assert any("APIFY_API_TOKEN" in w for w in result.warnings)


def test_apify_source_skips_when_actor_unset(monkeypatch) -> None:
    monkeypatch.setenv("APIFY_API_TOKEN", "tok-not-for-output")
    source = sources_module.ApifyPaperSource(_apify_config(actor_id=""))

    result = source.search("low-rank compression", limit=5, filters=SearchPaperFilters())

    assert result.status.status == "skipped"
    assert any("actor_id" in w for w in result.warnings)


def test_apify_source_maps_actor_items(monkeypatch) -> None:
    monkeypatch.setenv("APIFY_API_TOKEN", "tok-not-for-output")
    payload = json.dumps(
        [
            {
                "title": "TensorGPT compression", "url": "https://x.test/1",
                "authors": ["Ada Lovelace", "R. Feynman"],
                "abstract": "Tensor-train body.", "publishedDate": "2023-07-01",
                "doi": "10.1234/tensorgpt",
            },
            {
                "name": "no-url paper", "description": "alt fields",
                "authors": [{"name": "Grace Hopper"}],
            },
            {"description": "no title -> skipped"},
        ]
    ).encode("utf-8")
    captured: dict = {}
    source = sources_module.ApifyPaperSource(
        _apify_config(actor_id="owner/scholar-scraper"),
        opener=_apify_opener(payload, captured),
    )

    result = source.search("compression", limit=5, filters=SearchPaperFilters())

    assert result.status.status == "success"
    assert result.status.result_count == 2  # the title-less item is skipped
    first = result.results[0]
    assert first.source == "apify"
    assert first.title == "TensorGPT compression"
    assert first.url == "https://x.test/1"
    assert first.text == "Tensor-train body."
    assert first.authors == ["Ada Lovelace", "R. Feynman"]
    assert first.external_ids["doi"] == "10.1234/tensorgpt"  # normalized
    assert result.results[1].title == "no-url paper"
    assert result.results[1].authors == ["Grace Hopper"]
    # owner/name -> owner~name in the run-sync REST path; token + POST threaded
    assert "owner~scholar-scraper/run-sync-get-dataset-items" in captured["request"].full_url
    assert "token=tok-not-for-output" in captured["request"].full_url
    assert captured["request"].method == "POST"


def test_apify_timeout_fallback_matches_config_default(monkeypatch) -> None:
    """Regression for the drifted `_get(self.config, "timeout_seconds", 60.0)`
    fallback: config.py's ApifySourceConfig pins 400.0 (a live actor run measured
    ~292s), so a duck-typed config lacking timeout_seconds must resolve 400.0,
    not the stale 60.0 that timed out every call."""
    monkeypatch.setenv("APIFY_API_TOKEN", "tok-not-for-output")
    captured: dict = {}
    config_without_timeout = SimpleNamespace(
        base_url="https://api.apify.com",
        require_api_key_env="APIFY_API_TOKEN",
        actor_id="owner/scholar-scraper",
        query_field="query",
        extra_input={},
        query_max_chars=300,
        max_items=15,
        # timeout_seconds intentionally omitted -> must fall back to config default
    )
    source = sources_module.ApifyPaperSource(
        config_without_timeout,
        opener=_apify_opener(json.dumps([]).encode("utf-8"), captured),
    )

    source.search("low-rank compression", limit=5, filters=SearchPaperFilters())

    assert captured["timeout"] == 400.0


def test_source_base_urls_fall_back_to_config_class_defaults() -> None:
    """Regression: the module-level `_XXX_DEFAULT_URL` constants used as `_get(...)`
    fallbacks must resolve the same URL as the real config.py pydantic defaults,
    whether that resolution comes from a restated literal or the config class
    itself (guards crossref/openalex/europepmc/apify base_url drift)."""
    captured: dict = {}

    crossref = sources_module.CrossrefPaperSource(
        SimpleNamespace(mailto=None, select=None),
        opener=_crossref_opener(json.dumps({"message": {"items": []}}).encode("utf-8"), captured),
    )
    crossref.search("q", limit=1, filters=SearchPaperFilters())
    assert captured["request"].full_url.startswith(
        CrossrefSourceConfig.model_fields["base_url"].default
    )

    openalex = sources_module.OpenAlexPaperSource(
        SimpleNamespace(require_api_key=False, key_source="env"),
        opener=_crossref_opener(json.dumps({"results": []}).encode("utf-8"), captured),
    )
    openalex.search("q", limit=1, filters=SearchPaperFilters())
    assert captured["request"].full_url.startswith(
        OpenAlexSourceConfig.model_fields["base_url"].default
    )

    europepmc_search_payload = json.dumps({"resultList": {"result": []}}).encode("utf-8")
    europepmc = sources_module.EuropePmcPaperSource(
        SimpleNamespace(), opener=_crossref_opener(europepmc_search_payload, captured)
    )
    europepmc.search("q", limit=1, filters=SearchPaperFilters())
    assert captured["request"].full_url.startswith(
        EuropePmcSourceConfig.model_fields["base_url"].default
    )
    oa_record = europepmc._normalize_record(
        {"id": "PMC1", "source": "PMC", "title": "t", "isOpenAccess": "Y"}
    )
    assert oa_record.metadata["jats_fulltext_url"].startswith(
        EuropePmcSourceConfig.model_fields["fulltext_base_url"].default
    )

    apify = sources_module.ApifyPaperSource(
        SimpleNamespace(actor_id="owner/actor", query_field="query", extra_input={}),
    )
    assert apify._build_url("owner/actor", "tok", 1).startswith(
        ApifySourceConfig.model_fields["base_url"].default
    )


def test_source_user_agents_share_single_constant(monkeypatch) -> None:
    """Regression: crossref/openalex/europepmc/apify must all send the same
    User-Agent as `sources_module.USER_AGENT` (crossref appends `mailto:` when
    configured), guarding against the literal drifting between call sites."""
    monkeypatch.setenv("APIFY_API_TOKEN", "tok-not-for-output")
    captured: dict = {}

    crossref = sources_module.CrossrefPaperSource(
        SimpleNamespace(mailto=None, select=None),
        opener=_crossref_opener(json.dumps({"message": {"items": []}}).encode("utf-8"), captured),
    )
    crossref.search("q", limit=1, filters=SearchPaperFilters())
    assert captured["request"].get_header("User-agent") == sources_module.USER_AGENT

    openalex = sources_module.OpenAlexPaperSource(
        SimpleNamespace(require_api_key=False, key_source="env"),
        opener=_crossref_opener(json.dumps({"results": []}).encode("utf-8"), captured),
    )
    openalex.search("q", limit=1, filters=SearchPaperFilters())
    assert captured["request"].get_header("User-agent") == sources_module.USER_AGENT

    europepmc = sources_module.EuropePmcPaperSource(
        SimpleNamespace(),
        opener=_crossref_opener(json.dumps({"resultList": {"result": []}}).encode("utf-8"), captured),
    )
    europepmc.search("q", limit=1, filters=SearchPaperFilters())
    assert captured["request"].get_header("User-agent") == sources_module.USER_AGENT

    apify = sources_module.ApifyPaperSource(
        SimpleNamespace(actor_id="owner/actor", query_field="query", extra_input={}),
        opener=_apify_opener(json.dumps([]).encode("utf-8"), captured),
    )
    apify.search("q", limit=1, filters=SearchPaperFilters())
    assert captured["request"].get_header("User-agent") == sources_module.USER_AGENT


@pytest.mark.live_apify
def test_apify_live_returns_results() -> None:  # opt-in; needs token and actor
    import os

    token = os.environ.get("APIFY_API_TOKEN")
    actor = os.environ.get("APIFY_ACTOR_ID")
    if not token or not actor:
        pytest.skip("APIFY_API_TOKEN / APIFY_ACTOR_ID not set")
    source = sources_module.ApifyPaperSource(_apify_config(actor_id=actor))

    result = source.search(
        "large language model compression", limit=3, filters=SearchPaperFilters()
    )

    assert result.status.status in {"success", "failed"}
    if result.status.status == "success":
        assert all(r.title for r in result.results)
