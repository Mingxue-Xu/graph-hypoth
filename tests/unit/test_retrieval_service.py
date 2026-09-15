from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from src.config import RetrievalConfig as AppRetrievalConfig
from src.events import stable_hash_payload
from src.log_store import SQLiteLogStore
from src.retrieval import service as service_module
from src.retrieval.models import (
    SearchPaperFilters,
    SourceResult,
    SourceStatus,
)
from src.retrieval.resilience import ResilientSource
from src.retrieval.service import RetrievalService
from src.retrieval.sources import SourceSearchResult


@dataclass
class ToolBudgetConfig:
    max_calls_per_run: int | None = 12
    max_calls_per_agent_round: int | None = None


@dataclass
class RankingConfig:
    source_order: list[str] = field(default_factory=lambda: ["arxiv", "exa", "fake"])
    trust_tier_priority: dict[str, int] = field(
        default_factory=lambda: {
            "authoritative_preprint": 100,
            "web_research_paper": 60,
            "deterministic_test": 100,
        }
    )


@dataclass
class RetrievalConfig:
    sources: list[str] = field(default_factory=lambda: ["arxiv"])
    optional_sources: list[str] = field(default_factory=list)
    final_top_k: int = 5
    per_source_top_k: int = 5
    fake_mode: bool = False
    cache_replay: bool = True
    ranking: RankingConfig = field(default_factory=RankingConfig)
    tool_budget: ToolBudgetConfig = field(default_factory=ToolBudgetConfig)
    source_limits: dict[str, Any] = field(default_factory=dict)
    trust_policy: dict[str, str] = field(
        default_factory=lambda: {
            "arxiv": "authoritative_preprint",
            "exa": "web_research_paper",
            "fake": "deterministic_test",
        }
    )

    def selected_source_names(self) -> list[str]:
        if self.fake_mode:
            return ["fake"]
        return [*self.sources, *self.optional_sources]


class RecordingSource:
    def __init__(
        self,
        name: str,
        *,
        result_count: int = 1,
        status: str = "success",
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []
        self.result_count = result_count
        self.status = status
        self.warnings = warnings or []
        self.errors = errors or []

    def search(self, query: str, *, limit: int, filters: Any) -> SourceSearchResult:
        self.calls.append({"query": query, "limit": limit, "filters": filters})
        results = [
            SourceResult(
                source=self.name,
                source_id=f"{self.name}-{index}",
                title=f"{self.name} paper {index}",
                authors=["Example Author"],
                published_date="2026-05-10",
                url=f"https://example.test/{self.name}/{index}",
                text=f"{self.name} evidence {index} for {query}",
                score=1.0 / index,
                metadata={"index": index},
            )
            for index in range(1, self.result_count + 1)
        ]
        return SourceSearchResult(
            results=results,
            status=SourceStatus(
                source=self.name,
                status=self.status,  # type: ignore[arg-type]
                source_query=query,
                result_count=len(results),
                warnings=list(self.warnings),
                errors=list(self.errors),
            ),
            warnings=list(self.warnings),
            errors=list(self.errors),
        )


class StaticSource:
    def __init__(self, name: str, results: list[SourceResult]) -> None:
        self.name = name
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, *, limit: int, filters: Any) -> SourceSearchResult:
        self.calls.append({"query": query, "limit": limit, "filters": filters})
        results = self.results[:limit]
        return SourceSearchResult(
            results=results,
            status=SourceStatus(
                source=self.name,
                status="success",
                source_query=query,
                result_count=len(results),
            ),
        )


def _store(tmp_path: Path) -> SQLiteLogStore:
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    return store


def _service(
    tmp_path: Path,
    config: RetrievalConfig | None = None,
) -> RetrievalService:
    return RetrievalService(config=config or RetrievalConfig(), log_store=_store(tmp_path))


def test_service_wraps_sources_with_resilience_when_enabled(tmp_path: Path) -> None:
    service = RetrievalService(
        config=AppRetrievalConfig(),
        log_store=_store(tmp_path),
    )

    assert isinstance(service.sources["arxiv"], ResilientSource)


def test_service_leaves_sources_unwrapped_when_resilience_disabled(
    tmp_path: Path,
) -> None:
    config = AppRetrievalConfig()
    config.resilience.enabled = False
    service = RetrievalService(config=config, log_store=_store(tmp_path))

    assert not isinstance(service.sources["arxiv"], ResilientSource)


def test_fake_mode_forces_fake_source_even_when_real_sources_requested(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, RetrievalConfig(fake_mode=True))
    fake = RecordingSource("fake")
    arxiv = RecordingSource("arxiv")
    exa = RecordingSource("exa")
    service.sources = {"fake": fake, "arxiv": arxiv, "exa": exa}

    result = service.search_papers(
        run_id="run-1",
        query="retrieval augmented debate",
        sources=["arxiv", "exa"],
        limit=2,
        filters=None,
        called_by="builder",
        tool_call_id="tool-1",
    )

    assert result.sources == ["fake"]
    assert len(fake.calls) == 1
    assert arxiv.calls == []
    assert exa.calls == []


def test_unsupported_source_raises_value_error(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="unsupported retrieval source: semantic_scholar"):
        service.search_papers(
            run_id="run-1",
            query="retrieval augmented debate",
            sources=["semantic_scholar"],
            limit=2,
            filters=None,
            called_by="builder",
            tool_call_id="tool-1",
        )


def test_configured_sources_are_selected_when_sources_are_not_requested(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path,
        RetrievalConfig(sources=["arxiv"], optional_sources=["exa"]),
    )
    arxiv = RecordingSource("arxiv")
    exa = RecordingSource("exa")
    service.sources = {"fake": RecordingSource("fake"), "arxiv": arxiv, "exa": exa}

    result = service.search_papers(
        run_id="run-1",
        query="retrieval augmented debate",
        sources=None,
        limit=3,
        filters=None,
        called_by="builder",
        tool_call_id="tool-1",
    )

    assert result.sources == ["arxiv", "exa"]
    assert len(arxiv.calls) == 1
    assert len(exa.calls) == 1


def test_cache_replay_returns_cache_hit_without_calling_source_again(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    arxiv = RecordingSource("arxiv")
    service.sources = {"fake": RecordingSource("fake"), "arxiv": arxiv, "exa": RecordingSource("exa")}

    first = service.search_papers(
        run_id="run-1",
        query="retrieval augmented debate",
        sources=["arxiv"],
        limit=2,
        filters={"text": True},
        called_by="builder",
        tool_call_id="tool-1",
    )
    second = service.search_papers(
        run_id="run-1",
        query="retrieval augmented debate",
        sources=["arxiv"],
        limit=2,
        filters={"text": True},
        called_by="verifier",
        tool_call_id="tool-2",
    )

    assert len(arxiv.calls) == 1
    assert second.tool_call_id == "tool-2"
    assert second.input_hash == first.input_hash
    assert second.source_statuses[0].metadata["cache"] == "hit"
    assert first.evidence[0]["metadata"]["retrieval_batch_id"]
    assert (
        second.evidence[0]["metadata"]["retrieval_batch_id"]
        == first.evidence[0]["metadata"]["retrieval_batch_id"]
    )


def test_failed_result_cache_replay_preserves_diagnostics(tmp_path: Path) -> None:
    service = _service(tmp_path, RetrievalConfig(sources=["exa"]))
    exa = RecordingSource(
        "exa",
        result_count=0,
        status="failed",
        warnings=["provider warning"],
        errors=["provider unavailable"],
    )
    service.sources = {"exa": exa}

    first = service.search_papers(
        run_id="run-failed",
        query="failed query",
        sources=["exa"],
        limit=2,
        filters=None,
        called_by="retrieve_evidence",
        tool_call_id="tool-1",
    )
    replay = service.search_papers(
        run_id="run-failed",
        query="failed query",
        sources=["exa"],
        limit=2,
        filters=None,
        called_by="retrieve_evidence",
        tool_call_id="tool-2",
    )

    assert len(exa.calls) == 1
    assert first.source_statuses[0].status == "failed"
    assert replay.source_statuses[0].status == "failed"
    assert replay.source_statuses[0].metadata["cache"] == "hit"
    assert replay.warnings == ["provider warning"]
    assert replay.errors == ["provider unavailable"]


def test_distinct_retrieval_queries_have_distinct_batch_ids(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.sources = {"arxiv": RecordingSource("arxiv")}

    first = service.search_papers(
        run_id="run-1",
        query="first query",
        sources=["arxiv"],
        limit=1,
        filters=None,
        called_by="builder",
        tool_call_id="tool-1",
    )
    second = service.search_papers(
        run_id="run-1",
        query="second query",
        sources=["arxiv"],
        limit=1,
        filters=None,
        called_by="builder",
        tool_call_id="tool-2",
    )

    assert first.evidence[0]["evidence_id"] == "ev_000001"
    assert second.evidence[0]["evidence_id"] == "ev_000001"
    assert (
        first.evidence[0]["metadata"]["retrieval_batch_id"]
        != second.evidence[0]["metadata"]["retrieval_batch_id"]
    )


def test_query_reranker_scores_complete_pool_before_global_limit(
    tmp_path: Path,
) -> None:
    ranking = RankingConfig(
        source_order=["crossref", "openalex", "codex_web"],
        trust_tier_priority={"metadata": 100, "web_research_paper": 10},
    )
    config = RetrievalConfig(
        sources=["crossref", "openalex"],
        optional_sources=["codex_web"],
        final_top_k=5,
        per_source_top_k=5,
        ranking=ranking,
        trust_policy={
            "crossref": "metadata",
            "openalex": "metadata",
            "codex_web": "web_research_paper",
        },
    )

    def result(source: str, index: int, text: str) -> SourceResult:
        return SourceResult(
            source=source,
            source_id=f"{source}-{index}",
            title=f"{source} methods record {index}: {text}",
            authors=[],
            published_date="2026-01-01",
            url=f"https://example.test/{source}/{index}",
            text=text,
            score=1.0 / index,
        )

    irrelevant_crossref = [
        result("crossref", index, "marine ecology population survey")
        for index in range(1, 4)
    ]
    irrelevant_openalex = [
        result("openalex", index, "clinical records cohort analysis")
        for index in range(1, 4)
    ]
    relevant_web = [
        result(
            "codex_web",
            1,
            "activation-aware structured pruning ablation benchmark",
        )
    ]

    def embed(texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0]
            if "activation-aware" in text.lower()
            else [0.0, 1.0]
            for text in texts
        ]

    service = RetrievalService(
        config=config,
        log_store=_store(tmp_path),
        rerank_embedder=embed,
    )
    service.sources = {
        "crossref": StaticSource("crossref", irrelevant_crossref),
        "openalex": StaticSource("openalex", irrelevant_openalex),
        "codex_web": StaticSource("codex_web", relevant_web),
    }

    result_payload = service.search_papers(
        run_id="run-1",
        query="activation-aware pruning experiment methods",
        sources=None,
        limit=5,
        filters=None,
        called_by="experiment_design",
        tool_call_id="tool-1",
    )

    assert len(result_payload.evidence) == 5
    assert result_payload.evidence[0]["source"] == "codex_web"
    assert result_payload.evidence[0]["rank"] == 1
    assert result_payload.evidence[0]["metadata"]["query_relevance"] == 1.0
    assert result_payload.evidence[0]["metadata"]["query_reranker"]


def test_query_reranker_is_opt_in_and_has_separate_cache_identity(
    tmp_path: Path,
) -> None:
    config = RetrievalConfig(
        sources=["arxiv"],
        optional_sources=["exa"],
        final_top_k=1,
        per_source_top_k=1,
    )
    store = _store(tmp_path)

    legacy = RetrievalService(config=config, log_store=store)
    legacy.sources = {
        "arxiv": RecordingSource("arxiv"),
        "exa": RecordingSource("exa"),
    }
    legacy_result = legacy.search_papers(
        run_id="run-1",
        query="relevant experiment",
        sources=None,
        limit=1,
        filters=None,
        called_by="builder",
        tool_call_id="tool-1",
    )

    calls = [0]

    def embed(texts: list[str]) -> list[list[float]]:
        calls[0] += 1
        return [[1.0] for _text in texts]

    reranked = RetrievalService(
        config=config,
        log_store=store,
        rerank_embedder=embed,
    )
    reranked.sources = {
        "arxiv": RecordingSource("arxiv"),
        "exa": RecordingSource("exa"),
    }
    reranked_result = reranked.search_papers(
        run_id="run-1",
        query="relevant experiment",
        sources=None,
        limit=1,
        filters=None,
        called_by="experiment_design",
        tool_call_id="tool-2",
    )
    replay = reranked.search_papers(
        run_id="run-1",
        query="relevant experiment",
        sources=None,
        limit=1,
        filters=None,
        called_by="experiment_design",
        tool_call_id="tool-3",
    )

    assert "query_relevance" not in legacy_result.evidence[0]["metadata"]
    legacy_config_hash = stable_hash_payload(legacy._config_payload())
    assert legacy_result.input_hash == stable_hash_payload(
        {
            "tool_name": "search_papers",
            "query": "relevant experiment",
            "sources": ["arxiv", "exa"],
            "limit": 1,
            "filters": legacy._filters_payload(legacy._parse_filters(None)),
            "retrieval_config_hash": legacy_config_hash,
        }
    )
    assert reranked_result.input_hash != legacy_result.input_hash
    assert (
        reranked_result.evidence[0]["metadata"]["retrieval_batch_id"]
        != legacy_result.evidence[0]["metadata"]["retrieval_batch_id"]
    )
    assert calls[0] == 2  # query batch + paper batch; replay uses the cached result
    assert replay.evidence == [
        {
            **reranked_result.evidence[0],
            "tool_call_id": "tool-3",
        }
    ]


def test_partial_source_failure_is_preserved_with_successful_evidence(
    tmp_path: Path,
) -> None:
    service = _service(
        tmp_path,
        RetrievalConfig(sources=["arxiv"], optional_sources=["exa"]),
    )
    service.sources = {
        "fake": RecordingSource("fake"),
        "arxiv": RecordingSource("arxiv"),
        "exa": RecordingSource(
            "exa",
            result_count=0,
            status="partial_failure",
            warnings=["exa warning"],
            errors=["exa timed out"],
        ),
    }

    result = service.search_papers(
        run_id="run-1",
        query="retrieval augmented debate",
        sources=None,
        limit=5,
        filters=None,
        called_by="builder",
        tool_call_id="tool-1",
    )
    payload = service.result_to_dict(result)

    assert [status.source for status in result.source_statuses] == ["arxiv", "exa"]
    assert result.source_statuses[1].status == "partial_failure"
    assert result.errors == ["exa timed out"]
    assert result.warnings == ["exa warning"]
    assert [item["source"] for item in payload["evidence"]] == ["arxiv"]
    assert isinstance(payload, dict)
    assert isinstance(payload["source_statuses"][0], dict)


def test_exa_fallback_backfills_when_a_non_exa_source_fails(tmp_path: Path) -> None:
    # arxiv is the only selected source and it fails; exa is the safety-net fallback that
    # retries the same keyword search and backfills the pool.
    service = _service(
        tmp_path,
        RetrievalConfig(sources=["arxiv"], optional_sources=[], per_source_top_k=2),
    )
    exa = RecordingSource("exa", result_count=2)
    service.sources = {
        "arxiv": RecordingSource("arxiv", result_count=0, status="failed"),
        "exa": exa,
    }

    result = service.search_papers(
        run_id="run-1",
        query="visual language models text usage efficiency",
        sources=None,
        limit=5,
        filters=None,
        called_by="builder",
        tool_call_id="tool-1",
    )

    statuses = {status.source: status for status in result.source_statuses}
    assert "exa_fallback" in statuses
    assert statuses["exa_fallback"].result_count == 2
    # exa was queried once, at the full result_limit (to backfill), not per_source_top_k=2
    assert len(exa.calls) == 1
    assert exa.calls[0]["limit"] == 5
    # the pool is filled from exa even though the selected source returned nothing
    sources = [item["source"] for item in service.result_to_dict(result)["evidence"]]
    assert sources == ["exa", "exa"]


def test_exa_fallback_is_skipped_when_selected_sources_succeed(tmp_path: Path) -> None:
    service = _service(tmp_path, RetrievalConfig(sources=["arxiv"], optional_sources=[]))
    exa = RecordingSource("exa", result_count=2)
    service.sources = {"arxiv": RecordingSource("arxiv", result_count=1), "exa": exa}

    result = service.search_papers(
        run_id="run-1", query="q", sources=None, limit=5, filters=None,
        called_by="builder", tool_call_id="tool-1",
    )

    assert "exa_fallback" not in {status.source for status in result.source_statuses}
    assert exa.calls == []  # exa never queried because arxiv succeeded


def test_exa_fallback_fires_on_partial_failure_even_with_some_results(tmp_path: Path) -> None:
    # arxiv returns 1 result but is flagged partial_failure (rate-limited / truncated PDF);
    # exa still backfills additively (the arxiv result is kept).
    service = _service(tmp_path, RetrievalConfig(sources=["arxiv"], optional_sources=[]))
    exa = RecordingSource("exa", result_count=2)
    service.sources = {
        "arxiv": RecordingSource("arxiv", result_count=1, status="partial_failure"),
        "exa": exa,
    }

    result = service.search_papers(
        run_id="run-1", query="q", sources=None, limit=5, filters=None,
        called_by="builder", tool_call_id="tool-1",
    )

    statuses = {status.source: status for status in result.source_statuses}
    assert statuses["exa_fallback"].result_count == 2
    assert len(exa.calls) == 1
    sources = sorted(item["source"] for item in service.result_to_dict(result)["evidence"])
    assert sources == ["arxiv", "exa", "exa"]  # arxiv result kept, exa backfilled additively


def test_service_uses_exa_highlight_query_for_quote_selection(
    tmp_path: Path,
) -> None:
    class ExaHighlightSource(RecordingSource):
        def __init__(self) -> None:
            super().__init__("exa")

        def search(self, query: str, *, limit: int, filters: Any) -> SourceSearchResult:
            self.calls.append({"query": query, "limit": limit, "filters": filters})
            return SourceSearchResult(
                results=[
                    SourceResult(
                        source="exa",
                        source_id="exa-highlight-query",
                        title="Exa Highlight Query",
                        text=(
                            "Discovery words only describe the paper title. "
                            "Compression lowers perplexity for language models."
                        ),
                        summary="Generated summary.",
                        metadata={
                            "exa_highlights": ["A clean but unrelated highlight."],
                            "exa_highlight_query": "compression perplexity",
                        },
                    )
                ],
                status=SourceStatus(
                    source="exa",
                    status="success",
                    source_query=query,
                    result_count=1,
                ),
            )

    service = _service(tmp_path, RetrievalConfig(sources=["exa"], cache_replay=False))
    exa = ExaHighlightSource()
    service.sources = {"fake": RecordingSource("fake"), "arxiv": RecordingSource("arxiv"), "exa": exa}

    result = service.search_papers(
        run_id="run-quote-selection",
        query="discovery paper title",
        sources=["exa"],
        limit=1,
        filters={"highlight_query": "compression perplexity"},
        called_by="builder",
        tool_call_id="tool-quote-selection",
    )

    evidence = result.evidence[0]
    assert evidence["quote"] == "Compression lowers perplexity for language models."
    assert evidence["metadata"]["quote_selection"] == {
        "source": "text",
        "highlight_index": None,
        "query": "compression perplexity",
        "candidate_excerpt": "Compression lowers perplexity for language models.",
        "verified_quote": "Compression lowers perplexity for language models.",
        "verification_status": "accepted",
        "match_type": "exact",
    }


def test_service_search_enriches_coherence_fields_from_source_references(
    tmp_path: Path,
) -> None:
    config = AppRetrievalConfig(
        sources=["crossref"],
        optional_sources=["openalex"],
        cache_replay=False,
    )
    config.coherence.embedding_enabled = False
    config.coherence.embedding_weight = 0.0
    config.coherence.citation_weight = 1.0
    config.ranking.source_order = ["crossref", "openalex"]
    service = RetrievalService(config=config, log_store=_store(tmp_path))
    service.sources = {
        "crossref": StaticSource(
            "crossref",
            [
                SourceResult(
                    source="crossref",
                    source_id="10.5/a",
                    title="Citing Paper",
                    text="Citing paper text.",
                    external_ids={"doi": "10.5/a"},
                    metadata={"doi": "10.5/a", "references": ["10.5/b"]},
                )
            ],
        ),
        "openalex": StaticSource(
            "openalex",
            [
                SourceResult(
                    source="openalex",
                    source_id="10.5/b",
                    title="Cited Paper",
                    text="Cited paper text.",
                    external_ids={"doi": "10.5/b"},
                    metadata={"doi": "10.5/b", "references": []},
                )
            ],
        ),
    }

    result = service.search_papers(
        run_id="run-coherence",
        query="citation relatedness",
        sources=None,
        limit=5,
        filters=None,
        called_by="builder",
        tool_call_id="tool-coherence",
    )

    by_source = {item["source"]: item for item in result.evidence}
    assert by_source["crossref"]["metadata"]["r2"]["citation"] == 1.0
    assert by_source["openalex"]["metadata"]["r2"]["citation"] == 1.0
    assert by_source["crossref"]["relatedness_score"] == 1.0
    assert by_source["openalex"]["relatedness_score"] == 1.0


def test_augment_full_text_is_opt_in_and_upgrades_oa_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str | None] = []

    def fake_fetch(result, config):  # type: ignore[no-untyped-def]
        calls.append(result.source_id)
        return "FULL TEXT BODY"

    monkeypatch.setattr(service_module, "fetch_oa_fulltext", fake_fetch)
    service = _service(tmp_path)
    oa = SourceResult(
        source="openalex",
        source_id="10.1/x",
        title="t",
        metadata={"is_oa": True, "oa_url": "https://example.org/p.pdf"},
    )

    # No text filter -> no fetch, no upgrade.
    service._augment_full_text([oa], SearchPaperFilters(), [])
    assert calls == []
    assert oa.text is None

    # filters.text=True -> opt-in fetch upgrades the text in place.
    service._augment_full_text([oa], SearchPaperFilters(text=True), [])
    assert calls == ["10.1/x"]
    assert oa.text == "FULL TEXT BODY"


class _BareTopKConfig:
    """Duck-typed config that omits final_top_k/per_source_top_k entirely (unlike
    the ``RetrievalConfig`` dataclass above, whose class-level literal defaults
    would survive a plain ``del instance.attr`` and mask the fallback path)."""

    def __init__(self) -> None:
        self.sources = ["arxiv"]
        self.optional_sources: list[str] = []
        self.fake_mode = False
        self.cache_replay = True
        self.ranking = RankingConfig()
        self.tool_budget = ToolBudgetConfig()
        self.source_limits: dict[str, Any] = {}
        self.trust_policy = {"arxiv": "authoritative_preprint"}

    def selected_source_names(self) -> list[str]:
        return self.sources


def test_search_papers_final_top_k_fallback_matches_config_default(
    tmp_path: Path,
) -> None:
    """The `_get(self.config, "final_top_k", 8)` fallback must track
    src.config.RetrievalConfig.final_top_k's pydantic default
    rather than restate it as an independent literal that can silently drift."""
    service = _service(tmp_path, _BareTopKConfig())
    source = RecordingSource("arxiv", result_count=20)
    service.sources = {"arxiv": source}

    service.search_papers(
        run_id="run-1",
        query="retrieval augmented debate",
        sources=None,
        limit=None,
        filters=None,
        called_by="builder",
        tool_call_id="tool-1",
    )

    assert source.calls[0]["limit"] == AppRetrievalConfig.model_fields["final_top_k"].default
