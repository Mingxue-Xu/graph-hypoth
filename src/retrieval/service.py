from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from src.config import RetrievalConfig as AppRetrievalConfig
from src.events import stable_hash_payload
from src.log_store import SQLiteLogStore
from src.progress import progress_operation
from src.retrieval._util import dump as _base_dump
from src.retrieval._util import get_attr_or_key as _get
from src.retrieval.claude_web import (
    ClaudeWebPaperSource,
    claude_web_call_budget_for,
)
from src.retrieval.codex_web import (
    CodexWebPaperSource,
    codex_web_call_budget_for,
)
from src.retrieval.coherence import EvidenceEnricher
from src.retrieval.fulltext import fetch_oa_fulltext
from src.retrieval.identity import retrieval_batch_id
from src.retrieval.models import SearchPaperFilters
from src.retrieval.resilience import ResilientSource
from src.retrieval.similarity import Embedder, _cosine
from src.retrieval.sources import (
    ApifyPaperSource,
    ArxivPaperSource,
    CrossrefPaperSource,
    EuropePmcPaperSource,
    ExaPaperSource,
    FakePaperSource,
    OpenAlexPaperSource,
    PaperSource,
    SourceResult,
    SourceSearchResult,
    SourceStatus,
)

try:
    from src.retrieval.ledger import EvidenceLedger
except ModuleNotFoundError:
    EvidenceLedger = None  # type: ignore[assignment, misc]

try:
    from src.retrieval.models import RetrievalToolResult  # type: ignore
except ModuleNotFoundError:

    @dataclass
    class RetrievalToolResult:  # type: ignore[no-redef]
        tool_call_id: str
        query: str
        filters: dict[str, Any]
        sources: list[str]
        evidence: list[dict[str, Any]]
        source_statuses: list[SourceStatus]
        warnings: list[str] = field(default_factory=list)
        errors: list[str] = field(default_factory=list)
        elapsed_ms: int = 0
        input_hash: str = ""
        retrieval_config_hash: str = ""


def _cfg_default(cls: type[BaseModel], name: str) -> Any:
    """The pydantic field default for ``cls.name`` in config.py — so an inline
    ``_get(self.config, name, ...)`` fallback tracks the real config default
    instead of restating it as a literal that can silently drift."""
    return cls.model_fields[name].default


def _dump(value: Any) -> Any:
    return _base_dump(value, tuples=True, str_keys=True, callables=True)


_QUERY_RERANKER_ID = "query-embedding-cosine-v1"


class RetrievalService:
    def __init__(
        self,
        config: Any,
        log_store: SQLiteLogStore,
        *,
        rerank_embedder: Embedder | None = None,
    ) -> None:
        self.config = config
        self.log_store = log_store
        self.rerank_embedder = rerank_embedder
        source_limits = _get(config, "source_limits")
        raw_sources: dict[str, PaperSource] = {
            "fake": FakePaperSource(),
            "arxiv": ArxivPaperSource(_get(source_limits, "arxiv")),
            "exa": ExaPaperSource(_get(source_limits, "exa")),
            "crossref": CrossrefPaperSource(_get(source_limits, "crossref")),
            "openalex": OpenAlexPaperSource(_get(source_limits, "openalex")),
            "europepmc": EuropePmcPaperSource(_get(source_limits, "europepmc")),
            "apify": ApifyPaperSource(_get(source_limits, "apify")),
        }
        selector = _get(config, "selected_source_names")
        if callable(selector):
            configured_sources = set(selector())
        else:
            configured_sources = set(_get(config, "sources", []) or [])
            configured_sources.update(
                _get(config, "optional_sources", []) or []
            )
        # Keep this potentially paid source completely absent unless it was
        # explicitly selected. Its runner also resolves the CLI lazily.
        if "codex_web" in configured_sources:
            raw_sources["codex_web"] = CodexWebPaperSource(
                _get(source_limits, "codex_web"),
                call_budget=codex_web_call_budget_for(log_store),
            )
        # Same rule as codex_web: a paid, tool-enabled subprocess source stays
        # completely absent unless it was explicitly selected.
        if "claude_web" in configured_sources:
            raw_sources["claude_web"] = ClaudeWebPaperSource(
                _get(source_limits, "claude_web"),
                call_budget=claude_web_call_budget_for(log_store),
            )
        self.sources: dict[str, PaperSource] = {
            name: self._maybe_wrap_resilient(name, source)
            for name, source in raw_sources.items()
        }

    def _maybe_wrap_resilient(self, name: str, source: PaperSource) -> PaperSource:
        resilience = _get(self.config, "resilience")
        if resilience is None or not _get(resilience, "enabled", True):
            return source
        policy_for = getattr(resilience, "policy_for", None)
        if callable(policy_for):
            policy = policy_for(name)
        else:
            per_source = _get(resilience, "per_source", {}) or {}
            policy = per_source.get(name, _get(resilience, "default"))
        if policy is None or not _get(policy, "enabled", True):
            return source
        return ResilientSource(source, policy=policy)

    def search_papers(
        self,
        *,
        run_id: str,
        query: str,
        sources: list[str] | None,
        limit: int | None,
        filters: SearchPaperFilters | dict[str, Any] | None,
        called_by: str,
        tool_call_id: str,
    ) -> RetrievalToolResult:
        started_at = time.monotonic()
        selected_sources = self._select_sources(sources)
        result_limit = int(
            limit or _get(self.config, "final_top_k", _cfg_default(AppRetrievalConfig, "final_top_k"))
        )
        per_source_limit = min(
            result_limit,
            int(_get(self.config, "per_source_top_k", result_limit)),
        )
        parsed_filters = self._parse_filters(filters)
        filter_payload = self._filters_payload(parsed_filters)
        retrieval_config_hash = stable_hash_payload(self._config_payload())
        input_payload = {
            "tool_name": "search_papers",
            "query": query,
            "sources": selected_sources,
            "limit": result_limit,
            "filters": filter_payload,
            "retrieval_config_hash": retrieval_config_hash,
        }
        if self.rerank_embedder is not None:
            # Opt-in query reranking changes both the selected evidence and its
            # order. Keep it in the cache identity so a legacy result can never
            # replay into a reranked call (or vice versa). Omitting the key when
            # disabled preserves all pre-reranker cache hashes byte-for-byte.
            input_payload["query_reranker"] = _QUERY_RERANKER_ID
        input_hash = stable_hash_payload(input_payload)
        batch_id = retrieval_batch_id(
            run_id=run_id,
            input_hash=input_hash,
            retrieval_config_hash=retrieval_config_hash,
        )

        cached_payload = None
        if _get(self.config, "cache_replay", True):
            cached_payload = self.log_store.get_retrieval_cache(
                run_id=run_id,
                input_hash=input_hash,
                retrieval_config_hash=retrieval_config_hash,
            )
        if cached_payload is not None:
            with progress_operation("replaying cached retrieval"):
                return self._result_from_cache(
                    payload=cached_payload,
                    tool_call_id=tool_call_id,
                    elapsed_ms=self._elapsed_ms(started_at),
                    batch_id=batch_id,
                )

        all_results: list[SourceResult] = []
        source_statuses: list[SourceStatus] = []
        warnings: list[str] = []
        errors: list[str] = []
        for source_name in selected_sources:
            try:
                source = self.sources[source_name]
                run_search = getattr(source, "search_for_run", None)
                if source_name == "codex_web" and callable(run_search):
                    source_result = run_search(
                        query=query,
                        limit=per_source_limit,
                        filters=parsed_filters,
                        run_id=run_id,
                    )
                else:
                    source_result = source.search(
                        query=query,
                        limit=per_source_limit,
                        filters=parsed_filters,
                    )
            except Exception as exc:
                source_result = SourceSearchResult(
                    results=[],
                    status=SourceStatus(
                        source=source_name,
                        status="failed",
                        source_query=query,
                        result_count=0,
                        errors=[str(exc)],
                    ),
                    errors=[str(exc)],
                )
            all_results.extend(source_result.results)
            source_statuses.append(source_result.status)
            warnings.extend(source_result.warnings)
            errors.extend(source_result.errors)

        self._exa_fallback_backfill(
            query=query,
            filters=parsed_filters,
            result_limit=result_limit,
            all_results=all_results,
            source_statuses=source_statuses,
            warnings=warnings,
            errors=errors,
        )

        with progress_operation(f"fetching full text for {len(all_results)} records"):
            self._augment_full_text(all_results, parsed_filters, warnings)

        with progress_operation("normalizing and ranking evidence"):
            evidence = self._normalize_evidence(
                results=all_results,
                query=query,
                tool_call_id=tool_call_id,
                called_by=called_by,
            )
            evidence = self._rerank_evidence(evidence, query=query)[:result_limit]
        self._stamp_retrieval_batch(evidence, batch_id=batch_id)
        result = RetrievalToolResult(
            tool_call_id=tool_call_id,
            query=query,
            filters=filter_payload,
            sources=selected_sources,
            evidence=evidence,
            source_statuses=source_statuses,
            warnings=warnings,
            errors=errors,
            elapsed_ms=self._elapsed_ms(started_at),
            input_hash=input_hash,
            retrieval_config_hash=retrieval_config_hash,
        )
        if _get(self.config, "cache_replay", True):
            self.log_store.put_retrieval_cache(
                run_id=run_id,
                input_hash=input_hash,
                retrieval_config_hash=retrieval_config_hash,
                payload=self.result_to_dict(result),
            )
        return result

    def _exa_fallback_backfill(
        self,
        *,
        query: str,
        filters: SearchPaperFilters,
        result_limit: int,
        all_results: list[SourceResult],
        source_statuses: list[SourceStatus],
        warnings: list[str],
        errors: list[str],
    ) -> None:
        """When a non-Exa source fails or returns nothing, retry the SAME keyword search with
        Exa (the rich full-text source) to backfill the pool. Exa is the safety net for flaky or
        metadata-only sources: a higher limit pulls extra pages, results are deduped against what
        was already collected, and the fallback never crashes the run."""
        failed = [
            status.source
            for status in source_statuses
            if status.source not in {"exa", "codex_web"}
            and (status.result_count == 0 or status.status in ("failed", "partial_failure"))
        ]
        if not failed or "exa" not in self.sources:
            return
        seen = {self._result_key(result) for result in all_results}
        try:
            exa_result = self.sources["exa"].search(
                query=query, limit=result_limit, filters=filters
            )
        except Exception as exc:  # noqa: BLE001 - the fallback must never crash the search.
            errors.append(f"exa fallback failed: {exc}")
            source_statuses.append(
                SourceStatus(
                    source="exa_fallback", status="failed", source_query=query,
                    result_count=0, errors=[str(exc)],
                )
            )
            return
        added = [r for r in exa_result.results if self._result_key(r) not in seen]
        all_results.extend(added)
        warnings.extend(exa_result.warnings)
        errors.extend(exa_result.errors)
        # Only record the fallback when it actually contributed (or errored): a pure no-op
        # (e.g. Exa unavailable, nothing new) must stay invisible so it never perturbs the
        # "all selected sources failed" detection downstream.
        if added or exa_result.errors:
            source_statuses.append(
                SourceStatus(
                    source="exa_fallback",
                    status="success" if added else exa_result.status.status,
                    source_query=query,
                    result_count=len(added),
                    warnings=[f"backfilled {len(added)} results for failed sources: {failed}"],
                    errors=list(exa_result.errors),
                )
            )

    @staticmethod
    def _result_key(result: SourceResult) -> str:
        return (
            getattr(result, "source_id", None)
            or getattr(result, "url", None)
            or (getattr(result, "title", "") or "").strip().lower()
        )

    def _select_sources(self, requested_sources: list[str] | None) -> list[str]:
        if requested_sources is not None:
            selected_sources = requested_sources
        elif bool(_get(self.config, "fake_mode", False)):
            selected_sources = ["fake"]
        else:
            selector = _get(self.config, "selected_source_names")
            selected_sources = selector() if callable(selector) else []
        unsupported_sources = [
            source for source in selected_sources if source not in self.sources
        ]
        if unsupported_sources:
            raise ValueError(f"unsupported retrieval source: {unsupported_sources[0]}")
        if bool(_get(self.config, "fake_mode", False)):
            return ["fake"]
        return list(selected_sources)

    def _parse_filters(
        self,
        filters: SearchPaperFilters | dict[str, Any] | None,
    ) -> SearchPaperFilters:
        if isinstance(filters, SearchPaperFilters):
            return filters
        return SearchPaperFilters.model_validate(filters or {})

    def _filters_payload(self, filters: Any) -> dict[str, Any]:
        if isinstance(filters, SearchPaperFilters):
            return filters.model_dump(mode="json", exclude_computed_fields=True)
        dumped = _dump(filters)
        if dumped is None:
            return {}
        if isinstance(dumped, dict):
            return dumped
        return {}

    def _config_payload(self) -> dict[str, Any]:
        payload = _dump(self.config)
        if not isinstance(payload, dict):
            return {}
        payload.pop("selected_source_names", None)
        return payload

    def _augment_full_text(
        self,
        results: list[SourceResult],
        filters: SearchPaperFilters,
        warnings: list[str],
    ) -> None:
        """Lazily upgrade OA results to full text when the caller asked for it.

        Only runs when filters.text is set, so the extra fetch stays opt-in. The
        bytes are parsed in memory and discarded by the fetcher (no local store).
        """
        if not getattr(filters, "text", None):
            return
        fulltext_config = _get(self.config, "fulltext")
        if fulltext_config is not None and not _get(fulltext_config, "enabled", True):
            return
        for result in results:
            if getattr(result, "text", None):
                continue
            text = fetch_oa_fulltext(result, fulltext_config)
            if text:
                result.text = text

    def _enrich_coherence(self, items: list[Any], claim: str) -> list[Any]:
        coherence_config = _get(self.config, "coherence")
        if coherence_config is None or not _get(coherence_config, "enabled", True):
            return items
        enricher = self._coherence_enricher(coherence_config)
        if enricher is None:
            return items
        return enricher.enrich(claim, items)

    def _coherence_enricher(self, coherence_config: Any) -> EvidenceEnricher | None:
        cached = getattr(self, "_enricher", None)
        if cached is None:
            cached = EvidenceEnricher(coherence_config)
            self._enricher = cached
        return cached

    def _normalize_evidence(
        self,
        *,
        results: list[SourceResult],
        query: str,
        tool_call_id: str,
        called_by: str,
    ) -> list[dict[str, Any]]:
        deduped: dict[str, SourceResult] = {}
        if EvidenceLedger is not None:
            ranking = _get(self.config, "ranking")
            ledger = EvidenceLedger(
                trust_policy=_get(self.config, "trust_policy", {}),
                trust_tier_priority=_get(ranking, "trust_tier_priority", {}),
                source_order=list(_get(ranking, "source_order", [])),
            )
            ledger.add_many(
                results=results,
                query=query,
                retrieved_by=called_by,
                tool_call_id=tool_call_id,
            )
            enriched = self._enrich_coherence(ledger.all_evidence(), query)
            return [_dump(item) for item in enriched]

        for result in results:
            key = (
                result.source_id
                or result.url
                or f"{result.title.lower()}:{result.published_date or ''}"
            )
            deduped.setdefault(key, result)
        ordered = sorted(
            deduped.values(),
            key=lambda result: (
                -self._trust_priority(result.source),
                self._source_order_index(result.source),
                -(result.score or 0.0),
                result.title,
            ),
        )
        evidence: list[dict[str, Any]] = []
        for index, result in enumerate(ordered, start=1):
            payload = _dump(result)
            quote = result.text or result.summary or result.title
            evidence.append(
                {
                    "evidence_id": f"ev_{index:06d}",
                    "source": result.source,
                    "source_id": result.source_id,
                    "title": result.title,
                    "authors": result.authors,
                    "published_date": result.published_date,
                    "url": result.url,
                    "quote": quote,
                    "relevance": f"Retrieved for query: {query}",
                    "retrieval_method": "search_papers",
                    "retrieved_by": called_by,
                    "tool_call_id": tool_call_id,
                    "score": result.score,
                    "rank": index,
                    "trust_tier": _get(self.config, "trust_policy", {}).get(
                        result.source,
                        "unknown",
                    ),
                    "redacted": False,
                    "metadata": payload.get("metadata", {}),
                }
            )
        return evidence

    def _rerank_evidence(
        self,
        evidence: list[dict[str, Any]],
        *,
        query: str,
    ) -> list[dict[str, Any]]:
        """Query-rank the complete normalized cross-source pool when enabled.

        Source scores are not comparable across providers and the normalizer's
        default trust-first ordering can therefore fill a small final cut with
        off-topic metadata records. The optional embedder supplies one common
        cosine surface over ``query`` and every paper's ``title + quote`` before
        the caller's global limit is applied. Equal scores preserve the complete
        normalized order, making the rerank deterministic and backward-compatible.
        """
        if self.rerank_embedder is None or not evidence:
            return evidence

        query_vector = self.rerank_embedder([query])[0]
        texts = [
            f"{item.get('title') or ''} {item.get('quote') or ''}".strip()
            for item in evidence
        ]
        vectors = self.rerank_embedder(texts)
        if len(vectors) != len(evidence):
            raise ValueError(
                "query rerank embedder returned a different number of vectors than inputs"
            )

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for original_index, (item, vector) in enumerate(zip(evidence, vectors)):
            relevance = _cosine(query_vector, vector)
            metadata = item.get("metadata")
            item["metadata"] = {
                **(metadata if isinstance(metadata, dict) else {}),
                "query_relevance": relevance,
                "query_reranker": _QUERY_RERANKER_ID,
            }
            scored.append((relevance, original_index, item))

        reranked = [
            item
            for _score, _original_index, item in sorted(
                scored,
                key=lambda value: (-value[0], value[1]),
            )
        ]
        for rank, item in enumerate(reranked, start=1):
            item["rank"] = rank
        return reranked

    def _trust_priority(self, source: str) -> int:
        trust_policy = _get(self.config, "trust_policy", {})
        trust_tier = trust_policy.get(source, "unknown")
        ranking = _get(self.config, "ranking")
        priorities = _get(ranking, "trust_tier_priority", {})
        return int(priorities.get(trust_tier, 0))

    def _source_order_index(self, source: str) -> int:
        ranking = _get(self.config, "ranking")
        source_order = _get(ranking, "source_order", [])
        try:
            return list(source_order).index(source)
        except ValueError:
            return 999

    def _result_from_cache(
        self,
        *,
        payload: dict[str, Any],
        tool_call_id: str,
        elapsed_ms: int,
        batch_id: str,
    ) -> RetrievalToolResult:
        replay_payload = dict(payload)
        replay_payload["tool_call_id"] = tool_call_id
        replay_payload["elapsed_ms"] = elapsed_ms
        replay_payload["source_statuses"] = [
            SourceStatus(
                **{
                    **status,
                    "metadata": {**status.get("metadata", {}), "cache": "hit"},
                },
            )
            for status in replay_payload.get("source_statuses", [])
        ]
        for evidence in replay_payload.get("evidence", []):
            evidence["tool_call_id"] = tool_call_id
            evidence.pop("evidence_hash", None)
        self._stamp_retrieval_batch(
            replay_payload.get("evidence", []), batch_id=batch_id
        )
        if isinstance(replay_payload.get("filters"), dict):
            replay_payload["filters"].pop("request_kind", None)
        return RetrievalToolResult(**replay_payload)

    @staticmethod
    def _stamp_retrieval_batch(
        evidence: list[dict[str, Any]], *, batch_id: str
    ) -> None:
        """Attach stable cache-batch identity without changing local evidence IDs."""
        for item in evidence:
            metadata = item.get("metadata")
            item["metadata"] = {
                **(metadata if isinstance(metadata, dict) else {}),
                "retrieval_batch_id": batch_id,
            }

    def _elapsed_ms(self, started_at: float) -> int:
        return int((time.monotonic() - started_at) * 1000)

    def result_to_dict(self, result: RetrievalToolResult) -> dict[str, Any]:
        payload = result.model_dump(mode="json")
        if not isinstance(payload, dict):
            raise TypeError("RetrievalToolResult must serialize to a dict")
        if isinstance(payload.get("filters"), dict):
            payload["filters"].pop("request_kind", None)
        return payload
