from __future__ import annotations

import importlib
import json
import logging
import os
import re
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import fcntl
from pydantic import BaseModel

from src.config import (
    ApifySourceConfig,
    ArxivSourceConfig,
    CrossrefSourceConfig,
    EuropePmcSourceConfig,
    ExaSourceConfig,
    OpenAlexSourceConfig,
)
from src.credentials import resolve_api_key
from src.paper_titles import (
    USER_AGENT,
    canonical_source_url,
    pdf_url_from_values,
    real_paper_title,
    resolve_pdf_title_from_url,
    source_text_keys,
    strip_markup,
    title_from_source_text,
)
from src.retrieval._util import dump as _dump
from src.retrieval._util import get_attr_or_key as _get
from src.retrieval.models import (
    SearchPaperFilters,
    SourceResult,
    SourceStatus,
)
from src.runtime_trace import record_runtime_event


try:
    from langchain_community.document_loaders import ArxivLoader
except Exception:  # pragma: no cover - exercised when optional extra absent.
    ArxivLoader = None  # type: ignore[assignment, misc]

try:
    from exa_py import Exa
except Exception:  # pragma: no cover - exercised when optional extra absent.
    Exa = None  # type: ignore[assignment, misc]


@dataclass
class SourceSearchResult:
    results: list[SourceResult]
    status: SourceStatus
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class PaperSource(Protocol):
    # Read-only property (not a settable attribute) so that BOTH a plain class
    # attribute (`name = "arxiv"`) and a @property (ResilientSource.name) satisfy
    # the contract under structural type-checking.
    @property
    def name(self) -> str: ...

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchPaperFilters,
    ) -> SourceSearchResult: ...


def _cfg_default(cls: type[BaseModel], name: str) -> Any:
    """The pydantic field default for ``cls.name`` in config.py — so an inline
    ``_get(self.config, name, ...)`` fallback tracks the real config default
    instead of restating it as a literal that can silently drift (see the apify
    timeout: a stale 60.0 literal here vs config.py's pinned 400.0)."""
    return cls.model_fields[name].default


@contextmanager
def _quiet_pymupdf_full_pdf_load() -> Iterator[None]:
    try:
        fitz_module = importlib.import_module("fitz")
    except Exception:
        yield
        return

    missing_nested_alias = not hasattr(fitz_module, "fitz")
    previous_nested_alias = getattr(fitz_module, "fitz", None)
    if missing_nested_alias:
        setattr(fitz_module, "fitz", fitz_module)

    tools = getattr(fitz_module, "TOOLS", None)
    display_errors = getattr(tools, "mupdf_display_errors", None)
    display_warnings = getattr(tools, "mupdf_display_warnings", None)
    previous_errors = display_errors() if callable(display_errors) else None
    previous_warnings = display_warnings() if callable(display_warnings) else None

    try:
        if callable(display_errors):
            display_errors(False)
        if callable(display_warnings):
            display_warnings(False)
        yield
    finally:
        if callable(display_errors) and previous_errors is not None:
            display_errors(previous_errors)
        if callable(display_warnings) and previous_warnings is not None:
            display_warnings(previous_warnings)
        if missing_nested_alias:
            delattr(fitz_module, "fitz")
        else:
            setattr(fitz_module, "fitz", previous_nested_alias)


@contextmanager
def _capture_arxiv_loader_warnings() -> Iterator[list[str]]:
    messages: list[str] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = self.format(record)
            if message:
                messages.append(f"arxiv loader warning: {message}")

    logger = logging.getLogger("langchain_community.utilities.arxiv")
    previous_disabled = logger.disabled
    previous_handlers = list(logger.handlers)
    previous_level = logger.level
    previous_propagate = logger.propagate
    handler = CaptureHandler(level=logging.WARNING)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.disabled = False
    logger.handlers = [handler]
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        yield messages
    finally:
        logger.disabled = previous_disabled
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def _source_result(**kwargs: Any) -> SourceResult:
    return SourceResult(**kwargs)


class FakePaperSource:
    name = "fake"

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchPaperFilters,
    ) -> SourceSearchResult:
        del filters
        normalized_query = " ".join(query.split()) or "empty query"
        results = [
            _source_result(
                source=self.name,
                source_id=f"fake-{index:03d}",
                title=f"Deterministic Retrieval Result {index} for {normalized_query}",
                authors=["GraphHypoth Test Harness"],
                published_date=f"2026-05-{index:02d}",
                url=f"https://example.test/fake/{index}",
                text=(
                    f"Deterministic evidence excerpt {index} for query: "
                    f"{normalized_query}"
                ),
                summary=f"Stable fake summary {index} for {normalized_query}.",
                score=1.0 / index,
                metadata={"query": query, "rank": index},
            )
            for index in range(1, max(limit, 0) + 1)
        ]
        status = SourceStatus(
            source=self.name,
            status="success",
            source_query=query,
            result_count=len(results),
        )
        return SourceSearchResult(results=results, status=status)


_CROSSREF_DEFAULT_URL = _cfg_default(CrossrefSourceConfig, "base_url")


def _first_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list | tuple):
        for item in value:
            if item:
                return str(item).strip() or None
        return None
    text = str(value).strip()
    return text or None


def _normalize_doi(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text or None


def _normalize_pmid(value: Any) -> str | None:
    if not value:
        return None
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else None


def _normalize_pmcid(value: Any) -> str | None:
    if not value:
        return None
    match = re.search(r"pmc\d+", str(value), re.IGNORECASE)
    return match.group(0).upper() if match else None


def _normalize_arxiv(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip().lower()
    for prefix in ("arxiv:", "https://arxiv.org/abs/", "http://arxiv.org/abs/"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    text = re.sub(r"v\d+$", "", text)  # strip version suffix (e.g. v2)
    return text or None


def _ids_map(**ids: str | None) -> dict[str, str]:
    """Assemble a normalized external-id map, dropping empty/None values."""
    return {key: value for key, value in ids.items() if value}


def _invert_abstract(index: Any) -> str | None:
    if not isinstance(index, dict) or not index:
        return None
    positions: list[tuple[int, str]] = []
    for word, locations in index.items():
        if not isinstance(locations, list | tuple):
            continue
        for location in locations:
            try:
                positions.append((int(location), str(word)))
            except (TypeError, ValueError):
                continue
    if not positions:
        return None
    positions.sort(key=lambda item: item[0])
    text = " ".join(word for _, word in positions)
    return text or None


_MAX_RESPONSE_BYTES = 10_000_000


def _http_get_json(
    opener: Callable[..., Any],
    url: str,
    *,
    timeout: float,
    user_agent: str,
    max_bytes: int = _MAX_RESPONSE_BYTES,
) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with opener(request, timeout=timeout) as response:
        raw = response.read(max_bytes + 1)
    if isinstance(raw, bytes | bytearray):
        if len(raw) > max_bytes:
            raise ValueError("response exceeded max bytes")
        raw = bytes(raw).decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed


def _http_post_json(
    opener: Callable[..., Any],
    url: str,
    *,
    payload: dict[str, Any],
    timeout: float,
    user_agent: str,
    max_bytes: int = _MAX_RESPONSE_BYTES,
) -> Any:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"User-Agent": user_agent, "Content-Type": "application/json"},
    )
    with opener(request, timeout=timeout) as response:
        raw = response.read(max_bytes + 1)
    if isinstance(raw, bytes | bytearray):
        if len(raw) > max_bytes:
            raise ValueError("response exceeded max bytes")
        raw = bytes(raw).decode("utf-8")
    return json.loads(raw)  # Apify run-sync-get-dataset-items returns a JSON array


class CrossrefPaperSource:
    name = "crossref"

    def __init__(self, config: Any, *, opener: Callable[..., Any] | None = None) -> None:
        self.config = config
        self._opener = opener or urllib.request.urlopen

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchPaperFilters,
    ) -> SourceSearchResult:
        record_runtime_event(
            "agent_with_api",
            actor="retrieval",
            target="crossref",
            direction="request",
            payload={
                "query": query,
                "limit": limit,
                "filters": _dump(filters),
                "config": _dump(self.config),
            },
        )
        warnings: list[str] = []
        unused_filters = (
            filters.ignored_for_source("crossref")
            if filters is not None and hasattr(filters, "ignored_for_source")
            else []
        )
        query_max_chars = int(
            _get(self.config, "query_max_chars", _cfg_default(CrossrefSourceConfig, "query_max_chars"))
        )
        source_query = query[:query_max_chars]
        if len(query) > query_max_chars:
            warnings.append(f"crossref query truncated to {query_max_chars} characters")

        url = self._build_url(source_query, limit)
        try:
            payload = self._fetch(url)
        except Exception as exc:
            error = f"crossref search failed: {exc}"
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="failed",
                    source_query=source_query,
                    result_count=0,
                    warnings=warnings,
                    errors=[error],
                    unused_filters=unused_filters,
                ),
                warnings=warnings,
                errors=[error],
            )

        items = (((payload or {}).get("message") or {}).get("items")) or []
        results = [
            result
            for item in items
            if (result := self._normalize_item(item)) is not None
        ]
        if not results:
            warnings.append("crossref returned zero works")
        status = SourceStatus(
            source=self.name,
            status="success",
            source_query=source_query,
            result_count=len(results),
            warnings=warnings,
            unused_filters=unused_filters,
        )
        response = SourceSearchResult(results=results, status=status, warnings=warnings)
        record_runtime_event(
            "agent_with_api",
            actor="retrieval",
            target="crossref",
            direction="response",
            payload=_dump(response),
        )
        return response

    def _build_url(self, source_query: str, limit: int) -> str:
        base_url = str(_get(self.config, "base_url", _CROSSREF_DEFAULT_URL))
        params: dict[str, str] = {
            "query": source_query,
            "rows": str(max(int(limit), 0)),
        }
        mailto = _get(self.config, "mailto", None)
        if mailto:
            params["mailto"] = str(mailto)
        select = _get(self.config, "select", None)
        if select:
            params["select"] = str(select)
        return f"{base_url}?{urllib.parse.urlencode(params)}"

    def _fetch(self, url: str) -> dict[str, Any]:
        return _http_get_json(
            self._opener,
            url,
            timeout=float(
                _get(self.config, "timeout_seconds", _cfg_default(CrossrefSourceConfig, "timeout_seconds"))
            ),
            user_agent=self._user_agent(),
        )

    def _user_agent(self) -> str:
        base = USER_AGENT
        mailto = _get(self.config, "mailto", None)
        return f"{base} mailto:{mailto}" if mailto else base

    def _normalize_item(self, item: Any) -> SourceResult | None:
        if not isinstance(item, dict):
            return None
        title = _first_str(item.get("title"))
        if not title:
            return None
        doi = _normalize_doi(item.get("DOI"))
        url = item.get("URL") or (f"https://doi.org/{doi}" if doi else None)
        references = [
            normalized
            for ref in (item.get("reference") or [])
            if isinstance(ref, dict)
            and (normalized := _normalize_doi(ref.get("DOI"))) is not None
        ]
        metadata = {
            "doi": doi,
            "references": references,
            "cited_by_count": item.get("is-referenced-by-count"),
            "container_title": _first_str(item.get("container-title")),
            "type": item.get("type"),
            "subject": item.get("subject") or [],
        }
        return _source_result(
            source=self.name,
            source_id=doi or (str(url) if url else None),
            external_ids=_ids_map(doi=doi),
            title=title,
            authors=self._normalize_authors(item.get("author") or []),
            published_date=self._published_date(item),
            url=str(url) if url else None,
            text=None,
            summary=strip_markup(item.get("abstract")),
            score=None,
            metadata=metadata,
        )

    def _normalize_authors(self, authors: Any) -> list[str]:
        if not isinstance(authors, list | tuple):
            return []
        names: list[str] = []
        for author in authors:
            if not isinstance(author, dict):
                continue
            name = author.get("name")
            if not name:
                given = str(author.get("given", "")).strip()
                family = str(author.get("family", "")).strip()
                name = " ".join(part for part in (given, family) if part)
            name = str(name).strip()
            if name:
                names.append(name)
        return names

    def _published_date(self, item: Any) -> str | None:
        for key in ("published", "issued", "published-print", "published-online"):
            block = item.get(key)
            if not isinstance(block, dict):
                continue
            date_parts = block.get("date-parts")
            if not isinstance(date_parts, list) or not date_parts:
                continue
            parts = date_parts[0]
            if not isinstance(parts, list) or not parts:
                continue
            if not isinstance(parts[0], int):
                return None
            segments = [f"{parts[0]:04d}"]
            if len(parts) > 1 and isinstance(parts[1], int):
                segments.append(f"{parts[1]:02d}")
                if len(parts) > 2 and isinstance(parts[2], int):
                    segments.append(f"{parts[2]:02d}")
            return "-".join(segments)
        return None


_OPENALEX_DEFAULT_URL = _cfg_default(OpenAlexSourceConfig, "base_url")


class OpenAlexPaperSource:
    name = "openalex"

    def __init__(self, config: Any, *, opener: Callable[..., Any] | None = None) -> None:
        self.config = config
        self._opener = opener or urllib.request.urlopen

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchPaperFilters,
    ) -> SourceSearchResult:
        record_runtime_event(
            "agent_with_api",
            actor="retrieval",
            target="openalex",
            direction="request",
            payload={
                "query": query,
                "limit": limit,
                "filters": _dump(filters),
                "config": _dump(self.config),
            },
        )
        warnings: list[str] = []
        unused_filters = (
            filters.ignored_for_source("openalex")
            if filters is not None and hasattr(filters, "ignored_for_source")
            else []
        )
        api_key_env = str(_get(self.config, "require_api_key_env", "OPENALEX_API_KEY"))
        key_source = str(_get(self.config, "key_source", "env"))
        api_key = resolve_api_key(api_key_env, key_source=key_source)
        if (
            bool(_get(self.config, "require_api_key", False))
            and key_source != "proxy"
            and not api_key
        ):
            warning = f"{api_key_env} is not set; skipping openalex"
            warnings.append(warning)
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="skipped",
                    source_query=query,
                    result_count=0,
                    warnings=warnings,
                    unused_filters=unused_filters,
                ),
                warnings=warnings,
            )

        query_max_chars = int(
            _get(self.config, "query_max_chars", _cfg_default(OpenAlexSourceConfig, "query_max_chars"))
        )
        source_query = query[:query_max_chars]
        if len(query) > query_max_chars:
            warnings.append(
                f"openalex query truncated to {query_max_chars} characters"
            )

        if int(limit) <= 0:
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="success",
                    source_query=source_query,
                    result_count=0,
                    warnings=warnings,
                    unused_filters=unused_filters,
                ),
                warnings=warnings,
            )

        url = self._build_url(source_query, limit, api_key)
        try:
            payload = self._fetch(url)
        except Exception as exc:
            error = f"openalex search failed: {exc}"
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="failed",
                    source_query=source_query,
                    result_count=0,
                    warnings=warnings,
                    errors=[error],
                    unused_filters=unused_filters,
                ),
                warnings=warnings,
                errors=[error],
            )

        works = (payload or {}).get("results") or []
        results = [
            result
            for work in works
            if (result := self._normalize_work(work)) is not None
        ]
        if not results:
            warnings.append("openalex returned zero works")
        status = SourceStatus(
            source=self.name,
            status="success",
            source_query=source_query,
            result_count=len(results),
            warnings=warnings,
            unused_filters=unused_filters,
        )
        response = SourceSearchResult(results=results, status=status, warnings=warnings)
        record_runtime_event(
            "agent_with_api",
            actor="retrieval",
            target="openalex",
            direction="response",
            payload=_dump(response),
        )
        return response

    def _build_url(self, source_query: str, limit: int, api_key: str | None) -> str:
        base_url = str(_get(self.config, "base_url", _OPENALEX_DEFAULT_URL))
        per_page = min(max(int(limit), 1), 200)
        params: dict[str, str] = {"search": source_query, "per_page": str(per_page)}
        mailto = _get(self.config, "mailto", None)
        if mailto:
            params["mailto"] = str(mailto)
        if api_key:
            params["api_key"] = api_key
        select = _get(self.config, "select", None)
        if select:
            params["select"] = str(select)
        return f"{base_url}?{urllib.parse.urlencode(params)}"

    def _fetch(self, url: str) -> dict[str, Any]:
        return _http_get_json(
            self._opener,
            url,
            timeout=float(
                _get(self.config, "timeout_seconds", _cfg_default(OpenAlexSourceConfig, "timeout_seconds"))
            ),
            user_agent=USER_AGENT,
        )

    def _normalize_work(self, work: Any) -> SourceResult | None:
        if not isinstance(work, dict):
            return None
        title = _first_str(work.get("display_name") or work.get("title"))
        if not title:
            return None
        doi = _normalize_doi(work.get("doi"))
        ids_block = work.get("ids")
        if not isinstance(ids_block, dict):
            ids_block = {}
        openalex_id = work.get("id")
        primary_location = work.get("primary_location")
        landing = (
            primary_location.get("landing_page_url")
            if isinstance(primary_location, dict)
            else None
        )
        url = f"https://doi.org/{doi}" if doi else (landing or openalex_id)
        open_access = work.get("open_access") or {}
        if not isinstance(open_access, dict):
            open_access = {}
        metadata = {
            "doi": doi,
            "openalex_id": openalex_id,
            "references": list(work.get("referenced_works") or []),
            "cited_by_count": work.get("cited_by_count"),
            "cited_by_api_url": work.get("cited_by_api_url"),
            "topics": [
                topic.get("display_name")
                for topic in (work.get("topics") or [])
                if isinstance(topic, dict) and topic.get("display_name")
            ],
            "open_access": open_access,
            "best_oa_location": work.get("best_oa_location"),
            "oa_url": open_access.get("oa_url"),
            "is_oa": open_access.get("is_oa"),
        }
        return _source_result(
            source=self.name,
            source_id=doi or (str(openalex_id) if openalex_id else None),
            external_ids=_ids_map(
                doi=doi or _normalize_doi(ids_block.get("doi")),
                pmid=_normalize_pmid(ids_block.get("pmid")),
                pmcid=_normalize_pmcid(ids_block.get("pmcid")),
            ),
            title=title,
            authors=self._normalize_authors(work.get("authorships") or []),
            published_date=self._published_date(work),
            url=str(url) if url else None,
            text=None,
            summary=_invert_abstract(work.get("abstract_inverted_index")),
            score=None,
            metadata=metadata,
        )

    def _normalize_authors(self, authorships: Any) -> list[str]:
        if not isinstance(authorships, list | tuple):
            return []
        names: list[str] = []
        for authorship in authorships:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author")
            display = (
                author.get("display_name") if isinstance(author, dict) else None
            )
            display = str(display).strip() if display else ""
            if display:
                names.append(display)
        return names

    def _published_date(self, work: Any) -> str | None:
        date = work.get("publication_date")
        if date:
            return str(date)
        year = work.get("publication_year")
        return str(year) if year else None


_EUROPEPMC_DEFAULT_SEARCH_URL = _cfg_default(EuropePmcSourceConfig, "base_url")
_EUROPEPMC_DEFAULT_FULLTEXT_URL = _cfg_default(EuropePmcSourceConfig, "fulltext_base_url")


class EuropePmcPaperSource:
    name = "europepmc"

    def __init__(self, config: Any, *, opener: Callable[..., Any] | None = None) -> None:
        self.config = config
        self._opener = opener or urllib.request.urlopen

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchPaperFilters,
    ) -> SourceSearchResult:
        record_runtime_event(
            "agent_with_api",
            actor="retrieval",
            target="europepmc",
            direction="request",
            payload={
                "query": query,
                "limit": limit,
                "filters": _dump(filters),
                "config": _dump(self.config),
            },
        )
        warnings: list[str] = []
        unused_filters = (
            filters.ignored_for_source("europepmc")
            if filters is not None and hasattr(filters, "ignored_for_source")
            else []
        )
        query_max_chars = int(
            _get(self.config, "query_max_chars", _cfg_default(EuropePmcSourceConfig, "query_max_chars"))
        )
        source_query = query[:query_max_chars]
        if len(query) > query_max_chars:
            warnings.append(
                f"europepmc query truncated to {query_max_chars} characters"
            )

        url = self._build_url(source_query, limit)
        try:
            payload = self._fetch(url)
        except Exception as exc:
            error = f"europepmc search failed: {exc}"
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="failed",
                    source_query=source_query,
                    result_count=0,
                    warnings=warnings,
                    errors=[error],
                    unused_filters=unused_filters,
                ),
                warnings=warnings,
                errors=[error],
            )

        records = (((payload or {}).get("resultList") or {}).get("result")) or []
        results = [
            result
            for record in records
            if (result := self._normalize_record(record)) is not None
        ]
        if not results:
            warnings.append("europepmc returned zero results")
        status = SourceStatus(
            source=self.name,
            status="success",
            source_query=source_query,
            result_count=len(results),
            warnings=warnings,
            unused_filters=unused_filters,
        )
        response = SourceSearchResult(results=results, status=status, warnings=warnings)
        record_runtime_event(
            "agent_with_api",
            actor="retrieval",
            target="europepmc",
            direction="response",
            payload=_dump(response),
        )
        return response

    def _build_url(self, source_query: str, limit: int) -> str:
        base_url = str(_get(self.config, "base_url", _EUROPEPMC_DEFAULT_SEARCH_URL))
        params = {
            "query": source_query,
            "format": "json",
            "pageSize": str(min(max(int(limit), 1), 100)),
            "resultType": str(_get(self.config, "result_type", "core")),
        }
        return f"{base_url}?{urllib.parse.urlencode(params)}"

    def _fetch(self, url: str) -> dict[str, Any]:
        return _http_get_json(
            self._opener,
            url,
            timeout=float(
                _get(self.config, "timeout_seconds", _cfg_default(EuropePmcSourceConfig, "timeout_seconds"))
            ),
            user_agent=USER_AGENT,
        )

    def _normalize_record(self, record: Any) -> SourceResult | None:
        if not isinstance(record, dict):
            return None
        title = _first_str(record.get("title"))
        if not title:
            return None
        doi = _normalize_doi(record.get("doi"))
        ext_source = record.get("source")
        ext_id = record.get("id")
        is_oa = str(record.get("isOpenAccess", "")).upper() == "Y"
        metadata: dict[str, Any] = {
            "doi": doi,
            "pmid": record.get("pmid"),
            "pmcid": record.get("pmcid"),
            "europepmc_source": ext_source,
            "europepmc_id": ext_id,
            "cited_by_count": record.get("citedByCount"),
            "is_oa": is_oa,
        }
        ext_source_q = urllib.parse.quote(str(ext_source), safe="")
        ext_id_q = urllib.parse.quote(str(ext_id), safe="")
        if is_oa and ext_source and ext_id:
            base = str(_get(self.config, "fulltext_base_url", _EUROPEPMC_DEFAULT_FULLTEXT_URL))
            metadata["jats_fulltext_url"] = (
                f"{base}/{ext_source_q}/{ext_id_q}/fullTextXML"
            )
        url = (
            f"https://doi.org/{doi}"
            if doi
            else (
                f"https://europepmc.org/article/{ext_source_q}/{ext_id_q}"
                if ext_id
                else None
            )
        )
        return _source_result(
            source=self.name,
            source_id=doi or (f"{ext_source}/{ext_id}" if ext_id else None),
            external_ids=_ids_map(
                doi=doi,
                pmid=_normalize_pmid(record.get("pmid")),
                pmcid=_normalize_pmcid(record.get("pmcid")),
            ),
            title=title,
            authors=self._normalize_authors(record.get("authorString")),
            published_date=_first_str(record.get("firstPublicationDate"))
            or _first_str(record.get("pubYear")),
            url=url,
            text=None,
            summary=_first_str(record.get("abstractText")),
            score=None,
            metadata=metadata,
        )

    def _normalize_authors(self, author_string: Any) -> list[str]:
        if not author_string:
            return []
        text = str(author_string).strip().rstrip(".")
        return [name.strip() for name in text.split(",") if name.strip()]


def _parse_published(value: Any) -> datetime | None:
    """Best-effort parse of a source ``published_date`` into a tz-aware UTC datetime.

    Returns None when the value is missing or unparseable (the record is then treated
    as undated). Tolerates date-only (``YYYY-MM-DD``) and full ISO timestamps, and a
    trailing ``Z``."""
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    for candidate in (text, text[:10]):
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _filter_by_recency(
    results: list[SourceResult],
    *,
    max_age_hours: int | None,
    now: datetime,
) -> tuple[list[SourceResult], int]:
    """Drop results older than ``now - max_age_hours``; KEEP undated/unparseable ones
    (never drop what we cannot prove is stale). A disabled cutoff (None/<=0) is a no-op.

    Honest scope: this filters only the page a source actually returned — arxiv is not
    date-sorted, so it cannot surface older-but-unreturned recent papers. Returns
    ``(kept, dropped_count)``."""
    if not max_age_hours or max_age_hours <= 0:
        return results, 0
    cutoff = now - timedelta(hours=max_age_hours)
    kept: list[SourceResult] = []
    dropped = 0
    for result in results:
        published = _parse_published(result.published_date)
        if published is None or published >= cutoff:
            kept.append(result)
        else:
            dropped += 1
    return kept, dropped


class ArxivPaperSource:
    name = "arxiv"
    _lock = threading.Lock()
    _last_call_at = 0.0

    def __init__(self, config: Any) -> None:
        self.config = config

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchPaperFilters,
    ) -> SourceSearchResult:
        record_runtime_event(
            "agent_with_api",
            actor="retrieval",
            target="arxiv",
            direction="request",
            payload={
                "query": query,
                "limit": limit,
                "filters": _dump(filters),
                "config": _dump(self.config),
            },
        )
        warnings: list[str] = []
        unused_filters = (
            filters.ignored_for_source("arxiv")
            if filters is not None and hasattr(filters, "ignored_for_source")
            else []
        )
        query_max_chars = int(
            _get(self.config, "query_max_chars", _cfg_default(ArxivSourceConfig, "query_max_chars"))
        )
        source_query = query[:query_max_chars]
        if len(query) > query_max_chars:
            warnings.append(f"arxiv query truncated to {query_max_chars} characters")

        try:
            importlib.import_module("arxiv")
            if _get(self.config, "mode", "full_pdf") == "full_pdf":
                self._ensure_pymupdf_available()
        except Exception as exc:
            error = f"arxiv dependency unavailable: {exc}"
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="failed",
                    source_query=source_query,
                    result_count=0,
                    warnings=warnings,
                    errors=[error],
                    unused_filters=unused_filters,
                ),
                warnings=warnings,
                errors=[error],
            )

        if ArxivLoader is None:
            error = "langchain_community ArxivLoader unavailable"
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="failed",
                    source_query=source_query,
                    result_count=0,
                    warnings=warnings,
                    errors=[error],
                    unused_filters=unused_filters,
                ),
                warnings=warnings,
                errors=[error],
            )

        try:
            documents, load_warnings = self._load_documents_with_warnings(
                source_query,
                limit,
            )
            warnings.extend(load_warnings)
        except Exception as exc:
            error = f"arxiv search failed: {exc}"
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="failed",
                    source_query=source_query,
                    result_count=0,
                    warnings=warnings,
                    errors=[error],
                    unused_filters=unused_filters,
                ),
                warnings=warnings,
                errors=[error],
            )

        results = [self._normalize_document(document) for document in documents]
        max_age_hours = (
            getattr(filters, "max_age_hours", None) if filters is not None else None
        )
        if max_age_hours:
            results, dropped = _filter_by_recency(
                results, max_age_hours=int(max_age_hours), now=datetime.now(timezone.utc)
            )
            if dropped:
                warnings.append(
                    f"arxiv recency filter dropped {dropped} document(s) older than "
                    f"{max_age_hours}h (fetched page only; arxiv is not date-sorted)"
                )
        if not results:
            warnings.append("arxiv returned zero documents")
        status_value = "partial_failure" if load_warnings else "success"
        status = SourceStatus(
            source=self.name,
            status=status_value,
            source_query=source_query,
            result_count=len(results),
            warnings=warnings,
            unused_filters=unused_filters,
        )
        response = SourceSearchResult(results=results, status=status, warnings=warnings)
        record_runtime_event(
            "agent_with_api",
            actor="retrieval",
            target="arxiv",
            direction="response",
            payload=_dump(response),
        )
        return response

    def _ensure_pymupdf_available(self) -> None:
        importlib.import_module("pymupdf")
        fitz_module = importlib.import_module("fitz")
        fitz_open = getattr(fitz_module, "open", None)
        fitz_document = getattr(fitz_module, "Document", None)
        if not callable(fitz_open) or fitz_document is None:
            raise ImportError("PyMuPDF fitz alias is unavailable or malformed")

    def _load_documents(self, source_query: str, limit: int) -> list[Any]:
        documents, _warnings = self._load_documents_with_warnings(source_query, limit)
        return documents

    def _load_documents_with_warnings(
        self,
        source_query: str,
        limit: int,
    ) -> tuple[list[Any], list[str]]:
        min_delay_seconds = float(_get(self.config, "min_delay_seconds", 3))
        loader_warnings: list[str] = []
        with self._paced_call(min_delay_seconds=min_delay_seconds):
            with tempfile.TemporaryDirectory() as tmpdir:
                previous_cwd = Path.cwd()
                os.chdir(tmpdir)
                try:
                    loader = ArxivLoader(
                        query=source_query,
                        top_k_results=limit,
                        load_max_docs=_get(self.config, "load_max_docs", 100),
                        load_all_available_meta=_get(
                            self.config,
                            "load_all_available_meta",
                            True,
                        ),
                        doc_content_chars_max=_get(
                            self.config,
                            "doc_content_chars_max",
                            _cfg_default(ArxivSourceConfig, "doc_content_chars_max"),
                        ),
                        continue_on_failure=_get(
                            self.config,
                            "continue_on_failure",
                            True,
                        ),
                    )
                    with _capture_arxiv_loader_warnings() as captured_warnings:
                        if _get(self.config, "mode", "full_pdf") == "metadata_only":
                            if hasattr(loader, "get_summaries_as_docs"):
                                documents = loader.get_summaries_as_docs()
                            else:
                                documents = loader.load()
                        else:
                            with _quiet_pymupdf_full_pdf_load():
                                documents = loader.load()
                    loader_warnings.extend(captured_warnings)
                    documents = list(documents)
                finally:
                    os.chdir(previous_cwd)
                record_runtime_event(
                    "agent_with_api",
                    actor="retrieval",
                    target="arxiv",
                    direction="raw_response",
                    payload={
                        "source_query": source_query,
                        "document_count": len(documents),
                        "documents": [
                            {
                                "metadata": dict(
                                    getattr(document, "metadata", {}) or {}
                                ),
                                "page_content": getattr(
                                    document,
                                    "page_content",
                                    None,
                                ),
                            }
                            for document in documents
                        ],
                    },
                )
        return list(documents), loader_warnings

    @contextmanager
    def _paced_call(self, *, min_delay_seconds: float):
        lock_path = self._rate_limit_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                last_call_at = max(
                    self.__class__._last_call_at,
                    self._read_rate_limit_timestamp(lock_file),
                )
                elapsed = time.monotonic() - last_call_at
                if last_call_at and elapsed < min_delay_seconds:
                    time.sleep(min_delay_seconds - elapsed)
                yield
            finally:
                now = time.monotonic()
                self.__class__._last_call_at = now
                self._write_rate_limit_timestamp(lock_file, now)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _rate_limit_lock_path(self) -> Path:
        configured_path = _get(self.config, "rate_limit_lock_path", None)
        if configured_path:
            return Path(str(configured_path))
        return Path(tempfile.gettempdir()) / "graph-hypoth-arxiv-rate-limit.lock"

    def _read_rate_limit_timestamp(self, lock_file: Any) -> float:
        lock_file.seek(0)
        raw_value = lock_file.read().strip()
        if not raw_value:
            return 0.0
        try:
            return float(raw_value)
        except ValueError:
            return 0.0

    def _write_rate_limit_timestamp(self, lock_file: Any, timestamp: float) -> None:
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(timestamp))
        lock_file.flush()
        os.fsync(lock_file.fileno())

    def _normalize_document(self, document: Any) -> SourceResult:
        metadata = dict(getattr(document, "metadata", {}) or {})
        title = str(
            metadata.get("Title")
            or metadata.get("title")
            or metadata.get("entry_id")
            or "Untitled arXiv result"
        )
        authors = self._normalize_authors(
            metadata.get("Authors") or metadata.get("authors") or []
        )
        entry_id = metadata.get("entry_id") or metadata.get("Entry ID")
        url = metadata.get("source") or metadata.get("url") or entry_id
        return _source_result(
            source=self.name,
            source_id=str(entry_id) if entry_id is not None else None,
            external_ids=_ids_map(
                arxiv=_normalize_arxiv(entry_id),
                doi=_normalize_doi(metadata.get("doi")),
            ),
            title=title,
            authors=authors,
            published_date=self._string_or_none(
                metadata.get("Published") or metadata.get("published")
            ),
            url=str(url) if url is not None else None,
            text=getattr(document, "page_content", None),
            summary=self._string_or_none(
                metadata.get("Summary") or metadata.get("summary")
            ),
            score=None,
            metadata=metadata,
        )

    def _normalize_authors(self, authors: Any) -> list[str]:
        if isinstance(authors, str):
            return [author.strip() for author in authors.split(",") if author.strip()]
        if isinstance(authors, list | tuple):
            return [str(author).strip() for author in authors if str(author).strip()]
        return [str(authors).strip()] if authors else []

    def _string_or_none(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)


def _apify_first(item: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """First non-empty string among ``keys`` of an actor dataset item."""
    for key in keys:
        value = _first_str(item.get(key))
        if value:
            return value
    return None


def _apify_authors(item: dict[str, Any]) -> list[str]:
    raw = item.get("authors") or item.get("author") or []
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(raw, list | tuple):
        return []
    names: list[str] = []
    for entry in raw:
        if isinstance(entry, dict):
            name = _first_str(entry.get("name")) or _first_str(entry.get("fullName"))
        else:
            name = _first_str(entry)
        if name:
            names.append(name)
    return names


_APIFY_DEFAULT_URL = _cfg_default(ApifySourceConfig, "base_url")


class ApifyPaperSource:
    """Retrieve papers via an Apify actor (run-sync-get-dataset-items).

    Generic over the actor: the actor id and the query input field are configured, and the
    dataset items are mapped to ``SourceResult`` by best-effort field extraction (title /
    url / authors / abstract / doi). The exact item schema is actor-specific, so the field
    mapping below may need tuning for the chosen scraper — validate with the ``live_apify``
    test. Skips (never fails the run) when the API token or the actor id is unset.
    """

    name = "apify"

    def __init__(self, config: Any, *, opener: Callable[..., Any] | None = None) -> None:
        self.config = config
        self._opener = opener or urllib.request.urlopen

    def _skip(self, query: str, warning: str) -> SourceSearchResult:
        return SourceSearchResult(
            results=[],
            status=SourceStatus(
                source=self.name, status="skipped", source_query=query,
                result_count=0, warnings=[warning],
            ),
            warnings=[warning],
        )

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchPaperFilters,
    ) -> SourceSearchResult:
        record_runtime_event(
            "agent_with_api", actor="retrieval", target="apify", direction="request",
            payload={"query": query, "limit": limit, "filters": _dump(filters),
                     "config": _dump(self.config)},
        )
        api_key_env = str(_get(self.config, "require_api_key_env", "APIFY_API_TOKEN"))
        token = os.environ.get(api_key_env)
        if not token:
            return self._skip(query, f"{api_key_env} is not set; skipping apify")
        actor_id = str(_get(self.config, "actor_id", "") or "")
        if not actor_id:
            return self._skip(query, "apify actor_id is not configured; skipping apify")

        warnings: list[str] = []
        query_max_chars = int(
            _get(self.config, "query_max_chars", _cfg_default(ApifySourceConfig, "query_max_chars"))
        )
        source_query = query[:query_max_chars]
        if len(query) > query_max_chars:
            warnings.append(f"apify query truncated to {query_max_chars} characters")

        url = self._build_url(actor_id, token, limit)
        actor_input = {
            str(_get(self.config, "query_field", "query")): source_query,
            **(_get(self.config, "extra_input", {}) or {}),
        }
        try:
            items = self._fetch(url, actor_input)
        except Exception as exc:
            error = f"apify search failed: {exc}"
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name, status="failed", source_query=source_query,
                    result_count=0, warnings=warnings, errors=[error],
                ),
                warnings=warnings, errors=[error],
            )

        items = items if isinstance(items, list) else []
        results = [r for item in items if (r := self._normalize_item(item)) is not None]
        if not results:
            warnings.append("apify returned zero items")
        status = SourceStatus(
            source=self.name, status="success", source_query=source_query,
            result_count=len(results), warnings=warnings,
        )
        response = SourceSearchResult(results=results, status=status, warnings=warnings)
        record_runtime_event(
            "agent_with_api", actor="retrieval", target="apify", direction="response",
            payload=_dump(response),
        )
        return response

    def _build_url(self, actor_id: str, token: str, limit: int) -> str:
        base_url = str(_get(self.config, "base_url", _APIFY_DEFAULT_URL)).rstrip("/")
        # Apify actor ids use `~` for the owner/name separator in REST paths.
        actor_path = actor_id.replace("/", "~")
        max_items = int(_get(self.config, "max_items", 15))
        params = {"token": token, "maxItems": str(max(min(int(limit), max_items), 1))}
        return (
            f"{base_url}/v2/acts/{actor_path}/run-sync-get-dataset-items"
            f"?{urllib.parse.urlencode(params)}"
        )

    def _fetch(self, url: str, actor_input: dict[str, Any]) -> Any:
        return _http_post_json(
            self._opener, url, payload=actor_input,
            # THE bug: this literal used to restate a stale 60.0 (config.py's
            # ApifySourceConfig pins 400.0 — a live actor run measured ~292s, so
            # 60.0 timed out every call and tripped the circuit breaker).
            timeout=float(
                _get(self.config, "timeout_seconds", _cfg_default(ApifySourceConfig, "timeout_seconds"))
            ),
            user_agent=USER_AGENT,
        )

    def _normalize_item(self, item: Any) -> SourceResult | None:
        if not isinstance(item, dict):
            return None
        title = _apify_first(item, ("title", "name", "paperTitle"))
        if not title:
            return None
        url = _apify_first(item, ("url", "link", "paperUrl", "downloadUrl"))
        abstract = _apify_first(item, ("abstract", "description", "snippet", "summary", "text"))
        published = _apify_first(item, ("publishedDate", "publicationDate", "date", "year"))
        doi = _normalize_doi(item.get("doi") or item.get("DOI"))
        return _source_result(
            source=self.name,
            source_id=url or title,
            external_ids=_ids_map(doi=doi),
            title=title,
            authors=_apify_authors(item),
            published_date=published,
            url=url,
            text=abstract,
            metadata={"apify_actor": str(_get(self.config, "actor_id", ""))},
        )


class ExaPaperSource:
    name = "exa"

    def __init__(self, config: Any) -> None:
        self.config = config
        self._observed_cost_dollars = 0.0
        self._last_call_at = 0.0

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchPaperFilters,
    ) -> SourceSearchResult:
        record_runtime_event(
            "agent_with_api",
            actor="retrieval",
            target="exa",
            direction="request",
            payload={
                "query": query,
                "limit": limit,
                "filters": _dump(filters),
                "config": _dump(self.config),
            },
        )
        warnings: list[str] = []
        errors: list[str] = []
        api_key_env = str(_get(self.config, "require_api_key_env", "EXA_API_KEY"))
        if not os.environ.get(api_key_env):
            warning = f"{api_key_env} is not set; skipping exa"
            warnings.append(warning)
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="skipped",
                    source_query=query,
                    result_count=0,
                    warnings=warnings,
                ),
                warnings=warnings,
            )

        max_cost = float(
            _get(self.config, "max_cost_dollars_per_run", _cfg_default(ExaSourceConfig, "max_cost_dollars_per_run"))
        )
        if self._observed_cost_dollars >= max_cost:
            warning = "exa budget exhausted; skipping exa"
            warnings.append(warning)
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="skipped",
                    source_query=query,
                    result_count=0,
                    warnings=warnings,
                    metadata={"costDollars": self._observed_cost_dollars},
                ),
                warnings=warnings,
            )

        include_text = _get(filters, "include_text", None)
        if include_text is None:
            include_text = _get(self.config, "include_text", None)
        exclude_text = _get(filters, "exclude_text", None)
        if exclude_text is None:
            exclude_text = _get(self.config, "exclude_text", None)
        include_domains = _get(filters, "include_domains", None)
        if include_domains is None:
            include_domains = _get(self.config, "include_domains", None)
        exclude_domains = _get(filters, "exclude_domains", None)
        if exclude_domains is None:
            exclude_domains = _get(self.config, "exclude_domains", None)
        urls = _get(filters, "urls", None) or _get(filters, "target_urls", None)
        validation_error = self._validate_text_filters(include_text, exclude_text)
        if validation_error is not None:
            errors.append(validation_error)
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="failed",
                    source_query=query,
                    result_count=0,
                    errors=errors,
                ),
                errors=errors,
            )

        text = _get(filters, "text", None)
        if text is None:
            text = bool(_get(self.config, "text", False))
        highlights = _get(filters, "highlights", None)
        if highlights is None:
            configured_highlights = _get(self.config, "highlights", True)
            highlights = (
                True if configured_highlights is None else bool(configured_highlights)
            )
        highlight_query = _get(filters, "highlight_query", None)
        if highlight_query is None:
            highlight_query = _get(self.config, "highlight_query", None)
        highlight_query = str(highlight_query or query)
        highlight_max_characters = self._resolve_character_limit(
            _get(filters, "highlight_max_characters", None),
            config_name="highlight_max_characters",
            default=600,
        )
        text_max_characters = self._resolve_character_limit(
            _get(filters, "text_max_characters", None),
            config_name="text_max_characters",
            default=5000,
        )
        max_age_hours = _get(filters, "max_age_hours", None)
        if max_age_hours is None:
            max_age_hours = _get(self.config, "max_age_hours", None)
        livecrawl_timeout = _get(filters, "livecrawl_timeout", None)
        if livecrawl_timeout is None:
            livecrawl_timeout = _get(self.config, "livecrawl_timeout", None)
        include_html_tags = _get(filters, "include_html_tags", None)
        if include_html_tags is None:
            include_html_tags = bool(_get(self.config, "include_html_tags", False))
        target_scope = _get(filters, "target_scope", None)
        request_kind = "contents" if urls else "search"
        anchor_queries = _get(filters, "anchor_queries", None)
        resolved_anchor_queries = (
            list(anchor_queries)
            if request_kind == "contents" and anchor_queries
            else [highlight_query]
        )
        content_modes = self._content_modes(
            text=bool(text), highlights=bool(highlights)
        )
        projected_cost = (
            self._estimate_contents_call_cost_dollars(
                url_count=len(urls or []),
                text=bool(text),
                highlight_request_count=(
                    len(resolved_anchor_queries) if highlights else 0
                ),
            )
            if request_kind == "contents"
            else self._estimate_call_cost_dollars(
                limit=limit,
                text=bool(text),
            )
        )
        max_call_cost = float(
            _get(
                self.config,
                "max_cost_dollars_per_call",
                _cfg_default(ExaSourceConfig, "max_cost_dollars_per_call"),
            )
        )
        budget_metadata = {
            "estimatedCostDollars": projected_cost,
            "maxCostDollarsPerCall": max_call_cost,
            "observedCostDollars": self._observed_cost_dollars,
        }
        if projected_cost > max_call_cost:
            warning = "exa per-call cost cap exceeded; skipping exa"
            warnings.append(warning)
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="skipped",
                    source_query=query,
                    result_count=0,
                    warnings=warnings,
                    metadata=budget_metadata,
                ),
                warnings=warnings,
            )
        if self._observed_cost_dollars + projected_cost > max_cost:
            warning = "exa run cost cap would be exceeded; skipping exa"
            warnings.append(warning)
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="skipped",
                    source_query=query,
                    result_count=0,
                    warnings=warnings,
                    metadata={
                        **budget_metadata,
                        "maxCostDollarsPerRun": max_cost,
                    },
                ),
                warnings=warnings,
            )
        if Exa is None:
            error = "exa-py Exa client unavailable"
            errors.append(error)
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="failed",
                    source_query=query,
                    result_count=0,
                    errors=errors,
                ),
                errors=errors,
            )
        self._pace_call()
        try:
            payloads: list[tuple[str, list[str], dict[str, Any]]] = []
            for anchor_index, anchor_query in enumerate(resolved_anchor_queries):
                call_text = bool(text) and (
                    request_kind != "contents" or anchor_index == 0
                )
                call_modes = self._content_modes(
                    text=call_text,
                    highlights=bool(highlights),
                )
                raw_payload = self._call_exa_sdk(
                    api_key=os.environ[api_key_env],
                    query=query,
                    limit=limit,
                    urls=urls,
                    text=call_text,
                    highlights=bool(highlights),
                    highlight_query=anchor_query,
                    highlight_max_characters=highlight_max_characters,
                    text_max_characters=text_max_characters,
                    max_age_hours=max_age_hours,
                    livecrawl_timeout=livecrawl_timeout,
                    include_html_tags=bool(include_html_tags),
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    include_text=include_text,
                    exclude_text=exclude_text,
                )
                payload = _dump(raw_payload)
                payload = payload if isinstance(payload, dict) else {}
                record_runtime_event(
                    "agent_with_api",
                    actor="retrieval",
                    target="exa",
                    direction="raw_response",
                    payload=payload,
                )
                payloads.append((anchor_query, call_modes, payload))
        except Exception as exc:
            error = f"exa {request_kind} call failed: {exc}"
            errors.append(error)
            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="failed",
                    source_query=query,
                    result_count=0,
                    errors=errors,
                    metadata={"raw": {"error": error}},
                ),
                errors=errors,
            )
        for _, _, payload in payloads:
            if payload.get("error"):
                error = str(payload["error"])
                errors.append(error)
                return SourceSearchResult(
                    results=[],
                    status=SourceStatus(
                        source=self.name,
                        status="failed",
                        source_query=query,
                        result_count=0,
                        errors=errors,
                        metadata={"raw": {"error": error}},
                    ),
                    errors=errors,
                )

        support_payloads = list(payloads)
        cost_values = [
            cost
            for _, _, payload in payloads
            if (cost := self._extract_cost(payload)) is not None
        ]
        if not bool(text):
            title_lookup_urls = self._missing_title_urls(payloads, limit=limit)
            if title_lookup_urls:
                title_lookup_cost = self._estimate_contents_call_cost_dollars(
                    url_count=len(title_lookup_urls),
                    text=True,
                    highlights=False,
                )
                observed_after_search = self._observed_cost_dollars + sum(cost_values)
                if title_lookup_cost > max_call_cost:
                    warnings.append(
                        "exa title lookup skipped; per-call cost cap exceeded"
                    )
                elif observed_after_search + title_lookup_cost > max_cost:
                    warnings.append(
                        "exa title lookup skipped; run cost cap would be exceeded"
                    )
                else:
                    try:
                        raw_payload = self._call_exa_sdk(
                            api_key=os.environ[api_key_env],
                            query=query,
                            limit=len(title_lookup_urls),
                            urls=title_lookup_urls,
                            text=True,
                            highlights=False,
                            highlight_query=highlight_query,
                            highlight_max_characters=highlight_max_characters,
                            text_max_characters=min(text_max_characters, 5000),
                            max_age_hours=max_age_hours,
                            livecrawl_timeout=livecrawl_timeout,
                            include_html_tags=False,
                            include_domains=None,
                            exclude_domains=None,
                            include_text=None,
                            exclude_text=None,
                        )
                    except Exception as exc:
                        warnings.append(f"exa title lookup failed: {exc}")
                    else:
                        payload = _dump(raw_payload)
                        payload = payload if isinstance(payload, dict) else {}
                        record_runtime_event(
                            "agent_with_api",
                            actor="retrieval",
                            target="exa",
                            direction="raw_response",
                            payload=payload,
                        )
                        if payload.get("error"):
                            warnings.append(
                                f"exa title lookup failed: {payload['error']}"
                            )
                        else:
                            support_payloads.append((query, ["text"], payload))
                            lookup_cost = self._extract_cost(payload)
                            if lookup_cost is not None:
                                cost_values.append(lookup_cost)
        cost = sum(cost_values) if cost_values else None
        if cost is not None:
            self._observed_cost_dollars += cost
        statuses: list[Any] = []
        for _, _, payload in support_payloads:
            statuses.extend(self._extract_statuses(payload))
        text_by_key = self._source_text_by_key(support_payloads)
        pdf_title_by_key = self._pdf_title_by_key(
            payloads,
            text_by_key=text_by_key,
            warnings=warnings,
            limit=limit,
        )
        results: list[SourceResult] = []
        for anchor_index, (anchor_query, call_modes, payload) in enumerate(
            payloads,
            start=1,
        ):
            for index, result in enumerate(
                self._extract_results(payload)[:limit],
                start=1,
            ):
                result = self._with_fallback_text(result, text_by_key)
                result = self._with_fallback_pdf_title(result, pdf_title_by_key)
                results.append(
                    self._normalize_result(
                        result,
                        index,
                        content_modes=call_modes,
                        statuses=statuses,
                        highlight_query=anchor_query,
                        request_kind=request_kind,
                        target_scope=target_scope,
                        anchor_query=(
                            anchor_query
                            if request_kind == "contents" and anchor_queries
                            else None
                        ),
                        anchor_index=(
                            anchor_index
                            if request_kind == "contents" and anchor_queries
                            else None
                        ),
                    )
                )
        results = results[:limit]
        metadata = {
            "result_count": len(results),
            "exa_request_kind": request_kind,
            "exa_content_modes": content_modes,
            "exa_statuses": statuses,
            "exa_highlight_query": highlight_query
            if "highlights" in content_modes
            else None,
            "target_scope": target_scope,
        }
        if request_kind == "contents" and anchor_queries:
            metadata["exa_anchor_queries"] = resolved_anchor_queries
        if cost is not None:
            metadata["costDollars"] = cost
        status = SourceStatus(
            source=self.name,
            status="success",
            source_query=query,
            result_count=len(results),
            warnings=warnings,
            metadata=metadata,
        )
        response = SourceSearchResult(results=results, status=status, warnings=warnings)
        record_runtime_event(
            "agent_with_api",
            actor="retrieval",
            target="exa",
            direction="response",
            payload=_dump(response),
        )
        return response

    def _estimate_call_cost_dollars(self, *, limit: int, text: bool) -> float:
        del text
        search_cost = 0.007
        extra_result_cost = max(0, limit - 10) * 0.001
        return search_cost + extra_result_cost

    def _estimate_contents_call_cost_dollars(
        self,
        *,
        url_count: int,
        text: bool,
        highlights: bool | None = None,
        highlight_request_count: int | None = None,
    ) -> float:
        if highlight_request_count is None:
            highlight_request_count = int(bool(highlights))
        content_type_count = int(text) + max(highlight_request_count, 0)
        return max(url_count, 0) * max(content_type_count, 0) * 0.001

    def _pace_call(self) -> None:
        max_qps = float(_get(self.config, "max_qps", 10))
        min_delay = 1 / max_qps
        now = time.monotonic()
        elapsed = now - self._last_call_at
        if self._last_call_at and elapsed < min_delay:
            time.sleep(min_delay - elapsed)
            now = time.monotonic()
        self._last_call_at = now

    def _call_exa_sdk(
        self,
        *,
        api_key: str,
        query: str,
        limit: int,
        urls: list[str] | None,
        text: bool,
        highlights: bool,
        highlight_query: str,
        highlight_max_characters: int,
        text_max_characters: int,
        max_age_hours: int | None,
        livecrawl_timeout: int | None,
        include_html_tags: bool,
        include_domains: list[str] | None,
        exclude_domains: list[str] | None,
        include_text: list[str] | None,
        exclude_text: list[str] | None,
    ) -> Any:
        client = Exa(api_key=api_key)
        contents, _ = self._build_contents(
            query=query,
            text=text,
            highlights=highlights,
            highlight_query=highlight_query,
            highlight_max_characters=highlight_max_characters,
            text_max_characters=text_max_characters,
            include_html_tags=include_html_tags,
        )
        if urls:
            kwargs = self._contents_kwargs(
                contents=contents,
                max_age_hours=max_age_hours,
                livecrawl_timeout=livecrawl_timeout,
            )
            return client.get_contents(urls, **kwargs)

        kwargs = {
            "query": query,
            "type": _get(self.config, "search_type", "auto"),
            "category": _get(self.config, "category", "research paper"),
            "contents": contents,
            "num_results": limit,
        }
        if include_domains is not None:
            kwargs["include_domains"] = include_domains
        if exclude_domains is not None:
            kwargs["exclude_domains"] = exclude_domains
        if include_text is not None:
            kwargs["include_text"] = include_text
        if exclude_text is not None:
            kwargs["exclude_text"] = exclude_text
        return client.search(**kwargs)

    def _extract_cost(self, payload: Any) -> float | None:
        if isinstance(payload, dict):
            for key in ("costDollars", "cost_dollars", "cost"):
                cost = self._coerce_cost(payload.get(key))
                if cost is not None:
                    return cost
        return None

    def _coerce_cost(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        if isinstance(value, dict):
            for key in (
                "costDollars",
                "cost_dollars",
                "total",
                "search",
                "contents",
            ):
                cost = self._coerce_cost(value.get(key))
                if cost is not None:
                    return cost
            for nested_value in value.values():
                cost = self._coerce_cost(nested_value)
                if cost is not None:
                    return cost
        return None

    def _extract_results(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            results = payload.get("results") or payload.get("data") or []
            if isinstance(results, list):
                return [item for item in results if isinstance(item, dict)]
        return []

    def _extract_statuses(self, payload: Any) -> list[Any]:
        if not isinstance(payload, dict):
            return []
        statuses = payload.get("statuses")
        if statuses is None:
            statuses = payload.get("status")
        return self._normalize_list(statuses)

    def _missing_title_urls(
        self,
        payloads: list[tuple[str, list[str], dict[str, Any]]],
        *,
        limit: int,
    ) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()
        for _, _, payload in payloads:
            for result in self._extract_results(payload)[:limit]:
                text = self._string_or_none(result.get("text"))
                if self._real_title(
                    result.get("title")
                ) or self._title_from_source_text(text):
                    continue
                url = self._string_or_none(result.get("url") or result.get("id"))
                if not url or not url.startswith(("http://", "https://")):
                    continue
                if url in seen:
                    continue
                seen.add(url)
                urls.append(url)
        return urls

    def _normalize_result(
        self,
        result: dict[str, Any],
        index: int,
        *,
        content_modes: list[str],
        statuses: list[Any],
        highlight_query: str,
        request_kind: str,
        target_scope: str | None,
        anchor_query: str | None = None,
        anchor_index: int | None = None,
    ) -> SourceResult:
        published_date = (
            result.get("publishedDate")
            or result.get("published_date")
            or result.get("published")
        )
        authors = self._normalize_authors(result.get("authors") or result.get("author"))
        text = self._string_or_none(result.get("text"))
        title = self._real_title(result.get("title")) or self._title_from_source_text(
            text
        )
        highlight_scores = result.get("highlightScores")
        if highlight_scores is None:
            highlight_scores = result.get("highlight_scores")
        metadata = {key: _dump(value) for key, value in result.items()}
        metadata.update(
            {
                "exa_highlights": self._normalize_list(result.get("highlights")),
                "exa_highlight_scores": self._normalize_list(highlight_scores),
                "exa_highlight_query": (
                    highlight_query if "highlights" in content_modes else None
                ),
                "exa_statuses": statuses,
                "exa_content_modes": list(content_modes),
                "exa_request_kind": request_kind,
                "exa_anchor_query": anchor_query,
                "exa_anchor_index": anchor_index,
                "target_scope": target_scope,
                "source_text_chars_available": len(text or ""),
            }
        )
        source_id = self._string_or_none(result.get("id") or result.get("url"))
        if anchor_index is not None and source_id is not None:
            source_id = f"{source_id}#anchor-{anchor_index}"
        return _source_result(
            source=self.name,
            source_id=source_id,
            external_ids=_ids_map(doi=_normalize_doi(result.get("doi"))),
            title=title or "",
            authors=authors,
            published_date=self._string_or_none(published_date),
            url=self._string_or_none(result.get("url")),
            text=text,
            summary=self._string_or_none(result.get("summary")),
            score=float(result["score"]) if result.get("score") is not None else None,
            metadata=metadata,
        )

    def _normalize_authors(self, authors: Any) -> list[str]:
        if authors is None:
            return []
        if isinstance(authors, str):
            return [author.strip() for author in authors.split(",") if author.strip()]
        if isinstance(authors, list | tuple):
            return [str(author).strip() for author in authors if str(author).strip()]
        return [str(authors).strip()]

    def _string_or_none(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    def _real_title(self, value: Any) -> str | None:
        return real_paper_title(value)

    def _title_from_source_text(self, value: Any) -> str | None:
        return title_from_source_text(value)

    def _validate_text_filters(
        self,
        include_text: list[str] | None,
        exclude_text: list[str] | None,
    ) -> str | None:
        for label, values in (
            ("include_text", include_text),
            ("exclude_text", exclude_text),
        ):
            if values is None:
                continue
            if len(values) > 1:
                return f"exa {label} accepts at most one string"
            if values and len(values[0].split()) > 5:
                return f"exa {label} string must be at most five words"
        return None

    def _resolve_character_limit(
        self,
        filter_value: int | None,
        *,
        config_name: str,
        default: int,
    ) -> int:
        value = filter_value
        if value is None:
            value = _get(self.config, config_name, default)
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            return default
        return max(1, resolved)

    def _build_contents(
        self,
        *,
        query: str,
        text: bool,
        highlights: bool,
        highlight_query: str,
        highlight_max_characters: int,
        text_max_characters: int,
        include_html_tags: bool,
    ) -> tuple[dict[str, Any] | bool, list[str]]:
        contents: dict[str, Any] = {}
        if highlights:
            contents["highlights"] = {
                "query": highlight_query or query,
                "max_characters": highlight_max_characters,
            }
        if text:
            contents["text"] = {
                "max_characters": text_max_characters,
                "include_html_tags": include_html_tags,
            }
        content_modes = self._content_modes(text=text, highlights=highlights)
        return (contents if contents else False), content_modes

    def _contents_kwargs(
        self,
        *,
        contents: dict[str, Any] | bool,
        max_age_hours: int | None,
        livecrawl_timeout: int | None,
    ) -> dict[str, Any]:
        if not isinstance(contents, dict):
            kwargs: dict[str, Any] = {}
        else:
            kwargs = dict(contents)
        if max_age_hours is not None:
            kwargs["max_age_hours"] = int(max_age_hours)
        if livecrawl_timeout is not None:
            kwargs["livecrawl_timeout"] = int(livecrawl_timeout)
        return kwargs

    def _source_text_by_key(
        self,
        payloads: list[tuple[str, list[str], dict[str, Any]]],
    ) -> dict[str, str]:
        text_by_key: dict[str, str] = {}
        for _, _, payload in payloads:
            for result in self._extract_results(payload):
                text = self._string_or_none(result.get("text"))
                if not text:
                    continue
                for key in self._source_text_keys(result):
                    text_by_key[key] = text
        return text_by_key

    def _pdf_title_by_key(
        self,
        payloads: list[tuple[str, list[str], dict[str, Any]]],
        *,
        text_by_key: dict[str, str],
        warnings: list[str],
        limit: int,
    ) -> dict[str, str]:
        title_by_key: dict[str, str] = {}
        title_by_url: dict[str, str | None] = {}
        for _, _, payload in payloads:
            for result in self._extract_results(payload)[:limit]:
                result_with_text = self._with_fallback_text(result, text_by_key)
                if self._real_title(
                    result.get("title")
                ) or self._title_from_source_text(result_with_text.get("text")):
                    continue
                url = self._pdf_url_for_result(result)
                if not url:
                    continue
                canonical_url = self._canonical_source_url(url) or url
                if canonical_url not in title_by_url:
                    try:
                        title_by_url[canonical_url] = resolve_pdf_title_from_url(url)
                    except Exception as exc:
                        title_by_url[canonical_url] = None
                        warnings.append(f"exa direct PDF title lookup failed: {exc}")
                    else:
                        if title_by_url[canonical_url] is None:
                            warnings.append(
                                "exa direct PDF title lookup did not recover title"
                            )
                title = self._real_title(title_by_url[canonical_url])
                if not title:
                    continue
                for key in self._source_text_keys(result):
                    title_by_key[key] = title
        return title_by_key

    def _with_fallback_text(
        self,
        result: dict[str, Any],
        text_by_key: dict[str, str],
    ) -> dict[str, Any]:
        if result.get("text"):
            return result
        for key in self._source_text_keys(result):
            if key in text_by_key:
                updated = dict(result)
                updated["text"] = text_by_key[key]
                return updated
        return result

    def _with_fallback_pdf_title(
        self,
        result: dict[str, Any],
        title_by_key: dict[str, str],
    ) -> dict[str, Any]:
        if self._real_title(result.get("title")):
            return result
        if self._title_from_source_text(result.get("text")):
            return result
        for key in self._source_text_keys(result):
            if key in title_by_key:
                updated = dict(result)
                updated["title"] = title_by_key[key]
                updated["pdf_title_lookup"] = "direct_pdf"
                return updated
        return result

    def _pdf_url_for_result(self, result: dict[str, Any]) -> str | None:
        return pdf_url_from_values(
            self._string_or_none(result.get("url")),
            self._string_or_none(result.get("id")),
        )

    def _source_text_keys(self, result: dict[str, Any]) -> list[str]:
        return source_text_keys(result)

    def _canonical_source_url(self, value: str) -> str | None:
        return canonical_source_url(value)

    def _content_modes(self, *, text: bool, highlights: bool) -> list[str]:
        modes: list[str] = []
        if highlights:
            modes.append("highlights")
        if text:
            modes.append("text")
        return modes

    def _normalize_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list | tuple):
            return [_dump(item) for item in value]
        return [_dump(value)]
