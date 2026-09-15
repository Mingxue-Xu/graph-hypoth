"""Live contract tests for the OpenAlex capabilities the cross-domain graph relies on.

These hit the real OpenAlex API (``https://api.openalex.org``) and verify the four
platform features the retrieval subsystem depends on:

1. **Topic taxonomy** — the 4-level domain -> field -> subfield -> topic hierarchy and
   filtering works by ``primary_topic.id``.
2. **Bidirectional citations** — outbound ``referenced_works`` and inbound ``cites:``
   describe the same edge (a work that references R appears among R's citers).
3. **Author / institution IDs** — OpenAlex ``A...`` author IDs (+ ORCID) and ``I...``
   institution IDs (+ ROR), and filtering works by them.
4. **Semantic search** — the ``search.semantic=`` embedding query (which the configured
   ``OPENALEX_API_KEY`` premium pool serves) returns relevance-ranked results.

It also exercises the project's own ``OpenAlexPaperSource`` adapter against the live API
to confirm it surfaces the citation graph + topics into normalized evidence.

Live tests are opt-in (see ``tests/conftest.py``): run with ``-m live`` /
``-m live_openalex`` or invoke this file directly. The polite pool is used via a
``mailto`` query param; no key is required except where noted.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

import pytest

from src.config import OpenAlexSourceConfig
from src.retrieval.models import SearchPaperFilters
from src.retrieval.sources import OpenAlexPaperSource

_API_BASE = "https://api.openalex.org"
_MAILTO = "graph-hypoth-tests@graph-hypoth.ai"
_USER_AGENT = f"GraphHypoth/0.1 (mailto:{_MAILTO})"
_TIMEOUT_SECONDS = 30.0


def _short_id(value: str) -> str:
    """``https://openalex.org/W123`` -> ``W123`` (OpenAlex accepts the bare id in filters)."""
    return value.rstrip("/").rsplit("/", 1)[-1]


def _get(path: str, *, api_key: str | None = None, **params: str) -> dict:
    """GET an OpenAlex endpoint via the polite pool and return parsed JSON.

    Raises ``urllib.error.HTTPError`` on non-2xx so negative-path tests can assert on it.
    """
    query = dict(params)
    query.setdefault("mailto", _MAILTO)
    if api_key:
        query["api_key"] = api_key
    url = f"{_API_BASE}/{path}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


@pytest.mark.live
@pytest.mark.live_openalex
@pytest.mark.drift_watchdog
def test_openalex_topic_taxonomy_hierarchy_and_work_filtering() -> None:
    # The published OpenAlex taxonomy is a fixed 4-level tree. These counts only change
    # with a major taxonomy revision, so drift here is a meaningful upstream signal.
    assert _get("domains", per_page="1")["meta"]["count"] == 4
    assert _get("fields", per_page="1")["meta"]["count"] == 26
    assert _get("subfields", per_page="1")["meta"]["count"] == 252

    topics = _get("topics", per_page="1")
    assert topics["meta"]["count"] >= 4000
    topic = topics["results"][0]
    assert topic["id"].startswith("https://openalex.org/T")
    for level in ("domain", "field", "subfield"):
        assert topic[level]["id"], f"topic missing {level} id"
        assert topic[level]["display_name"], f"topic missing {level} display_name"

    # Works carry the same hierarchy via primary_topic and are filterable by it.
    work = _get(
        "works",
        search="machine learning",
        per_page="1",
        select="id,primary_topic",
    )["results"][0]
    primary_topic = work["primary_topic"]
    assert primary_topic["id"].startswith("https://openalex.org/T")
    for level in ("domain", "field", "subfield"):
        assert primary_topic[level]["display_name"], f"primary_topic missing {level}"

    topic_id = _short_id(primary_topic["id"])
    by_topic = _get(
        "works",
        filter=f"primary_topic.id:{topic_id}",
        per_page="1",
        select="id",
    )
    assert by_topic["meta"]["count"] >= 1


@pytest.mark.live
@pytest.mark.live_openalex
@pytest.mark.drift_watchdog
def test_openalex_bidirectional_citations_roundtrip() -> None:
    # Find a work with outbound edges (referenced_works).
    search = _get(
        "works",
        search="retrieval augmented generation",
        per_page="5",
        select="id,referenced_works,referenced_works_count",
    )
    work = next((w for w in search["results"] if w.get("referenced_works")), None)
    assert work is not None, "expected at least one work with referenced_works"
    assert work["referenced_works_count"] >= 1

    work_id = _short_id(work["id"])
    referenced_id = _short_id(work["referenced_works"][0])

    # Inbound view: cited_by_count (cached on the work) and the live cites: filter both
    # count "works that cite R" — they describe the same edge set.
    referenced = _get(f"works/{referenced_id}", select="id,cited_by_count")
    citing = _get("works", filter=f"cites:{referenced_id}", per_page="1", select="id")
    cited_by_count = referenced["cited_by_count"]
    cites_count = citing["meta"]["count"]
    assert cited_by_count >= 1
    assert cites_count >= 1
    # The cached count tracks the live index (allow tiny lag between the two requests).
    assert abs(cites_count - cited_by_count) <= max(3, round(cited_by_count * 0.02))

    # Strong proof of bidirectionality: the work that *references* R (outbound) is found
    # among R's *citers* (inbound) — the intersection is exactly that one work.
    edge = _get(
        "works",
        filter=f"cites:{referenced_id},ids.openalex:{work_id}",
        per_page="1",
        select="id",
    )
    assert edge["meta"]["count"] == 1


@pytest.mark.live
@pytest.mark.live_openalex
@pytest.mark.drift_watchdog
def test_openalex_author_and_institution_ids() -> None:
    # Institution lookup by ROR is stable: MIT's ROR maps to a permanent OpenAlex I-id.
    mit_ror = "https://ror.org/042nb2s44"
    institutions = _get(
        "institutions",
        filter=f"ror:{mit_ror}",
        per_page="1",
        select="id,ror,display_name,country_code,type",
    )
    assert institutions["meta"]["count"] == 1
    institution = institutions["results"][0]
    assert institution["id"].startswith("https://openalex.org/I")
    assert institution["ror"] == mit_ror
    assert institution["country_code"] == "US"

    institution_id = _short_id(institution["id"])
    institution_works = _get(
        "works",
        filter=f"authorships.institutions.id:{institution_id}",
        per_page="1",
        select="id",
    )
    assert institution_works["meta"]["count"] >= 1

    # Author IDs are derived live from a work's authorships (no hardcoded person).
    works = _get("works", search="deep learning", per_page="5", select="id,authorships")
    author = None
    for candidate in works["results"]:
        for authorship in candidate.get("authorships") or []:
            if (authorship.get("author") or {}).get("id"):
                author = authorship["author"]
                break
        if author is not None:
            break
    assert author is not None, "expected an authorship carrying an author id"
    assert author["id"].startswith("https://openalex.org/A")

    author_id = _short_id(author["id"])
    author_works = _get(
        "works",
        filter=f"authorships.author.id:{author_id}",
        per_page="1",
        select="id",
    )
    assert author_works["meta"]["count"] >= 1

    # ORCID round-trips to the same OpenAlex author entity.
    orcid_author = _get(
        "authors",
        filter="has_orcid:true",
        per_page="1",
        select="id,orcid,display_name",
    )["results"][0]
    assert orcid_author["orcid"].startswith("https://orcid.org/")
    by_orcid = _get(
        "authors",
        filter=f"orcid:{orcid_author['orcid']}",
        per_page="1",
        select="id,orcid",
    )
    assert by_orcid["meta"]["count"] == 1
    assert by_orcid["results"][0]["id"] == orcid_author["id"]


@pytest.mark.live
@pytest.mark.live_openalex
@pytest.mark.drift_watchdog
def test_openalex_semantic_search_returns_relevance_ranked_results() -> None:
    query = "how does retrieval augmented generation improve factual accuracy"
    semantic = _get(
        "works",
        per_page="10",
        select="id,display_name,relevance_score",
        **{"search.semantic": query},
    )
    results = semantic["results"]
    assert len(results) >= 1
    scores = [item["relevance_score"] for item in results]
    assert all(score is not None for score in scores)
    # Results come back sorted by semantic relevance (descending).
    assert scores == sorted(scores, reverse=True)

    # `search.semantic` is the recognized parameter; an unknown search param is rejected.
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get("works", per_page="1", **{"semantic_search": query})
    assert excinfo.value.code == 400

    # Semantic (embedding cosine) and keyword (BM25 + citation weighting) use different
    # scoring scales: keyword relevance for a multi-term query dwarfs semantic cosine.
    keyword = _get("works", per_page="1", search=query, select="id,relevance_score")
    assert scores[0] < keyword["results"][0]["relevance_score"]


@pytest.mark.live
@pytest.mark.live_openalex
def test_openalex_semantic_search_accepts_configured_api_key() -> None:
    api_key = os.environ.get("OPENALEX_API_KEY")
    if not api_key:
        pytest.skip("OPENALEX_API_KEY is required to verify the premium semantic-search key")

    semantic = _get(
        "works",
        api_key=api_key,
        per_page="3",
        select="id,relevance_score",
        **{"search.semantic": "multi-agent debate for factual verification"},
    )
    assert semantic["meta"]["count"] >= 1
    results = semantic["results"]
    assert results
    assert all(item.get("relevance_score") is not None for item in results)


@pytest.mark.live
@pytest.mark.live_openalex
@pytest.mark.drift_watchdog
def test_openalex_paper_source_live_surfaces_citation_graph_and_topics() -> None:
    source = OpenAlexPaperSource(OpenAlexSourceConfig(mailto=_MAILTO, timeout_seconds=30.0))
    result = source.search(
        "retrieval augmented generation factuality",
        limit=5,
        filters=SearchPaperFilters(),
    )

    assert result.status.status == "success"
    assert result.status.result_count >= 1
    items = result.results
    assert all(item.source == "openalex" for item in items)
    assert all(item.metadata.get("openalex_id") for item in items)

    # Citation graph: outbound references + inbound cited_by_count surfaced to metadata.
    assert any(item.metadata.get("references") for item in items)
    assert any(item.metadata.get("cited_by_count") is not None for item in items)
    # Topic taxonomy surfaced.
    assert any(item.metadata.get("topics") for item in items)
    # Cross-source identity bridge for dedup.
    assert any(item.external_ids.get("doi") for item in items)
    # Inverted abstract -> summary and authorships -> authors.
    assert any(item.summary for item in items)
    assert any(item.authors for item in items)
