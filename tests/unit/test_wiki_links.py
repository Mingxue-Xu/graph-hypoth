"""Wiki-link resolver ( Critic Panel): verify-before-link chain wikipedia-summary ->
opensearch (re-verified) -> wikidata (label guard) -> scholar-search fallback.
Pure stdlib; ALL tests inject fetch_json/opener/sleep fakes — ZERO live network."""

from __future__ import annotations

import json
from urllib.parse import quote_plus

import pytest


def _std_summary(title, url):
    return (
        200,
        {
            "type": "standard",
            "title": title,
            "content_urls": {"desktop": {"page": url}},
        },
    )


def _router(routes):
    """fetch_json fake: first (substring, response) route matching the url wins.
    A response may be a list (queue, popped per call) or a single (status, data)."""
    calls = []

    def fetch(url):
        calls.append(url)
        for needle, response in routes:
            if needle in url:
                if isinstance(response, list):
                    return response.pop(0) if response else (0, None)
                return response
        return (0, None)

    return fetch, calls


def _no_sleep(_seconds):
    return None


# --- resolve_term chain ----------------------------------------------------------------------
def test_standard_summary_resolves_to_wikipedia_link():
    from src.wiki_links import resolve_term

    fetch, calls = _router(
        [("/page/summary/", _std_summary("Belief propagation", "https://en.wikipedia.org/wiki/Belief_propagation"))]
    )
    link = resolve_term("belief propagation", fetch_json=fetch, sleep=_no_sleep)
    assert link is not None
    assert link.kind == "wikipedia"
    assert link.url == "https://en.wikipedia.org/wiki/Belief_propagation"
    assert link.title == "Belief propagation"
    assert len(calls) == 1
    assert "belief_propagation" in calls[0]  # spaces become underscores in the path


def test_disambiguation_skips_to_scholar_never_guessing_a_sense():
    from src.wiki_links import resolve_term

    fetch, calls = _router([("/page/summary/", (200, {"type": "disambiguation"}))])
    link = resolve_term("mercury", fetch_json=fetch, sleep=_no_sleep)
    assert link.kind == "scholar-search"
    assert link.url == "https://scholar.google.com/scholar?q=" + quote_plus("mercury")
    assert len(calls) == 1  # no opensearch/wikidata guessing


def test_404_falls_to_opensearch_and_reverifies_before_linking():
    from src.wiki_links import resolve_term

    fetch, calls = _router(
        [
            ("/page/summary/factor_graphs", (404, None)),
            ("action=opensearch", (200, ["factor graphs", ["Factor graph"], [""], [""]])),
            ("/page/summary/Factor_graph", _std_summary("Factor graph", "https://en.wikipedia.org/wiki/Factor_graph")),
        ]
    )
    link = resolve_term("factor graphs", fetch_json=fetch, sleep=_no_sleep)
    assert link.kind == "wikipedia"
    assert link.url == "https://en.wikipedia.org/wiki/Factor_graph"
    assert len(calls) == 3  # summary -> opensearch -> RE-VERIFY summary (max 3/term)
    assert "Factor_graph" in calls[2]  # the re-verify used the opensearch title


# --- regression: the five observed mislink shapes (verify-r 2026-07-02) ----------------------
def test_summary_redirect_to_wrong_concept_falls_through():  # DESs -> 'Dessert'
    from src.wiki_links import resolve_term

    page = (200, {"type": "standard", "title": "Dessert",
                  "extract": "Dessert is a course that concludes a meal.",
                  "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Dessert"}}})
    fetch, calls = _router([
        ("/page/summary/", page),
        ("wbsearchentities", (200, {"search": []})),
    ])
    link = resolve_term("DESs", fetch_json=fetch, sleep=_no_sleep)
    assert link.kind == "scholar-search"      # never links a page that doesn't NAME the term
    assert len(calls) == 2                    # summary -> wikidata (strict) -> scholar


def test_summary_redirect_kept_when_extract_names_term():  # SNAr redirect stays linked
    from src.wiki_links import resolve_term

    page = (200, {"type": "standard", "title": "Nucleophilic aromatic substitution",
                  "extract": "A nucleophilic aromatic substitution (SNAr) is a substitution reaction.",
                  "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Nucleophilic_aromatic_substitution"}}})
    fetch, _ = _router([("/page/summary/", page)])
    link = resolve_term("SNAr", fetch_json=fetch, sleep=_no_sleep)
    assert link.kind == "wikipedia"
    assert link.url.endswith("Nucleophilic_aromatic_substitution")


def test_opensearch_reverified_page_must_name_the_term():  # 'low toxicity' -> 'Tonicity'
    from src.wiki_links import resolve_term

    fetch, calls = _router([
        ("/page/summary/low_toxicity", (404, None)),
        ("action=opensearch", (200, ["low toxicity", ["Tonicity"], [""], [""]])),
        ("/page/summary/Tonicity", (200, {"type": "standard", "title": "Tonicity",
                                          "extract": "Tonicity is a measure of the effective osmotic pressure gradient.",
                                          "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Tonicity"}}})),
    ])
    link = resolve_term("low toxicity", fetch_json=fetch, sleep=_no_sleep)
    assert link.kind == "scholar-search"      # re-verified page fails the naming bar
    assert len(calls) == 3                    # budget respected


def test_wikidata_alias_fuzz_rejected():  # NADES -> gamer tag via alias
    from src.wiki_links import resolve_term

    hit = {"search": [{"id": "Q99", "label": "Nadeshot", "aliases": ["NADES", "nades"]}]}
    fetch, _ = _router([
        ("/page/summary/", (404, None)),
        ("action=opensearch", (200, ["NADES", [], [], []])),
        ("wbsearchentities", (200, hit)),
    ])
    link = resolve_term("NADES", fetch_json=fetch, sleep=_no_sleep)
    assert link.kind == "scholar-search"      # aliases no longer accepted — main label only


def test_short_term_sense_veto_and_rescue():  # 'Nades' the village vs NADES the solvents; SNAr rescue
    from src.wiki_links import resolve_term

    solvent_ctx = "natural deep eutectic solvents made from plant metabolites"
    village = (200, {"type": "standard", "title": "Nades",
                     "extract": "Nades is a village in Lorestan province, Iran.",
                     "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Nades"}}})
    fetch, _ = _router([("/page/summary/", village), ("wbsearchentities", (200, {"search": []}))])
    link = resolve_term("NADES", context=solvent_ctx, fetch_json=fetch, sleep=_no_sleep)
    assert link.kind == "scholar-search"      # title-equal but WRONG SENSE -> vetoed

    snar_ctx = "a nucleophilic aromatic substitution reaction on activated arenes"
    redirect = (200, {"type": "standard", "title": "Nucleophilic aromatic substitution",
                      "extract": "A reaction where a nucleophile displaces a leaving group on an aromatic ring.",
                      "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Nucleophilic_aromatic_substitution"}}})
    fetch2, _ = _router([("/page/summary/", redirect)])
    link2 = resolve_term("SNAr", context=snar_ctx, fetch_json=fetch2, sleep=_no_sleep)
    assert link2.kind == "wikipedia"          # extract omits the acronym; strong context overlap rescues


def test_wikidata_multiword_term_needs_context_agreement():  # verify-gbp regression
    from src.wiki_links import resolve_term

    company = {"search": [{"id": "Q131633603", "label": "Spectrum Control",
                           "description": "French electronics company"}]}
    fetch, _ = _router([
        ("/page/summary/", (404, None)),
        ("action=opensearch", (200, ["spectrum control", [], [], []])),
        ("wbsearchentities", (200, company)),
    ])
    ctx = "keeping eigenvalue spectra of quantized message precision matrices bounded"
    link = resolve_term("spectrum control", context=ctx, fetch_json=fetch, sleep=_no_sleep)
    assert link.kind == "scholar-search"      # label-equal but a company, not the concept


def test_wikidata_short_term_needs_context_agreement():  # description-level sense check
    from src.wiki_links import resolve_term

    gamer = {"search": [{"id": "Q99", "label": "NADES", "description": "professional video gamer"}]}
    fetch, _ = _router([
        ("/page/summary/", (404, None)),
        ("action=opensearch", (200, ["NADES", [], [], []])),
        ("wbsearchentities", (200, gamer)),
    ])
    ctx = "natural deep eutectic solvents made from plant metabolites"
    assert resolve_term("NADES", context=ctx, fetch_json=fetch, sleep=_no_sleep).kind == "scholar-search"


def test_plural_and_hyphen_tolerant_title_equality():  # legit near-equal titles stay linked
    from src.wiki_links import resolve_term

    page = _std_summary("Deep eutectic solvent", "https://en.wikipedia.org/wiki/Deep_eutectic_solvent")
    fetch, _ = _router([("/page/summary/", page)])
    link = resolve_term("deep-eutectic solvents", fetch_json=fetch, sleep=_no_sleep)
    assert link.kind == "wikipedia"           # plural + hyphen normalize to the same canon


def test_wikidata_accepted_only_on_label_guard():
    from src.wiki_links import resolve_term

    hit = {"search": [{"id": "Q170058", "label": "Factor graph", "aliases": []}]}
    fetch, calls = _router(
        [
            ("/page/summary/", (404, None)),
            ("action=opensearch", (200, ["factor graph", [], [], []])),
            ("wbsearchentities", (200, hit)),
        ]
    )
    link = resolve_term("factor graph", fetch_json=fetch, sleep=_no_sleep)
    assert link.kind == "wikidata"
    assert link.url == "https://www.wikidata.org/wiki/Q170058"
    assert len(calls) == 3

    # guard rejects a non-matching label -> honest scholar-search fallback
    miss = {"search": [{"id": "Q1", "label": "completely different thing", "aliases": []}]}
    fetch2, _ = _router(
        [
            ("/page/summary/", (404, None)),
            ("action=opensearch", (200, ["factor graph", [], [], []])),
            ("wbsearchentities", (200, miss)),
        ]
    )
    assert resolve_term("factor graph", fetch_json=fetch2, sleep=_no_sleep).kind == (
        "scholar-search"
    )


def test_exhausted_chain_falls_back_to_scholar_search():
    from src.wiki_links import resolve_term

    fetch, _ = _router(
        [
            ("/page/summary/", (404, None)),
            ("action=opensearch", (200, ["x", [], [], []])),
            ("wbsearchentities", (200, {"search": []})),
        ]
    )
    link = resolve_term("posterior-mean bottleneck", fetch_json=fetch, sleep=_no_sleep)
    assert link.kind == "scholar-search"
    assert link.url == (
        "https://scholar.google.com/scholar?q=" + quote_plus("posterior-mean bottleneck")
    )


def test_transport_failure_returns_none_and_is_never_cached():
    from src.wiki_links import resolve_term, resolve_terms

    fetch, calls = _router([])  # everything is a transport error (0, None)
    assert resolve_term("anything", fetch_json=fetch, sleep=_no_sleep) is None

    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cache_path = pathlib.Path(tmp) / "cache.json"
        out = resolve_terms(["t1"], cache_path=cache_path, fetch_json=fetch, sleep=_no_sleep)
        assert out["t1"] is None
        if cache_path.exists():  # transport errors must never negative-cache
            assert json.loads(cache_path.read_text(encoding="utf-8")) == {}


def test_network_down_latch_stops_fetching_after_5_consecutive_failures():
    from src.wiki_links import resolve_terms

    fetch, calls = _router([])  # all transport errors
    terms = [f"term {i}" for i in range(8)]
    out = resolve_terms(terms, fetch_json=fetch, sleep=_no_sleep)
    assert all(link is None for link in out.values())
    assert len(calls) == 5  # latch after 5 consecutive transport failures


def test_definitive_miss_is_negative_cached_as_null(tmp_path):
    from src.wiki_links import resolve_terms

    fetch, _ = _router(
        [
            ("/page/summary/", (404, None)),
            ("action=opensearch", (200, ["x", [], [], []])),
            ("wbsearchentities", (200, {"search": []})),
        ]
    )
    cache_path = tmp_path / "cache.json"
    out = resolve_terms(["Mystery Term"], cache_path=cache_path, fetch_json=fetch, sleep=_no_sleep)
    assert out["Mystery Term"].kind == "scholar-search"  # still an honest link
    stored = json.loads(cache_path.read_text(encoding="utf-8"))
    assert stored["mystery term"] is None  # chain exhausted with network up -> null


def test_cache_roundtrip_and_precedence(tmp_path):
    from src.wiki_links import load_cache, resolve_terms, save_cache

    cache_path = tmp_path / "cache.json"
    entry = {
        "url": "https://en.wikipedia.org/wiki/Cumulant",
        "kind": "wikipedia",
        "title": "Cumulant",
        "resolved_at": "2026-07-02T00:00:00+00:00",
    }
    save_cache(cache_path, {"cumulant": entry, "known miss": None})
    assert load_cache(cache_path) == {"cumulant": entry, "known miss": None}

    fetch, calls = _router([])  # any fetch would be a transport error
    out = resolve_terms(
        ["Cumulant", "known miss"], cache_path=cache_path, fetch_json=fetch, sleep=_no_sleep
    )
    assert calls == []  # cache wins: zero fetches
    assert out["Cumulant"].url == entry["url"] and out["Cumulant"].kind == "wikipedia"
    assert out["known miss"].kind == "scholar-search"  # null miss -> derived search link


def test_missing_or_corrupt_cache_loads_empty(tmp_path):
    from src.wiki_links import load_cache

    assert load_cache(tmp_path / "absent.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_cache(bad) == {}


def test_sleep_is_called_between_serialized_http_calls():
    from src.wiki_links import resolve_term, resolve_terms

    sleeps = []
    fetch, _ = _router(
        [
            ("/page/summary/factor_graphs", (404, None)),
            ("action=opensearch", (200, ["factor graphs", ["Factor graph"], [""], [""]])),
            ("/page/summary/Factor_graph", _std_summary("Factor graph", "https://x/wiki/f")),
        ]
    )
    resolve_term("factor graphs", fetch_json=fetch, sleep=sleeps.append)
    assert sleeps == [0.15, 0.15]  # between calls 1-2 and 2-3, not before the first

    sleeps2 = []
    fetch2, _ = _router([
        ("/page/summary/alpha", _std_summary("Alpha", "https://x/wiki/alpha")),
        ("/page/summary/beta", _std_summary("Beta", "https://x/wiki/beta")),
    ])
    resolve_terms(["alpha", "beta"], fetch_json=fetch2, sleep=lambda s: sleeps2.append(s))
    assert sleeps2 == [0.15]  # one sleep between the two terms' single calls


# --- attach_key_term_links --------------------------------------------------------------------
def _elabs():
    return {
        "h1": {
            "headline": "H",
            "key_terms": [
                {"term": "belief propagation", "plain_meaning": "message passing"},
                {
                    "term": "GBP",
                    "plain_meaning": "Gaussian BP",
                    "link": "https://existing.example/gbp",
                    "link_kind": "manual",
                },
            ],
        },
        "h2": {
            "headline": "H2",
            "key_terms": [{"term": "Belief Propagation", "plain_meaning": "dup, casefold"}],
        },
    }


def test_attach_disabled_returns_input_untouched_with_zero_fetches(tmp_path):
    from src.wiki_links import attach_key_term_links

    fetch, calls = _router([])
    elabs = _elabs()
    out = attach_key_term_links(
        elabs, cache_path=tmp_path / "c.json", enabled=False, fetch_json=fetch
    )
    assert out is elabs
    assert calls == []


def test_attach_writes_link_only_where_absent_and_dedups_terms(tmp_path):
    from src.wiki_links import attach_key_term_links

    fetch, calls = _router(
        [("/page/summary/", _std_summary("Belief propagation", "https://en.wikipedia.org/wiki/Belief_propagation"))]
    )
    elabs = _elabs()
    out = attach_key_term_links(
        elabs, cache_path=tmp_path / "c.json", fetch_json=fetch, sleep=_no_sleep
    )
    entry = out["h1"]["key_terms"][0]
    assert entry["link"] == "https://en.wikipedia.org/wiki/Belief_propagation"
    assert entry["link_kind"] == "wikipedia"
    # pre-linked entry untouched
    assert out["h1"]["key_terms"][1]["link"] == "https://existing.example/gbp"
    assert out["h1"]["key_terms"][1]["link_kind"] == "manual"
    # casefold-duplicate term resolved ONCE, but both cards get the link
    assert len(calls) == 1
    assert out["h2"]["key_terms"][0]["link"] == "https://en.wikipedia.org/wiki/Belief_propagation"
    # the input dict is not mutated
    assert "link" not in elabs["h1"]["key_terms"][0]


# --- _fetch_json ------------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize(
    ("env_contact", "expected_in_ua", "forbidden_in_ua"),
    [
        # No personal email in source: the default contact is the project URL.
        (None, "https://github.com/Mingxue-Xu/graph-hypoth", "tiny.mercy.lovelace@gmail.com"),
        # GRAPH_HYPOTH_WIKI_CONTACT overrides the default contact at call time.
        ("mailto:ops@example.org", "mailto:ops@example.org", None),
    ],
    ids=["default_contact", "env_contact"],
)
def test_fetch_json_sends_wiki_link_ua(
    monkeypatch, env_contact, expected_in_ua, forbidden_in_ua
):
    from src.wiki_links import _fetch_json, _wiki_link_ua

    if env_contact is None:
        monkeypatch.delenv("GRAPH_HYPOTH_WIKI_CONTACT", raising=False)
    else:
        monkeypatch.setenv("GRAPH_HYPOTH_WIKI_CONTACT", env_contact)
    assert expected_in_ua in _wiki_link_ua()
    if forbidden_in_ua is not None:
        assert forbidden_in_ua not in _wiki_link_ua()
    seen = {}

    def opener(request, timeout=None):
        seen["ua"] = request.get_header("User-agent")
        return _FakeResponse(200, json.dumps({"ok": True}).encode("utf-8"))

    status, data = _fetch_json("https://en.wikipedia.org/api/rest_v1/page/summary/X", opener=opener)
    assert (status, data) == (200, {"ok": True})
    assert seen["ua"] == _wiki_link_ua()


def test_fetch_json_never_raises():
    from urllib.error import HTTPError

    from src.wiki_links import _fetch_json

    def raising_opener(request, timeout=None):
        raise OSError("network down")

    assert _fetch_json("https://x", opener=raising_opener) == (0, None)

    def http_error_opener(request, timeout=None):
        raise HTTPError("https://x", 404, "not found", hdrs=None, fp=None)

    # 404 is a definitive ANSWER (drives the opensearch fallback), not a transport error
    assert _fetch_json("https://x", opener=http_error_opener) == (404, None)

    def garbage_opener(request, timeout=None):
        return _FakeResponse(200, b"not json")

    assert _fetch_json("https://x", opener=garbage_opener) == (0, None)


def test_default_wiki_cache_matches_historical_string():
    # Consolidated constant (cli-paths unit): every caller default MUST still
    # resolve to this exact path — a pure name for the repeated literal.
    from pathlib import Path

    from src.wiki_links import DEFAULT_WIKI_CACHE

    assert DEFAULT_WIKI_CACHE == Path("out/wiki-link-cache.json")
