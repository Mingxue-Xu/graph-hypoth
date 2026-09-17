# Paper Retrieval — User Guide

How the retrieval subsystem (`src/retrieval/`) finds papers
and returns their **abstracts** and **full text** as grounded evidence.

The guide describes the implemented adapters and their user-visible behavior.

---

## 1. What the system does

A single call — `RetrievalService.search_papers(...)` — fans a query out across one
or more **sources**, normalizes every hit into a uniform `SourceResult`, optionally
fetches open-access full text, then ranks and de-duplicates the hits into
**evidence** records. There is one retrieval mode: `source_only`
([config.py](../../src/config.py)).

The eight selectable sources:

| Source | `name` | Backed by | Default active? |
|--------|--------|-----------|-----------------|
| arXiv | `arxiv` | `ArxivLoader` (metadata or full-PDF) | ⬜ opt-in |
| Exa | `exa` | `exa-py` neural/keyword web search | ✅ (`optional_sources` default) |
| Crossref | `crossref` | Crossref REST API | ✅ (`sources` default) |
| OpenAlex | `openalex` | OpenAlex Works API | ✅ (`optional_sources` in shipped YAML) |
| Europe PMC | `europepmc` | Europe PMC REST API | ⬜ opt-in |
| Codex live web | `codex_web` | constrained `codex exec` native web search | ⬜ opt-in |
| Claude live web | `claude_web` | constrained `claude --print` with only the web tools | ⬜ opt-in |
| Fake | `fake` | deterministic synthetic data | test/debug only |

The default execution YAML selects required `crossref` plus optional `exa` and
`openalex`; arXiv and Europe PMC are opt-in. The bare `RetrievalConfig()`
constructor selects only `crossref` plus optional `exa`. The API example
configuration (`config/api.yaml`) explicitly enables all five HTTP/API sources;
that is a separate Setting 3 choice, not the default execution configuration.
The unbounded arXiv loader is available only when selected explicitly
([config.py](../../src/config.py)).
`fake_mode=True` replaces the configured source set with `["fake"]`; current
validation only rejects listing `fake` in `sources` / `optional_sources` when
`fake_mode=True` ([config.py](../../src/config.py)).

---

## 2. Abstract vs. full text — the data model

A result distinguishes the two by **field**
([models.py](../../src/retrieval/models.py)):

- **`summary`** → the **abstract** (or provider snippet).
- **`text`** → the **full text** (or source body), populated only when available.

Each scholarly source maps the abstract into `summary` and leaves `text=None` at
search time:

- Crossref: `summary = strip_markup(item.get("abstract"))` ([sources.py](../../src/retrieval/sources.py))
- OpenAlex: `summary = _invert_abstract(abstract_inverted_index)` ([sources.py](../../src/retrieval/sources.py))
- Europe PMC: `summary = abstractText` ([sources.py](../../src/retrieval/sources.py))

When evidence is built through the normal `RetrievalService` path, quote
selection is handled by `EvidenceLedger`: prefer a clean Exa highlight verified
against source `text`; otherwise select a query-relevant window from `text`;
otherwise use the shaped `summary` with `summary_fallback` metadata
([ledger.py](../../src/retrieval/ledger.py)).
There is no title fallback on the normal path; hits with no usable text or
summary get an empty quote. The raw `text or summary or title` branch is only
the service's import-fallback path ([service.py](../../src/retrieval/service.py)).

### Getting an abstract
Just run a normal search against Crossref / OpenAlex / Europe PMC (or arXiv/Exa).
At the source-adapter level, abstracts arrive in each `SourceResult.summary`.
Through `search_papers(...)`, abstract-only hits surface as evidence `quote`
values with `metadata.quote_selection.source == "summary"` and
`verification_status == "summary_fallback"`. **No extra config or fetch.**

### Getting full text
Two distinct paths:

1. **Inline full text** — returned directly by the source at search time:
   - **arXiv** in `full_pdf` mode → `text = page_content` (the parsed PDF, capped by
     `doc_content_chars_max`, default 4000) ([sources.py](../../src/retrieval/sources.py)).
   - **Exa** when `filters.text=True` → `text` carries the crawled source body
     ([sources.py](../../src/retrieval/sources.py)).

2. **Open-access fetch** — `fulltext.py` fetches OA full text *after* search,
   for metadata-only OpenAlex and Europe PMC results that expose OA locations.
   Crossref remains abstract/metadata-only because its adapter does not emit
   `is_oa` or full-text fetch locations. The upgrade is **opt-in and gated** —
   all of the following must hold:
   - `filters.text = True` on the request, and `fulltext.enabled = True` (default)
     ([service.py](../../src/retrieval/service.py)).
   - The result is open access: `metadata["is_oa"]` is truthy
     ([fulltext.py](../../src/retrieval/fulltext.py)).
   - A usable location exists, in priority order
     ([fulltext.py](../../src/retrieval/fulltext.py)):
     `jats_fulltext_url` (Europe PMC, [sources.py](../../src/retrieval/sources.py)) →
     `oa_url` / `best_oa_location.pdf_url` ending in `.pdf` (OpenAlex,
     [sources.py](../../src/retrieval/sources.py)).
   - For PDF (not JATS) extraction, **PyMuPDF** must be installed.

   Full-text config defaults ([config.py](../../src/config.py)):
   `timeout_seconds=15.0`, `max_bytes=20_000_000` (20 MB), `max_pages=40`. Fetches
   are SSRF-guarded (`_blocked_fetch_host`) and PDF payloads verified to start with
   `%PDF`. If any precondition is missing, the system **silently keeps the abstract**.

*Tests:* abstract→full-text upgrade is exercised by
`test_augment_full_text_is_opt_in_and_upgrades_oa_results`
(`tests/unit/test_retrieval_service.py:699`); OA detection and JATS extraction
by `tests/unit/test_retrieval_fulltext.py` (offline via an injected opener).
PDF URL detection and bounded PDF fetching are covered in
`tests/unit/test_paper_titles.py`; PDF-to-text extraction and
`fulltext.enabled=False` do not currently have dedicated tests.

### Codex live-web excerpts

`codex_web` is an online-material source, not a full-text fetcher. A fresh
retrieval-only Codex process performs native live web searches and returns a
schema-constrained title, public HTTP(S) URL, date/authors when known, and an
excerpt requested to be verbatim. GraphHypoth requires a successfully completed
Codex `web_search` JSONL event,
records the Codex thread ID/token usage, applies URL/domain/text/date filters,
and then admits the result through the same cache and evidence ledger as other
sources.

The host does not independently download every returned page to prove the
transcribed excerpt. To avoid claiming a self-comparison is quote verification,
the adapter stores the excerpt in `SourceResult.summary`, leaves `text=None`,
and tags it `web_search_observed_not_host_quote_verified`. It therefore reaches
evidence as a `summary_fallback` and uses the lower
`subagent_web_research` trust tier. Its model-reported DOI and title are not used
for cross-source ledger merging or citation/coherence identity, so the excerpt
cannot inherit a scholarly record's higher trust tier or create a false citation
edge. Corroborate material claims with a scholarly source or manual page review.

### Claude live-web excerpts

`claude_web` is the Claude analogue of `codex_web` and admits records under the
same rules: same strict records schema, same URL/domain admission, same recency
and excerpt caps, same `subagent_web_research` trust tier, and the same
`summary_fallback` treatment with `text=None` and the
`web_search_observed_not_host_quote_verified` tag. Its model-reported DOI and
title are likewise excluded from cross-source ledger merging and citation
identity. Those policy helpers are imported from `codex_web` rather than copied,
so the two sources cannot drift apart on what they will admit.

Only the transport and its proof of search differ. Codex reports a completed
`web_search` item in its JSONL; Claude reports `tool_use` blocks, which are
visible only under `--output-format stream-json`, so this source runs the CLI in
that mode and requires at least one observed `WebSearch` call before any record
is admitted. Any tool outside the configured allowlist rejects the whole
response.

Two boundaries are weaker than the Codex source and are disclosed rather than
implied away:

- **No `--output-schema`.** Codex constrains its final turn server-side; the
  Claude CLI has no equivalent, so the schema is embedded in the prompt and
  enforced host-side by the same pydantic models. That is strictly weaker — it
  can waste a call — but it cannot admit an off-schema record. Exactly one
  Markdown fence is tolerated around the JSON; nothing else is repaired.
- **No kernel sandbox.** The boundary is `--tools`, which names the available
  built-in tools exhaustively and is validated to contain only `WebSearch` and
  `WebFetch`. Because that surface is already read-only, `--permission-mode` is
  set so a headless turn cannot block on an approval prompt it has no way to
  answer.

Enable it by setting `source_limits.claude_web.model_id` and adding `claude_web`
to `sources`/`optional_sources`; like `codex_web` it is absent from the service
until explicitly selected, and it is capped to one resilience attempt and a
durable per-run subprocess budget. That `model_id` is a model name your saved
CLI login can use, not an OpenRouter catalog slug — the same key under an
`agents.*.model` block means something different.

### OpenAlex enrichment: citation graph & topics

Beyond the abstract, the **OpenAlex** adapter surfaces a citation graph and topic
labels into each result's `metadata`
([sources.py](../../src/retrieval/sources.py)):

| `metadata` key | Meaning |
|----------------|---------|
| `references` | outbound citations — list of OpenAlex IDs this work cites (`referenced_works`) |
| `cited_by_count` | inbound citation count |
| `cited_by_api_url` | OpenAlex API URL listing the citing works |
| `topics` | flat list of topic **display-names** (e.g. "Retrieval-Augmented Generation") |
| `openalex_id` | the work's OpenAlex `W…` id |
| `open_access`, `best_oa_location`, `oa_url`, `is_oa` | OA status / full-text location (drives the §2 full-text fetch) |

Cross-source identity for de-duplication is also captured: `external_ids` carries
`doi`, `pmid`, `pmcid`
([sources.py](../../src/retrieval/sources.py));
authors are display-names extracted from `authorships`
([sources.py](../../src/retrieval/sources.py)).
This is verified end-to-end against the live API by
`test_openalex_paper_source_live_surfaces_citation_graph_and_topics`
([test_openalex_live_e2e.py:262](../../tests/e2e/test_openalex_live_e2e.py)),
which asserts `references`, `cited_by_count`, `topics`, a `doi` external id,
`summary`, and `authors` are present.

> **Platform-only — not surfaced by the adapter.** OpenAlex itself exposes more
> than the adapter maps, as documented by the live contract tests in
> `tests/e2e/test_openalex_live_e2e.py`: the full topic **taxonomy** hierarchy
> (domain→field→subfield→topic) and `primary_topic.id` filtering ([:66](../../tests/e2e/test_openalex_live_e2e.py)),
> **bidirectional** citation filtering via the `cites:` query ([:106](../../tests/e2e/test_openalex_live_e2e.py)),
> and **author / institution IDs** (`A…`/ORCID, `I…`/ROR) ([:146](../../tests/e2e/test_openalex_live_e2e.py)).
> The adapter keeps only topic *names* (not the hierarchy or topic IDs), citation
> *counts + reference IDs* (no `cites:`-based querying), and author *display-names*
> (no author/institution IDs), and offers no filtering by topic/author/institution.

---

## 3. The four input modes

You can ask for papers along four input dimensions. They may be used alone or
combined. Support is **not uniform** across sources — this is the crux of the matrix.

| Input mode | Supported? | By which source | Mechanism (evidence) |
|------------|-----------|-----------------|----------------------|
| **1. Paper title** | Partial (free-text keyword, no title-only field) | all sources | the `query` string is sent as the API's free-text term: Crossref `query` ([sources.py](../../src/retrieval/sources.py)), OpenAlex `search` ([sources.py](../../src/retrieval/sources.py)), Europe PMC `query` ([sources.py](../../src/retrieval/sources.py)), arXiv loader ([sources.py](../../src/retrieval/sources.py)), Exa `query` ([sources.py](../../src/retrieval/sources.py)). Field-scoped syntax (`ti:`, `TITLE:`) works only if **you** embed it in the query. |
| **2. DOI** | ❌ No scholarly by-DOI lookup | `exa` workaround | Exa can fetch `urls=["https://doi.org/<doi>"]`. Codex exact-URL constraints reject the publisher URL after a DOI redirect, so they are not a reliable DOI resolver. |
| **3. Publisher** | ⚠️ Web sources only | `exa`, `codex_web` | Both honor `filters.include_domains` / `exclude_domains`; Codex also passes the effective allowlist to its native web-search tool and checks returned URLs host-side. |
| **4. Semantic search** | ⚠️ Web sources | `exa`, `codex_web` | Exa supplies neural search. Codex performs model-planned live web search; this is not an embedding-ranked scholarly index. OpenAlex's semantic API remains unwired — see note ‡. |

> **‡ OpenAlex semantic search — capability vs. wiring.** OpenAlex *can* do
> embedding-based semantic search (`search.semantic=`, relevance-ranked, served by
> the premium `OPENALEX_API_KEY` pool), as verified by the live contract tests
> `test_openalex_semantic_search_returns_relevance_ranked_results`
> ([test_openalex_live_e2e.py:213](../../tests/e2e/test_openalex_live_e2e.py)) and
> `..._accepts_configured_api_key` ([:241](../../tests/e2e/test_openalex_live_e2e.py)).
> The `OpenAlexPaperSource` adapter does not call `search.semantic`; it issues
> the keyword `search` query ([sources.py](../../src/retrieval/sources.py)),
> and `OpenAlexSourceConfig` has no semantic toggle
> ([config.py](../../src/config.py)). OpenAlex is therefore keyword-only
> through this retrieval pipeline.

**Key consequences**
- **Publisher scoping is available through Exa and Codex web retrieval.** Exa
  remains the wired source for true neural ranking; Codex supplies model-planned
  live-web research rather than a stable embedding-ranked index.
- **DOI is still a gap.** There is no "fetch paper by DOI" path in the scholarly
  APIs; DOIs are only *returned*, not *queried*. Resolve a DOI to a URL and use
  Exa `contents`; Codex exact-target mode is suited to stable, non-redirecting
  public URLs.

---

## 4. Comprehensive combination matrix

All 15 non-empty combinations of **T**itle, **D**OI, **P**ublisher,
**S**emantic. "Source" is the source that satisfies the combination; abstract is
always available on a successful hit (in `summary`), and full text follows the
rules in §2.

This matrix describes the deterministic scholarly/Exa paths. `codex_web` can
attempt the same combinations by combining its natural-language query with
target-URL and domain constraints, but it is probabilistic live-web research,
not a precise DOI/publisher API, and does not supply independently verified full
text. It is therefore not counted as turning an unsupported exact lookup into a
supported one below.

| # | Combination | Supported | Source / how | Abstract | Full text |
|---|-------------|-----------|--------------|----------|-----------|
| 1 | **T** (title only) | ✅ | any source: title → `query` | ✅ `summary` | arXiv `full_pdf` / Exa `text` / OA fetch |
| 2 | **D** (DOI only) | ❌ (workaround) | Exa `contents`, `urls=[doi-url]` | ✅ if page yields it | Exa `text` of the resolved page |
| 3 | **P** (publisher only) | ⚠️ needs a query | Exa `include_domains` + a query term | ✅ `summary` | Exa `text` / OA fetch |
| 4 | **S** (semantic only) | ✅ | Exa neural (`query` = NL question) | partial (`summary` if Exa returns) | Exa `text` / OA fetch |
| 5 | **T + D** | ❌ | DOI not a search param; degrades to title search, or Exa-contents on the DOI URL (ignores title) | — | — |
| 6 | **T + P** | ✅ | Exa: `query`=title, `include_domains`=publisher | ✅ | Exa `text` / OA fetch |
| 7 | **T + S** | ✅ | Exa neural: `query`=title text | partial | Exa `text` / OA fetch |
| 8 | **D + P** | ❌ | no DOI param; Exa-contents fetch ignores domain filter | — | — |
| 9 | **D + S** | ❌ | DOI lookup is a contents fetch, mutually exclusive with semantic search | — | — |
| 10 | **P + S** | ✅ | Exa neural + `include_domains` | partial | Exa `text` / OA fetch |
| 11 | **T + D + P** | ❌ | DOI blocks the combined precise lookup | — | — |
| 12 | **T + D + S** | ❌ | DOI blocks the combined precise lookup | — | — |
| 13 | **T + P + S** | ✅ | Exa neural + `query`=title + `include_domains` | partial | Exa `text` / OA fetch |
| 14 | **D + P + S** | ❌ | DOI + filters incompatible with contents fetch | — | — |
| 15 | **T + D + P + S** | ❌ | DOI blocks the combined precise lookup | — | — |

> **Why `contents` and filters don't mix:** supplying `urls`/`target_urls` flips the
> Exa request to `request_kind="contents"`
> ([models.py](../../src/retrieval/models.py)),
> a direct URL fetch — domain filters and semantic ranking apply to the `search`
> path, not the `contents` path.

**Bottom line.** To retrieve abstracts/full text you have effectively two levers:
the **query string** (title or natural-language — works on every source), the
**Exa source** (stable neural/domain-scoped web retrieval), and the opt-in
**Codex source** (model-planned live-web retrieval with domain/URL constraints).
**Precise DOI-keyed retrieval is unsupported** beyond resolving the DOI to a URL
and using Exa. If exact DOI lookup matters, that remains the feature to
add (e.g. a Crossref `/works/{doi}` or OpenAlex `doi:` filter path).

---

## 5. Wrapped interfaces (how to call it)

### 5.1 `RetrievalService` (programmatic)

```python
from src.config import RetrievalConfig
from src.log_store import SQLiteLogStore
from src.retrieval.service import RetrievalService

store = SQLiteLogStore(tmp_path / "events.sqlite"); store.setup()
service = RetrievalService(
    config=RetrievalConfig(sources=["arxiv"], optional_sources=["exa"]),
    log_store=store,
)

result = service.search_papers(
    run_id="run-1",
    query="retrieval augmented debate",
    sources=None,           # None -> configured defaults; or e.g. ["exa", "openalex"]
    limit=3,                # defaults to config.final_top_k (8)
    filters={"text": True}, # SearchPaperFilters or dict; set text=True to pull full text
    called_by="builder",
    tool_call_id="tool-1",
)
assert result.sources == ["arxiv", "exa"]
result.evidence[0]["quote"]    # selected evidence passage, or summary fallback
```

Signature ([service.py](../../src/retrieval/service.py)):
`search_papers(*, run_id, query, sources, limit, filters, called_by, tool_call_id) -> RetrievalToolResult`.
Source selection ([service.py](../../src/retrieval/service.py)):
the service first builds a candidate list from explicit `sources`, otherwise
from `["fake"]` when `fake_mode=True`, otherwise from
`config.selected_source_names()`. Unknown candidate names raise `ValueError`.
If `fake_mode=True`, the final selected source set is always `["fake"]`, even
when explicit real sources were requested. `result_to_dict(result)` returns a
JSON-able dict ([service.py](../../src/retrieval/service.py)).

### 5.2 The host-side `search_papers` wrapper

Graph-state orchestration calls `RetrievalToolRegistry.search_papers` directly.
The host wrapper adds budget enforcement, input sanitization, run-ledger
admission, audit events, and optional artifact output before and after
delegating to the service. Retrieval is not exposed to model providers as a
CAMEL or OpenAI function tool.

Tool call signature: `search_papers(query, sources=None, limit=None, filters=None)`.
Input normalization ([tool_registry.py](../../src/retrieval/tool_registry.py)):
bare domains passed in `sources` are folded into `filters.include_domains`, and
`include_text`/`exclude_text` are clamped to a single ≤5-word phrase.

Source selection remains constrained by the configured and available service
adapters. Returned records are re-admitted to the run-level evidence ledger,
which replaces call-local identifiers with stable run-local identifiers before
the event payload is written.

### 5.3 `SearchPaperFilters` — what you can set

([models.py](../../src/retrieval/models.py), `extra="forbid"`).
The fields relevant to the four input modes:

| Field | Type | Relevant to | Notes |
|-------|------|-------------|-------|
| `urls`, `target_urls` | list[str] | URL retrieval | public http/https domain URLs only for Codex; flips Exa to `contents`; constrains Codex to exact, non-redirected URLs |
| `include_domains`, `exclude_domains` | list[str] | **publisher** | lowercased; Exa and `codex_web` |
| `text` | bool | **full text** | turn on to pull inline/OA full text |
| `highlights`, `highlight_query`, `highlight_max_characters` | bool/str/int | semantic passages | Exa highlight selection |
| `anchor_queries` | list[str] (≤8) | semantic | per-anchor Exa contents calls |
| `include_text`, `exclude_text` | list[str] | filtering | ≤1 string, ≤5 words |
| `text_max_characters`, `max_age_hours`, `livecrawl_timeout`, `include_html_tags`, `target_scope`, `verify_quotes` | mixed | tuning | — |

> There is **no `category` field** on the request filters; Exa's
> `category="research paper"` lives on `ExaSourceConfig`
> ([config.py](../../src/config.py)), set per-deployment.

### 5.4 Result shape

`RetrievalToolResult` ([models.py](../../src/retrieval/models.py)):
`tool_call_id, query, filters, sources, evidence: list[dict], source_statuses:
list[SourceStatus], warnings, errors, elapsed_ms, input_hash,
retrieval_config_hash`. Each `evidence` dict carries `evidence_id, source, title,
authors, url, quote, relevance, trust_tier, score, rank, metadata`
([service.py](../../src/retrieval/service.py)).
`SourceStatus` ([models.py](../../src/retrieval/models.py))
reports per-source `status` (`success`/`partial_failure`/`failed`/`skipped`),
`result_count`, `warnings`, `errors`, and `unused_filters` (filters a non-Exa
source could not honor).

---

## 6. Configuring sources

```yaml
retrieval:
  sources: ["crossref"]                # required default source
  optional_sources: ["exa", "openalex"]  # appended, de-duped
  final_top_k: 100
  per_source_top_k: 25
  fulltext:
    enabled: true
  source_limits:
    exa:
      category: "research paper"
      include_domains: ["arxiv.org"]
```

Valid source names:
`arxiv, exa, crossref, openalex, europepmc, codex_web, claude_web, fake`
([config.py](../../src/config.py)). `fake_mode=True`
selects `["fake"]` and rejects configs that also list `fake` in
`sources`/`optional_sources` ([config.py](../../src/config.py));
with `fake_mode=False`, current validation treats `fake` as a valid configured
source name. API keys (Exa, OpenAlex) load from the environment per
[Local Setup And API Key Loading](local-setup-and-key-loading.md); a source with a
required-but-missing key returns `status="skipped"` (never an error).

### 6.1 Exa source settings

Exa is not a separate retrieval path. It is the `exa` source adapter inside the
same `RetrievalService.search_papers(...)` flow as Crossref, OpenAlex, arXiv,
Europe PMC, and the live-web sources. Keeping Exa behind that adapter ensures
its hits are normalized, ledgered, quote-checked, ranked, cached, and subject to
the same run-level audit trail.

Export the environment variable named by
`retrieval.source_limits.exa.require_api_key_env` before a live run. The default
name is `EXA_API_KEY`:

```bash
export EXA_API_KEY='...'
```

Do not put the key value in YAML or a tracked repository file. For external key
files, use the credential-loading options described in
[Local Setup And API Key Loading](local-setup-and-key-loading.md).

Exa can be required or optional:

```yaml
retrieval:
  sources:
    - crossref
  optional_sources:
    - exa
    - openalex
```

An optional Exa source that cannot run is recorded as skipped while the rest of
retrieval continues. Put `exa` under `sources` only when a run must require it.

Configure provider-specific behavior under `retrieval.source_limits.exa`:

```yaml
retrieval:
  source_limits:
    exa:
      require_api_key_env: EXA_API_KEY
      search_type: auto
      category: research paper
      text: true
      highlights: true
      highlight_max_characters: 1200
      text_max_characters: 20000
      max_qps: 10
      max_cost_dollars_per_call: 0.05
      max_cost_dollars_per_run: 0.25
```

The main settings are:

- `search_type`: `auto`, `neural`, or `keyword`.
- `category`: provider-side category, normally `research paper`.
- `text`: request extracted source text.
- `highlights`: request relevant passages for evidence selection.
- `highlight_query`: optional query used to target highlights.
- `include_domains` / `exclude_domains`: restrict web domains.
- `include_text` / `exclude_text`: provider text filters. Each accepts at most
  one phrase of five words or fewer.
- `max_qps`: local pacing ceiling.
- `max_cost_dollars_per_call` / `max_cost_dollars_per_run`: guards that skip a
  request before it would exceed the configured budget.

Result count is controlled by retrieval-level caps, not by an Exa-only setting.
Normal searches request `retrieval.per_source_top_k`; Exa fallback backfill uses
the larger `retrieval.final_top_k` so it can cover a failed source.

Normal retrieval sends the query to Exa search. Programmatic callers may pass
`urls` or `target_urls` through `SearchPaperFilters`; that switches the adapter
to an Exa contents request. Domain filters and semantic ranking apply to search
requests, not direct URL contents requests. The adapter prefers verified
highlights when source text is available, then falls back to a source-text
window or a shaped summary. Provider cost metadata is recorded when Exa supplies
it.

Common Exa troubleshooting:

- `EXA_API_KEY` missing: export the configured variable in the same process
  environment as GraphHypoth.
- Exa is skipped by a cost guard: raise the relevant limit deliberately, lower
  `retrieval.per_source_top_k`, or reduce direct-URL content options.
- `exa-py Exa client unavailable`: install the retrieval dependencies with
  `python -m pip install -e ".[retrieval]"`.
- Exa fails while another source succeeds: inspect per-source status and
  warnings in the retrieval artifacts; optional-source failure does not discard
  successful results from other sources.

`codex_web` is deliberately absent from both default source lists. Selecting it
requires `source_limits.codex_web.model_id`; saved `codex login` authentication;
and a Codex CLI version that supports strict config, `web_search="live"`,
`tools.web_search`, JSONL events, and `--output-schema`. See [Local CLI Subagent
Backends And Live-Web Sources](cli-subagent-backends.md) for the complete YAML
and security/cost boundaries. Its
`max_calls_per_run` value is persisted in the SQLite run store and counts Codex
subprocess attempts; a single turn may perform multiple native web searches.
The synthesist generates a fresh run ID by default, while explicitly reusing
`--run-id` also opts into retrieval-cache replay.

`claude_web` is likewise absent from both default source lists. Selecting it
requires `source_limits.claude_web.model_id`; saved Claude Code CLI
authentication; and, when resilience is enabled for it, exactly one attempt
(`max_attempts: 1`) — all three are enforced at config load
([config.py](../../src/config.py)). See [Local CLI Subagent Backends And
Live-Web Sources](cli-subagent-backends.md) for the complete YAML and the weaker
boundaries described above.

The strict override disables Code Mode with `features.code_mode=false`; do not
replace that with `features.code_mode_host=false`. Current Codex CLI schema
output uses the internal host even while the Code Mode capability is disabled,
so forcing the host switch off produces a startup error before live web search.

---

## 7. Quick recipes

| Goal | Call |
|------|------|
| Abstract by title | Source adapters store abstracts in `SourceResult.summary`; `search_papers(...)` returns normalized evidence, so abstract-only hits surface through `evidence[*]["quote"]` with `metadata.quote_selection.source == "summary"` / `verification_status == "summary_fallback"` |
| Full text (OA) by title | `search_papers(query="<title>", sources=["openalex","europepmc"], filters={"text": True})` |
| arXiv full text | Configure arXiv with `source_limits.arxiv.mode="full_pdf"` (default), then call `search_papers(query="<title>", sources=["arxiv"], filters=None)`; `filters.text` is ignored by arXiv and reported as unused |
| Semantic search | `search_papers(query="<natural-language question>", sources=["exa"], filters={"highlights": True})` |
| Scope to a publisher | `search_papers(query="<terms>", sources=["exa"], filters={"include_domains": ["nature.com"]})` |
| Live-web materials through Codex | Configure and select `codex_web`, then call `search_papers(query="<question>", sources=["codex_web"], filters={"include_domains": ["nature.com"]})` |
| By DOI (workaround) | `search_papers(query="", sources=["exa"], filters={"urls": ["https://doi.org/<doi>"], "text": True})` |

---

*Verified against `tests/unit/test_retrieval_sources.py`,
`test_retrieval_fulltext.py`, `test_retrieval_service.py`,
`test_retrieval_tool_registry.py`, `test_retrieval_config.py`,
`test_arxiv_source_truncation.py`, `test_codex_web_retrieval.py`,
`test_claude_web_retrieval.py`.*
