# Exa Retrieval Configuration

GraphHypoth uses the `exa-py` client through `src/retrieval/sources.py`. Keep Exa
behind that adapter so results are normalized, ledgered, quote-checked, and
subject to the configured call and cost limits.

## Credential

Export the environment variable named by
`retrieval.source_limits.exa.require_api_key_env` before a live run. The default
name is `EXA_API_KEY`:

```bash
export EXA_API_KEY='...'
```

Do not put the key value in YAML or a tracked repository file. For an external
credentials-file option, see [Local Setup And API Key Loading](local-setup-and-key-loading.md).

## Enable Exa

Exa can be a required or optional source. The repository configuration keeps
bounded Crossref metadata required and selects Exa and OpenAlex as optional
sources; arXiv is opt-in:

```yaml
retrieval:
  sources:
    - crossref
  optional_sources:
    - exa
    - openalex
```

An optional source that cannot run is recorded as skipped while the remaining
sources continue. Put `exa` under `sources` when the run must require it.

## Adapter Settings

Configure Exa under `retrieval.source_limits.exa`:

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
- Result count is not an Exa-block setting. The adapter requests exactly the
  caller's limit: `retrieval.per_source_top_k` for a normal search, and the
  larger `retrieval.final_top_k` for the Exa fallback backfill, which pulls
  extra pages to cover a failed source.
- `max_qps`: local pacing ceiling.
- `max_cost_dollars_per_call` / `max_cost_dollars_per_run`: guards that skip a
  request before it would exceed the configured budget.

## Search And Direct-URL Retrieval

Normal retrieval sends a query to Exa search. Programmatic callers may pass
`urls` or `target_urls` through `SearchPaperFilters`; this switches the adapter
to a direct contents request. Domain filters and semantic ranking apply to
search requests, not direct URL contents requests.

The adapter prefers verified highlights when source text is available. It can
fall back to a relevant source-text window or a shaped summary. Returned
provider cost metadata is recorded when Exa supplies it.

For the complete cross-source result model and examples, see
[Paper Retrieval](paper_retrieval.md).

## Troubleshooting

- `EXA_API_KEY` missing: export the configured variable in the same process
  environment as GraphHypoth.
- Exa is skipped by a cost guard: raise the relevant limit deliberately, or
  lower the requested result count (`retrieval.per_source_top_k`, capped by
  `retrieval.final_top_k`); for direct-URL contents requests, reduce the
  content options instead.
- `exa-py Exa client unavailable`: install the retrieval dependencies with
  `python -m pip install -e ".[retrieval]"`.
- Exa fails while arXiv succeeds: inspect the per-source status and warnings in
  the retrieval artifacts; optional-source failure does not discard successful
  results from other sources.
