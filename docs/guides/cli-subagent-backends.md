# Local CLI Subagent Backends And Live-Web Sources

GraphHypoth's default execution mode runs its model-backed roles through a
fresh, constrained local CLI process per call instead of a provider API. Two
completion backends exist, `claude-cli` (`claude --print`, the default) and
`codex-cli` (`codex exec`, optional), and two matching opt-in retrieval
sources, `claude_web` and `codex_web`, which are the only subprocesses allowed
to search the web. All four authenticate from the CLI's saved login; no
provider key is passed to the child process.

GraphHypoth stays the orchestrator: this is process-per-call model execution,
not the CLI's own multi-agent hierarchy.

## Live terminal progress

Both `graph-hypoth-synthesist` and `graph-hypoth-orchestrate` print flushed progress
lines to stderr by default. The example runner (`python scripts/run_example.py`)
uses the same reporter. It works with Claude CLI, Codex CLI, and API backends.

The full pipeline reports literature retrieval, claim mapping, evidence checking,
priorities, proposal/critique rounds, confirmation, experiment design, and export
and report rendering. Evidence checking precedes hypothesis generation, matching
the running system's order. Claimless and evidence-only runs report skipped work.

Updates include query/link/hypothesis counts where known, each active critic judge,
conditional third-judge calls, experiment revisions, and retrieval retry attempts.
During silent work a heartbeat runs every 30 seconds, including while a CLI
subprocess blocks. It identifies the active operation, its elapsed time, and the
provider during a model call. It reports waiting time, not a percentage or ETA.
Heartbeats pause while an interactive confirmation prompt awaits your selection.

```bash
python scripts/run_example.py \
  --config runtime_artifacts/config-claude-code.yaml \
  --run-dir runtime_artifacts/example-coding-agent \
  --progress-jsonl runtime_artifacts/example-coding-agent/progress.jsonl
```

Use `--quiet` to suppress progress on stderr; final results, errors, and interactive
prompts remain. `--progress-jsonl PATH` optionally appends flushed structured events
even with `--quiet`. Each JSON object includes `run_id`, UTC `timestamp`, `event`,
`stage`, `detail`, `status`, elapsed times, and optional current/total counters and
provider. Appending another run preserves existing events; distinguish runs by ID.
The progress log contains operational metadata; model prompts and responses remain
in the existing audit log. A completed model operation means the call returned;
later validation and commit outcomes are reported by the pipeline.

Direct Python library calls stay silent unless wrapped in
`src.progress.ProgressReporter(run_id, ...)`. The reporter's scope owns its
heartbeat thread and closes it on success, failure, or interruption. Progress
does not change subprocess output capture, timeout, or cancellation behavior.

## Where they apply

Every model role is answered by a direct `backend.run(messages)` seam, on both
`graph-hypoth-synthesist` and `graph-hypoth-orchestrate`, so both providers are
available everywhere. `provider: openrouter` roles still resolve through CAMEL's
model factory, which refuses a CLI provider with a clear error rather than
degrading.

| Role | CLI provider? |
|---|---|
| `builder`, `research_synthesist`, `experiment_designer`, `elaboration_writer`, `reader_translator` | yes (generative tier) |
| `skeptical_verifier`, `evidence_reviewer`, `critic_panel`, `experiment_validator`, `translation_verifier` | yes (critical tier) |

Unset generative roles inherit `builder` and unset critical roles inherit
`skeptical_verifier`, so pointing those two at a CLI provider moves every
direct-path seam onto it. Using one CLI model for every Critic Panel judge
removes the panel's distinct-model independence; mixed CLI and OpenRouter role
configurations are allowed.

Rules shared by both providers. In each role's `model:` block, `api_key_env`
and `base_url` must be `null`, `max_tokens` must be omitted (neither CLI exposes
a per-call token cap), `tools` must be `[]` (GraphHypoth function tools are not
exposed to the child; host-side retrieval still runs), `reasoning_effort`
defaults to `high`, and `timeout_seconds` defaults to 600 and bounds the whole
subprocess call. `reasoning_effort` is validated against each provider's own
vocabulary at config load: `claude-cli` accepts `low`, `medium`, `high`,
`xhigh`, `max`; `codex-cli` accepts `minimal`, `low`, `medium`, `high`,
`xhigh`. `model_id` is a model the saved login can use, not an OpenRouter slug.

Only a small environment allowlist reaches either child (`HOME`, `PATH`,
locale, proxy, TLS-bundle, and the CLI's own config-home variable,
`CLAUDE_CONFIG_DIR` or `CODEX_HOME`); every provider key and token is withheld,
so a saved login is mandatory.

## Claude Code CLI completion backend (default)

Install the Claude Code CLI, sign in for the account the work should bill to,
and confirm the login:

```bash
claude --version
```

The integration is verified against `claude-cli 2.1.247`. GraphHypoth checks
`PATH`, user-local installs, and common Homebrew locations automatically. If
those checks fail, set `GRAPH_HYPOTH_CLAUDE_BIN` to the executable's absolute
path; a symlink into an older version directory can break silently on update.
`claude_cli_preflight()` in
`src/claude_cli_backend.py` reports version and login state under exactly the
stripped environment a real call uses, without making a model call.

The shipped `config/evidence-evaluation.yaml` already selects this provider
for the three base blocks. The role shape is:

```yaml
agents:
  builder:
    temperature: 0.0  # required by the schema; the Claude CLI ignores it
    model:
      provider: claude-cli
      model_id: <MODEL_ID>       # a model your saved `claude` login can use
      api_key_env: null
      base_url: null
      reasoning_effort: medium   # low | medium | high | xhigh | max
      timeout_seconds: 600
```

Each call runs in a new empty temporary directory with every built-in tool
disabled (`--tools ""`), no skills, MCP servers, Chrome integration, user or
project settings, plugins, hooks, or agents, and no resumable session state.
Retrieval stays in the audited host pipeline.

The backend prompt, request framing, diagnostic redaction, and process-group
cancellation are imported from `src/codex_cli_backend.py` rather than
reimplemented, so the two transports send a byte-identical instruction and
cancel identically. That is the fairness precondition for comparing them, and
it means a provenance freeze has to hash both source files. Four boundaries
differ from the Codex backend:

- **No kernel sandbox.** The Claude CLI exposes no sandbox flag, so isolation
  rests on removing the whole tool surface and running in an empty directory.
- **No strict-config analogue.** Every constraint is a command-line flag and no
  configuration file is read, so the fail-closed mode is `unknown option` on a
  renamed flag. `tests/unit/test_claude_cli_strict_config.py` is the drift
  watchdog.
- **Failure signalling.** The CLI can report `is_error` with the failure text in
  `result` while `subtype` still reads `success`, so `is_error` is treated as
  authoritative and error text is never promoted to a completion.
- **Reasoning-effort vocabulary.** Claude has `max` and lacks `minimal`, and an
  unrecognised `--effort` only warns before silently falling back to the
  default, so the value is validated before a process is spawned.

The Claude backend additionally records the CLI's per-call `total_cost_usd` (a
CLI-reported figure, not a reconciled charge) and the model that actually served
the call, resolved from `modelUsage` by output tokens so an auxiliary fast-model
turn is never recorded as the generator. There is no cost budget to enforce it
against — `cost_tracking` was removed with the debate path — so treat the
reported cost as audit data, not a cap.

## Codex CLI completion backend (optional)

Install or update the [Codex CLI](https://learn.chatgpt.com/docs/codex/cli),
save a local login, and confirm the CLI can see it:

```bash
codex --version
codex login status
```

The integration is verified against a recent Codex CLI that supports
`--strict-config`, and it uses strict configuration deliberately, so an older
CLI that lacks a required safety flag fails instead of silently weakening it.
GraphHypoth checks `PATH`, user-local and standalone-package installs, Homebrew,
and ChatGPT's macOS application bundle automatically. If those checks fail, set
`GRAPH_HYPOTH_CODEX_BIN` to the executable's absolute path.

Copy `config/evidence-evaluation.yaml` to an untracked local file and replace
the three base `agents` entries:

```yaml
agents:
  builder:
    temperature: 0.0  # required by the GraphHypoth schema; Codex CLI ignores it
    model: &codex_model
      provider: codex-cli
      model_id: <MODEL_ID>   # a model your saved `codex` login can use
      api_key_env: null
      base_url: null
      reasoning_effort: high
      timeout_seconds: 600

  skeptical_verifier:
    temperature: 0.0
    model: *codex_model
```

Then run the pipeline with the local configuration:

```bash
graph-hypoth-synthesist \
  --profile examples/research_profile.yaml \
  --config config/codex-local.yaml \
  --events-db runtime_artifacts/codex/events.sqlite \
  --export-dir runtime_artifacts/codex \
  --trace-dir runtime_artifacts/codex/trace
```

The synthesist is a multi-call workflow: retrieval planning, extraction,
evidence review, synthesis, panel judging, and experiment design each start
Codex processes. Trace reports also generate plain-language prose by default,
using one more completion per surfaced hypothesis. Pass `--no-elaborate` to skip it.

Process isolation. Each completion runs in a new empty temporary directory with
a read-only sandbox; shell, web-search, local-image, and Codex multi-agent tools
disabled; approvals set to `never`; ephemeral session state; no inherited
model-shell environment; and user config and exec rules disabled. These
controls are defense in depth, not a formally verified confidentiality
boundary: `HOME` and `CODEX_HOME` stay readable for saved authentication, and
future CLI capabilities can change. Do not process hostile prompt-injection
content on a host holding sensitive readable files.

Known limits: Codex JSONL usage, latency, and thread IDs are recorded in
per-call audit events, but token usage is not aggregated into the run-level
cost report and no exact dollar reconciliation exists; the subprocess contract
has deterministic test coverage, but no live Codex call is part of the default
test suite.

## Optional Claude live-web retrieval (`claude_web`)

`claude_web` is a separate, opt-in retrieval source; using the `claude-cli`
completion backend does not enable it, and that backend's tool surface stays
empty. The source launches a fresh Claude process per source query with only
the read-only web tools granted and rejects the response unless the stream
contains an observed `WebSearch` tool call; any tool outside the configured
allowlist rejects the whole response.

It shares the Codex source's admission policy (records schema, URL and domain
rules, recency and excerpt caps, the `subagent_web_research` trust tier, and the
`summary_fallback` treatment described below) because those helpers are
imported from `src/retrieval/codex_web.py` rather than copied. Two boundaries
are weaker than the Codex source: there is no `--output-schema`, so the schema
is embedded in the prompt and enforced host-side by the same pydantic models (a
wasted call at worst, never an off-schema record); and there is no kernel
sandbox, so the boundary is `--tools`, validated to contain only `WebSearch` and
`WebFetch`, with `--permission-mode` set so a headless turn cannot block on an
approval prompt.

`claude_web` has to appear in four places for a config to load:
`source_limits`, `trust_policy`, `ranking.source_order`, and
`resilience.per_source`. The shipped `config/evidence-evaluation.yaml` already
carries all four with `model_id: null`, so the source stays off. To enable it,
set a model available to the saved Claude login and select the source:

```yaml
retrieval:
  optional_sources:
    - crossref
    - openalex
    - claude_web          # select it
  source_limits:
    claude_web:
      model_id: <MODEL_ID>      # required; null keeps the source off
      reasoning_effort: high    # low | medium | high | xhigh | max
      timeout_seconds: 600
      max_records: 8
      max_calls_per_run: 8      # durable per-run subprocess cap, not searches per turn
      max_excerpt_chars: 4000
      web_tools: [WebFetch, WebSearch]   # validated: read-only web tools only
      permission_mode: bypassPermissions
      allowed_domains: null     # or, for example, ["arxiv.org", "nature.com"]
      require_web_search_event: true
```

Each of the other three YAML blocks replaces its code default wholesale rather
than merging into it, so omitting a block is safe while defining it without
`claude_web` is not:

| Block | Omit the whole block | Define it, omit `claude_web` |
|---|---|---|
| `resilience.per_source` | fine; default gives one attempt | config rejected: `selected claude_web source requires exactly one resilience attempt` |
| `trust_policy` | fine; default gives the tier | config rejected: `source 'claude_web' has no trust_policy entry` |
| `ranking.source_order` | sorts last in its tier, silently | sorts last in its tier, silently |

`ranking.source_order` is the one to watch: it never errors, and the code
default (`arxiv`, `exa`, `codex_web`) does not list `claude_web` either. The two
live-web sources share a trust tier, so this list is what breaks their tie:

```yaml
retrieval:
  trust_policy:
    claude_web: subagent_web_research
  ranking:
    source_order:
      - arxiv
      - europepmc
      - openalex
      - crossref
      - exa
      - codex_web
      - claude_web
  resilience:
    per_source:
      claude_web:
        enabled: true
        max_attempts: 1
```

Its per-run subprocess budget is separate from the Codex source's, so the two
cannot spend each other's `max_calls_per_run`. A required `claude_web` source
is not preflighted; it fails at first use.

## Optional Codex live-web retrieval (`codex_web`)

`codex_web` is likewise a separate, opt-in source; using `provider: codex-cli`
for completions does not enable it. It launches a fresh Codex process per source
query with native live web search enabled, requires strict schema-conforming
JSON, and rejects even plausible citations unless the JSONL stream contains a
successfully completed `web_search` event. All known non-web tools are disabled
under strict config, and any unexpected tool item rejects the whole response.
This is a version-pinned denylist, not a guaranteed one-tool allowlist.

The subprocess disables Code Mode with `features.code_mode=false` and
deliberately leaves `features.code_mode_host` alone: forcing it off aborts the
schema-constrained turn before web search in the verified CLI.

Add the source to your untracked config and choose a model available to the
saved login:

```yaml
retrieval:
  optional_sources:
    - exa
    - crossref
    - openalex
    - codex_web
  ranking:
    source_order:
      - arxiv
      - europepmc
      - openalex
      - crossref
      - exa
      - codex_web
    trust_tier_priority:
      authoritative_preprint: 100
      peer_reviewed_oa: 90
      indexed_metadata: 80
      web_research_paper: 60
      subagent_web_research: 55
      deterministic_test: 100
  source_limits:
    codex_web:
      model_id: <MODEL_ID>   # a model your saved `codex` login can use
      reasoning_effort: high
      timeout_seconds: 900   # bounded whole research turn
      query_max_chars: 2000
      max_records: 8
      max_calls_per_run: 8
      max_excerpt_chars: 4000
      web_search_mode: live
      context_size: high
      allowed_domains: null  # or, for example, ["arxiv.org", "nature.com"]
      require_web_search_event: true
  resilience:
    per_source:
      codex_web:
        enabled: true
        max_attempts: 1
        backoff_base_seconds: 0.5
        backoff_max_seconds: 8.0
        circuit_failure_threshold: 5
        circuit_reset_seconds: 60.0
```

Selection is the opt-in: `codex_web` is absent from the shipped default source
set, and selecting it without `source_limits.codex_web.model_id` fails config
validation. The source is single-attempt and bounded by both `max_records` and
`max_calls_per_run`; the latter caps Codex subprocess attempts per run, not the
number of native searches inside one turn. A Codex failure never falls back to
Exa. Cache replay avoids another subprocess for the same run, input, and
retrieval config; the synthesist generates a fresh run ID by default, and
reusing `--run-id` opts into replay. The shipped config already carries the
`trust_policy.codex_web: subagent_web_research` mapping. Required-source
preflight confirms only that the executable exists; saved-login validity and
CLI capability compatibility are checked when the first Codex turn starts.

## What both live-web sources admit

Every accepted record has a public HTTP(S) URL and an observed web-search turn,
but GraphHypoth does not fetch each page to prove the excerpt is present
verbatim. Excerpts are therefore stored as `summary_fallback`, tagged
`web_search_observed_not_host_quote_verified`, and ranked below Exa's curated
web-paper tier. Their model-reported DOI and title are excluded from
cross-source ledger merging and from citation and coherence identity, so an
unverified excerpt cannot inherit a scholarly source's trust tier or create a
false citation edge. Accepted records then pass through the normal cache,
source status, ranking, de-duplication, evidence ledger, and retrieval-artifact
paths. Corroborate important citations with arXiv, Crossref, OpenAlex, Europe
PMC, or Exa.

## References

- [`codex exec` non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
- [Codex developer command reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli)
- [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
- The backend resolver lives in [`src/config.py`](../../src/config.py), with
  process transports in [`src/claude_cli_backend.py`](../../src/claude_cli_backend.py)
  and [`src/codex_cli_backend.py`](../../src/codex_cli_backend.py). The public
  setup path is summarized in the [README](../../README.md) and
  [model configuration guide](model-configuration.md).
