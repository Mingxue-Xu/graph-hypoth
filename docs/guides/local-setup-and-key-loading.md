# Local Setup And API Key Loading

This guide covers local installation, credential loading, and the available
GraphHypoth command-line workflows.

## Install From The Repo Root

Requires Python 3.11 or newer (`requires-python = ">=3.11"` in `pyproject.toml`).
Create the virtual environment with a 3.11+ interpreter.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[all]"
```

This exposes two console commands:

```bash
graph-hypoth-orchestrate --help
graph-hypoth-synthesist --help
```

The `.[all]` install includes the retrieval, retrieval-coherence, and
development dependencies. If you prefer a smaller install, `.[dev,retrieval]`
covers both modes: OpenRouter is served through `camel-ai`, and the CLI subagent
providers shell out to a locally installed CLI.

## Model Configuration

Choose or edit your model config before running a command. The default runtime
config is `config/evidence-evaluation.yaml`, and it ships with `<MODEL_ID>`
placeholders rather than runnable models. Copy it to an untracked local file,
replace both placeholders, and pass the copy with `--config` on every run.

GraphHypoth has two LLM execution modes.
1. **Coding-agent subagents (default).** Claude Code CLI subagents via `provider: claude-cli`; a Codex CLI variant via `provider: codex-cli` is optional. Both authenticate from your local CLI login and take no API key (`api_key_env: null`, `base_url: null`).
2. **OpenRouter, for direct LLM calls.** One credential, `OPENROUTER_API_KEY`, with `base_url: https://openrouter.ai/api/v1`. Each role's `model_id` is an OpenRouter catalog slug that you choose.

"Default" marks the recommended path, and within coding-agent mode it marks Claude Code over Codex. The shipped `config/evidence-evaluation.yaml` runs coding-agent mode: both base configurations use `provider: claude-cli`, so a saved `claude` login is all you need. Set each role's `model_id` to a model that login can use. To run API mode instead, follow the OpenRouter block in the config's header comment and export `OPENROUTER_API_KEY`.

The pipeline's seven workflow roles are described in the README's
[Agent Roles](../../README.md#agent-roles) section. The schema requires exactly
two configuration blocks, both of them model defaults. These are configuration
entries, not two additional workflow agents:

- `builder`: generative base model configuration.
- `skeptical_verifier`: critical base model configuration.

The Extraction Agent is configured through `builder`. Workflow roles that expose
an optional role-specific block under `agents` can override their base model
configuration; otherwise they use that default.

For the `agents.*.model` fields and the OpenRouter and CLI-subagent examples,
see [Model Configuration Guide](model-configuration.md). GraphHypoth does not
recommend a model. Choose a catalog slug for each role in the
[OpenRouter model catalog](https://openrouter.ai/models) — see the
[rankings](https://openrouter.ai/rankings) and the
[OpenRouter docs](https://openrouter.ai/docs).

## Load API Keys

The safest setup is to export API keys directly in the shell session where you
will use this repo, one variable at a time. This keeps raw secrets out of
repo-local files and makes the key lifetime limited to that terminal session.
The shipped configuration needs **no** model credential: it runs `claude-cli`
subagents, which authenticate from your saved `claude` login. You only need to
export a model key if you switch the roles to OpenRouter:

```bash
export OPENROUTER_API_KEY='...'   # only for provider: openrouter
export EXA_API_KEY='...'          # only when `exa` is a configured source
```

Only export keys for the providers your configuration actually uses. Direct LLM
calls need exactly one credential, `OPENROUTER_API_KEY`. The `claude-cli` and
`codex-cli` subagent providers take no API key at all — they use a saved local
CLI login. `EXA_API_KEY` is a retrieval-source credential and is only needed
when `exa` is configured.

For repeat use, keep a `NAME=value` file outside the repository, restrict its
permissions, and pass it to `graph-hypoth-orchestrate`:

```bash
chmod 600 <path-outside-repo>/graph-hypoth.env
graph-hypoth-orchestrate --api-keys-file <path-outside-repo>/graph-hypoth.env --claim "$CLAIM"
```

The file uses the same variable names, without shell commands:

```dotenv
OPENROUTER_API_KEY=...
EXA_API_KEY=...
```

`graph-hypoth-synthesist` reads the process environment rather than
`--api-keys-file`, so export the needed variables before invoking it. Do not
echo keys, enable shell tracing, paste raw values into YAML, or save credentials
in a tracked repository file. Config files should contain environment variable
names such as `OPENROUTER_API_KEY`, never secret values.

### Every Credential This Repo Reads

No variable below is hard-coded as a credential lookup except where noted: model
and retrieval keys are read through the environment variable *named in YAML*, so
renaming the variable in config renames what you export.

| Variable | Selected by | Needed when | If missing |
|---|---|---|---|
| `OPENROUTER_API_KEY` | `agents.*.model.api_key_env` | only if you switch roles to `provider: openrouter`; the shipped config uses `claude-cli` and needs no model key | run stops before graph construction |
| *(your name)* | `agents.*.model.api_key_env` | any custom key variable; the preflight reads whatever it names | run stops before graph construction |
| `EXA_API_KEY` | `retrieval.source_limits.exa.require_api_key_env` | `exa` listed in `retrieval.sources` | run stops at preflight |
| `OPENALEX_API_KEY` | `retrieval.source_limits.openalex.require_api_key_env` | never required; raises rate limits only | source degrades, run continues |
| `APIFY_API_TOKEN` | `retrieval.source_limits.apify.require_api_key_env` | never — `apify` is not selectable in this release | n/a |
| `GRAPH_HYPOTH_CLAUDE_BIN` | environment only | automatic discovery cannot find the Claude Code CLI | `claude-cli` / `claude_web` paths error out |
| `GRAPH_HYPOTH_CODEX_BIN` | environment only | automatic discovery cannot find the Codex CLI | `codex-cli` / `codex_web` paths error out |
| `GRAPH_HYPOTH_WIKI_CONTACT` | environment only | optional; not a secret | falls back to a default contact URL |
| `GRAPH_HYPOTH_RUNTIME_LOG_DIR` | environment only | opt-in runtime tracing; not a secret | no trace file is written |
| `GRAPH_HYPOTH_RUNTIME_RUN_LABEL` | environment only | optional trace-directory label | defaults to `run` |

Fail-fast coverage is deliberately uneven, and the difference matters when you
are debugging an empty result set:

- **Model keys** are checked by a preflight that names the missing role,
  provider, model, and variable. Both commands run it.
  `graph-hypoth-orchestrate` checks the two base configurations (`builder` and
  the `evidence_reviewer` role that falls back to `skeptical_verifier`);
  `graph-hypoth-synthesist` checks every role the run will use, adding
  `research_synthesist`, `critic_panel`, `experiment_designer`,
  `experiment_validator`, `skeptical_verifier` when `k_judges >= 3`, and the
  elaboration and reader-translation roles for trace reports unless `--no-elaborate` is set. The preflight
  reads each role's `api_key_env`, so a key-free `claude-cli` or `codex-cli`
  role passes it unconditionally and a `model_id` that login cannot use is
  reported only at that role's first call.
- **Required retrieval sources** are checked by a separate preflight covering
  `exa` and `codex_web`, and only for sources listed in `retrieval.sources`. A
  required `claude_web` source is not preflighted and fails at first use.
- **Everything else degrades quietly.** A missing `OPENALEX_API_KEY` or
  `APIFY_API_TOKEN` produces a warning on the retrieval result rather than an
  error, so the run completes with fewer sources than you configured. Grep the
  run's retrieval warnings before concluding a source returned nothing.

### OpenAlex

OpenAlex works without a key through its polite pool, so `require_api_key`
defaults to `false` and the key only raises rate limits. Set
`require_api_key: true` to skip the source when the variable is absent, which
degrades to Crossref and arXiv instead of issuing throttled requests:

```bash
export OPENALEX_API_KEY='...'
```

Setting `openalex.mailto` in YAML is the other half of polite-pool etiquette,
and it is a config value rather than an environment variable. Note that the
premium key also serves OpenAlex semantic search, but this repo's adapter issues
keyword queries only — see [Paper Retrieval](paper_retrieval.md) for that gap.

### Apify

Apify is not selectable from configuration in this release.
`RetrievalConfig.validate_sources` (`src/config.py`) rejects `apify`, so adding
it to `sources` or `optional_sources` fails config load — exporting
`APIFY_API_TOKEN` or setting `actor_id` does not enable it. The `apify` block in
`config/evidence-evaluation.yaml` and the `APIFY_API_TOKEN` variable are
reserved for a future release.

### CLI subagent saved-login authentication

The `claude-cli` (default) and `codex-cli` (optional) completion backends, and
the opt-in `claude_web` and `codex_web` retrieval sources, read no provider key
from GraphHypoth config. Install the CLI you intend to use, authenticate it
locally, and verify the saved session:

```bash
claude --version
codex --version
codex login status
```

GraphHypoth checks `PATH` and common user-local, Homebrew, and application install
locations automatically. If those checks fail, set `GRAPH_HYPOTH_CLAUDE_BIN` or
`GRAPH_HYPOTH_CODEX_BIN` to an absolute path. GraphHypoth passes only a small
environment allowlist needed for the saved login and transport — `HOME`, `PATH`,
locale, proxy, TLS-bundle, and the CLI's own config-home variable
(`CLAUDE_CONFIG_DIR`, `CODEX_HOME`). Anything named as a provider key or token
is withheld, so exporting any provider `*_API_KEY` or `*_TOKEN` will not
authenticate the child process and a saved login is mandatory. See [Local CLI
Subagent Backends And Live-Web Sources](cli-subagent-backends.md) before using
live-web retrieval, because their token usage is audited but exact dollar cost
cannot be enforced by GraphHypoth.

### Runtime Trace Output

Runtime tracing of LLM and tool calls is off unless a destination is set.
Exporting `GRAPH_HYPOTH_RUNTIME_LOG_DIR` turns it on for
`graph-hypoth-orchestrate` and `graph-hypoth-synthesist`, which write one
timestamped subdirectory per process holding `trace.jsonl` and `manifest.json`.
`GRAPH_HYPOTH_RUNTIME_RUN_LABEL` names that subdirectory and defaults to `run`:

```bash
export GRAPH_HYPOTH_RUNTIME_LOG_DIR="$PWD/runtime_logs"
export GRAPH_HYPOTH_RUNTIME_RUN_LABEL='exa-smoke'
```

Traces are redacted on write, but not uniformly. Only `OPENROUTER_API_KEY` and
`EXA_API_KEY` are scrubbed by exact value; every other credential relies on
shape heuristics — an `sk-` prefix, or a value labelled `api_key`,
`authorization`, or `bearer`. A key whose format matches neither, appearing as a
bare unlabelled value, can survive redaction. Treat a trace directory as
sensitive, and keep it outside the repository or under the ignored
`runtime_logs/` path.

### Live Test Credentials

Opt-in live tests are where extra variables appear. They are selected by pytest
markers (`live`, `live_exa`, `live_arxiv`, `live_openalex`, `live_apify`,
`live_llm_provider`, `live_openrouter`, `live_openai_compatible`) and skip
themselves when their credentials are absent, so the default `python -m pytest`
run needs none of them. `live_openai_compatible` exercises a self-hosted or
proxy OpenAI-wire endpoint; it is not a supported configuration path. Live
Apify tests additionally read `APIFY_ACTOR_ID`, and the provider-swap suite
reads `GRAPH_HYPOTH_LIVE_LLM_PROVIDERS` along with its own per-endpoint model
and base-URL variables to choose what to exercise.

Third-party benchmark corpora kept locally under an ignored `datasets/`
directory carry their own unrelated credentials, such as
`OPENAI_BASE_URL`, `IDEA_GRAPH_API_KEY`, and `IDEA_GRAPH_MODEL`. Those are read
by the vendored code only, are configured per that project's own README, and are
not part of GraphHypoth's configuration surface.

## Run The Hypothesis Pipeline

The full profile-driven hypothesis workflow accepts the canonical `field` and
`concepts` profile keys:

```bash
graph-hypoth-synthesist \
  --profile examples/research_profile.yaml \
  --config runtime_artifacts/config-live.yaml \
  --export-dir runtime_artifacts/example \
  --trace-dir runtime_artifacts/example/trace
```

Pass `--config`. Omitting it loads the shipped
`config/evidence-evaluation.yaml`, whose `model_id` values are the placeholder
`<MODEL_ID>`; the run then fails at the first model call, after retrieval has
already run. Copy the example profile to an untracked location before adding
private research context.

## Retrieval

The shipped `config/evidence-evaluation.yaml` enables source-only retrieval with
bounded Crossref metadata required and Exa and OpenAlex selected as optional
sources. arXiv, Europe PMC, and the two live-web sources are opt-in.
Live Exa calls require `EXA_API_KEY` and
are guarded by per-call and per-run cost limits in the YAML config. arXiv,
Crossref, and Europe PMC need no credential at all; OpenAlex and Apify are
covered in [Every Credential This Repo Reads](#every-credential-this-repo-reads).

For Exa setup details, see the
[Paper Retrieval Exa source settings](paper_retrieval.md#61-exa-source-settings).
For saved-login CLI retrieval, enable the separate `claude_web` or `codex_web`
source as documented in [Local CLI Subagent Backends And Live-Web
Sources](cli-subagent-backends.md); neither is part of the default source set.

## Development Checks

The deterministic suite passes fakes for every model-backed seam and needs no
provider credentials. Run it with:

```bash
python -m pytest
```

Live provider and retrieval tests are marked (`live`, `live_exa`,
`live_openalex`, and so on) and skipped unless explicitly selected and their
credentials are available. The [Live Test Credentials](#live-test-credentials)
section above lists the environment variables those opt-in checks read.

Run a deterministic no-credential smoke check:

```bash
python -m pytest -m smoke
```

Run the current graph-state CLI end to end without credentials or network:

```bash
python -m pytest \
  tests/e2e/test_cli_e2e.py::test_cli_graph_state_path_is_hermetic_end_to_end
```

That test launches the real CLI in a subprocess with a temporary fake Codex
executable and fake retrieval source, then verifies SQLite model-call events and
the exported graph artifacts. The production CLI also exposes `--adapter fake`,
which forces `retrieval.fake_mode` so retrieval is deterministic and needs no
retrieval credentials; the hermetic test above additionally stubs the model
backend, which `--adapter fake` does not.
