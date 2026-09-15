# Model Configuration Guide

GraphHypoth selects large-language-model providers from configuration. Copy
`config/evidence-evaluation.yaml` to an untracked local file and pass it with
`--config`; code changes are not needed for normal provider changes.

GraphHypoth has two LLM execution modes.
1. **Coding-agent subagents (default).** Claude Code CLI subagents via `provider: claude-cli`; a Codex CLI variant via `provider: codex-cli` is optional. Both authenticate from your local CLI login and take no API key (`api_key_env: null`, `base_url: null`).
2. **OpenRouter, for direct LLM calls.** One credential, `OPENROUTER_API_KEY`, with `base_url: https://openrouter.ai/api/v1`. Each role's `model_id` is an OpenRouter catalog slug that you choose.

"Default" marks the recommended path, and within coding-agent mode it marks Claude Code over Codex. The shipped `config/evidence-evaluation.yaml` runs coding-agent mode: both base configurations use `provider: claude-cli`, so a saved `claude` login is all you need. Set each role's `model_id` to a model that login can use. To run API mode instead, follow the OpenRouter block in the config's header comment and export `OPENROUTER_API_KEY`.

The graph-state pipeline has seven workflow roles, described in the README's
[Agent Roles](../../README.md#agent-roles) section. Its model configuration
starts from two **base model configurations**, and these are the only two blocks
the schema requires:

```yaml
agents:
  builder:
    model:
      provider: openrouter
      model_id: <MODEL_ID>   # OpenRouter catalog slug — https://openrouter.ai/models
      api_key_env: OPENROUTER_API_KEY
      base_url: https://openrouter.ai/api/v1
      max_tokens: 4096
      timeout_seconds: 60
```

Use the same model shape for both base configurations, `builder` and
`skeptical_verifier`. Graph-state workflow roles use one of the two as their
default; roles with a named optional block can override it. Extraction uses
`builder` directly. Research synthesis, elaboration, reader translation, and
experiment design fall back to `builder`; evidence review, the critic panel,
translation verification, and experiment validation fall back to
`skeptical_verifier`.

## Fields

- `provider`: one of exactly three literals — `openrouter` for direct LLM calls
  through CAMEL, or the coding-agent subagent backends `claude-cli` (default)
  and `codex-cli` (optional), which the direct graph-state pipelines resolve
  without CAMEL.
- `model_id`: for `openrouter`, the catalog slug passed to CAMEL; for
  `claude-cli` and `codex-cli`, the model selection passed to `claude --print`
  and `codex exec` respectively.
- `api_key_env`: environment variable that holds the OpenRouter key, normally
  `OPENROUTER_API_KEY`. It must be `null` for `claude-cli` and `codex-cli`,
  which authenticate from a saved local CLI login and take no API key.
- `base_url`: the OpenRouter endpoint, `https://openrouter.ai/api/v1`. Keep it
  `null` for `claude-cli` and `codex-cli`.
- `max_tokens` and `timeout_seconds`: per-role runtime limits for `openrouter`
  roles. For `claude-cli` and `codex-cli`, omit `max_tokens`; `timeout_seconds`
  bounds the whole subprocess call.
- `reasoning_effort`: optional CLI-subagent effort, validated when the config
  loads. `claude-cli` accepts `low`, `medium`, `high`, `xhigh`, or `max`;
  `codex-cli` accepts `minimal`, `low`, `medium`, `high`, or `xhigh`. The
  `openrouter` path ignores it.

Never put API key values in YAML. Put only the environment variable name in
`api_key_env`.

## Examples

Claude Code CLI subagents, the default coding-agent path:

```yaml
provider: claude-cli
model_id: <MODEL_ID>   # a model your saved `claude` login can use (not an OpenRouter slug)
api_key_env: null
base_url: null
```

Codex CLI subagents, the optional variant:

```yaml
provider: codex-cli
model_id: <MODEL_ID>   # a model your saved `codex` login can use (not an OpenRouter slug)
api_key_env: null
base_url: null
```

OpenRouter, for direct LLM calls:

```yaml
provider: openrouter
model_id: <MODEL_ID>   # OpenRouter catalog slug — https://openrouter.ai/models
api_key_env: OPENROUTER_API_KEY
base_url: https://openrouter.ai/api/v1
```

The `model_id` should be the raw OpenRouter catalog slug, including its
`<vendor>/<model>` prefix. Keep `base_url` explicit in YAML even though CAMEL
may provide its own OpenRouter default. GraphHypoth does not recommend a model.
Choose a catalog slug for each role in the
[OpenRouter model catalog](https://openrouter.ai/models) — see the
[rankings](https://openrouter.ai/rankings) and the
[OpenRouter docs](https://openrouter.ai/docs).

## Provider Dependencies

Neither documented path needs a vendor SDK extra: OpenRouter is served through
the installed `camel-ai` package, and the CLI subagent providers shell out to a
locally installed CLI. For the exact editable install commands, see
[Local Setup And API Key Loading](local-setup-and-key-loading.md).

OpenRouter model support is delegated to the installed `camel-ai` version. If
the installed `camel-ai` cannot serve the configured slug, GraphHypoth fails
before the run with a clear provider/model error.

`provider: claude-cli` and `provider: codex-cli` are direct subagent backends for
`graph-hypoth-synthesist` and the `graph-hypoth-orchestrate` graph-state path;
they do not pass through CAMEL. Both require `api_key_env: null`,
`base_url: null`, and no `max_tokens`, plus saved `claude auth login` or
`codex login` authentication respectively. See
[Local CLI Subagent Backends And Live-Web Sources](cli-subagent-backends.md)
for the complete configuration, security limitations, and runtime boundaries.

This agent model block controls **completion calls only**, whose subagent
processes have web search disabled. Online retrieval through a CLI subagent is
independently configured and explicitly selected as
`retrieval.source_limits.claude_web` / `retrieval.source_limits.codex_web` with
`retrieval.sources` (or `optional_sources`). It may be used with subagent
completion roles, OpenRouter-backed roles, or a mixed panel; neither setting
implicitly enables the other.
