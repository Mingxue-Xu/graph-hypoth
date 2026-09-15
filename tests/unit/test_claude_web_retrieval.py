"""Contract tests for the opt-in Claude CLI live-web retrieval source.

The Codex source is covered by ``test_codex_web_retrieval.py``; these tests
concentrate on what differs for Claude -- proof of search read from
``stream-json`` tool_use blocks, host-side schema enforcement in place of
``--output-schema``, and the explicitly named tool surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.config import ClaudeWebSourceConfig, RetrievalConfig
from src.retrieval.claude_web import (
    ClaudeCliWebRunner,
    ClaudeWebCallBudget,
    ClaudeWebPaperSource,
    ClaudeWebRunResult,
    ClaudeWebRunnerError,
    _stream_metadata,
)
from src.retrieval.models import SearchPaperFilters


def _record(
    *,
    title: str = "Source title",
    url: str = "https://example.org/source",
    excerpt: str = "A contiguous source sentence supports the requested claim.",
) -> dict[str, Any]:
    return {
        "title": title,
        "url": url,
        "authors": ["A. Researcher"],
        "published_date": "2026-08-29",
        "summary": "A neutral summary.",
        "excerpt": excerpt,
        "doi": "10.1000/example",
        "material_type": "research_paper",
    }


def _completion(records: list[dict[str, Any]] | None = None) -> str:
    return json.dumps(
        {
            "records": [_record()] if records is None else records,
            "search_queries": ["claim primary source"],
        }
    )


def _config(**overrides: Any) -> ClaudeWebSourceConfig:
    return ClaudeWebSourceConfig(model_id="claude-model-id", **overrides)


class _Runner:
    """Stand-in transport that returns a prepared run result."""

    def __init__(self, result: ClaudeWebRunResult | Exception) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        prompt: str,
        *,
        timeout: float,
        allowed_domains: list[str] | None,
        output_schema: dict[str, Any],
    ) -> ClaudeWebRunResult:
        self.calls.append(
            {
                "prompt": prompt,
                "timeout": timeout,
                "allowed_domains": allowed_domains,
                "output_schema": output_schema,
            }
        )
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _run(
    *,
    completion: str | None = None,
    searches: tuple[dict[str, Any], ...] = ({"id": "t1", "name": "WebSearch"},),
    disallowed: tuple[str, ...] = (),
) -> ClaudeWebRunResult:
    return ClaudeWebRunResult(
        completion=_completion() if completion is None else completion,
        session_id="sess-1",
        usage={"input_tokens": 10, "output_tokens": 20},
        cost_usd=0.02,
        resolved_model="claude-model-id",
        latency_ms=1234,
        web_searches=searches,
        disallowed_tools=disallowed,
    )


def _source(runner: _Runner, **overrides: Any) -> ClaudeWebPaperSource:
    return ClaudeWebPaperSource(
        _config(**overrides),
        runner=runner,
        call_budget=ClaudeWebCallBudget(),
    )


def _search(source: ClaudeWebPaperSource, **overrides: Any):
    kwargs: dict[str, Any] = {
        "limit": 5,
        "filters": SearchPaperFilters(),
    }
    kwargs.update(overrides)
    return source.search("primary source for the claim", **kwargs)


# --- proof of search -------------------------------------------------------


def test_records_are_admitted_when_a_websearch_tool_call_was_observed():
    source = _source(_Runner(_run()))

    result = _search(source)

    assert result.status.status == "success"
    assert [item.title for item in result.results] == ["Source title"]
    assert result.status.metadata["web_search_count"] == 1
    assert (
        result.results[0].metadata["excerpt_verification"]
        == "web_search_observed_not_host_quote_verified"
    )


def test_schema_valid_output_is_rejected_without_an_observed_search():
    source = _source(_Runner(_run(searches=())))

    result = _search(source)

    assert result.status.status == "failed"
    assert result.results == []
    assert any(
        "without an observed WebSearch tool call" in error
        for error in result.status.errors
    )


def test_any_disallowed_tool_use_rejects_the_whole_response():
    source = _source(_Runner(_run(disallowed=("Bash",))))

    result = _search(source)

    assert result.status.status == "failed"
    assert result.results == []
    assert any("disallowed tool use: Bash" in e for e in result.status.errors)


# --- host-side schema enforcement (no --output-schema) ---------------------


def test_off_schema_output_fails_closed():
    source = _source(_Runner(_run(completion='{"records": [{"title": "x"}]}')))

    result = _search(source)

    assert result.status.status == "failed"
    assert any("strict records schema" in e for e in result.status.errors)


def test_non_json_output_fails_closed():
    source = _source(_Runner(_run(completion="Here are the papers I found.")))

    result = _search(source)

    assert result.status.status == "failed"
    assert any("was not valid JSON" in e for e in result.status.errors)


def test_one_markdown_fence_is_tolerated_but_nothing_else_is():
    fenced = "```json\n" + _completion() + "\n```"
    result = _search(_source(_Runner(_run(completion=fenced))))
    assert result.status.status == "success"

    prefixed = "Sure! " + _completion()
    result = _search(_source(_Runner(_run(completion=prefixed))))
    assert result.status.status == "failed"


def test_records_beyond_the_requested_cap_are_rejected():
    payload = json.dumps(
        {
            "records": [
                _record(url=f"https://example.org/{index}") for index in range(3)
            ],
            "search_queries": [],
        }
    )
    source = _source(_Runner(_run(completion=payload)))

    result = _search(source, limit=2)

    assert result.status.status == "failed"
    assert any("records schema cap" in e for e in result.status.errors)


# --- transport failures ----------------------------------------------------


def test_runner_failure_is_reported_as_a_structured_source_failure():
    runner = _Runner(
        ClaudeWebRunnerError(
            "Claude CLI web retrieval timed out after 600s",
            result=ClaudeWebRunResult(completion="", latency_ms=600_000),
        )
    )

    result = _search(_source(runner))

    assert result.status.status == "failed"
    assert any("runner failed" in e for e in result.status.errors)
    assert result.status.metadata["latency_ms"] == 600_000


def test_per_run_call_budget_skips_rather_than_spending():
    runner = _Runner(_run())
    source = ClaudeWebPaperSource(
        _config(max_calls_per_run=1), runner=runner, call_budget=ClaudeWebCallBudget()
    )

    first = source.search_for_run(
        "q", limit=5, filters=SearchPaperFilters(), run_id="run-1"
    )
    second = source.search_for_run(
        "q", limit=5, filters=SearchPaperFilters(), run_id="run-1"
    )

    assert first.status.status == "success"
    assert second.status.status == "skipped"
    assert len(runner.calls) == 1


# --- stream parsing --------------------------------------------------------


_ALLOWED = frozenset({"WebSearch", "WebFetch"})


def test_stream_metadata_reads_tool_use_from_assistant_events():
    stdout = "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tu1",
                                "name": "WebSearch",
                                "input": {"query": "primary source"},
                            }
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "is_error": False,
                    "result": "final text",
                    "session_id": "s1",
                    "usage": {"input_tokens": 5},
                    "total_cost_usd": 0.01,
                    "modelUsage": {"claude-model-id": {"outputTokens": 40}},
                }
            ),
        ]
    )

    (
        session_id,
        usage,
        cost,
        model,
        completion,
        _error,
        is_error,
        _denials,
        searches,
        disallowed,
    ) = _stream_metadata(stdout, allowed_tools=_ALLOWED)

    assert session_id == "s1"
    assert completion == "final text"
    assert is_error is False
    assert usage == {"input_tokens": 5}
    assert cost == 0.01
    assert model == "claude-model-id"
    assert [item["query"] for item in searches] == ["primary source"]
    assert disallowed == ()


def test_stream_metadata_flags_tools_outside_the_allowlist():
    stdout = json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}
                ]
            },
        }
    )

    *_, searches, disallowed = _stream_metadata(stdout, allowed_tools=_ALLOWED)

    assert searches == ()
    assert disallowed == ("Bash",)


def test_stream_metadata_never_promotes_error_text_to_a_completion():
    stdout = json.dumps(
        {
            "type": "result",
            "is_error": True,
            "subtype": "success",
            "result": "Not logged in",
            "terminal_reason": "auth",
        }
    )

    (*_, completion, error, is_error, _d, _s, _dis) = _stream_metadata(
        stdout, allowed_tools=_ALLOWED
    )[3:]

    assert completion == ""
    assert is_error is True
    assert error is not None and "Not logged in" in error


def test_websearch_is_required_even_when_webfetch_ran():
    stdout = json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "WebFetch",
                        "input": {"url": "https://example.org/x"},
                    }
                ]
            },
        }
    )

    *_, searches, disallowed = _stream_metadata(stdout, allowed_tools=_ALLOWED)

    assert searches == ()  # fetching is not evidence that a search happened
    assert disallowed == ()  # but it is a granted tool, so not a violation


# --- command construction --------------------------------------------------


def test_command_names_the_tool_surface_and_streams_tool_events(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.claude_web.resolve_claude_executable", lambda _=None: "/bin/claude"
    )
    command = ClaudeCliWebRunner(_config())._command(executable="/bin/claude")

    assert command[:2] == ["/bin/claude", "--print"]
    assert "--output-format" in command
    assert command[command.index("--output-format") + 1] == "stream-json"
    assert command[command.index("--tools") + 1] == "WebFetch,WebSearch"
    assert "--safe-mode" in command and "--strict-mcp-config" in command
    assert "--no-session-persistence" in command
    assert command[command.index("--permission-mode") + 1] == "bypassPermissions"


def test_runner_requires_a_configured_model_id():
    with pytest.raises(RuntimeError, match="requires a configured model_id"):
        ClaudeCliWebRunner(ClaudeWebSourceConfig())._command(executable="/bin/claude")


# --- configuration ---------------------------------------------------------


def test_selected_source_requires_a_model_id():
    with pytest.raises(ValueError, match="source_limits.claude_web.model_id"):
        RetrievalConfig(optional_sources=["claude_web"])


def test_selected_source_requires_a_single_resilience_attempt():
    with pytest.raises(ValueError, match="exactly one resilience attempt"):
        RetrievalConfig(
            optional_sources=["claude_web"],
            source_limits={"claude_web": {"model_id": "claude-model-id"}},
            resilience={"per_source": {"claude_web": {"max_attempts": 2}}},
        )


def test_source_stays_absent_until_it_is_selected(tmp_path):
    from src.log_store import SQLiteLogStore
    from src.retrieval.service import RetrievalService

    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()

    default_service = RetrievalService(config=RetrievalConfig(), log_store=store)
    assert "claude_web" not in default_service.sources

    selected = RetrievalService(
        config=RetrievalConfig(
            optional_sources=["claude_web"],
            source_limits={"claude_web": {"model_id": "claude-model-id"}},
        ),
        log_store=store,
    )
    assert "claude_web" in selected.sources


def test_web_tools_must_stay_read_only_and_retain_websearch():
    with pytest.raises(ValueError, match="read-only web tools"):
        ClaudeWebSourceConfig(web_tools=["Bash", "WebSearch"])
    with pytest.raises(ValueError, match="must include WebSearch"):
        ClaudeWebSourceConfig(web_tools=["WebFetch"])


def test_trust_tier_matches_the_codex_live_web_source():
    config = RetrievalConfig()
    assert config.trust_policy["claude_web"] == "subagent_web_research"
    assert config.trust_policy["claude_web"] == config.trust_policy["codex_web"]


# --- shipped configuration -------------------------------------------------


def _shipped_retrieval_config() -> dict[str, Any]:
    import copy

    import yaml

    data = yaml.safe_load(Path("config/evidence-evaluation.yaml").read_text())
    return copy.deepcopy(data["retrieval"])


def test_shipped_config_can_actually_enable_claude_web():
    """The shipped YAML must support turning this source on.

    ``resilience.per_source`` in the YAML REPLACES the model default rather than
    merging into it, so a source omitted there silently inherits
    ``default.max_attempts`` (3) and is then rejected by the one-attempt
    invariant. Without the YAML entry, selecting claude_web fails config
    validation outright and the source cannot be used at all.
    """
    raw = _shipped_retrieval_config()
    raw["optional_sources"].append("claude_web")
    raw["source_limits"]["claude_web"]["model_id"] = "claude-model-id"

    config = RetrievalConfig.model_validate(raw)

    assert "claude_web" in config.selected_source_names()
    assert config.resilience.policy_for("claude_web").max_attempts == 1


def test_shipped_config_ranks_both_live_web_sources():
    """Both live-web sources share a trust tier, so source_order breaks their tie.

    An unlisted source falls back to a sentinel index (999 in the service,
    ``len(source_order)`` in the ledger), which would rank claude_web last within
    ``subagent_web_research`` purely by omission rather than by policy.
    """
    config = RetrievalConfig.model_validate(_shipped_retrieval_config())

    assert "claude_web" in config.ranking.source_order
    assert config.trust_policy["claude_web"] == config.trust_policy["codex_web"]


def test_shipped_config_still_loads_with_claude_web_unselected():
    config = RetrievalConfig.model_validate(_shipped_retrieval_config())

    assert "claude_web" not in config.selected_source_names()
    assert config.source_limits.claude_web.model_id is None
