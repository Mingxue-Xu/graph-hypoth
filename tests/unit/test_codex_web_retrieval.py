from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.retrieval.codex_web as codex_web
from src.config import CodexWebSourceConfig, RetrievalConfig
from src.log_store import SQLiteLogStore
from src.retrieval.codex_web import (
    CodexCliWebRunner,
    CodexWebCallBudget,
    CodexWebPaperSource,
    CodexWebRunResult,
    CodexWebRunnerError,
    codex_web_output_schema,
)
from src.retrieval.ledger import EvidenceLedger
from src.retrieval.models import SearchPaperFilters, SourceResult
from src.retrieval.resilience import ResilientSource
from src.retrieval.service import RetrievalService
from src.retrieval.preflight import preflight_required_retrieval
from src.retrieval.similarity import identifiers_of


def _record(
    *,
    title: str = "Source title",
    url: str = "https://example.org/source",
    excerpt: str = "A contiguous source sentence supports the requested claim.",
    published_date: str | None = "2026-08-29",
) -> dict[str, Any]:
    return {
        "title": title,
        "url": url,
        "authors": ["A. Researcher"],
        "published_date": published_date,
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


class _Runner:
    def __init__(
        self,
        result: CodexWebRunResult | None = None,
        *,
        raises: BaseException | None = None,
    ) -> None:
        self.result = result or CodexWebRunResult(
            completion=_completion(),
            thread_id="thread-web",
            usage={"input_tokens": 20, "output_tokens": 10},
            latency_ms=123,
            web_searches=({"id": "search-1", "query": "claim"},),
        )
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def invoke(self, prompt: str, **kwargs: Any) -> CodexWebRunResult:
        self.calls.append({"prompt": prompt, **kwargs})
        if self.raises is not None:
            raise self.raises
        return self.result


def _source(
    runner: _Runner | None = None,
    **config_overrides: Any,
) -> tuple[CodexWebPaperSource, _Runner]:
    configured_runner = runner or _Runner()
    config = CodexWebSourceConfig(model_id="gpt-test", **config_overrides)
    return CodexWebPaperSource(config, runner=configured_runner), configured_runner


def _executable(tmp_path: Path) -> Path:
    executable = tmp_path / "codex"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


class _FakeProcess:
    def __init__(
        self,
        command: list[str],
        *,
        capture: dict[str, Any],
        completion: str | None,
        planned_stdout: str,
        planned_stderr: str,
        returncode: int,
        timeout: bool,
        **popen_kwargs: Any,
    ) -> None:
        self.command = command
        self.capture = capture
        self.completion = completion
        self.stdout = planned_stdout
        self.stderr = planned_stderr
        self.returncode = returncode
        self.should_timeout = timeout
        self.pid = 43210
        capture.update(command=command, popen_kwargs=popen_kwargs, process=self)

    def communicate(
        self,
        input: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        self.capture["input"] = input
        self.capture["timeout"] = timeout
        if self.should_timeout:
            raise subprocess.TimeoutExpired(self.command, timeout)
        output_path = Path(
            self.command[self.command.index("--output-last-message") + 1]
        )
        schema_path = Path(
            self.command[self.command.index("--output-schema") + 1]
        )
        self.capture["output_path"] = output_path
        self.capture["schema_path"] = schema_path
        self.capture["schema"] = json.loads(schema_path.read_text(encoding="utf-8"))
        if self.completion is not None:
            output_path.write_text(self.completion, encoding="utf-8")
        return self.stdout, self.stderr


def _install_process(
    monkeypatch: pytest.MonkeyPatch,
    *,
    completion: str | None = None,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    timeout: bool = False,
) -> dict[str, Any]:
    capture: dict[str, Any] = {}

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        return _FakeProcess(
            command,
            capture=capture,
            completion=completion,
            planned_stdout=stdout,
            planned_stderr=stderr,
            returncode=returncode,
            timeout=timeout,
            **kwargs,
        )

    monkeypatch.setattr(codex_web.subprocess, "Popen", fake_popen)
    return capture


def test_source_normalizes_search_grounded_records_without_claiming_quote_verification() -> None:
    source, _ = _source()

    response = source.search(
        "Does the claim hold?", limit=4, filters=SearchPaperFilters()
    )

    assert response.status.status == "success"
    assert len(response.results) == 1
    item = response.results[0]
    assert item.source == "codex_web"
    assert item.source_id == "https://example.org/source"
    assert item.text is None
    assert item.summary == _record()["excerpt"]
    assert item.external_ids == {}
    assert item.metadata["model_reported_doi"] == "10.1000/example"
    assert identifiers_of(item) == set()
    assert item.metadata["excerpt_verification"] == (
        "web_search_observed_not_host_quote_verified"
    )
    assert response.status.metadata["codex_thread_id"] == "thread-web"
    assert response.status.metadata["usage"] == {
        "input_tokens": 20,
        "output_tokens": 10,
    }
    assert response.status.metadata["web_search_count"] == 1
    assert any("not independently" in warning for warning in response.warnings)


def test_source_rejects_valid_looking_output_without_web_search_event() -> None:
    runner = _Runner(CodexWebRunResult(completion=_completion()))
    source, _ = _source(runner)

    response = source.search("q", limit=2, filters=SearchPaperFilters())

    assert response.results == []
    assert response.status.status == "failed"
    assert "without an observed web_search event" in response.errors[0]


def test_source_rejects_any_observed_non_web_tool() -> None:
    runner = _Runner(
        CodexWebRunResult(
            completion=_completion(),
            web_searches=({"id": "search"},),
            disallowed_item_types=("command_execution",),
        )
    )
    source, _ = _source(runner)

    response = source.search("q", limit=2, filters=SearchPaperFilters())

    assert response.status.status == "failed"
    assert "command_execution" in response.errors[0]


@pytest.mark.parametrize(
    "completion",
    ["", "not json", "[]", '{"records":"wrong","search_queries":[]}'],
)
def test_source_returns_structured_failure_for_malformed_output(
    completion: str,
) -> None:
    runner = _Runner(
        CodexWebRunResult(
            completion=completion,
            web_searches=({"id": "search"},),
        )
    )
    source, _ = _source(runner)

    response = source.search("q", limit=2, filters=SearchPaperFilters())

    assert response.results == []
    assert response.status.status == "failed"
    assert response.errors


@pytest.mark.parametrize(
    "payload",
    [
        {"records": [_record()], "search_queries": [], "extra": True},
        {"records": [_record()]},
        {
            "records": [{**_record(), "authors": "A. Researcher"}],
            "search_queries": [],
        },
        {
            "records": [{**_record(), "summary": 7}],
            "search_queries": [],
        },
        {"records": [_record()], "search_queries": "query"},
    ],
)
def test_source_enforces_strict_output_schema_types_and_keys(
    payload: dict[str, Any],
) -> None:
    runner = _Runner(
        CodexWebRunResult(
            completion=json.dumps(payload),
            web_searches=({"id": "search"},),
        )
    )
    source, _ = _source(runner)

    response = source.search("q", limit=2, filters=SearchPaperFilters())

    assert response.status.status == "failed"
    assert "strict records schema" in response.errors[0]


def test_source_salvages_valid_records_and_enforces_public_urls() -> None:
    records = [
        _record(url="http://127.0.0.1/private"),
        _record(url="file:///tmp/private"),
        _record(title="", url="https://example.org/no-title"),
        _record(url="https://example.org/valid"),
    ]
    runner = _Runner(
        CodexWebRunResult(
            completion=_completion(records),
            web_searches=({"id": "search"},),
        )
    )
    source, _ = _source(runner)

    response = source.search("q", limit=8, filters=SearchPaperFilters())

    assert [item.url for item in response.results] == [
        "https://example.org/valid"
    ]
    assert response.status.status == "partial_failure"
    assert any("dropped 3" in warning for warning in response.warnings)


def test_source_enforces_domain_url_text_and_recency_filters() -> None:
    records = [
        _record(
            url="https://papers.example.org/keep",
            excerpt="Required phrase appears in this fresh source.",
        ),
        _record(
            url="https://outside.test/drop",
            excerpt="Required phrase appears here too.",
        ),
        _record(
            url="https://papers.example.org/stale",
            excerpt="Required phrase but this is stale.",
            published_date="2000-01-01",
        ),
    ]
    runner = _Runner(
        CodexWebRunResult(
            completion=_completion(records),
            web_searches=({"id": "search"},),
        )
    )
    source, capturing_runner = _source(
        runner, allowed_domains=["example.org"]
    )
    filters = SearchPaperFilters(
        include_domains=["papers.example.org"],
        include_text=["required phrase"],
        max_age_hours=24 * 365,
    )

    response = source.search("q", limit=8, filters=filters)

    assert [item.url for item in response.results] == [
        "https://papers.example.org/keep"
    ]
    assert capturing_runner.calls[0]["allowed_domains"] == [
        "papers.example.org"
    ]


def test_source_allows_url_only_retrieval() -> None:
    source, runner = _source()
    target_url = "https://example.org/source"

    response = source.search(
        "",
        limit=2,
        filters=SearchPaperFilters(urls=[target_url]),
    )

    assert response.status.status == "success"
    assert [item.url for item in response.results] == [target_url]
    assert '"query":""' in runner.calls[0]["prompt"]
    assert runner.calls[0]["allowed_domains"] == ["example.org"]


def test_source_skips_without_invocation_when_domain_allowlists_do_not_overlap() -> None:
    source, runner = _source(allowed_domains=["example.org"])

    response = source.search(
        "q",
        limit=2,
        filters=SearchPaperFilters(include_domains=["other.test"]),
    )

    assert response.status.status == "skipped"
    assert runner.calls == []


@pytest.mark.parametrize(
    "include_domain",
    ["127.0.0.1", "169.254.169.254", "service.internal"],
)
def test_source_rejects_non_public_request_domain_before_invocation(
    include_domain: str,
) -> None:
    source, runner = _source()

    response = source.search(
        "q",
        limit=2,
        filters=SearchPaperFilters(include_domains=[include_domain]),
    )

    assert response.status.status == "skipped"
    assert runner.calls == []


@pytest.mark.parametrize(
    "filters",
    [
        SearchPaperFilters(urls=["https://outside.test/source"]),
        SearchPaperFilters(
            urls=["https://example.org/source"],
            exclude_domains=["example.org"],
        ),
    ],
)
def test_source_skips_impossible_target_domain_policy_before_invocation(
    filters: SearchPaperFilters,
) -> None:
    source, runner = _source(allowed_domains=["example.org"])

    response = source.search("q", limit=2, filters=filters)

    assert response.status.status == "skipped"
    assert runner.calls == []


@pytest.mark.parametrize(
    "target_url",
    [
        "http://127.0.0.1/private",
        "http://localhost/private",
        "http://169.254.169.254/latest/meta-data/",
        "https://user:password@example.org/private",
    ],
)
def test_source_rejects_non_public_target_urls_before_invocation(
    target_url: str,
) -> None:
    source, runner = _source()

    response = source.search(
        "q",
        limit=2,
        filters=SearchPaperFilters(urls=[target_url]),
    )

    assert response.status.status == "failed"
    assert "public HTTP(S)" in response.errors[0]
    assert runner.calls == []


def test_source_enforces_query_result_and_call_caps() -> None:
    runner = _Runner(
        CodexWebRunResult(
            completion=_completion([_record()]),
            web_searches=({"id": "search"},),
        )
    )
    source, _ = _source(
        runner,
        query_max_chars=5,
        max_records=1,
        max_calls_per_run=1,
    )

    first = source.search("long query", limit=10, filters=SearchPaperFilters())
    second = source.search("again", limit=1, filters=SearchPaperFilters())

    assert first.status.source_query == "long"
    assert len(first.results) == 1
    assert any("truncated" in warning for warning in first.warnings)
    assert any("capped" in warning for warning in first.warnings)
    assert second.status.status == "skipped"
    assert len(runner.calls) == 1


def test_source_rejects_output_exceeding_dynamic_record_cap() -> None:
    runner = _Runner(
        CodexWebRunResult(
            completion=_completion(
                [_record(), _record(url="https://example.org/2")]
            ),
            web_searches=({"id": "search"},),
        )
    )
    source, _ = _source(runner, max_records=1)

    response = source.search("q", limit=10, filters=SearchPaperFilters())

    assert response.status.status == "failed"
    assert "exceeded" in response.errors[0]


def test_source_target_url_matching_preserves_non_default_ports() -> None:
    runner = _Runner(
        CodexWebRunResult(
            completion=_completion([_record(url="https://example.org/source")]),
            web_searches=({"id": "search"},),
        )
    )
    source, _ = _source(runner)

    response = source.search(
        "q",
        limit=1,
        filters=SearchPaperFilters(
            urls=["https://example.org:8443/source"]
        ),
    )

    assert response.results == []
    assert response.status.status == "partial_failure"


def test_call_budget_is_keyed_by_run_and_shared_across_source_instances() -> None:
    budget = CodexWebCallBudget()
    first_runner = _Runner()
    second_runner = _Runner()
    config = CodexWebSourceConfig(model_id="gpt-test", max_calls_per_run=1)
    first = CodexWebPaperSource(
        config,
        runner=first_runner,
        call_budget=budget,
    )
    second = CodexWebPaperSource(
        config,
        runner=second_runner,
        call_budget=budget,
    )

    first_result = first.search_for_run(
        "one",
        limit=1,
        filters=SearchPaperFilters(),
        run_id="run-a",
    )
    capped_result = second.search_for_run(
        "two",
        limit=1,
        filters=SearchPaperFilters(),
        run_id="run-a",
    )
    other_run_result = second.search_for_run(
        "three",
        limit=1,
        filters=SearchPaperFilters(),
        run_id="run-b",
    )

    assert first_result.status.status == "success"
    assert capped_result.status.status == "skipped"
    assert other_run_result.status.status == "success"
    assert len(first_runner.calls) == 1
    assert len(second_runner.calls) == 1


def test_source_runner_exception_never_escapes() -> None:
    source, _ = _source(_Runner(raises=RuntimeError("subprocess failed")))

    response = source.search("q", limit=2, filters=SearchPaperFilters())

    assert response.status.status == "failed"
    assert "subprocess failed" in response.errors[0]


def test_source_preserves_observed_usage_when_runner_fails() -> None:
    failed_run = CodexWebRunResult(
        completion="",
        thread_id="failed-thread",
        usage={"input_tokens": 31, "output_tokens": 2},
        latency_ms=456,
        web_searches=({"id": "search"},),
    )
    source, _ = _source(
        _Runner(
            raises=CodexWebRunnerError(
                "exit code 2",
                result=failed_run,
            )
        )
    )

    response = source.search("q", limit=2, filters=SearchPaperFilters())

    assert response.status.status == "failed"
    assert response.status.metadata["codex_thread_id"] == "failed-thread"
    assert response.status.metadata["usage"] == {
        "input_tokens": 31,
        "output_tokens": 2,
    }
    assert response.status.metadata["web_search_count"] == 1


def test_source_reports_only_genuinely_unused_filters() -> None:
    source, _ = _source()
    filters = SearchPaperFilters(
        include_domains=["example.org"],
        highlight_query="claim",
        text=True,
        livecrawl_timeout=10,
        verify_quotes=False,
    )

    response = source.search("q", limit=2, filters=filters)

    assert response.status.unused_filters == [
        "text",
        "livecrawl_timeout",
        "verify_quotes",
    ]


def test_runner_builds_live_web_only_isolated_command_and_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "web-1",
                        "type": "web_search",
                        "query": "primary source",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 9, "output_tokens": 4},
                }
            ),
        ]
    )
    capture = _install_process(
        monkeypatch,
        completion=_completion(),
        stdout=stdout,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("SOME_TOKEN", "also-secret")
    runner = CodexCliWebRunner(
        CodexWebSourceConfig(
            model_id="gpt-test",
            reasoning_effort="xhigh",
            context_size="medium",
        ),
        executable=_executable(tmp_path),
    )
    schema = codex_web_output_schema(max_records=3)
    injected = "literal $(touch nope) `uname`"

    result = runner.invoke(
        injected,
        timeout=42,
        allowed_domains=["example.org"],
        output_schema=schema,
    )

    command = capture["command"]
    assert command[:3] == [str(tmp_path / "codex"), "exec", "--strict-config"]
    assert 'web_search="live"' in command
    web_config = next(value for value in command if value.startswith("tools.web_search="))
    assert 'context_size = "medium"' in web_config
    assert 'allowed_domains = ["example.org"]' in web_config
    for disabled in (
        "agents.enabled=false",
        "features.shell_tool=false",
        "features.unified_exec=false",
        "features.shell_snapshot=false",
        "features.skill_search=false",
        "features.skill_mcp_dependency_install=false",
        "features.apps=false",
        "features.plugins=false",
        "features.plugin_sharing=false",
        "features.remote_plugin=false",
        "features.browser_use=false",
        "features.browser_use_external=false",
        "features.browser_use_full_cdp_access=false",
        "features.in_app_browser=false",
        "features.computer_use=false",
        "features.workspace_dependencies=false",
        "features.image_generation=false",
        "features.view_image=false",
        "features.multi_agent=false",
        "features.goals=false",
        "features.hooks=false",
        "features.tool_suggest=false",
        "features.auth_elicitation=false",
        "features.tool_call_mcp_elicitation=false",
        "features.code_mode=false",
    ):
        assert disabled in command
    assert "features.code_mode_host=false" not in command
    assert 'shell_environment_policy.inherit="none"' in command
    assert 'approval_policy="never"' in command
    for flag in (
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--ephemeral",
        "--json",
        "--output-schema",
    ):
        assert flag in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert injected not in command
    assert capture["input"] == injected
    assert capture["schema"] == schema
    assert capture["popen_kwargs"]["shell"] is False
    assert capture["popen_kwargs"]["start_new_session"] is True
    child_env = capture["popen_kwargs"]["env"]
    assert "OPENROUTER_API_KEY" not in child_env
    assert "SOME_TOKEN" not in child_env
    assert result.thread_id == "thread-1"
    assert result.usage == {"input_tokens": 9, "output_tokens": 4}
    assert result.web_search_count == 1
    assert not capture["output_path"].exists()
    assert not capture["popen_kwargs"]["cwd"].exists()


def test_runner_reports_disallowed_jsonl_tool_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "web", "type": "web_search", "query": "q"},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "cmd", "type": "command_execution"},
                }
            ),
        ]
    )
    _install_process(monkeypatch, completion=_completion(), stdout=stdout)
    runner = CodexCliWebRunner(
        CodexWebSourceConfig(model_id="gpt-test"),
        executable=_executable(tmp_path),
    )

    result = runner.invoke(
        "prompt",
        timeout=10,
        allowed_domains=None,
        output_schema=codex_web_output_schema(max_records=1),
    )

    assert result.disallowed_item_types == ("command_execution",)


def test_runner_counts_only_completed_web_search_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = json.dumps(
        {
            "type": "item.started",
            "item": {"id": "web", "type": "web_search", "query": "q"},
        }
    )
    _install_process(monkeypatch, completion=_completion(), stdout=stdout)
    runner = CodexCliWebRunner(
        CodexWebSourceConfig(model_id="gpt-test"),
        executable=_executable(tmp_path),
    )

    result = runner.invoke(
        "prompt",
        timeout=10,
        allowed_domains=None,
        output_schema=codex_web_output_schema(max_records=1),
    )

    assert result.web_search_count == 0


@pytest.mark.parametrize("event_type", ["item.started", "item.failed"])
def test_runner_never_uses_incomplete_agent_message_as_completion(
    event_type: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "web",
                        "type": "web_search",
                        "query": "q",
                    },
                }
            ),
            json.dumps(
                {
                    "type": event_type,
                    "item": {
                        "id": "message",
                        "type": "agent_message",
                        "text": _completion(),
                    },
                }
            ),
        ]
    )
    _install_process(monkeypatch, completion=None, stdout=stdout)
    runner = CodexCliWebRunner(
        CodexWebSourceConfig(model_id="gpt-test"),
        executable=_executable(tmp_path),
    )

    with pytest.raises(CodexWebRunnerError, match="returned no completion"):
        runner.invoke(
            "prompt",
            timeout=10,
            allowed_domains=None,
            output_schema=codex_web_output_schema(max_records=1),
        )


def test_runner_does_not_count_failed_completed_web_search_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "id": "web",
                "type": "web_search",
                "query": "q",
                "status": "failed",
            },
        }
    )
    _install_process(monkeypatch, completion=_completion(), stdout=stdout)
    runner = CodexCliWebRunner(
        CodexWebSourceConfig(model_id="gpt-test"),
        executable=_executable(tmp_path),
    )

    result = runner.invoke(
        "prompt",
        timeout=10,
        allowed_domains=None,
        output_schema=codex_web_output_schema(max_records=1),
    )

    assert result.web_search_count == 0


def test_runner_redacts_proxy_secret_on_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = "http://user:password@example.test:8080"
    monkeypatch.setenv("HTTPS_PROXY", proxy)
    _install_process(
        monkeypatch,
        completion=None,
        stdout="\n".join(
            [
                json.dumps(
                    {"type": "thread.started", "thread_id": "failed-thread"}
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"id": "web", "type": "web_search"},
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 12, "output_tokens": 1},
                    }
                ),
            ]
        ),
        stderr=f"failed via {proxy}",
        returncode=2,
    )
    runner = CodexCliWebRunner(
        CodexWebSourceConfig(model_id="gpt-test"),
        executable=_executable(tmp_path),
    )

    with pytest.raises(RuntimeError) as exc_info:
        runner.invoke(
            "prompt",
            timeout=10,
            allowed_domains=None,
            output_schema=codex_web_output_schema(max_records=1),
        )

    assert proxy not in str(exc_info.value)
    assert "[redacted]" in str(exc_info.value)
    assert isinstance(exc_info.value, CodexWebRunnerError)
    assert exc_info.value.result is not None
    assert exc_info.value.result.thread_id == "failed-thread"
    assert exc_info.value.result.usage == {
        "input_tokens": 12,
        "output_tokens": 1,
    }


def test_runner_timeout_stops_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _install_process(monkeypatch, timeout=True)
    stopped: list[Any] = []
    monkeypatch.setattr(codex_web, "_stop_process_group", stopped.append)
    runner = CodexCliWebRunner(
        CodexWebSourceConfig(model_id="gpt-test"),
        executable=_executable(tmp_path),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        runner.invoke(
            "prompt",
            timeout=1,
            allowed_domains=None,
            output_schema=codex_web_output_schema(max_records=1),
        )

    assert stopped == [capture["process"]]


def test_runner_interrupt_stops_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptProcess:
        pid = 43211
        returncode = None

        def communicate(self, **kwargs: Any) -> tuple[str, str]:
            del kwargs
            raise KeyboardInterrupt

    process = InterruptProcess()
    monkeypatch.setattr(
        codex_web.subprocess,
        "Popen",
        lambda command, **kwargs: process,
    )
    stopped: list[Any] = []
    monkeypatch.setattr(codex_web, "_stop_process_group", stopped.append)
    runner = CodexCliWebRunner(
        CodexWebSourceConfig(model_id="gpt-test"),
        executable=_executable(tmp_path),
    )

    with pytest.raises(KeyboardInterrupt):
        runner.invoke(
            "prompt",
            timeout=10,
            allowed_domains=None,
            output_schema=codex_web_output_schema(max_records=1),
        )

    assert stopped == [process]


def test_config_requires_explicit_model_and_keeps_source_opt_in() -> None:
    default = RetrievalConfig()

    assert "codex_web" not in default.selected_source_names()
    assert default.source_limits.codex_web.model_id is None
    with pytest.raises(ValueError, match="source_limits.codex_web.model_id"):
        RetrievalConfig(sources=["codex_web"], optional_sources=[])

    selected = RetrievalConfig(
        sources=["codex_web"],
        optional_sources=[],
        source_limits={"codex_web": {"model_id": "gpt-test"}},
    )
    assert selected.trust_policy["codex_web"] == "subagent_web_research"
    assert selected.resilience.policy_for("codex_web").max_attempts == 1
    with pytest.raises(ValueError, match="exactly one resilience attempt"):
        RetrievalConfig(
            sources=["codex_web"],
            optional_sources=[],
            source_limits={"codex_web": {"model_id": "gpt-test"}},
            resilience={
                "per_source": {"codex_web": {"max_attempts": 2}}
            },
        )


def test_config_rejects_non_live_mode_invalid_domains_and_unknown_fields() -> None:
    with pytest.raises(ValueError, match="web_search_mode"):
        CodexWebSourceConfig(web_search_mode="cached")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bare public domain"):
        CodexWebSourceConfig(allowed_domains=["https://example.org/path"])
    with pytest.raises(ValueError, match="bare public domain"):
        CodexWebSourceConfig(allowed_domains=["127.0.0.1"])
    with pytest.raises(ValueError, match="bare public domain"):
        CodexWebSourceConfig(allowed_domains=["service.internal"])
    with pytest.raises(ValueError, match="require_web_search_event"):
        CodexWebSourceConfig(require_web_search_event=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Extra inputs"):
        CodexWebSourceConfig.model_validate({"model": "gpt-test"})


@pytest.mark.parametrize("codex_first", [True, False])
def test_unverified_codex_identity_never_promotes_its_quote_into_scholarly_record(
    codex_first: bool,
) -> None:
    source, _ = _source()
    codex_response = source.search(
        "claim", limit=1, filters=SearchPaperFilters()
    )
    [codex_result] = codex_response.results
    scholarly = SourceResult(
        source="arxiv",
        source_id="2401.00001",
        external_ids={"doi": "10.1000/example"},
        title=codex_result.title,
        url="https://arxiv.org/abs/2401.00001",
        text="Short independently retrieved abstract.",
    )
    ledger = EvidenceLedger(
        trust_policy={
            "arxiv": "authoritative_preprint",
            "codex_web": "subagent_web_research",
        },
        trust_tier_priority={
            "authoritative_preprint": 100,
            "subagent_web_research": 55,
        },
        source_order=["arxiv", "codex_web"],
    )
    ordered = [codex_result, scholarly] if codex_first else [scholarly, codex_result]

    ledger.add_many(
        results=ordered,
        query="claim",
        retrieved_by="test",
        tool_call_id="tool",
    )

    items = ledger.all_evidence()
    assert len(items) == 2
    arxiv_item = next(item for item in items if item.source == "arxiv")
    assert arxiv_item.quote == "Short independently retrieved abstract."


def test_service_registers_codex_only_when_selected_without_eager_cli_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_lookup(*args: Any, **kwargs: Any) -> str:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(codex_web, "resolve_codex_executable", unexpected_lookup)
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()

    default_service = RetrievalService(config=RetrievalConfig(), log_store=store)
    assert "codex_web" not in default_service.sources

    selected = RetrievalConfig(
        sources=["codex_web"],
        optional_sources=[],
        source_limits={"codex_web": {"model_id": "gpt-test"}},
    )
    service = RetrievalService(config=selected, log_store=store)

    assert isinstance(service.sources["codex_web"], ResilientSource)
    assert isinstance(
        service.sources["codex_web"]._inner,
        CodexWebPaperSource,
    )


def test_optional_codex_web_does_not_require_cli_during_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = RetrievalConfig(
        sources=["arxiv"],
        optional_sources=["codex_web"],
        source_limits={"codex_web": {"model_id": "gpt-test"}},
    )
    config = SimpleNamespace(retrieval=retrieval)
    monkeypatch.setattr(
        "src.retrieval.preflight.resolve_codex_executable",
        lambda: (_ for _ in ()).throw(AssertionError("must stay lazy")),
    )

    preflight_required_retrieval(config)


def test_services_share_run_budget_through_log_store_and_resilience(
    tmp_path: Path,
) -> None:
    config = RetrievalConfig(
        sources=["codex_web"],
        optional_sources=[],
        cache_replay=False,
        source_limits={
            "codex_web": {
                "model_id": "gpt-test",
                "max_calls_per_run": 1,
            }
        },
    )
    store_path = tmp_path / "events.sqlite"
    first_store = SQLiteLogStore(store_path)
    second_store = SQLiteLogStore(store_path)
    first_store.setup()
    second_store.setup()
    first_service = RetrievalService(config=config, log_store=first_store)
    second_service = RetrievalService(config=config, log_store=second_store)
    first_runner = _Runner()
    second_runner = _Runner()
    first_service.sources["codex_web"]._inner._runner = first_runner
    second_service.sources["codex_web"]._inner._runner = second_runner

    first = first_service.search_papers(
        run_id="shared-run",
        query="one",
        sources=None,
        limit=1,
        filters=None,
        called_by="retrieve_evidence",
        tool_call_id="tool-1",
    )
    capped = second_service.search_papers(
        run_id="shared-run",
        query="two",
        sources=None,
        limit=1,
        filters=None,
        called_by="experiment_design",
        tool_call_id="tool-2",
    )

    assert first.source_statuses[0].status == "success"
    assert first.evidence
    assert first.evidence[0]["trust_tier"] == "subagent_web_research"
    assert first.evidence[0]["metadata"]["quote_selection"] == {
        "source": "summary",
        "highlight_index": None,
        "query": "one",
        "candidate_excerpt": _record()["excerpt"],
        "verified_quote": _record()["excerpt"],
        "verification_status": "summary_fallback",
        "match_type": "abstractive_summary",
    }
    assert capped.source_statuses[0].status == "skipped"
    assert len(first_runner.calls) == 1
    assert second_runner.calls == []


def test_service_replays_successful_codex_web_result_without_another_call(
    tmp_path: Path,
) -> None:
    config = RetrievalConfig(
        sources=["codex_web"],
        optional_sources=[],
        cache_replay=True,
        source_limits={"codex_web": {"model_id": "gpt-test"}},
    )
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    service = RetrievalService(config=config, log_store=store)
    runner = _Runner()
    service.sources["codex_web"]._inner._runner = runner

    first = service.search_papers(
        run_id="cached-run",
        query="query",
        sources=None,
        limit=1,
        filters=None,
        called_by="builder",
        tool_call_id="tool-1",
    )
    replay = service.search_papers(
        run_id="cached-run",
        query="query",
        sources=None,
        limit=1,
        filters=None,
        called_by="builder",
        tool_call_id="tool-2",
    )

    assert first.evidence
    assert replay.tool_call_id == "tool-2"
    assert replay.evidence[0]["quote"] == first.evidence[0]["quote"]
    assert replay.evidence[0]["url"] == first.evidence[0]["url"]
    assert replay.evidence[0]["metadata"]["schema_version"] == "codex-web-v1"
    assert len(runner.calls) == 1


def test_codex_only_failure_does_not_silently_invoke_exa(
    tmp_path: Path,
) -> None:
    config = RetrievalConfig(
        sources=["codex_web"],
        optional_sources=[],
        cache_replay=False,
        source_limits={"codex_web": {"model_id": "gpt-test"}},
    )
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()
    service = RetrievalService(config=config, log_store=store)

    class FailedSource:
        name = "codex_web"

        def search(self, *args: Any, **kwargs: Any):
            del args, kwargs
            from src.retrieval.sources import SourceSearchResult
            from src.retrieval.models import SourceStatus

            return SourceSearchResult(
                results=[],
                status=SourceStatus(
                    source=self.name,
                    status="failed",
                    source_query="q",
                    errors=["failed"],
                ),
                errors=["failed"],
            )

    class UnexpectedExa:
        name = "exa"

        def search(self, *args: Any, **kwargs: Any):
            raise AssertionError((args, kwargs))

    service.sources["codex_web"] = FailedSource()
    service.sources["exa"] = UnexpectedExa()

    result = service.search_papers(
        run_id="run",
        query="q",
        sources=None,
        limit=2,
        filters=None,
        called_by="builder",
        tool_call_id="tool",
    )

    assert [status.source for status in result.source_statuses] == ["codex_web"]
    assert result.evidence == []
