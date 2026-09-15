from __future__ import annotations

from types import SimpleNamespace

from src.retrieval.models import SearchPaperFilters, SourceStatus
from src.retrieval.resilience import ResilientSource
from src.retrieval.sources import SourceSearchResult


def _policy(**overrides):
    base = {
        "enabled": True,
        "max_attempts": 3,
        "backoff_base_seconds": 0.5,
        "backoff_max_seconds": 8.0,
        "circuit_failure_threshold": 2,
        "circuit_reset_seconds": 60.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _ok(name: str = "openalex") -> SourceSearchResult:
    return SourceSearchResult(
        results=[],
        status=SourceStatus(
            source=name, status="success", source_query="q", result_count=0
        ),
    )


def _failed(name: str = "openalex") -> SourceSearchResult:
    return SourceSearchResult(
        results=[],
        status=SourceStatus(
            source=name, status="failed", source_query="q", result_count=0,
            errors=["boom"],
        ),
        errors=["boom"],
    )


class _ScriptedSource:
    """Yields a scripted sequence of outcomes; an Exception item is raised."""

    name = "openalex"

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def search(self, query, *, limit, filters):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _clock():
    state = {"t": 0.0}

    def monotonic() -> float:
        return state["t"]

    def sleep(seconds: float) -> None:
        state["t"] += seconds

    return state, monotonic, sleep


def test_retries_then_succeeds_without_raising() -> None:
    state, monotonic, sleep = _clock()
    inner = _ScriptedSource([_failed(), _failed(), _ok()])
    wrapped = ResilientSource(
        inner, policy=_policy(max_attempts=3, circuit_failure_threshold=5),
        sleep=sleep, monotonic=monotonic,
    )

    result = wrapped.search("q", limit=3, filters=SearchPaperFilters())

    assert result.status.status == "success"
    assert inner.calls == 3
    assert state["t"] > 0  # backoff slept between attempts


def test_progress_reports_failed_attempt_retry_and_success(tmp_path):
    import json

    from src.progress import ProgressReporter, report_progress

    path = tmp_path / "progress.jsonl"
    source = ResilientSource(
        _ScriptedSource([_failed(), _ok()]), policy=_policy(), sleep=lambda _: None,
    )
    with ProgressReporter("r", quiet=True, jsonl_path=path):
        report_progress("Retrieving literature", "search query", current=2, total=3)
        result = source.search("PRIVATE QUERY", limit=3, filters=SearchPaperFilters())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    operations = [row for row in rows if row["event"] == "operation_finished"]
    assert [row["status"] for row in operations] == ["failed", "completed", "completed"]
    assert any(row["status"] == "retrying" for row in rows)
    assert all((row["current"], row["total"]) == (2, 3) for row in operations)
    assert "PRIVATE QUERY" not in path.read_text()
    assert result.status.status == "success"


def test_retry_exhausted_returns_failed_status() -> None:
    _state, monotonic, sleep = _clock()
    inner = _ScriptedSource([RuntimeError("down"), RuntimeError("down")])
    wrapped = ResilientSource(
        inner, policy=_policy(max_attempts=2, circuit_failure_threshold=10),
        sleep=sleep, monotonic=monotonic,
    )

    result = wrapped.search("q", limit=3, filters=SearchPaperFilters())

    assert result.status.status == "failed"
    assert inner.calls == 2
    assert any("search failed" in err for err in result.status.errors)


def test_circuit_opens_after_threshold_and_short_circuits_to_skipped() -> None:
    _state, monotonic, sleep = _clock()
    # 4 failures available, but circuit opens after 2 consecutive failures.
    inner = _ScriptedSource([_failed(), _failed(), _ok(), _ok()])
    wrapped = ResilientSource(
        inner,
        policy=_policy(max_attempts=1, circuit_failure_threshold=2,
                       circuit_reset_seconds=60.0),
        sleep=sleep, monotonic=monotonic,
    )

    first = wrapped.search("q", limit=1, filters=SearchPaperFilters())
    second = wrapped.search("q", limit=1, filters=SearchPaperFilters())
    third = wrapped.search("q", limit=1, filters=SearchPaperFilters())

    assert first.status.status == "failed"
    assert second.status.status == "failed"
    # circuit now open -> no inner call, skipped
    assert third.status.status == "skipped"
    assert inner.calls == 2
    assert any("circuit open" in w for w in third.status.warnings)


def test_circuit_half_opens_after_reset_window() -> None:
    state, monotonic, sleep = _clock()
    inner = _ScriptedSource([_failed(), _failed(), _ok()])
    wrapped = ResilientSource(
        inner,
        policy=_policy(max_attempts=1, circuit_failure_threshold=2,
                       circuit_reset_seconds=30.0),
        sleep=sleep, monotonic=monotonic,
    )

    wrapped.search("q", limit=1, filters=SearchPaperFilters())  # fail 1
    wrapped.search("q", limit=1, filters=SearchPaperFilters())  # fail 2 -> opens
    state["t"] += 31.0  # advance past reset window
    recovered = wrapped.search("q", limit=1, filters=SearchPaperFilters())

    assert recovered.status.status == "success"
    assert inner.calls == 3


def test_skipped_status_passes_through_without_retry() -> None:
    _state, monotonic, sleep = _clock()
    skipped = SourceSearchResult(
        results=[],
        status=SourceStatus(
            source="openalex", status="skipped", source_query="q", result_count=0,
            warnings=["OPENALEX_API_KEY not set"],
        ),
        warnings=["OPENALEX_API_KEY not set"],
    )
    inner = _ScriptedSource([skipped])
    wrapped = ResilientSource(
        inner, policy=_policy(max_attempts=3), sleep=sleep, monotonic=monotonic
    )

    result = wrapped.search("q", limit=1, filters=SearchPaperFilters())

    assert result.status.status == "skipped"
    assert inner.calls == 1  # no retry on skipped


def test_half_open_probe_is_single_call_and_reopens_on_failure() -> None:
    # Fix #6: after the reset window the breaker admits exactly ONE probe call;
    # if that probe fails the breaker re-opens immediately (next search skipped).
    state, monotonic, sleep = _clock()
    inner = _ScriptedSource([_failed(), _failed(), _failed(), _failed()])
    wrapped = ResilientSource(
        inner,
        policy=_policy(max_attempts=3, circuit_failure_threshold=2,
                       circuit_reset_seconds=30.0),
        sleep=sleep, monotonic=monotonic,
    )

    wrapped.search("q", limit=1, filters=SearchPaperFilters())  # fail 1
    wrapped.search("q", limit=1, filters=SearchPaperFilters())  # fail 2 -> opens
    calls_after_open = inner.calls
    state["t"] += 31.0  # advance past reset window
    probe = wrapped.search("q", limit=1, filters=SearchPaperFilters())
    # Exactly one probe call despite max_attempts=3.
    assert inner.calls == calls_after_open + 1
    assert probe.status.status == "failed"
    # Probe failed -> breaker re-opened -> immediate next search is skipped.
    skipped = wrapped.search("q", limit=1, filters=SearchPaperFilters())
    assert skipped.status.status == "skipped"
    assert inner.calls == calls_after_open + 1  # no additional inner call


def test_skipped_does_not_reset_failure_streak() -> None:
    # Fix #10: a real failure, then a skipped, then a failure must accrue 2
    # consecutive failures (skipped is neither success nor failure).
    state, monotonic, sleep = _clock()
    skipped = SourceSearchResult(
        results=[],
        status=SourceStatus(
            source="openalex", status="skipped", source_query="q", result_count=0,
            warnings=["OPENALEX_API_KEY not set"],
        ),
        warnings=["OPENALEX_API_KEY not set"],
    )
    inner = _ScriptedSource([_failed(), skipped, _failed()])
    wrapped = ResilientSource(
        inner,
        policy=_policy(max_attempts=1, circuit_failure_threshold=2,
                       circuit_reset_seconds=60.0),
        sleep=sleep, monotonic=monotonic,
    )

    wrapped.search("q", limit=1, filters=SearchPaperFilters())  # fail 1
    wrapped.search("q", limit=1, filters=SearchPaperFilters())  # skipped (no reset)
    third = wrapped.search("q", limit=1, filters=SearchPaperFilters())  # fail 2 -> opens

    assert third.status.status == "failed"
    # Two real failures reached threshold -> breaker open -> next is skipped.
    fourth = wrapped.search("q", limit=1, filters=SearchPaperFilters())
    assert fourth.status.status == "skipped"


def test_zero_reset_seconds_does_not_wedge_breaker_open() -> None:
    # Fix #11: circuit_reset_seconds == 0 means "no cooldown"; the breaker must
    # admit a probe rather than skipping forever.
    state, monotonic, sleep = _clock()
    inner = _ScriptedSource([_failed(), _failed(), _ok()])
    wrapped = ResilientSource(
        inner,
        policy=_policy(max_attempts=1, circuit_failure_threshold=2,
                       circuit_reset_seconds=0),
        sleep=sleep, monotonic=monotonic,
    )

    wrapped.search("q", limit=1, filters=SearchPaperFilters())  # fail 1
    wrapped.search("q", limit=1, filters=SearchPaperFilters())  # fail 2 -> opens
    recovered = wrapped.search("q", limit=1, filters=SearchPaperFilters())

    assert recovered.status.status == "success"
    assert inner.calls == 3
