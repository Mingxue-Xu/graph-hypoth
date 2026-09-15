from __future__ import annotations

import time
from typing import Any, Callable

from src.progress import progress_operation
from src.retrieval._util import get_attr_or_key as _get
from src.retrieval.models import SearchPaperFilters, SourceStatus
from src.retrieval.sources import PaperSource, SourceSearchResult


class ResilientSource:
    """Wrap any PaperSource with retry/backoff + a per-source circuit breaker.

    Composition (not inheritance): a ResilientSource is itself a PaperSource, so
    the service treats it identically. It never raises — exhausted retries and
    open circuits are returned as ``failed``/``skipped`` SourceStatus so the
    service's per-source loop keeps degrading gracefully.
    """

    def __init__(
        self,
        inner: PaperSource,
        *,
        policy: Any,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._policy = policy
        self._sleep = sleep
        self._monotonic = monotonic
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open = False

    @property
    def name(self) -> str:
        return str(getattr(self._inner, "name", "unknown"))

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchPaperFilters,
    ) -> SourceSearchResult:
        return self._search(
            query,
            limit=limit,
            filters=filters,
            run_id=None,
        )

    def search_for_run(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchPaperFilters,
        run_id: str,
    ) -> SourceSearchResult:
        return self._search(
            query,
            limit=limit,
            filters=filters,
            run_id=run_id,
        )

    def _search(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchPaperFilters,
        run_id: str | None,
    ) -> SourceSearchResult:
        if self._circuit_open():
            with progress_operation(f"{self.name}: circuit open") as outcome:
                outcome.status = "skipped"
            return self._skipped(query, f"circuit open for {self.name}; skipping")

        max_attempts = max(int(_get(self._policy, "max_attempts", 3)), 1)
        last_result: SourceSearchResult | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                with progress_operation(f"{self.name}: attempt {attempt}/{max_attempts}") as outcome:
                    run_search = getattr(self._inner, "search_for_run", None)
                    if run_id is not None and callable(run_search):
                        result = run_search(
                            query,
                            limit=limit,
                            filters=filters,
                            run_id=run_id,
                        )
                    else:
                        result = self._inner.search(
                            query,
                            limit=limit,
                            filters=filters,
                        )
                    outcome.status = (
                        "completed" if result.status.status == "success" else result.status.status
                    )
            except Exception as exc:  # noqa: BLE001 - never propagate to the loop.
                self._record_failure()
                last_result = self._failed(query, f"{self.name} search failed: {exc}")
            else:
                if result.status.status == "skipped":
                    # Neither success nor failure for breaker accounting.
                    return result
                if result.status.status != "failed":
                    self._record_success()
                    return result
                self._record_failure()
                last_result = result

            if attempt < max_attempts and not self._circuit_open():
                with progress_operation(f"{self.name}: retry backoff", status="retrying"):
                    self._sleep_backoff(attempt)
                continue
            break

        return last_result if last_result is not None else self._failed(
            query, f"{self.name} produced no result"
        )

    def _sleep_backoff(self, attempt: int) -> None:
        base = float(_get(self._policy, "backoff_base_seconds", 0.5))
        ceiling = float(_get(self._policy, "backoff_max_seconds", 8.0))
        if base <= 0:
            return
        delay = min(base * (2 ** (attempt - 1)), ceiling)
        if delay > 0:
            self._sleep(delay)

    def _record_failure(self) -> None:
        if self._half_open:
            # The half-open probe failed: re-open immediately without waiting
            # for the threshold again.
            self._half_open = False
            self._opened_at = self._monotonic()
            return
        self._consecutive_failures += 1
        threshold = int(_get(self._policy, "circuit_failure_threshold", 5))
        if self._consecutive_failures >= threshold:
            self._opened_at = self._monotonic()

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None
        self._half_open = False

    def _circuit_open(self) -> bool:
        if self._opened_at is None:
            return False
        reset_seconds = float(_get(self._policy, "circuit_reset_seconds", 60.0))
        elapsed = self._monotonic() - self._opened_at
        if reset_seconds <= 0 or elapsed >= reset_seconds:
            # Half-open: admit exactly one trial call without clearing the
            # failure counter; _record_failure re-opens on a failed probe.
            self._half_open = True
            return False
        return True

    def _skipped(self, query: str, warning: str) -> SourceSearchResult:
        return SourceSearchResult(
            results=[],
            status=SourceStatus(
                source=self.name,
                status="skipped",
                source_query=query,
                result_count=0,
                warnings=[warning],
            ),
            warnings=[warning],
        )

    def _failed(self, query: str, error: str) -> SourceSearchResult:
        return SourceSearchResult(
            results=[],
            status=SourceStatus(
                source=self.name,
                status="failed",
                source_query=query,
                result_count=0,
                errors=[error],
            ),
            errors=[error],
        )
