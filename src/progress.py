"""Run-scoped progress for CLI users; library calls stay silent by default.

The heartbeat lives in a separate thread so blocking model and retrieval calls
remain visible. It only reads reporter state; it never touches the pipeline or
its subprocesses. ContextVar keeps independent runs from sharing a reporter.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Self, TextIO


@dataclass(frozen=True)
class _Activity:
    stage: str
    detail: str = ""
    status: str = "running"
    current: int | None = None
    total: int | None = None
    provider: str | None = None
    started: float = 0.0


@dataclass
class ProgressOutcome:
    status: str = "completed"


_REPORTER: ContextVar[ProgressReporter | None] = ContextVar("progress", default=None)


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


class ProgressReporter:
    """A scoped, flushed stderr reporter with optional append-only JSONL events.

    Output failures disable the affected sink without failing the research run.
    An explicitly requested JSONL file is opened before the run starts, so an
    invalid destination fails early. Use one reporter per sequential workflow.
    """

    def __init__(
        self,
        run_id: str,
        *,
        quiet: bool = False,
        jsonl_path: Path | None = None,
        stream: TextIO | None = None,
        heartbeat_seconds: float = 30.0,
    ) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self.run_id = run_id
        self.stream = None if quiet else (stream if stream is not None else sys.stderr)
        self.jsonl_path = jsonl_path
        self.heartbeat_seconds = heartbeat_seconds
        self._jsonl: TextIO | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = time.monotonic()
        self._last_output = self._started
        self._activity = _Activity("Starting run", started=self._started)

    def __enter__(self) -> Self:
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl = self.jsonl_path.open("a", encoding="utf-8")
        self._started = time.monotonic()
        self._activity = _Activity("Starting run", started=self._started)
        self._token = _REPORTER.set(self)
        self._emit("run_started", self._activity)
        if self.stream is not None or self._jsonl is not None:
            self._thread = threading.Thread(
                target=self._tick, name="graph-hypoth-progress", daemon=True
            )
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        try:
            status = "completed" if exc_type is None else "failed"
            if isinstance(exc, KeyboardInterrupt):
                status = "interrupted"
            self._emit("run_finished", replace(self._activity, status=status))
        finally:
            _REPORTER.reset(self._token)
            if self._jsonl is not None:
                try:
                    self._jsonl.close()
                except OSError:
                    pass

    def update(
        self,
        stage: str | None,
        detail: str = "",
        *,
        status: str = "running",
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        with self._lock:
            self._activity = _Activity(
                stage=stage or self._activity.stage,
                detail=detail,
                status=status,
                current=current,
                total=total,
                started=time.monotonic(),
            )
            self._emit("progress", self._activity)

    @contextmanager
    def operation(
        self,
        detail: str,
        *,
        provider: str | None = None,
        status: str = "running",
    ) -> Iterator[ProgressOutcome]:
        with self._lock:
            parent = self._activity
            activity = replace(
                parent,
                detail=" · ".join(filter(None, (parent.detail, detail))),
                provider=provider,
                status=status,
                started=time.monotonic(),
            )
            self._activity = activity
            self._emit("operation_started", activity)
        outcome = ProgressOutcome()
        try:
            yield outcome
        except BaseException as exc:
            self._emit(
                "operation_finished",
                replace(
                    activity,
                    status="interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "failed",
                ),
            )
            raise
        else:
            self._emit("operation_finished", replace(activity, status=outcome.status))
        finally:
            with self._lock:
                self._activity = parent

    def heartbeat(self) -> None:
        """Emit only during idle output periods, never over an interactive prompt."""
        with self._lock:
            if (
                not self._stop.is_set()
                and self._activity.status in {"running", "retrying"}
                and time.monotonic() - self._last_output >= self.heartbeat_seconds
            ):
                self._emit("heartbeat", self._activity)

    def _tick(self) -> None:
        while True:
            with self._lock:
                delay = max(
                    0.01,
                    self.heartbeat_seconds - (time.monotonic() - self._last_output),
                )
                if self._activity.status not in {"running", "retrying"}:
                    delay = self.heartbeat_seconds
            if self._stop.wait(delay):
                return
            self.heartbeat()

    def _emit(self, event: str, activity: _Activity) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._started
            operation_elapsed = now - activity.started
            row = {
                **{k: v for k, v in asdict(activity).items() if k != "started"},
                "event": event,
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": self.run_id,
                "elapsed_seconds": round(elapsed, 3),
                "operation_elapsed_seconds": round(operation_elapsed, 3),
            }
            label = activity.stage
            if activity.detail:
                label += f" — {activity.detail}"
            if activity.current is not None and activity.total is not None:
                label += f" {activity.current}/{activity.total}"
            if activity.provider:
                label += f" · {activity.provider}"
            if event == "heartbeat":
                label = (
                    f"Still waiting: {label} · {_duration(operation_elapsed)} elapsed"
                )
            elif event == "run_finished":
                label = f"Run {activity.status} · {label}"
            elif activity.status != "running":
                label += f" · {activity.status}"
            if event == "operation_finished":
                label += f" · {_duration(operation_elapsed)} elapsed"
            self._last_output = now
            if self.stream is not None:
                try:
                    print(
                        f"[{_duration(elapsed)}] {label}", file=self.stream, flush=True
                    )
                except (OSError, ValueError):
                    self.stream = None
            if self._jsonl is not None:
                try:
                    self._jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                    self._jsonl.flush()
                except (OSError, ValueError):
                    try:
                        self._jsonl.close()
                    except OSError:
                        pass
                    self._jsonl = None
                    if self.stream is not None:
                        try:
                            print(
                                "Progress file unavailable; continuing run.",
                                file=self.stream,
                                flush=True,
                            )
                        except (OSError, ValueError):
                            self.stream = None


def report_progress(stage: str | None, detail: str = "", **kwargs) -> None:
    reporter = _REPORTER.get()
    if reporter is not None:
        reporter.update(stage, detail, **kwargs)


@contextmanager
def progress_operation(
    detail: str,
    *,
    provider: str | None = None,
    status: str = "running",
) -> Iterator[ProgressOutcome]:
    reporter = _REPORTER.get()
    if reporter is None:
        yield ProgressOutcome()
    else:
        with reporter.operation(detail, provider=provider, status=status) as outcome:
            yield outcome


def add_progress_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress on stderr; final results, prompts and errors remain.",
    )
    parser.add_argument(
        "--progress-jsonl",
        type=Path,
        default=None,
        metavar="PATH",
        help="Append structured progress events to PATH (also works with --quiet).",
    )
