"""Recover a hypothesis's detailed scores from a run's logged ``audit_events``.

The surfaced trace persists only the ranking-outcome scores (field novelty, saturation, HypScore,
RankScore, cross-concept, common-sense). The proposer's finer per-candidate self-assessments
(novelty, duplication/overlap, testability, scope_fit, plausibility, expected_yield, centrality,
mechanism_specificity) are NOT persisted there — they live in the proposer's logged LLM response,
which is captured in ``audit_events`` only when the run logged its LLM calls. This module pulls them
back out so the per-hypothesis render can show the full, detailed score set in user-facing language;
when the run did not log calls (``audit_events`` empty), it returns nothing and the caller falls back
to the persisted outcome scores. Pure stdlib (``sqlite3``/``json``, via ``run_artifacts``)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.run_artifacts import read_payload_rows


def load_audit_events(db_path: str | Path, run_id: str | None = None) -> list[dict[str, Any]]:
    """Every logged event ({..., content}) from a SQLiteLogStore ``audit_events`` table. When
    ``run_id`` is given, only that run's events are returned (a single DB may log more than one run,
    and candidate ids like ``h1`` recur across runs — filtering prevents cross-run mixing). Empty
    when the table/DB is absent or the run did not log LLM calls."""
    out: list[dict[str, Any]] = []
    for payload in read_payload_rows(db_path, "audit_events", run_id):
        try:
            out.append(json.loads(payload))
        except (ValueError, TypeError):
            continue
    return out


def events_db_for_run(db_paths: Sequence[str | Path], run_id: str) -> str | None:
    """The DB that logged ``run_id`` (its ``audit_events`` carry that ``run_id``). Returns ``None`` if
    no candidate DB logged this run — so per-candidate signals are read from THIS run's own log or not
    at all, never from a different run's log (candidate ids like ``h1`` are not globally unique)."""
    for db in db_paths:
        if read_payload_rows(db, "audit_events", run_id):
            return str(db)
    return None


def proposer_signals(events: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``{candidate_id: {signal_name: value}}`` from the proposer's logged per-candidate
    ``llm_signals``. Found content-by-shape (any event whose ``content`` parses to a dict with a
    ``candidates`` list carrying ``candidate_id`` + ``llm_signals``), so it does not depend on the
    event's role label. Empty when no such event exists (run logged no LLM calls)."""
    out: dict[str, dict[str, Any]] = {}
    for event in events:
        content = event.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (ValueError, TypeError):
                continue
        if not isinstance(content, dict):
            continue
        candidates = content.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if (
                isinstance(candidate, dict)
                and candidate.get("candidate_id")
                and isinstance(candidate.get("llm_signals"), dict)
            ):
                out[str(candidate["candidate_id"])] = candidate["llm_signals"]
    return out
