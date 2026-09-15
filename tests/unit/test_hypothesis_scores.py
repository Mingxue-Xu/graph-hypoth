"""Unit tests for hypothesis_scores — recover the proposer's per-candidate self-assessed signals
(novelty, duplication/overlap, testability, scope_fit, ...) from a run's logged audit_events.
Available only when the run logged LLM calls; empty otherwise (graceful)."""

from __future__ import annotations

import json
import sqlite3

from src.hypothesis_scores import (
    events_db_for_run,
    load_audit_events,
    proposer_signals,
)

PROPOSER_CONTENT = json.dumps(
    {
        "candidates": [
            {"candidate_id": "h1", "llm_signals": {"novelty": 0.82, "duplication": 0.12, "testability": 0.85}},
            {"candidate_id": "h2", "llm_signals": {"novelty": 0.80, "duplication": 0.15, "testability": 0.78}},
        ]
    }
)
EVENTS = [
    {"node_name": "retrieval", "content": "not json at all"},
    {"node_name": "hypothesis_proposer", "content": PROPOSER_CONTENT},
]


def test_proposer_signals_extracts_per_candidate():
    sig = proposer_signals(EVENTS)
    assert set(sig) == {"h1", "h2"}
    assert sig["h1"]["duplication"] == 0.12
    assert sig["h2"]["novelty"] == 0.80


def test_proposer_signals_empty_without_proposer_event():
    assert proposer_signals([{"content": "hello"}, {"content": json.dumps({"foo": 1})}]) == {}


def test_proposer_signals_handles_content_as_dict():
    ev = [{"content": {"candidates": [{"candidate_id": "h9", "llm_signals": {"novelty": 0.5}}]}}]
    assert proposer_signals(ev) == {"h9": {"novelty": 0.5}}


def test_proposer_signals_ignores_candidates_without_signals():
    ev = [{"content": json.dumps({"candidates": [{"candidate_id": "h1"}]})}]  # no llm_signals
    assert proposer_signals(ev) == {}


def test_load_audit_events_reads_rows(tmp_path):
    db = tmp_path / "events.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("create table audit_events(sequence integer, payload text)")
    con.execute("insert into audit_events(payload) values (?)", (json.dumps({"content": PROPOSER_CONTENT}),))
    con.commit()
    con.close()
    evs = load_audit_events(str(db))
    assert len(evs) == 1
    assert proposer_signals(evs) == {
        "h1": {"novelty": 0.82, "duplication": 0.12, "testability": 0.85},
        "h2": {"novelty": 0.80, "duplication": 0.15, "testability": 0.78},
    }


def test_load_audit_events_missing_table_returns_empty(tmp_path):
    db = tmp_path / "empty.sqlite"
    sqlite3.connect(str(db)).close()
    assert load_audit_events(str(db)) == []


def _mk_audit_db(path, run_id, content):
    con = sqlite3.connect(str(path))
    con.execute("create table audit_events(run_id text, payload text)")
    con.execute("insert into audit_events values(?,?)", (run_id, json.dumps({"content": content})))
    con.commit()
    con.close()


def test_events_db_for_run_selects_the_runs_own_db(tmp_path):
    a, b = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
    _mk_audit_db(a, "run-A", PROPOSER_CONTENT)
    _mk_audit_db(b, "run-B", "{}")
    # order-independent: returns the DB whose audit_events carry the run_id
    assert events_db_for_run([str(b), str(a)], "run-A") == str(a)
    # no DB logged this run -> None (NEVER fall back to another run's log)
    assert events_db_for_run([str(a), str(b)], "run-Z") is None


def test_load_audit_events_filters_by_run_id(tmp_path):
    db = tmp_path / "multi.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("create table audit_events(run_id text, payload text)")
    con.execute("insert into audit_events values(?,?)", ("run-A", json.dumps({"content": PROPOSER_CONTENT})))
    other = json.dumps({"candidates": [{"candidate_id": "hZ", "llm_signals": {"novelty": 0.1}}]})
    con.execute("insert into audit_events values(?,?)", ("run-B", json.dumps({"content": other})))
    con.commit()
    con.close()
    assert set(proposer_signals(load_audit_events(str(db), run_id="run-A"))) == {"h1", "h2"}
    assert set(proposer_signals(load_audit_events(str(db), run_id="run-B"))) == {"hZ"}
