"""Regression fixtures for the shared SQLiteLogStore payload reader consolidated out of
hypothesis_scores and provenance_links, pinning its fail-soft behavior."""

from __future__ import annotations

import json
import sqlite3

from src.run_artifacts import read_payload_rows


def test_read_payload_rows_reads_all_rows(tmp_path):
    db = tmp_path / "events.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("create table audit_events(sequence integer, payload text)")
    con.execute("insert into audit_events(payload) values (?)", (json.dumps({"a": 1}),))
    con.execute("insert into audit_events(payload) values (?)", (json.dumps({"a": 2}),))
    con.commit()
    con.close()
    rows = read_payload_rows(str(db), "audit_events")
    assert [json.loads(r) for r in rows] == [{"a": 1}, {"a": 2}]


def test_read_payload_rows_missing_table_returns_empty(tmp_path):
    db = tmp_path / "empty.sqlite"
    sqlite3.connect(str(db)).close()
    assert read_payload_rows(str(db), "audit_events") == []


def test_read_payload_rows_missing_db_file_returns_empty(tmp_path):
    # sqlite3.connect() lazily creates the file on first write, but no table exists yet either
    # way -> same OperationalError -> [] fail-soft path as an existing-but-tableless DB.
    assert read_payload_rows(str(tmp_path / "does-not-exist.sqlite"), "audit_events") == []


def test_read_payload_rows_run_id_filter_selects_matching_rows_only(tmp_path):
    db = tmp_path / "multi.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("create table audit_events(run_id text, payload text)")
    con.execute("insert into audit_events values(?,?)", ("run-A", json.dumps({"a": 1})))
    con.execute("insert into audit_events values(?,?)", ("run-B", json.dumps({"a": 2})))
    con.commit()
    con.close()
    rows = read_payload_rows(str(db), "audit_events", run_id="run-A")
    assert [json.loads(r) for r in rows] == [{"a": 1}]
    assert read_payload_rows(str(db), "audit_events", run_id="run-Z") == []
