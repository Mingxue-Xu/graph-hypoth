"""Shared, low-level reader for on-disk run artifacts (SQLiteLogStore payload tables). Its
consumers — hypothesis_scores and provenance_links — used to re-implement the same
"connect -> select payload rows -> OperationalError -> []" sqlite shape. This module holds only
that shared mechanics; each consumer keeps its own domain projection (JSON shape, filtering) local.

Pure stdlib (``sqlite3``/``pathlib``) — the offline render scripts import this module without the
heavy app-stack deps, so it must import nothing from the app stack.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def read_payload_rows(db_path: str | Path, table: str, run_id: str | None = None) -> list[str]:
    """Every row's raw ``payload`` column text from a SQLiteLogStore ``table``, optionally filtered
    to one ``run_id`` (tables that carry a run_id column may log more than one run, and row
    identities like candidate ids can recur across runs). ``[]`` when the table or DB is absent, or
    the ``run_id`` filter matches nothing — never raises."""
    con = sqlite3.connect(str(db_path))
    try:
        if run_id is None:
            rows = con.execute(f"select payload from {table}").fetchall()
        else:
            rows = con.execute(f"select payload from {table} where run_id=?", (run_id,)).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()
    return [row[0] for row in rows]
