"""Resolve a graph concept node to the source paper(s) it was mined from.

A literature-mined concept node records ``matched_quote_span`` (the verbatim phrase the miner read it
from) + ``paper_id`` (the miner's passage index, not a stable citation). The span is a substring of
one retrieved evidence quote, so the source paper is recovered by VERBATIM quote-match against the
run's retrieval pool (the SQLiteLogStore ``retrieval_cache``). This is a faithful port of the verified
``attach_provenance`` resolver (causal-standalone) — pure stdlib (``re``/``json``, plus ``sqlite3`` via
``run_artifacts``), so it is importable by the render scripts without the heavy ML deps and is
unit-tested in isolation.

When the miner lightly joined/reworded the span, we match on the LONGEST contiguous word-window of the
span that appears verbatim in a pool quote (an ellipsis joins non-contiguous text, so a window crossing
it simply fails to match — self-correcting). The matched window is a real verbatim phrase from that
paper, so the resolution is faithful, never a guess; unresolved spans return no paper rather than guess.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.retrieval.identity import retrieval_batch_id


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _retrieval_cache_rows(
    db_path: str | Path, *, run_id: str | None
) -> list[tuple[str, str, str, str]]:
    """Read cache-key columns with a payload-only fallback for historical test/artifact DBs."""
    connection = sqlite3.connect(str(db_path))
    try:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(retrieval_cache)").fetchall()
        }
        if "payload" not in columns:
            return []
        select_columns = [
            column if column in columns else f"'' AS {column}"
            for column in ("run_id", "input_hash", "retrieval_config_hash")
        ]
        query = f"SELECT {', '.join(select_columns)}, payload FROM retrieval_cache"
        parameters: tuple[str, ...] = ()
        if run_id is not None and "run_id" in columns:
            query += " WHERE run_id = ?"
            parameters = (run_id,)
        return [
            tuple(str(value or "") for value in row)
            for row in connection.execute(query, parameters).fetchall()
        ]
    except sqlite3.OperationalError:
        return []
    finally:
        connection.close()


def load_retrieval_batches(
    db_path: str | Path, *, run_id: str | None = None
) -> list[dict[str, Any]]:
    """Load retrieval-cache rows without discarding their query/cache identity.

    ``evidence_id`` is local to one retrieval result, so consumers that resolve citations must keep
    these row boundaries. New payloads carry ``metadata.retrieval_batch_id`` on every evidence item;
    legacy payloads derive the same deterministic identity from the cache-key fields retained in the
    result. ``run_id`` is optional for old one-run databases, but callers that know it should pass it
    so both row filtering and the derived identity use the exact logical run.
    """
    batches: list[dict[str, Any]] = []
    for row_run_id, row_input_hash, row_config_hash, raw_payload in _retrieval_cache_rows(
        db_path, run_id=run_id
    ):
        payload = json.loads(raw_payload)
        evidence = payload.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = []
        batch_run_id = str(row_run_id or payload.get("run_id") or run_id or "")
        input_hash = str(row_input_hash or payload.get("input_hash") or "")
        config_hash = str(
            row_config_hash or payload.get("retrieval_config_hash") or ""
        )
        # Cache-key columns are authoritative. Evidence metadata carries the same ID for downstream
        # propagation, but must not be able to override or forge the identity of its containing row.
        batch_id = retrieval_batch_id(
            run_id=batch_run_id,
            input_hash=input_hash,
            retrieval_config_hash=config_hash,
        )
        batches.append(
            {
                "batch_id": batch_id,
                "run_id": batch_run_id,
                "input_hash": input_hash,
                "retrieval_config_hash": config_hash,
                "query": str(payload.get("query") or ""),
                "evidence": evidence,
            }
        )
    return batches


def load_retrieval_pool(
    db_path: str | Path, *, run_id: str | None = None
) -> list[dict[str, Any]]:
    """All retrieved evidence ({quote, title, url, source, ...}) from a run's SQLiteLogStore
    ``retrieval_cache`` (every cached row concatenated). Empty when the table/DB is absent."""
    return [
        item
        for batch in load_retrieval_batches(db_path, run_id=run_id)
        for item in batch["evidence"]
    ]


def resolve_paper(
    span: str, pool: Sequence[dict[str, Any]], *, min_chars: int = 18
) -> list[dict[str, str]]:
    """Resolve the source paper(s) for one literature span by verbatim quote-match against the pool.
    Returns ``[{title, url, source}]`` (deduped by url-or-title, in first-seen order); ``[]`` if the
    span matches no pool quote (unresolved -> no guess)."""
    words = re.sub(r"\.\.\.|…", " ", span or "").split()
    for length in range(len(words), 1, -1):
        best: list[dict[str, str]] = []
        seen: set[str] = set()
        for i in range(0, len(words) - length + 1):
            window = " ".join(words[i : i + length])
            n = _norm(window)
            if len(n) < min_chars:
                continue
            for e in pool:
                if n in _norm(e.get("quote") or ""):
                    key = (e.get("url") or "") or (e.get("title") or "")
                    if key and key not in seen:
                        seen.add(key)
                        best.append(
                            {
                                "title": e.get("title") or "",
                                "url": e.get("url") or "",
                                "source": e.get("source") or "",
                            }
                        )
        if best:
            return best
    return []


def node_sources(node: dict[str, Any], pool: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """The source paper(s) a node's literature-mined provenance traces to, each carrying the verbatim
    ``quote`` (the matched span) + ``paper_id`` for display. Claim/empty provenance -> no paper.
    Deduped by url-or-title across all of the node's literature spans."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for p in node.get("provenance") or []:
        if p.get("source") != "literature_enrichment":
            continue
        span = p.get("matched_quote_span") or ""
        for paper in resolve_paper(span, pool):
            key = paper["url"] or paper["title"]
            if key in seen:
                continue
            seen.add(key)
            out.append({**paper, "quote": span, "paper_id": str(p.get("paper_id") or "")})
    return out


def select_best_db(
    db_paths: Sequence[str | Path], mined_nodes: Sequence[dict[str, Any]]
) -> str | None:
    """Pick, among candidate event DBs, the one whose retrieval pool resolves the MOST of the given
    mined nodes — the run's own pool resolves its own spans, an unrelated run's pool resolves ~none,
    so this self-selects the right DB without a fragile filename convention. ``None`` if no candidate
    has a usable pool."""
    best_path: str | None = None
    best_hits = -1
    for db in db_paths:
        pool = load_retrieval_pool(db)
        if not pool:
            continue
        hits = sum(1 for n in mined_nodes if node_sources(n, pool))
        if hits > best_hits:
            best_path, best_hits = str(db), hits
    return best_path
