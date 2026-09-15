"""Unit tests for provenance_links — resolve a concept node to the source paper(s) it was mined
from, by VERBATIM quote-match against a run's retrieval pool (faithful port of the attach_provenance
resolver). Pure deterministic string matching, so hermetic fixtures + exact assertions."""

from __future__ import annotations

import json
import sqlite3

from src.provenance_links import (
    load_retrieval_batches,
    load_retrieval_pool,
    node_sources,
    resolve_paper,
    select_best_db,
)
from src.retrieval.identity import retrieval_batch_id

POOL = [
    {
        "quote": "We find that performance is strongly shaped by tokenizer efficiency and "
        "channel-specific inductive biases across modalities.",
        "title": "Decoding Across Modalities",
        "url": "http://arxiv.org/abs/2511.00001v1",
        "source": "arxiv",
    },
    {
        "quote": "An unrelated passage about gravitational wave detection pipelines.",
        "title": "GW Pipelines",
        "url": "http://arxiv.org/abs/2601.00002v1",
        "source": "arxiv",
    },
]

SPAN = "strongly shaped by tokenizer efficiency and channel-specific inductive biases"


def test_resolve_paper_matches_longest_verbatim_window():
    papers = resolve_paper(SPAN, POOL)
    assert [p["url"] for p in papers] == ["http://arxiv.org/abs/2511.00001v1"]
    assert papers[0]["title"] == "Decoding Across Modalities"
    assert papers[0]["source"] == "arxiv"


def test_resolve_paper_unresolved_returns_empty():
    assert resolve_paper("a clause appearing in none of the retrieved quotes whatsoever", POOL) == []


def test_resolve_paper_ignores_too_short_spans():
    # below the 18-char minimum window -> no match attempted
    assert resolve_paper("the a of", POOL) == []


def test_resolve_paper_matches_two_word_span_at_minimum_length():
    pool = [{
        "quote": (
            "LoRAP organically combines Low-Rank matrix approximation "
            "And structured Pruning."
        ),
        "title": "LoRAP",
        "url": "https://proceedings.mlr.press/v235/li24bi.html",
        "source": "codex_web",
    }]
    papers = resolve_paper("structured Pruning", pool)
    assert [paper["title"] for paper in papers] == ["LoRAP"]


def test_resolve_paper_rejects_short_two_word_span_even_when_verbatim():
    pool = [{
        "quote": "We study neural networks under compression.",
        "title": "Generic paper",
        "url": "https://example.test/generic",
        "source": "test",
    }]
    assert resolve_paper("neural networks", pool) == []


def test_resolve_paper_still_rejects_single_word_span_over_min_chars():
    pool = [{
        "quote": "We measure electroencephalography signals.",
        "title": "Signals",
        "url": "https://example.test/signals",
        "source": "test",
    }]
    assert resolve_paper("electroencephalography", pool) == []


def test_node_sources_resolves_literature_enrichment_provenance():
    node = {
        "label": "sub-lexical wordplay degradation",
        "provenance": [
            {"source": "literature_enrichment", "paper_id": "p6", "matched_quote_span": SPAN}
        ],
    }
    srcs = node_sources(node, POOL)
    assert len(srcs) == 1
    assert srcs[0]["url"] == "http://arxiv.org/abs/2511.00001v1"
    assert srcs[0]["quote"] == SPAN  # the verbatim span carried through for display


def test_node_sources_claim_node_has_no_paper():
    node = {"label": "image/video efficiency", "provenance": [{"source": "claim", "span": [0, 5]}]}
    assert node_sources(node, POOL) == []


def test_load_retrieval_pool_reads_all_cache_rows(tmp_path):
    db = tmp_path / "events.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("create table retrieval_cache(payload text)")
    con.execute("insert into retrieval_cache(payload) values (?)", (json.dumps({"evidence": POOL[:1]}),))
    con.execute("insert into retrieval_cache(payload) values (?)", (json.dumps({"evidence": POOL[1:]}),))
    con.commit()
    con.close()
    pool = load_retrieval_pool(str(db))
    assert len(pool) == 2
    assert {p["title"] for p in pool} == {"Decoding Across Modalities", "GW Pipelines"}


def test_load_retrieval_batches_preserves_duplicate_id_scopes_and_filters_run(tmp_path):
    db = tmp_path / "events.sqlite"
    con = sqlite3.connect(str(db))
    con.execute(
        "create table retrieval_cache("
        "run_id text, input_hash text, retrieval_config_hash text, payload text)"
    )
    for run_id, input_hash, title in (
        ("run-a", "input-a", "Paper A"),
        ("run-a", "input-b", "Paper B"),
        ("run-b", "input-c", "Paper C"),
    ):
        payload = {
            "input_hash": input_hash,
            "retrieval_config_hash": "config",
            "query": f"query for {title}",
            "evidence": [{
                "evidence_id": "ev_000001",
                "title": title,
                "metadata": {"retrieval_batch_id": "untrusted-stamp"},
            }],
        }
        con.execute(
            "insert into retrieval_cache values (?, ?, ?, ?)",
            (run_id, input_hash, "config", json.dumps(payload)),
        )
    con.commit()
    con.close()

    batches = load_retrieval_batches(db, run_id="run-a")

    assert [batch["run_id"] for batch in batches] == ["run-a", "run-a"]
    assert [batch["input_hash"] for batch in batches] == ["input-a", "input-b"]
    assert [batch["evidence"][0]["evidence_id"] for batch in batches] == [
        "ev_000001", "ev_000001"
    ]
    assert batches[0]["batch_id"] != batches[1]["batch_id"]
    assert {batch["evidence"][0]["title"] for batch in batches} == {"Paper A", "Paper B"}

    all_batches = load_retrieval_batches(db)
    assert [batch["run_id"] for batch in all_batches] == ["run-a", "run-a", "run-b"]
    assert all_batches[0]["batch_id"] == retrieval_batch_id(
        run_id="run-a", input_hash="input-a", retrieval_config_hash="config"
    )
    assert all_batches[0]["batch_id"] != "untrusted-stamp"


def test_load_retrieval_pool_missing_table_returns_empty(tmp_path):
    db = tmp_path / "empty.sqlite"
    sqlite3.connect(str(db)).close()
    assert load_retrieval_pool(str(db)) == []


def test_select_best_db_picks_pool_with_more_resolutions(tmp_path):
    def mk(path, evidence):
        con = sqlite3.connect(str(path))
        con.execute("create table retrieval_cache(payload text)")
        con.execute("insert into retrieval_cache(payload) values (?)", (json.dumps({"evidence": evidence}),))
        con.commit()
        con.close()

    good, bad = tmp_path / "good.sqlite", tmp_path / "bad.sqlite"
    mk(good, POOL)
    mk(bad, [POOL[1]])  # cannot resolve the SPAN
    node = {"provenance": [{"source": "literature_enrichment", "matched_quote_span": SPAN}]}
    assert select_best_db([str(bad), str(good)], [node]) == str(good)
