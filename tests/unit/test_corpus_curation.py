"""Deterministic three-angle corpus curation.

Classifies retrieved full-text records into researcher-named angles by keyword, filters short
bodies, keeps the richest-bodied per angle up to a quota, and emits the miner passages + the judge
saturation reference list. Pure + deterministic — no LLM, no network.
"""

from __future__ import annotations


def _angles():
    from src.corpus_curation import CorpusAngle

    return [
        CorpusAngle("IT", ("mutual information", "mdl", "bits-per-parameter"), quota=2),
        CorpusAngle("behavioral", ("calibration", "factuality", "benchmark"), quota=2),
        CorpusAngle("SVD", ("truncated svd", "low-rank"), quota=1),
    ]


def _records():
    return [
        {"evidence_id": "e1", "title": "MDL meets compression", "body": "x" * 800 + " mutual information"},
        {"evidence_id": "e2", "title": "Calibration trade-offs", "body": "y" * 700 + " calibration"},
        {"evidence_id": "e3", "title": "too short", "body": "brief mutual information note"},  # body < 600
        {"evidence_id": "e4", "title": "SVD pruning", "body": "z" * 650 + " low-rank"},
        {"evidence_id": "e5", "title": "memorization bits", "body": "w" * 900 + " bits-per-parameter"},
    ]


def test_curate_corpus_classifies_filters_quotas_and_orders():  # three-angle curation
    from src.corpus_curation import curate_corpus

    curated = curate_corpus(_records(), _angles(), min_body_chars=600)
    # IT (richest first: e5=919, e1=819) -> behavioral (e2) -> SVD (e4); e3 filtered (body < 600).
    assert [r["evidence_id"] for r in curated] == ["e5", "e1", "e2", "e4"]


def test_curate_corpus_respects_per_angle_quota():
    from src.corpus_curation import CorpusAngle, curate_corpus

    angles = [CorpusAngle("IT", ("mutual information",), quota=1)]
    curated = curate_corpus(_records(), angles, min_body_chars=600)
    assert [r["evidence_id"] for r in curated] == ["e1"]  # only e1 mentions "mutual information"; quota 1


def test_corpus_passages_carry_evidence_markers():  # miner provenance contract
    from src.corpus_curation import corpus_passages

    passages = corpus_passages([{"evidence_id": "e1", "paper_id": "p1", "body": "the body text"}])
    assert len(passages) == 1
    assert "e1" in passages[0] and "p1" in passages[0] and "the body text" in passages[0]


def test_corpus_paper_list_numbers_titles():  # judge saturation reference set
    from src.corpus_curation import corpus_paper_list

    out = corpus_paper_list([{"title": "Paper A"}, {"title": "Paper B"}])
    assert "P1: Paper A" in out and "P2: Paper B" in out
