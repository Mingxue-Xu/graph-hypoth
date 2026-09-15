"""Authored-priority emphasis reranking for claimless discovery.

When a researcher authors weighted concepts, the surfaced Synthesist ranking is re-ordered so hypotheses
touching high-priority concepts rise, per ``emphasis_policy`` (author_directed vs blended). With NO
authored priority the rerank is the identity: the original RankScore order is preserved,
keeping claim-mode runs without priorities byte-identical.
"""

from __future__ import annotations

import pytest


def _scored(cid, *, labels, rank_score):
    from src.cycles.hypothesis import HypothesisCandidate, ScoredCandidate
    from src.delta import build_node

    nodes = tuple(build_node(label=lab) for lab in labels)
    cand = HypothesisCandidate(candidate_id=cid, new_nodes=nodes)
    return ScoredCandidate(candidate=cand, hyp_score=rank_score, rank_score=rank_score)


def test_author_priority_is_max_matching_weight():
    from src.emphasis import author_priority

    spec = [("tensor", 0.88), ("rank", 0.70), ("singular value", 0.90)]
    # "tensor" and "rank" both match; the MAX authored weight (0.88) wins; "singular value" doesn't.
    assert author_priority(["tensor train rank"], spec) == pytest.approx(0.88, abs=1e-6)
    assert author_priority(["unrelated concept"], spec) == 0.0  # no term matches
    assert author_priority(["tensor train rank"], []) == 0.0  # no authored priority


def test_rerank_author_directed_promotes_authored_priority_over_rankscore():
    from src.emphasis import rerank_by_emphasis

    # b has the higher spec'd RankScore, but a matches an authored priority term -> a wins (lexico).
    a = _scored("a", labels=["tensor train rank"], rank_score=0.40)
    b = _scored("b", labels=["unrelated"], rank_score=0.90)
    out = rerank_by_emphasis(
        [b, a], authored_concepts=[("tensor", 0.9)], policy="author_directed"
    )
    assert [sc.candidate.candidate_id for sc in out] == ["a", "b"]


def test_rerank_blended_mixes_priority_and_rankscore():
    from src.emphasis import rerank_by_emphasis

    # blended emphasis = w_author*author_priority + w_rank*rank_score.
    # a: 0.5*0.6 + 0.5*0.40 = 0.50 ; b: 0.5*0.0 + 0.5*0.90 = 0.45 -> a first.
    a = _scored("a", labels=["tensor train rank"], rank_score=0.40)
    b = _scored("b", labels=["unrelated"], rank_score=0.90)
    out = rerank_by_emphasis(
        [b, a], authored_concepts=[("tensor", 0.6)], policy="blended",
        weights={"author": 0.5, "rank": 0.5},
    )
    assert [sc.candidate.candidate_id for sc in out] == ["a", "b"]


def test_rerank_is_identity_without_authored_priority():  # safety invariant: golden runs unchanged
    from src.emphasis import rerank_by_emphasis

    a = _scored("a", labels=["x"], rank_score=0.40)
    b = _scored("b", labels=["y"], rank_score=0.90)
    assert rerank_by_emphasis([b, a], authored_concepts=[]) == [b, a]


def test_emphasis_reranker_binds_profile_priority_and_policy():  # the cycle's reranker-hook builder
    from src.emphasis import emphasis_reranker

    a = _scored("a", labels=["tensor train rank"], rank_score=0.40)
    b = _scored("b", labels=["unrelated"], rank_score=0.90)
    rerank = emphasis_reranker([("tensor", 0.9)], "author_directed")
    assert [sc.candidate.candidate_id for sc in rerank([b, a])] == ["a", "b"]
