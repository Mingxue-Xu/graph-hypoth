from __future__ import annotations

import json
from typing import Any

import pytest

from src.research_profile import ResearchProfile
from src.retrieval.planner import (
    LLMRetrievalPlanner,
    RetrievalPlan,
    plan_t1,
    rrf_scores,
)


class _FakeBackend:
    """Canned CAMEL-shaped backend: ``.run(messages)`` returns a fixed completion."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[Any] = []

    def run(self, messages: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(messages)
        return {"choices": [{"message": {"content": self._content}}]}


class _ObjectMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _ObjectChoice:
    def __init__(self, content: str) -> None:
        self.message = _ObjectMessage(content)


class _ObjectResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_ObjectChoice(content)]


class _ObjectShapedBackend:
    """A CAMEL backend returning an attribute-shaped response (no ``__getitem__``) — the
    shape ``backend_json``'s ``_first_response_message``/``_message_content`` handle but raw
    ``response["choices"][0]["message"]["content"]`` indexing does not (routing the planner
    through ``backend_json`` handles)."""

    def __init__(self, content: str) -> None:
        self._content = content

    def run(self, messages: Any, *args: Any, **kwargs: Any) -> _ObjectResponse:
        return _ObjectResponse(self._content)


def test_plan_t1_derives_deterministic_subqueries_from_profile() -> None:
    # The deterministic planner is a field-agnostic fallback: the claim plus claim-and-topic variants,
    # ordered by priority weight, identical on repeat calls.
    profile = ResearchProfile(
        claim="C",
        concepts=[{"term": "alpha", "weight": 0.5}, {"term": "beta", "weight": 0.9}],
    )
    plans = plan_t1(profile)
    queries = [p.query for p in plans]
    assert queries[0] == "C"
    assert "C beta" in queries and "C alpha" in queries
    assert queries.index("C beta") < queries.index("C alpha")  # higher weight first
    assert queries == [p.query for p in plan_t1(profile)]  # deterministic
    assert all(isinstance(p, RetrievalPlan) and p.sources is None for p in plans)


def test_plan_t1_respects_n_cap() -> None:
    profile = ResearchProfile(
        claim="C", concepts=[{"term": f"t{i}", "weight": 0.5} for i in range(20)]
    )
    assert len(plan_t1(profile, n_cap=3)) == 3


def test_plan_t1_anchors_on_lens_for_claimless_profile() -> None:
    # Without a claim, the first retrieval plan must still produce real queries anchored on the lens
    # (field + expertise), never a ``None`` query from the absent claim.
    profile = ResearchProfile(
        field="tensor factorization for LLM compression",
        expertise="weight-matrix geometry",
        concepts=[{"term": "tensor train", "weight": 0.9}],
    )
    queries = [p.query for p in plan_t1(profile)]
    assert queries[0] == profile.anchor()  # the lens anchors the base query
    assert all(q and "None" not in q for q in queries)  # no None-query from the missing claim
    assert any("tensor train" in q for q in queries)  # priority topic variant still appended


def test_llm_planner_prompt_uses_lens_not_none_claim_for_claimless() -> None:
    # A claimless planner prompt must anchor on the lens and never emit ``Claim: None``.
    backend = _FakeBackend(json.dumps({"queries": [{"query": "q"}]}))
    planner = LLMRetrievalPlanner(backend)
    planner.plan(ResearchProfile(field="tensor methods", expertise="LLM compression"))
    user_msg = backend.calls[0][1]["content"]  # messages == [system, user]
    assert "Claim: None" not in user_msg
    assert "tensor methods" in user_msg  # the lens anchors the prompt instead


def test_llm_planner_keeps_research_question_open() -> None:
    backend = _FakeBackend(json.dumps({"queries": [{"query": "pruning tradeoffs"}]}))
    LLMRetrievalPlanner(backend).plan(
        ResearchProfile(research_question="When does pruning preserve facts?")
    )
    prompt = backend.calls[0][1]["content"]
    assert "Research question (open, not an established claim): When does pruning preserve facts?" in prompt
    assert "Claim:" not in prompt


def test_llm_planner_parses_json_contract_into_plans() -> None:
    content = json.dumps(
        {
            "queries": [
                {
                    "query": "efficient VLM token pruning",
                    "sources": None,
                    "max_age_hours": 17520,
                    "include_text": ["CVPR"],
                },
                {"query": "vision language model compression", "max_age_hours": 8760},
            ]
        }
    )
    planner = LLMRetrievalPlanner(_FakeBackend(content))
    plans = planner.plan(
        ResearchProfile(claim="C", concepts=[{"term": "x", "weight": 0.5}])
    )
    assert [p.query for p in plans] == [
        "efficient VLM token pruning",
        "vision language model compression",
    ]
    assert plans[0].filters.max_age_hours == 17520
    assert plans[0].filters.include_text == ["CVPR"]
    assert plans[1].filters.max_age_hours == 8760


def test_llm_planner_caps_subqueries_at_n_cap() -> None:
    content = json.dumps({"queries": [{"query": f"q{i}"} for i in range(20)]})
    planner = LLMRetrievalPlanner(_FakeBackend(content), n_cap=4)
    assert len(planner.plan(ResearchProfile(claim="C"))) == 4


def test_llm_planner_degrades_to_t1_on_malformed_response() -> None:
    profile = ResearchProfile(claim="C", concepts=[{"term": "x", "weight": 0.9}])
    planner = LLMRetrievalPlanner(_FakeBackend("not json at all"))
    plans = planner.plan(profile)
    assert [p.query for p in plans] == [p.query for p in plan_t1(profile)]


def test_llm_planner_drops_oversized_include_text_without_crashing() -> None:
    # include_text entries are capped at 5 words; an over-long phrase must not crash the
    # plan — the query survives with the bad filter field dropped.
    content = json.dumps(
        {
            "queries": [
                {
                    "query": "good query",
                    "include_text": ["one two three four five six seven"],
                }
            ]
        }
    )
    planner = LLMRetrievalPlanner(_FakeBackend(content))
    plans = planner.plan(ResearchProfile(claim="C"))
    assert [p.query for p in plans] == ["good query"]
    assert plans[0].filters.include_text is None


def test_llm_planner_parses_object_shaped_response() -> None:
    # object-shaped responses (attributes, not dict keys) now parse via backend_json instead of
    # falling into the broad except -> plan_t1 (deliberate robustness improvement, unit backend-json).
    content = json.dumps({"queries": [{"query": "object-shaped query"}]})
    planner = LLMRetrievalPlanner(_ObjectShapedBackend(content))
    plans = planner.plan(ResearchProfile(claim="C"))
    assert [p.query for p in plans] == ["object-shaped query"]


def test_rrf_scores_reward_items_ranked_high_across_queries() -> None:
    scores = rrf_scores([["a", "b"], ["a", "c"]])
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["c"]
    assert scores["a"] == pytest.approx(2 / 61, abs=1e-9)
