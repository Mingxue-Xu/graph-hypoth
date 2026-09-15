"""The relationship is explicit prose grounded in the seed, not a graph inference."""

from types import SimpleNamespace

import pytest

from src.connected_render import render_page
from src.report_context import relationship_section
from src.trace_render import _elaboration_user_prompt, render_hypothesis_html
from tests.unit.test_trace_render import _row


@pytest.mark.parametrize("seed_kind,label", [
    ("claim", "starting claim"), ("research_goal", "research goal"),
    ("raw_message", "starting input"),
])
def test_both_reports_include_saved_relationship_with_correct_label(seed_kind, label):
    card = {
        "headline": "Pruning may lose its cost advantage.",
        "relationship_to_seed": "Challenges the cost claim when recovery cost exceeds savings < 5%.",
    }
    graph = {"version": 1, "nodes": {"n": {"label": "cost", "provenance": []}}, "edges": {}}
    row = {"cid": "h1", "rank": 1, "scores": {}}
    connected, _ = render_page(
        graph, "run", row, ["n"], [], {}, plain=card, claim="Starting input text",
        seed_kind=seed_kind,
    )
    trace = render_hypothesis_html(
        _row(), claim="Starting input text", elaboration=card, seed_kind=seed_kind,
    )
    for page in (connected, trace):
        assert f"Relationship to the {label}" in page
        assert "Challenges the cost claim" in page
        assert "savings &lt; 5%." in page
    assert connected.index("Starting input text") < connected.index("Proposed hypothesis</h2>")
    assert connected.index("Starting input text") < connected.index("Relationship to the")
    assert connected.index("Relationship to the") < connected.index("Possible explanation")


def test_old_cards_do_not_get_an_invented_relationship():
    assert relationship_section({"headline": "An old proposal."}, "claim") == ""
    assert relationship_section({"relationship_to_seed": {}}, "claim") == ""
    assert relationship_section(None, None) == ""


def test_question_precedes_hypothesis_and_connection_in_both_reports():
    card = {"headline": "Pruning may lose its cost advantage.",
            "relationship_to_seed": "Tests whether recovery costs erase savings."}
    graph = {"version": 1, "nodes": {"n": {"label": "cost", "provenance": []}}, "edges": {}}
    connected, _ = render_page(
        graph, "run", {"cid": "h1", "rank": 1, "scores": {}}, ["n"], [], {},
        plain=card, claim="When does pruning save cost?", seed_kind="research_question",
    )
    trace = render_hypothesis_html(
        _row(), claim="When does pruning save cost?", elaboration=card,
        seed_kind="research_question",
    )
    for page in (connected, trace):
        assert page.index("Research question") < page.index("Pruning may lose its cost advantage.")
        assert page.index("Pruning may lose its cost advantage.") < page.index(
            "How this hypothesis addresses the question"
        ) < page.index("Possible explanation")
        assert page.count(card["headline"]) == 1
        assert "Claim to investigate" not in page


@pytest.mark.parametrize("kind", ["claim", "research_goal", "research_question"])
def test_elaborator_receives_actual_starting_input(kind):
    profile = SimpleNamespace(claim="Preserve facts at lower cost.", seed_kind=kind)
    prompt = _elaboration_user_prompt(_row(), {}, profile)
    assert "Preserve facts at lower cost." in prompt
    assert "Starting input (" in prompt
    assert ("Research goal" in prompt) is (kind == "research_goal")
