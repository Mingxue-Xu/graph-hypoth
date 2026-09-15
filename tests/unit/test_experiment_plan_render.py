"""Tests for shared ExperimentPlan presentation and normalization.

The logic is factored out of trace_render.py, cycles/experiment.py, and
scripts/render_hypothesis_connected.py so every renderer shares one field list, one design
vocabulary, and one placeholder rule. Pure deterministic string/tree logic, so hermetic fixtures +
exact assertions."""

from __future__ import annotations

from src.experiment_plan_render import (
    DESIGN_PLAIN_LANGUAGE,
    EXPERIMENT_PLAN_FIELDS,
    NO_EVIDENCE_PLACEHOLDERS,
    collect_evidence_references,
    optional_evidence_id,
    plain_design_text,
    real_evidence_id,
)
from src.graph_store import (
    ExperimentDesign,
    ExperimentPlan,
    GroundingItem,
    MaterialItem,
)


# --- (a) canonical ordered display-field definition list ---------------------------------
def test_experiment_plan_fields_is_ordered_with_unique_labeled_keys():
    keys = [key for key, _ in EXPERIMENT_PLAN_FIELDS]
    assert keys and len(keys) == len(set(keys))
    assert all(label.strip() for _, label in EXPERIMENT_PLAN_FIELDS)


# --- Design-enum to plain-language mapping -----------------------------------------------
def test_every_design_enum_has_non_empty_plain_language():
    for design in ExperimentDesign:
        text = plain_design_text({"design": design.value})
        assert text.strip()
        assert text == DESIGN_PLAIN_LANGUAGE[design.value]


def test_unknown_design_value_renders_without_crashing():
    text = plain_design_text({"design": "qualitative_case_study"})
    assert text == "Use a qualitative case study design."
    assert plain_design_text({}) == "Use a  design."  # no design at all: no crash either


def test_plain_design_text_appends_the_rationale_when_present():
    text = plain_design_text({"design": "ablation", "design_rationale": "isolates one factor"})
    assert text == (
        "Use an ablation: remove or vary one component at a time and measure how the result "
        "changes. In this plan: isolates one factor"
    )


# --- (c) placeholder evidence-id normalization (moved from cycles/experiment.py) ---------
def test_placeholder_evidence_ids_are_stripped_everywhere():
    for token in NO_EVIDENCE_PLACEHOLDERS:
        assert real_evidence_id(token) == ""
        assert real_evidence_id(token.upper()) == ""
        assert optional_evidence_id(token) is None
    assert real_evidence_id(None) == ""
    assert real_evidence_id("ev_001") == "ev_001"
    assert optional_evidence_id("ev_001") == "ev_001"

    plan = {
        "grounding": [
            {"evidence_id": "none_retrieved", "quote_span": "no source"},
            {"evidence_id": "ev_001", "quote_span": "real"},
        ],
        "materials_or_data": [{"item": "x", "evidence_id": "N/A"}],
        "metrics": [{"metric": "y", "evidence_id": "null"}],
    }
    refs = collect_evidence_references(plan)
    assert refs == [("grounding[1].evidence_id", "ev_001")]


# --- (d) recursive evidence-reference collector -------------------------------------------
def test_nested_evidence_ids_found_with_path_labels():
    plan = {
        "grounding": [{"evidence_id": "ev_001", "quote_span": "a"}],
        "materials_or_data": [{"item": "x", "evidence_id": "ev_002"}],
        "metrics": [{"metric": "y", "evidence_id": "ev_003"}],
        # controls_and_confounders carries no evidence_id in today's schema; proves the walker
        # is genuinely recursive, not hardcoded to grounding/materials_or_data/metrics.
        "controls_and_confounders": [
            {"factor": "model size", "from_graph": True, "handling": "hold fixed",
             "evidence_id": "ev_004"},
        ],
    }
    refs = collect_evidence_references(plan)
    assert refs == [
        ("grounding[0].evidence_id", "ev_001"),
        ("materials_or_data[0].evidence_id", "ev_002"),
        ("metrics[0].evidence_id", "ev_003"),
        ("controls_and_confounders[0].evidence_id", "ev_004"),
    ]


def test_duplicate_evidence_ids_keep_every_citing_path_label():
    plan = {
        "grounding": [{"evidence_id": "ev_001", "quote_span": "a"}],
        "materials_or_data": [{"item": "x", "evidence_id": "ev_001"}],
    }
    refs = collect_evidence_references(plan)
    assert refs == [
        ("grounding[0].evidence_id", "ev_001"),
        ("materials_or_data[0].evidence_id", "ev_001"),
    ]
    assert {eid for _, eid in refs} == {"ev_001"}  # one unique id...
    assert len(refs) == 2                          # ...cited from two distinct paths, both kept


def test_collect_evidence_references_accepts_a_pydantic_model():
    plan = ExperimentPlan(
        hypothesis_under_test="e",
        design=ExperimentDesign.ABLATION,
        grounding=[GroundingItem(evidence_id="ev_001", quote_span="q")],
        materials_or_data=[MaterialItem(item="x", evidence_id="none_retrieved")],
    )
    refs = collect_evidence_references(plan)
    assert refs == [("grounding[0].evidence_id", "ev_001")]
